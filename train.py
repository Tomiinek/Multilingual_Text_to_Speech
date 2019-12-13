import os
import time
import datetime
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import TextToSpeechDatasetCollection, TextToSpeechCollate
from params.params import Params as hp
from utils import audio, text
from modules.tacotron2 import Tacotron, TacotronLoss
from utils.logging import Logger
from utils.optimizers import Ranger


def to_gpu(x):
    if x is None: return x
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def cos_decay(global_step, decay_steps):
    global_step = min(global_step, decay_steps)
    return 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))


def train(logging_start_epoch, epoch, data, model, criterion, optimizer):
    model.train() 
    learning_rate = optimizer.param_groups[0]['lr']
    done = 0
    for i, batch in enumerate(data):     
        start_time = time.time() 
        global_step = done + epoch * len(data)
        optimizer.zero_grad() 

        # Parse batch
        batch = list(map(to_gpu, batch))
        src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

        # Get teacher forcing ratio
        if hp.constant_teacher_forcing: tf = hp.teacher_forcing
        else: tf = cos_decay(max(global_step - hp.teacher_forcing_start_steps, 0), hp.teacher_forcing_steps)

        # Run the model
        post_pred, pre_pred, stop_pred, alignment = model(src, src_len, trg_mel, trg_len, spkrs, langs, tf)
        
        # Evaluate loss function
        post_trg = trg_lin if hp.predict_linear else trg_mel
        loss, batch_losses = criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, post_trg, stop_pred, stop_trg, alignment)
        
        # Comptute gradients and make a step
        loss.backward()      
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clipping)
        optimizer.step()   
        
        # Logg training progress
        if not hp.guided_attention_loss: batch_losses.pop('guided_att')
        if epoch >= logging_start_epoch:
            Logger.training(global_step, batch_losses, gradient, learning_rate, time.time() - start_time) 
        done += 1 
    

def evaluate(epoch, data, model, criterion):     
    model.eval()
    mcd, mcd_count = 0, 0
    eval_losses = {}
    with torch.no_grad():  
        for i, batch in enumerate(data):

            # Parse batch
            batch = map(to_gpu, batch)
            src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

            # Run the model (twice, with and without teacher forcing)
            post_pred, pre_pred, stop_pred, alignment = model(src, src_len, trg_mel, trg_len, spkrs, langs, 1.0)
            post_pred_0, _, stop_pred_0, alignment_0 = model(src, src_len, trg_mel, trg_len, spkrs, langs, 0.0)
            stop_pred_probs = torch.sigmoid(stop_pred_0)

            # Evaluate loss function
            post_trg = trg_lin if hp.predict_linear else trg_mel
            loss, batch_losses = criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, post_trg, stop_pred, stop_trg, alignment)
            
            # Compute mel cepstral distorsion
            for j, (gen, ref, stop) in enumerate(zip(post_pred_0, trg_mel, stop_pred_probs)):
                stop_idxes = np.where(stop.cpu().numpy() > 0.5)[0]
                stop_idx = min(np.min(stop_idxes) + hp.stop_frames, gen.size()[1]) if len(stop_idxes) > 0 else gen.size()[1]
                gen = gen[:, :stop_idx].data.cpu().numpy()
                ref = ref[:, :trg_len[j]].data.cpu().numpy()
                if hp.normalize_spectrogram:
                    gen = audio.denormalize_spectrogram(gen, not hp.predict_linear)
                    ref = audio.denormalize_spectrogram(ref, True)
                if hp.predict_linear: gen = audio.linear_to_mel(gen)
                mcd = (mcd_count * mcd + audio.mel_cepstral_distorision(gen, ref, 'stretch')) / (mcd_count+1)
                mcd_count += 1

            for k, v in batch_losses.items(): 
                eval_losses[k] = v + eval_losses[k] if k in eval_losses else v 
    
    # Normalize loss per batch
    for k in eval_losses.keys():
        eval_losses[k] /= len(data)

    # Logg evaluation
    if not hp.guided_attention_loss: eval_losses.pop('guided_att')
    Logger.evaluation(epoch+1, eval_losses, mcd, src_len, trg_len, src, post_trg, post_pred, post_pred_0, stop_pred_probs, stop_trg, alignment_0)
    
    return sum(eval_losses.values())


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


if __name__ == '__main__':
    import argparse
    import os
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default=".", help="Base directory of the project.")
    parser.add_argument('--fine_tuning', action='store_true')
    parser.add_argument("--checkpoint", type=str, default=None, help="Name of the initial checkpoint.")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints", help="Base directory of checkpoints.")
    parser.add_argument("--data_root", type=str, default="data", help="Base directory of datasets.")
    parser.add_argument("--flush_seconds", type=int, default=60, help="How often to flush pending summaries to tensorboard.")
    parser.add_argument('--hyper_parameters', type=str, default="train_en", help="Name of the hyperparameters file.")
    parser.add_argument('--logging_start', type=int, default=1, help="First epoch to be logged")
    parser.add_argument('--max_gpus', type=int, default=2, help="Maximal number of GPUs of the local machine to use.")
    parser.add_argument('--loader_workers', type=int, default=2, help="Number of subprocesses to use for data loading.")
    args = parser.parse_args()

    # set up seeds and the target torch device
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare directory for checkpoints 
    checkpoint_dir = os.path.join(args.base_directory, args.checkpoint_root)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # load checkpoint (dict) with saved hyper-parameters (let some of them be overwritten because of fine-tuning)
    if args.checkpoint:
        checkpoint = os.path.join(checkpoint_dir, args.checkpoint)
        checkpoint_state = torch.load(checkpoint)
        hp.load_state_dict(checkpoint_state['parameters'])      
        used_input_characters = hp.phonemes if hp.use_phonemes else hp.characters

    # load hyperparameters
    hp_path = os.path.join(args.base_directory, 'params', f'{args.hyper_parameters}.json')
    hp.load(hp_path)

    # find characters which were not used in the input embedding of checkpoint
    if args.fine_tuning: 
        new_input_characters = hp.phonemes if hp.use_phonemes else hp.characters
        extra_input_characters = [x for x in new_input_characters if x not in used_input_characters]
        ordered_new_input_characters = used_input_characters + ''.join(extra_input_characters)
        if hp.use_phonemes: hp.phonemes = ordered_new_input_characters
        else: hp.characters = ordered_new_input_characters

    # load dataset
    dataset = TextToSpeechDatasetCollection(os.path.join(args.data_root, hp.dataset))
    train_data = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True, shuffle=True, collate_fn=TextToSpeechCollate(), num_workers=args.loader_workers)
    eval_data = DataLoader(dataset.dev, batch_size=hp.batch_size, drop_last=False, shuffle=False, collate_fn=TextToSpeechCollate(), num_workers=args.loader_workers)

    # acquire dataset-dependent constatnts, these should probably be the same while going from checkpoint
    if not args.checkpoint:
        # compute per-channel constants for spectrogram normalization
        hp.mel_normalize_mean, hp.mel_normalize_variance = dataset.train.get_normalization_constants(True)
        hp.lin_normalize_mean, hp.lin_normalize_variance = dataset.train.get_normalization_constants(False)
        # find out number of unique speakers and languages (because of embedding dimension)
        hp.speaker_number = 0 if not hp.multi_speaker else dataset.train.get_num_speakers()
        hp.language_number = 0 if not hp.multi_language else len(hp.languages)

    # set the input charset to match characters of the checkpoint (in order to load model) 
    if args.fine_tuning: 
        if hp.use_phonemes: hp.phonemes = used_input_characters
        else: hp.characters = used_input_characters

    # instantiate model, loss function, optimizer and learning rate scheduler
    if torch.cuda.is_available(): 
        model = Tacotron().cuda()
        if args.max_gpus > 1 and torch.cuda.device_count() > 1:
            model = DataParallelPassthrough(model, device_ids=list(range(args.max_gpus)))    
    else: model = Tacotron()

    # instantiate optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    #optimizer = Ranger(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hp.learning_rate_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hp.learning_rate_decay_each / len(train_data), gamma=hp.learning_rate_decay)
    criterion = TacotronLoss(hp.guided_attention_steps, hp.guided_attention_toleration, hp.guided_attention_gain)

    # load model weights and optimizer, scheduler states from checkpoint state dictionary
    initial_epoch = 0
    if args.checkpoint:
        model.load_state_dict(checkpoint_state['model'])
        if not args.fine_tuning:
            initial_epoch = checkpoint_state['epoch'] + 1
            optimizer.load_state_dict(checkpoint_state['optimizer'])
            scheduler.load_state_dict(checkpoint_state['scheduler'])
        else:
            # make speaker and language embeddings constant
            hp.embedding_type = "constant"
            if hp.multi_speaker:
                embedding = model._speaker_embedding.weight.mean(dim=0)
                model._speaker_embedding = ConstantEmbedding(embedding)
            if hp.multi_language:
                embedding = model._language_embedding.weight.mean(dim=0)
                model._language_embedding = ConstantEmbedding(embedding)
            # enlarge the input embedding to fit all new characters
            if hp.use_phonemes: hp.phonemes = ordered_new_input_characters
            else: hp.characters = ordered_new_input_characters
            num_news = len(extra_input_characters)
            input_embedding = model._embedding.weight     
            with torch.no_grad():
                input_embedding = torch.nn.functional.pad(input_embedding, (0,0,0, num_news))
                torch.nn.init.xavier_uniform_(input_embedding[-num_news:, :])
            model._embedding = torch.nn.Embedding.from_pretrained(input_embedding, padding_idx=0, freeze=False)
            
    # initialize logger
    log_dir = os.path.join(args.base_directory, "logs", f'{hp.version}-{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    Logger.initialize(log_dir, args.flush_seconds)

    # training loop
    best_eval = float('inf')
    for epoch in range(initial_epoch, hp.epochs):
        train(args.logging_start, epoch, train_data, model, criterion, optimizer)
        criterion.update_states(len(train_data))  
        if hp.learning_rate_decay_start - hp.learning_rate_decay_each < epoch * len(train_data):
            scheduler.step()
        eval_loss = evaluate(epoch, eval_data, model, criterion)   
        if (epoch + 1) % hp.checkpoint_each_epochs == 0:
            # save checkpoint together with hyper-parameters, optimizer and scheduler states
            checkpoint_file = f'{checkpoint_dir}/{hp.version}_loss-{epoch}-{eval_loss:2.3f}'
            state_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'parameters': hp.state_dict()
            }
            torch.save(state_dict, checkpoint_file)

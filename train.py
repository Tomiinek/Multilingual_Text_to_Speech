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


def train(epoch, data, model, criterion, optimizer):
    model.train() 
    learning_rate = optimizer.param_groups[0]['lr']
    done = 0
    for i, batch in enumerate(data):     
        start_time = time.time() 
        batch = list(map(to_gpu, batch))
        src_len, src, trg_mel_spec, trg_lin_spec, trg_stop, trg_len, spkrs = batch
        optimizer.zero_grad()         
        if hp.constant_teacher_forcing:
            teacher_forcing_ratio = hp.teacher_forcing
        else:
            global_step = done + epoch * len(data)
            teacher_forcing_ratio = cos_decay(max(global_step - hp.teacher_forcing_start_steps, 0), hp.teacher_forcing_steps)
        post_prediction, pre_prediction, stop, alignment = model(src, src_len, trg_mel_spec, trg_len, spkrs, None, teacher_forcing_ratio)
        post_trg_spec = trg_lin_spec if hp.predict_linear else trg_mel_spec
        loss, batch_losses = criterion(src_len, pre_prediction, trg_mel_spec, post_prediction, post_trg_spec, trg_len, stop, trg_stop, alignment)
        loss.backward()      
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clipping)
        optimizer.step()   
        if not hp.guided_attention_loss: 
            batch_losses.pop('guided_att')
        Logger.training(done + epoch * len(data), batch_losses, gradient, learning_rate, time.time() - start_time) 
        done += 1 
    

def evaluate(epoch, data, model, criterion, teacher_forcing):      
    model.eval()
    eval_losses = {}
    with torch.no_grad():   
        for i, item in enumerate(data):
            item = map(to_gpu, item)
            src_len, src, trg_mel_spec, trg_lin_spec, trg_stop, trg_len, spkrs = list(item)
            post_prediction, pre_prediction, stop, alignment = model(src, src_len, trg_mel_spec, trg_len, spkrs, None, teacher_forcing)
            post_trg_spec = trg_lin_spec if hp.predict_linear else trg_mel_spec
            loss, batch_losses = criterion(src_len, pre_prediction, trg_mel_spec, post_prediction, post_trg_spec, trg_len, stop, trg_stop, alignment)
            for k, v in batch_losses.items(): 
                eval_losses[k] = v + eval_losses[k] if k in eval_losses else v 
    for k in eval_losses.keys():
        eval_losses[k] /= len(data)
    if not hp.guided_attention_loss: 
        eval_losses.pop('guided_att')
    Logger.evaluation(f"eval_{teacher_forcing}", epoch+1, eval_losses, src, trg_lin_spec, post_prediction, trg_len, src_len, trg_stop, torch.sigmoid(stop), alignment)
    return sum(eval_losses.values())


def load_checkpoint(checkpoint, model, optimizer, scheduler):
    state = torch.load(checkpoint)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    hp.load_state_dict(sttate['parameters'])
    return state['epoch']


def save_checkpoint(checkpoint_path, epoch, model, optimizer, sheduler):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': sheduler.state_dict(),
        'parameters': hp.state_dict()
    }
    torch.save(state_dict, checkpoint_path)


if __name__ == '__main__':
    import argparse
    import os
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default=".", help="Base directory of the project.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Name of the initial checkpoint.")
    parser.add_argument("--data_root", type=str, default="data", help="Base directory of datasets.")
    parser.add_argument("--flush_seconds", type=int, default=60, help="How often to flush pending summaries to tensorboard.")
    parser.add_argument('--hyper_parameters', type=str, default="train_en", help="Name of the hyperparameters file.")
    parser.add_argument('--max_gpus', type=int, default=2, help="Maximal number of GPUs of the local machine to use.")
    args = parser.parse_args()

    # load hyperparameters
    hp_path = os.path.join(args.base_directory, 'params', f'{args.hyper_parameters}.json')
    hp.load(hp_path)

    # prepare directory for checkpoints 
    checkpoint_dir = os.path.join(args.base_directory, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # initialize logger
    log_dir = os.path.join(args.base_directory, "logs", f'{hp.version}-{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    Logger.initialize(log_dir, args.flush_seconds)

    # set up seeds and the target torch device
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = hp.cudnn_enabled
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load dataset
    dataset = TextToSpeechDatasetCollection(os.path.join(args.data_root, hp.dataset))
    train_data = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True, shuffle=True, collate_fn=TextToSpeechCollate())
    eval_data = DataLoader(dataset.dev, batch_size=hp.batch_size, drop_last=False, shuffle=False, collate_fn=TextToSpeechCollate())

    # compute per-channel constants for spectrogram normalization
    hp.mel_normalize_mean, hp.mel_normalize_variance = dataset.train.get_normalization_constants(True)
    hp.lin_normalize_mean, hp.lin_normalize_variance = dataset.train.get_normalization_constants(False)

    # find out number of unique speakers and languages (because of embedding dimension)
    hp.speaker_number = 0 if not hp.multi_speaker else dataset.train.get_num_speakers()
    hp.language_number = 0 if not hp.multi_language else dataset.train.get_num_languages()
    
    # instantiate model, loss function, optimizer and learning rate scheduler
    if torch.cuda.is_available(): 
        model = Tacotron().cuda()
        if args.max_gpus > 1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.max_gpus)))    
    else: model = Tacotron()

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    #optimizer = Ranger(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hp.learning_rate_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hp.learning_rate_decay_each / len(train_data), gamma=hp.learning_rate_decay)
    criterion = TacotronLoss(hp.guided_attention_steps, hp.guided_attention_toleration, hp.guided_attention_gain)

    # load checkpoint
    if args.checkpoint:
        checkpoint = os.path.join(checkpoint_dir, args.checkpoint)
        initial_epoch = load_checkpoint(checkpoint, model, optimizer, scheduler) + 1
    else: initial_epoch = 0

    # training loop
    best_eval = float('inf')
    for epoch in range(initial_epoch, hp.epochs):
        train(epoch, train_data, model, criterion, optimizer)
        criterion.update_states(len(train_data))  
        if hp.learning_rate_decay_start < epoch * len(train_data):
            scheduler.step()
        # evaluate without teacher forcing
        evaluate(epoch, eval_data, model, criterion, 0.0)   
        # evaluate with teacher forcing
        eval_loss = evaluate(epoch, eval_data, model, criterion, 1.0)   
        if (epoch + 1) % hp.checkpoint_each_epochs == 0:
            checkpoint_file = f'{checkpoint_dir}/{hp.version}_loss-{epoch}-{eval_loss:2.3f}'
            save_checkpoint(checkpoint_file, epoch, model, optimizer, scheduler)

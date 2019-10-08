import os
import time
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import TextToSpeechDatasetCollection, TextToSpeechCollate
from params.params import Params as hp
from utils import audio, text
from modules.tacotron2 import Tacotron, StopTokenLoss, MelLoss, GuidedAttentionLoss
from utils.logging import Logger


def to_gpu(x):
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def train(epoch, data, model, optimizer, criterions):
    model.train() 
    learning_rate = optimizer.param_groups[0]['lr']
    done = 0
    epoch_losses = {}
    for k in criterions.keys():
        epoch_losses[k] = 0
    start_time = time.time()  
    for i, batch in enumerate(data):      
        batch = list(map(to_gpu, batch))
        src_len, src, trg_spec, trg_stop, trg_len = batch
        optimizer.zero_grad()     
        prediction, residual_prediction, stop, alignment = model(src, src_len, trg_spec, teacher_forcing=True)
        batch_losses = {
            'mel_pre' : criterions['mel_pre'](prediction, trg_spec, trg_len),
            'mel_res' : criterions['mel_res'](residual_prediction, trg_spec, trg_len),
            'stop_token' : criterions['stop_token'](stop, trg_stop),
            'guided_att' : criterions['guided_att'](alignment, src_len, trg_len)
        }
        loss = sum(batch_losses.values())
        loss.backward()      
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clipping)
        optimizer.step()   
        for k in criterions.keys():
            epoch_losses[k] += batch_losses[k].item()
        done += 1       
        if i % args.skip_logging == 0:
            reduced_epoch_losses = {}
            for k in criterions.keys():
                reduced_epoch_losses[k] = epoch_losses[k] / done
            Logger.training_progress(epoch+1, done + epoch * len(data), reduced_epoch_losses, learning_rate, done / len(data)) 
        break
    reduced_epoch_losses = {}
    for k in criterions.keys():
        reduced_epoch_losses[k] = epoch_losses[k] / len(data)        
    Logger.training(epoch+1, reduced_epoch_losses, learning_rate, int(time.time() - start_time))


def evaluate(epoch, data, model, criterions):      
    model.eval()
    learning_rate = optimizer.param_groups[0]['lr']
    eval_losses = {}
    for k in criterions.keys():
        eval_losses[k] = 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    with torch.no_grad():   
        for i, item in enumerate(data):
            item = map(to_gpu, item)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            src_len, src, trg_spec, trg_stop, trg_len = list(item)
            prediction, residual_prediction, stop, alignment = model(src, src_len, trg_spec, teacher_forcing=False)
            eval_losses['mel_pre'] += criterions['mel_pre'](prediction, trg_spec, trg_len)
            eval_losses['mel_res'] += criterions['mel_res'](residual_prediction, trg_spec, trg_len)
            eval_losses['stop_token'] += criterions['stop_token'](stop, trg_stop)
            eval_losses['guided_att'] += criterions['guided_att'](alignment, src_len, trg_len) 
            break
    for k in criterions.keys():
        eval_losses[k] /= len(data)
    Logger.evaluation(epoch+1, eval_losses, learning_rate, trg_spec, prediction, trg_stop, torch.sigmoid(stop), alignment)
    return sum(eval_losses.values())


def load_checkpoint(checkpoint, model, optimizer, scheduler):
    state = torch.load(checkpoint)
    model.load_state_dict(state['optimizer'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    return state['epoch']


def save_checkpoint(checkpoint_path, epoch, model, optimizer, sheduler):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': sheduler.state_dict()
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
    parser.add_argument("--evaluate_each", type=int, default=1, help="Evaluate each this number epochs.")
    parser.add_argument("--flush_seconds", type=int, default=60, help="How often to flush pending summaries to tensorboard.")
    parser.add_argument('--hyper_parameters', type=str, default="train_en", help="Name of the hyperparameters file.")
    parser.add_argument('--max_gpus', type=int, default=2, help="Maximal number of GPUs of the local machine to use.")
    parser.add_argument("--min_checkpoint_loss", type=float, default=10000, help="Minimal required loss of a checkpoint to save.")
    parser.add_argument("--skip_logging", type=int, default=5, help="Log each of these steps.")
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
    Logger.initialize(log_dir, args.flush_seconds, to_console=False)

    # set up seeds and the target torch device
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = hp.cudnn_enabled
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # instantiate model, loss function, optimizer and learning rate scheduler
    if torch.cuda.is_available(): 
        model = Tacotron().cuda()
        if args.max_gpus > 1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.max_gpus)))    
    else: model = Tacotron()

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    # TODO: scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hp.learning_rate_decay)
    criterions = torch.nn.ModuleDict([
        ['mel_res', MelLoss()],
        ['mel_pre', MelLoss()],
        ['stop_token', StopTokenLoss()],
        ['guided_att', GuidedAttentionLoss(hp.guided_attention_toleration, hp.guided_attention_gain)]
    ])

    # load checkpoint
    if args.checkpoint:
        checkpoint = os.path.join(args.checkpoint, checkpoint_dir)
        initial_epoch = load_checkpoint(checkpoint, model, optimizer, scheduler) + 1
    else: initial_epoch = 0

    # load dataset
    dataset = TextToSpeechDatasetCollection(os.path.join(args.data_root, hp.dataset))
    train_data = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True, shuffle=True, collate_fn=TextToSpeechCollate())
    eval_data = DataLoader(dataset.dev, batch_size=hp.batch_size, drop_last=False, shuffle=False, collate_fn=TextToSpeechCollate())

    # training loop
    best_eval = float('inf')
    for epoch in range(initial_epoch, hp.epochs):
        train(epoch, train_data, model, optimizer, criterions)
        for c in criterions.values():
            c.update_state()    
        scheduler.step()
        if epoch % args.evaluate_each != args.evaluate_each - 1:
            Logger.skipped_evaluation()
            continue
        eval_loss = evaluate(epoch, eval_data, model, criterions)      
        if eval_loss < best_eval and eval_loss < args.min_checkpoint_loss:
            best_eval = eval_loss
            checkpoint_file = f'{checkpoint_dir}/{hp.version}_loss-{eval_loss:2.3f}'
            save_checkpoint(checkpoint_file, epoch, model, optimizer, scheduler)

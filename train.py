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
from modules.tacotron2 import Tacotron, StopTokenLoss, MelLoss, GuidedAttentionLoss
from utils.logging import Logger
from utils.optimizers import Ranger


def to_gpu(x):
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def cos_decay(global_step, decay_steps):
    global_step = min(global_step, decay_steps)
    return 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))


def train(epoch, data, model, optimizer, criterions):
    model.train() 
    learning_rate = optimizer.param_groups[0]['lr']
    done = 0
    for i, batch in enumerate(data):     
        start_time = time.time() 
        batch = list(map(to_gpu, batch))
        src_len, src, trg_spec, trg_stop, trg_len = batch
        optimizer.zero_grad()     
        global_step = done + epoch * len(data)
        if hp.constant_teacher_forcing:
            teacher_forcing_ratio = hp.teacher_forcing
        else:
            teacher_forcing_ratio = cos_decay(max(global_step - hp.teacher_forcing_start_steps, 0), hp.teacher_forcing_steps)
        prediction, residual_prediction, stop, alignment = model(src, src_len, trg_spec, trg_len, teacher_forcing_ratio)
        batch_losses = {
            'mel_pre' : criterions['mel_pre'](prediction, trg_spec, trg_len),
            'mel_res' : criterions['mel_res'](residual_prediction, trg_spec, trg_len),
            'stop_token' : criterions['stop_token'](stop, trg_stop),
            'guided_att' : criterions['guided_att'](alignment, src_len, trg_len)
        }
        if not hp.guided_attention_loss: criterions['guided_att'] = 0
        loss = sum(batch_losses.values())
        loss.backward()      
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clipping)
        optimizer.step()   
        if not hp.guided_attention_loss: 
            batch_losses.pop('guided_att')
        Logger.training(done + epoch * len(data), batch_losses, gradient, learning_rate, time.time() - start_time) 
        done += 1 
    

def evaluate(epoch, data, model, criterions, teacher_forcing):      
    model.eval()
    eval_losses = {}
    for k in criterions.keys():
        eval_losses[k] = 0
    with torch.no_grad():   
        for i, item in enumerate(data):
            item = map(to_gpu, item)
            src_len, src, trg_spec, trg_stop, trg_len = list(item)
            prediction, residual_prediction, stop, alignment = model(src, src_len, trg_spec, trg_len, teacher_forcing)
            eval_losses['mel_pre'] += criterions['mel_pre'](prediction, trg_spec, trg_len)
            eval_losses['mel_res'] += criterions['mel_res'](residual_prediction, trg_spec, trg_len)
            eval_losses['stop_token'] += criterions['stop_token'](stop, trg_stop)
            eval_losses['guided_att'] += criterions['guided_att'](alignment, src_len, trg_len) 
    for k in criterions.keys():
        eval_losses[k] /= len(data)
    if not hp.guided_attention_loss: 
        eval_losses.pop('guided_att')
    Logger.evaluation(f"eval_{teacher_forcing}", epoch+1, eval_losses, src, trg_spec, prediction, trg_len, src_len, trg_stop, torch.sigmoid(stop), alignment)
    return sum(eval_losses.values())


def load_checkpoint(checkpoint, model, optimizer, scheduler):
    state = torch.load(checkpoint)
    model.load_state_dict(state['model'])
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
    
    # instantiate model, loss function, optimizer and learning rate scheduler
    if torch.cuda.is_available(): 
        model = Tacotron().cuda()
        if args.max_gpus > 1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.max_gpus)))    
    else: model = Tacotron()

    #optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    optimizer = Ranger(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hp.learning_rate_decay)
    criterions = torch.nn.ModuleDict([
        ['mel_res', MelLoss()],
        ['mel_pre', MelLoss()],
        ['stop_token', StopTokenLoss()],
        ['guided_att', GuidedAttentionLoss(hp.guided_attention_toleration, hp.guided_attention_gain)]
    ])

    # load checkpoint
    if args.checkpoint:
        checkpoint = os.path.join(checkpoint_dir, args.checkpoint)
        initial_epoch = load_checkpoint(checkpoint, model, optimizer, scheduler) + 1
    else: initial_epoch = 0

    # load dataset
    dataset = TextToSpeechDatasetCollection(os.path.join(args.data_root, hp.dataset))
    train_data = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True, shuffle=True, collate_fn=TextToSpeechCollate())
    eval_data = DataLoader(dataset.dev, batch_size=hp.batch_size, drop_last=False, shuffle=False, collate_fn=TextToSpeechCollate())
    hp.normalize_mean, hp.normalize_variance = dataset.train.get_normalization_constants()
    #hp.normalize_mean, hp.normalize_variance = -69, 16
  
    # training loop
    best_eval = float('inf')
    for epoch in range(initial_epoch, hp.epochs):
        train(epoch, train_data, model, optimizer, criterions)
        for c in criterions.values():
            c.update_state()    
        if hp.learning_rate_decay_start < epoch * len(train_data):
            scheduler.step()
        # evaluate without teacher forcing
        evaluate(epoch, eval_data, model, criterions, 0.0)   
        # evaluate with teacher forcing
        eval_loss = evaluate(epoch, eval_data, model, criterions, 1.0)   
        if (epoch + 1) % hp.checkpoint_each == 0:
            checkpoint_file = f'{checkpoint_dir}/{hp.version}_loss-{eval_loss:2.3f}'
            save_checkpoint(checkpoint_file, epoch, model, optimizer, scheduler)

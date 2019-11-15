import sys
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from params.params import Params as hp
from utils import audio, text
from modules.tacotron2 import Tacotron


def remove_dataparallel_prefix(state_dict): 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def build_model(checkpoint):   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device)
    hp.load_state_dict(state['parameters'])
    model = Tacotron()
    model.load_state_dict(remove_dataparallel_prefix(state['model']))   
    model.to(device)
    return model


if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    parser.add_argument("--output", type=str, default=".", help="Path to output directory.")
    args = parser.parse_args()

    model = build_model(args.checkpoint)
    model.eval()

    inputs = [l.rstrip() for l in sys.stdin.readlines() if l]

    spectrograms = []
    for i in inputs:
        i = torch.LongTensor(text.to_sequence(i, use_phonemes=hp.use_phonemes))
        if torch.cuda.is_available(): i = i.cuda(non_blocking=True)
        spectrograms.append(model.inference(i).cpu().detach().numpy())

    for i, s in enumerate(spectrograms):
        s = audio.denormalize_spectrogram(s, not hp.predict_linear)
        w = audio.inverse_spectrogram(s, not hp.predict_linear)
        audio.save(w, os.path.join(args.output, f'{str(i).zfill(3)}-{datetime.now()}.wav'))
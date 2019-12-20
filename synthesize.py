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

    # Expected inputs is in case of
    # - mono-lingual and single-speaker model:  single input utterance per line
    # - otherwise:                              single input utterance|speaker|language
    inputs = [l.rstrip().split('|') for l in sys.stdin.readlines() if l]

    spectrograms = []
    for i in inputs:
        t = torch.LongTensor(text.to_sequence(i[0], use_phonemes=hp.use_phonemes))
        l = torch.LongTensor([hp.languages.index(i[2])]) if hp.multi_language else None
        s = torch.LongTensor([hp.unique_speakers.index(i[1])]) if hp.multi_speaker else None

        if torch.cuda.is_available(): 
            t = t.cuda(non_blocking=True)
            if l: l = l.cuda(non_blocking=True)
            if s: s = s.cuda(non_blocking=True)

        spectrograms.append(model.inference(t, speaker=s, language=l).cpu().detach().numpy())

    for i, s in enumerate(spectrograms):
        s = audio.denormalize_spectrogram(s, not hp.predict_linear)
        w = audio.inverse_spectrogram(s, not hp.predict_linear)
        audio.save(w, os.path.join(args.output, f'{str(i).zfill(3)}-{datetime.now()}.wav'))
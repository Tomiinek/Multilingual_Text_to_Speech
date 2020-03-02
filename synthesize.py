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


def build_model(checkpoint, force_cpu=False):   
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    state = torch.load(checkpoint, map_location=device)
    hp.load_state_dict(state['parameters'])
    model = Tacotron()
    # remove_dataparallel_prefix(state['model'])
    model.load_state_dict(state['model'])   
    model.to(device)
    return model


if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    parser.add_argument("--output", type=str, default=".", help="Path to output directory.")
    parser.add_argument("--cpu", action='store_true', help="Force to run o CPU.")
    parser.add_argument("--save_spec", action='store_true', help="Saves also spectrograms if set.")
    parser.add_argument("--ignore_wav", action='store_true', help="Does not save waveforms if set.")
    args = parser.parse_args()

    print("Building model ...")

    model = build_model(args.checkpoint, args.cpu)
    model.eval()

    # Expected inputs is in case of
    # - mono-lingual and single-speaker model:  id|single input utterance per line
    # - otherwise:                              id|single input utterance|speaker|language
    # - with per-character language:            id|single input utterance|speaker|l1-(length of l1),l2-(length of l2),l1
    #                                           where the last language takes all remaining character
    #                                           exmaple: "guten tag jean-paul.|speaker|de-10,fr-9,de"
    # id is used as output file name!

    inputs = [l.rstrip().split('|') for l in sys.stdin.readlines() if l]

    spectrograms = []
    for i, item in enumerate(inputs):

        print(f'Synthesizing({i+1}/{len(inputs)}): "{item[1]}"')

        clean_text = item[1]

        if not hp.use_punctuation: 
            clean_text = text.remove_punctuation(clean_text)
        if not hp.case_sensitive: 
            clean_text = text.to_lower(clean_text)
        if hp.remove_multiple_wspaces: 
            clean_text = text.remove_odd_whitespaces(clean_text)

        t = torch.LongTensor(text.to_sequence(clean_text, use_phonemes=hp.use_phonemes))

        if hp.multi_language:     
            l_tokens = item[3].split(',')
            t_length = len(item[1]) + 1
            l = []
            for token in l_tokens:
                l_d = token.split('-')
                language = hp.languages.index(l_d[0])
                language_length = (int(l_d[1]) if len(l_d) == 2 else t_length)
                l += [language] * language_length
                t_length -= language_length     
            l = torch.LongTensor([l])
        else:
            l = None

        s = torch.LongTensor([hp.unique_speakers.index(item[2])]) if hp.multi_speaker else None

        if torch.cuda.is_available() and not args.cpu: 
            t = t.cuda(non_blocking=True)
            if l: l = l.cuda(non_blocking=True)
            if s: s = s.cuda(non_blocking=True)

        s = model.inference(t, speaker=s, language=l).cpu().detach().numpy()
        s = audio.denormalize_spectrogram(s, not hp.predict_linear)

        if args.save_spec:
            np.save(os.path.join(args.output, f'{item[0]}.npy'), s)

        if not args.ignore_wav:
            w = audio.inverse_spectrogram(s, not hp.predict_linear)
            audio.save(w, os.path.join(args.output, f'{item[0]}.wav'))

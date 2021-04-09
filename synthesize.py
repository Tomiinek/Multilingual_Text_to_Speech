import sys
import os
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch

from utils import audio, text
from utils import build_model
from params.params import Params as hp
from modules.tacotron2 import Tacotron


"""

******************************************************** INSTRUCTIONS ********************************************************
*                                                                                                                            *
*   The script expects input utterances on stdin, every example on a separate line.                                          *
*                                                                                                                            *
*   Different models expect different lines, some have to specify speaker, language, etc.:                                   *
*   ID is used as name of the output file.                                                                                   *
*   Speaker and language IDs have to be the same as in parameters (see hp.languages and hp.speakers).                        *
*                                                                                                                            *
*   MONO-lingual and SINGLE-speaker:    id|single input utterance per line                                                   *
*   OTHERWISE                           id|single input utterance|speaker|language                                           *
*   OTHERWISE with PER-CHARACTER lang:  id|single input utterance|speaker|l1-(length of l1),l2-(length of l2),l1             *
*                                           where the last language takes all remaining character                            *
*                                           exmaple: "01|guten tag jean-paul.|speaker|de-10,fr-9,de"                         *
*   OTHERWISE with accent control:      id|single input utterance|speaker|l1-(len1),l2*0.75:l3*0.25-(len2),l1                *
*                                           accent can be controlled by weighting per-language characters                    *
*                                           language codes must be separated by : and weights are assigned using '*number'   *
*                                           example: "01|guten tag jean-paul.|speaker|de-10,fr*0.75:de*0.25-9,de"            *
*                                           the numbers do not have to sum up to one because they are normalized later       *
*                                                                                                                            *
******************************************************************************************************************************

"""


def synthesize(model, input_data, force_cpu=False):

    item = input_data.split('|')
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
        t_length = len(clean_text) + 1
        l = []
        for token in l_tokens:
            l_d = token.split('-')
 
            language = [0] * hp.language_number
            for l_cw in l_d[0].split(':'):
                l_cw_s = l_cw.split('*')
                language[hp.languages.index(l_cw_s[0])] = 1 if len(l_cw_s) == 1 else float(l_cw_s[1])

            language_length = (int(l_d[1]) if len(l_d) == 2 else t_length)
            l += [language] * language_length
            t_length -= language_length     
        l = torch.FloatTensor([l])
    else:
        l = None

    s = torch.LongTensor([hp.unique_speakers.index(item[2])]) if hp.multi_speaker else None

    if torch.cuda.is_available() and not force_cpu: 
        t = t.cuda(non_blocking=True)
        if l is not None: l = l.cuda(non_blocking=True)
        if s is not None: s = s.cuda(non_blocking=True)

    s = model.inference(t, speaker=s, language=l).cpu().detach().numpy()
    s = audio.denormalize_spectrogram(s, not hp.predict_linear)

    return s


if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    parser.add_argument("--output", type=str, default=".", help="Path to output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Torch random seed.")
    parser.add_argument("--cpu", action='store_true', help="Force to run on CPU.")
    parser.add_argument("--save_spec", action='store_true', help="Saves also spectrograms if set.")
    parser.add_argument("--ignore_wav", action='store_true', help="Does not save waveforms if set.")
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Random seed set to {args.seed}")
        torch.manual_seed(args.seed)

    print("Building model ...")

    model = build_model(args.checkpoint, args.cpu)
    model.eval()

    #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Builded model with {total_params} parameters")

    inputs = [l.rstrip() for l in sys.stdin.readlines() if l]
    progress = tqdm(enumerate(inputs), total=len(inputs), desc='Synthesizing')

    spectrograms = []
    for i, item in progress:
        progress.set_postfix_str(item)

        item_id = item.split("|")[0]
        if item_id == "":
            item_id = i

        s = synthesize(model, item, args.cpu)

        if not os.path.exists(args.output):
            os.makedirs(args.output)

        if args.save_spec:
            np.save(os.path.join(args.output, f'{item_id}.npy'), s)

        if not args.ignore_wav:
            w = audio.inverse_spectrogram(s, not hp.predict_linear)
            audio.save(w, os.path.join(args.output, f'{item_id}.wav'))

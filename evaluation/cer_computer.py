import os
import json
import ast
import numpy as np
import scipy.stats
from utils import text

"""

**************************************** INSTRUCTIONS ***************************************
*                                                                                           *
*   Usage: python cer_computer.py --language german --model ground-truth                    *
*                                                                                           *
*   For each utterance in a meta-file, find the output of ASR and compute CER between       *
*   these two texts, saves into a file with basic statistics.                               *
*                                                                                           *
*********************************************************************************************

"""

def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]


def sample_cer(reference, hypothesis):
    _, (s, i, d) = levenshtein(reference, hypothesis)
    return (s + i + d) / len(reference)
        

def clean(text, case, punctuation):

    punctuations_out = '—「」、。，"(),.:;¿？：！《》“”?⑸¡!\\'
    punctuations_in  = '\'-'
  
    if not case:
        text = text.lower()

    if not punctuation:
        punct_re = '[' + punctuations_out + punctuations_in + ']'
        text = re.sub(punct_re.replace('-', '\-'), '', text)
        
    text = ' '.join(text.split())
    
    return text


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="Language to be synthesized.")
    parser.add_argument("--model", type=str, required=True, help="Model specific folder.")
    parser.add_argument("--where", type=str, required=True, help="Data specific folder.")
    parser.add_argument('--case_sensitive', action='store_true', help="Enable case sensitivity.")
    parser.add_argument("--punctuation", action='store_true', help="Enable punctuation.")
    args = parser.parse_args()

    cers = []

    meta_file = os.path.join(args.where, 'all_meta_files', f'{args.language}.txt')
    with open(meta_file, 'r', encoding='utf-8') as f:
        for l in f:
			
            tokens = l.rstrip().split('|')
            idx = tokens[0]
            if args.language == "japanese" or args.language == "chinese":
                truth = tokens[2]
            else:
                truth = tokens[1]

            asr_path = os.path.join(args.where, args.model, 'asr', args.language, f'{idx}.json')		
            if not os.path.exists(asr_path):
                print(f'Missing ASR results of {idx}!') 
                continue
           
            with open(asr_path, 'r') as df:   
                asr = ast.literal_eval(df.read())
            transcript = asr[0]["alternatives"][0]["transcript"]
			
            cer = sample_cer(
                    clean(truth, args.case_sensitive, args.punctuation), 
                    clean(transcript, args.case_sensitive, args.punctuation))

            if len(asr) > 1:
                all_transcripts = [h["alternatives"][0]["transcript"] for h in asr]
                all_transcripts = ''.join(all_transcripts)
                cer = min(cer, sample_cer(
                    clean(truth, args.case_sensitive, args.punctuation),
                    clean(all_transcripts, args.case_sensitive, args.punctuation)))

            cers.append((idx, cer))
			
    values = [x for i, x in cers]
    cer_mean = np.mean(values)
    cer_std = np.std(values)

    output_path = os.path.join(args.where, args.model, 'cer')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cer_lower, cer_upper = confidence_interval(values)

    output_file = os.path.join(output_path, f'{args.language}.txt')
    with open(output_file, 'w+', encoding='utf-8') as of:
        print(f'Total mean CER: {cer_mean}', file=of)
        print(f'Std. dev. of CER: {cer_std}', file=of)
        print(f'Conf. interval: ({cer_lower}, {cer_upper})', file=of)
        
        for i, c in cers:
            print(f'{i}|{c}', file=of)
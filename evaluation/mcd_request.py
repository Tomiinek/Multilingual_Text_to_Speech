import os
import librosa
import librosa.feature
import numpy as np
import scipy.stats
from fastdtw import fastdtw

"""

**************************************** INSTRUCTIONS ***************************************
*                                                                                           *
*   Usage: python mcd_computer.py --language german --model simple                          *
*                                                                                           *
*   For each utterance in a meta-file, find the ground-truth spectrogram and a synthesized  *
*   spectrogram and compute Mel Cepstral Distorsion of them, saves into a file with basic   *
*   statistics.                                                                             * 
*                                                                                           *
*********************************************************************************************

"""


def get_spectrogram_mfcc(S, num_mfcc):
    return librosa.feature.mfcc(n_mfcc=num_mfcc, S=(S/10))


def mel_cepstral_distorision(S1, S2, num_mfcc):
  
    def mcd(s1, s2):
        diff = s1 - s2
        return np.average(np.sqrt(np.sum(diff*diff, axis=0)))

    x, y = get_spectrogram_mfcc(S1, num_mfcc)[1:], get_spectrogram_mfcc(S2, num_mfcc)[1:]

    x, y = x.T, y.T
    _, path = fastdtw(x, y, dist=mcd)     
    pathx, pathy = map(list,zip(*path))    
    x, y = x[pathx].T, y[pathy].T     
        
    return mcd(x, y)


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
    parser.add_argument('--num_mfcc', type=int, default=13, help="Number of MFCC coefficients.")
    parser.add_argument("--where", type=str, required=True, help="Data specific folder.")
    args = parser.parse_args()

    mcds = []

    meta_file = os.path.join(args.where, 'all_meta_files', f'{args.language}.txt')
    with open(meta_file, 'r', encoding='utf-8') as f:
        for l in f:
			
            tokens = l.rstrip().split('|')
            idx = tokens[0]
            
            spec_path = os.path.join(args.where, args.model, 'spectrograms', args.language, f'{idx}.npy')		
            if not os.path.exists(spec_path):
                print(f'Missing spectrogram of {idx}!') 
                continue
            gen = np.load(spec_path)

            ref_path = os.path.join(args.where, 'ground-truth', 'spectrograms', f'{idx}.npy')	
            ref = np.load(ref_path)

            mcd = mel_cepstral_distorision(gen, ref, args.num_mfcc)

            mcds.append((idx, mcd))
			
    values = [x for i, x in mcds]
    mcd_mean = np.mean(values)
    mcd_std = np.std(values)

    output_path = os.path.join(args.where, args.model, 'mcd')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mcd_lower, mcd_upper = confidence_interval(values)

    output_file = os.path.join(output_path, f'{args.language}.txt')
    with open(output_file, 'w+', encoding='utf-8') as of:
        for i, c in mcds:
            print(f'{c}', file=of) # {i}|
        print(f'Total mean MCD: {mcd_mean}', file=of)
        print(f'Std. dev. of MCD: {mcd_std}', file=of)
        print(f'Conf. interval: ({mcd_lower}, {mcd_upper})', file=of)
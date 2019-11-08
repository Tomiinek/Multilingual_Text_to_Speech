from datetime import date
import sys
import os

from params.params import Params as hp
from utils import audio, text
from modules.tacotron2 import Tacotron


def build_model(checkpoint, model):   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device)
    hp.load_state_dict(state['parameters'])
    model = Tacotron()
    model.load_state_dict(state['model'])   
    model.to(device)
    return model


if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    parser.add_argument("--output", type=str, default=".", help="Path to output directory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.checkpoint)

    inputs = [x for x in sys.stdin]

    spectrograms = []
    for i in inputs:
        i = torch.LongTensor(text.to_sequence(i, use_phonemes=hp.use_phonemes))
        if torch.cuda.is_available(): i = i.cuda(non_blocking=True)
        spectrograms.append(model.inference(i).cpu().numpy())

    for i, s in enumerate(spectrograms):
        w = audio.inverse_spectrogram(s, not hp.predict_linear)
        audio.save(w, os.path.join(args.output, f'{str(i).zfill(3)}-{datetime.now()}.wav'))
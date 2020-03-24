import sys
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import audio, text
from utils import build_model, to_gpu
from params.params import Params as hp
from modules.tacotron2 import Tacotron
from dataset.dataset import TextToSpeechDataset, TextToSpeechDatasetCollection, TextToSpeechCollate
from utils.samplers import PerfectBatchSampler

"""

*********************************************** INSTRUCTIONS ************************************************
*                                                                                                           *
*   The script computes and saves ground-truth aligned (GTA) spectrograms of a dataset it was trained on.   * 
*   The GTA spectrograms are useful for training vocoders like WaveNet, WaveRNN or WaveGlow.                *
*   This script uses the standard DataLoader as main trining loop, you can specify speakers whose will be   *
*   utterances will be synthesized.                                                                         *
*   Different models expect different lines, some have to specify speaker, language, etc.:                  *
*   ID is used as name of the output file.                                                                  *
*   Speaker and language IDs have to be the same as in parameters (see hp.languages and hp.speakers).       *
*                                                                                                           *
*************************************************************************************************************

"""

if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default=".", help="Base directory of the project.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    parser.add_argument("--output", type=str, default="gta_output", help="Path to output directory.", required=True)
    parser.add_argument("--data_root", type=str, default="data", help="Base directory of datasets.")
    parser.add_argument("--speakers", nargs='+', type=str, help="List of desired speakers.", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.", required=False)
    parser.add_argument("--loader_workers", type=int, default=1, help="Number of CPUs used by data loaders.", required=False)
    args = parser.parse_args()

    output_dir = os.path.join(args.base_directory, args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the model from checkpoint
    checkpoint_dir = os.path.join(args.base_directory, args.checkpoint)
    model = build_model(checkpoint_dir)
    model.eval()

    # Load dataset metafile   
    dataset = TextToSpeechDatasetCollection(os.path.join(args.data_root, hp.dataset))

    # Remove speakers we actualy do not want in the dataset
    speakers = [hp.unique_speakers.index(i) for i in args.speakers]
    filtered = [x for x in dataset.train.items if x["speaker"] in speakers]
    dataset.train.items = filtered

    # Prepare dataloaders
    if hp.multi_language and hp.balanced_sampling and hp.perfect_sampling:
        sampler = PerfectBatchSampler(dataset.train, hp.languages, args.batch_size, shuffle=False)
        data = DataLoader(dataset.train, batch_sampler=sampler, 
                          collate_fn=TextToSpeechCollate(False), num_workers=args.loader_workers)
    else:
        data = DataLoader(dataset.train, batch_size=args.batch_size, drop_last=False, shuffle=False,
                         collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)

    with torch.no_grad():   
        serial_number = 0      
        for i, batch in enumerate(data):

            batch = list(map(to_gpu, batch))             
            src, src_len, trg_mel, _, trg_len, _, spkrs, langs = batch

            # Run the model with enbaled teacher forcing (1.0)
            predictions = model(src, src_len, trg_mel, trg_len, spkrs, langs, 1.0)
            prediction = predictions[0].data.cpu().numpy()
        
            for idx in range(len(prediction)):
                speaker = spkrs[idx] if spkrs is not None else 0
                mel = prediction[idx, :, :trg_len[idx]]
                if hp.normalize_spectrogram:
                    mel = audio.denormalize_spectrogram(mel, not hp.predict_linear)           
                np.save(os.path.join(output_dir, f'{serial_number:05}-{speaker}.npy'), mel, allow_pickle=False)
                serial_number += 1

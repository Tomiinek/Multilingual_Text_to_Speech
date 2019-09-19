import os
import numpy as np
import collections
import torch
import random
import progressbar

import loaders
from utils import audio
from utils import text
from utils import logging
from training.params import Params as hp


class TextToSpeechDataset(torch.utils.data.Dataset):
    """Text to speech dataset.
    
    1) Load dataset metadata/data.
    2) Perform cleaning operations on the loaded utterances.
    3) Compute mel-spectrograms (cached).
    4) Compute phonemes (cached).
    5) Convert text into sequences of indices.
    """

    def __init__(self, name, root_dir, meta_files=None, cache_dir_spectrogram=None, cache_file_phonemes=None):
        """
        Keyword arguments:
            dataset (string): Dataset type, must match a method name in dataset.loader of the certain dataset. 
            root_dir (string): Root Directory of the dataset.
            meta_files (list of string, optional): Specific meta files to use.
            cache_dir_spectrogram (string): Directory with cached mel-spectrograms.
            cache_file_phonemes (string): Path to the file with cached phonemized uttrances.
        """
        random.seed(1234)
        self.root_dir = root_dir
        self.cache_dir_spectrogram = cache_dir_spectrogram
        self.cache_file_phonemes = cache_file_phonemes

        # load metafiles, an item is a list like: [text, audiopath, speaker_id]
        self.items = loaders.get_loader_by_name(name)(root_dir, meta_files)

        # clean text with basic stuff -- multiple spaces, case sensitivity and punctuation
        for idx in range(len(self.items)):
            utterance = self.items[idx][0]
            if not hp.use_punctuation: utterance = text.remove_punctuation(utterance)
            if not hp.case_sensitive: utterance = text.to_lower(utterance)
            if hp.remove_multiple_wspaces: utterance = text.remove_multiple_whitespaces(utterance)
            self.items[idx][0] = utterance

        # preload phonemized text if desired
        if hp.use_phonemes: self.load_phonemes()

        # sort dataset by utterance lengths in ascending order
        if hp.sort_by_length: sort_by_length_asc()

        # convert text into squence of character ids
        for idx in range(len(self.items)):
                self.items[idx][0] = text.to_sequence(self.items[idx][0])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.load_rich_item(index)

    def load_item(self, index, load_waveform=False):
        item = self.items[index]
        utterance, audio_path, speaker_idx = item[0], item[1], item[2]
        if load_waveform: audio_data = audio.load(audio_path)
        else: audio_data = self.load_mel(index)
        return (utterance, audio_data)

    def load_mel(self, audio_item_idx):
        audio_path = self.items[text_item_idx][1]
        if hp.cache_spectrograms:
            assert self.cache_dir_spectrogram is not None, (
                f'Directory of the cache for spectrogram was not provided.')
            if not os.path.exists(self.cache_dir_spectrogram): os.makedirs(self.cache_dir_spectrogram)
            spec_path = os.path.join(self.cache_dir_spectrogram, f'{audio_item_idx}.npy')
            try:
                melspec = np.load(spec_path)
            except (IOError, ValueError):
                audio_data = audio.load(audio_path)
                melspec = audio.mel_spectrogram(audio_data)
                np.save(spec_path, melspec)
        else:
            audio_data = audio.load(audio_path)
            melspec = audio.mel_spectrogram(audio_data)
        assert np.shape(melspec)[0] == hp.num_mels, (
                f'Mel dimension mismatch: given {np.shape(melspec)[0]}, expected {hp.num_mels}')
        return melspec

    def load_phonemes(self):
        num_items = len(self.items)
        if hp.cache_phonemes:
            phonemes = []
            assert self.cache_file_phonemes is not None, (
                f'Cache file of phonemes was not provided.')
            dirname = os.path.dirname(self.cache_file_phonemes)
            if not os.path.exists(dirname): os.makedirs(dirname)
            try:
                with open(self.cache_file_phonemes, 'r', encoding='utf-8') as f:
                    phonemes = [l[:-(l[-1] == '\n') or len(l)+1] for l in f]
            except (IOError, ValueError):
                with open(self.cache_file_phonemes, 'w', encoding='utf-8') as f:
                    logging.progress(0, num_items, prefix='Phonemizing utterances: ')
                    for idx in range(num_items):
                        utterance = self.items[idx][0]
                        phonemes.append(text.to_phoneme(utterance))
                        print(phonemes[-1], file=f)
                        logging.progress(idx + 1, num_items, prefix='Phonemizing utterances: ')
            assert len(phonemes) == num_items, (
                f'The number of utterances in dataset does not match number of phonemized utterances.')    
            for idx in range(num_items):
                self.items[idx][0] = phonemes[idx]
        else:
            logging.progress(0, num_items, prefix='Phonemizing utterances: ')
            for idx in range(num_items):
                self.items[idx][0] = text.to_phoneme(self.items[idx][0])
                logging.progress(idx + 1, num_items, prefix='Phonemizing utterances: ')

    def sort_by_length_asc(self):
        self.items.sort(key=lambda item: len(item[0]), reverse=False)
  
    # TODO: collate function to zero-pad uterrances in batches ...
    #       and provide them with vector of lengths
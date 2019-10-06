import os
import numpy as np
import collections
import torch
import random

from dataset import loaders
from utils import audio
from utils import text
from utils.logging import Logger
from params.params import Params as hp


class TextToSpeechDatasetCollection():
    """Collection of training, validation and test sets.
    
    Metadata format:
        The meta-file of a dataset (and corresponding spectrograms and phonemized utterances) can be 
        created by running the static TextToSpeechDataset.create_meta_file method!
        See the method for details about the format of the meta-file.
        
    Keyword arguments:
        dataset_root_dir (string): Root Directory of the dataset.
        training_file (string, default 'train.txt'): Relative path to the meta-file of the training set.
        validation_file (string, default 'val.txt'): Relative path to the meta-file of the validation set.
        test_file (string, default None): Relative path to the meta-file of the test set. Set None to ignore the test set.
    """
    def __init__(self, dataset_root_dir, training_file="train.txt", validation_file="val.txt", test_file=None):
        
        # create training set
        train_full_path = os.path.join(dataset_root_dir, training_file)
        if not os.path.exists(train_full_path):
            raise IOError(f'The training set meta-file not found, given: {train_full_path}')
        self.train = TextToSpeechDataset(train_full_path, dataset_root_dir)
        
        # create validation set
        val_full_path = os.path.join(dataset_root_dir, validation_file)
        if not os.path.exists(val_full_path):
            raise IOError(f'The validation set meta-file not found, given: {val_full_path}')
        self.dev = TextToSpeechDataset(val_full_path, dataset_root_dir)       
        
        # create test set
        if test_file:
            test_full_path = os.path.join(dataset_root_dir, test_file)
            if not os.path.exists(test_full_path):
                raise IOError(f'The test set meta-file not found, given: {test_full_path}')
            self.test = TextToSpeechDataset(test_full_path, dataset_root_dir)


class TextToSpeechDataset(torch.utils.data.Dataset):
    """Text to speech dataset.
    
        1) Load dataset metadata/data.
        2) Perform cleaning operations on the loaded utterances (phonemized).
        3) Compute mel-spectrograms (cached).
        4) Convert text into sequences of indices.

    Metadata format:
        The meta-file of a dataset (and corresponding spectrograms and phonemized utterances) can be 
        created by running the static TextToSpeechDataset.create_meta_file method!
        See the method for details about the format of the meta-file.
        
    Keyword arguments:
        meta_file (string): Meta-file of the dataset.
        dataset_root_dir (string): Root Directory of the dataset.
    """

    def __init__(self, meta_file, dataset_root_dir):
        random.seed(1234)
        self.root_dir = dataset_root_dir

        # read meta-file: id|speaker_id|audio_file_path|spectrogram_file_path|text|phonemized_text
        self.items = []
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_tokens = line[:-1].split('|')
                item = {
                    'id': line_tokens[0],
                    'speaker': line_tokens[1],
                    'audio': line_tokens[2],
                    'spectrogram': line_tokens[3],
                    'text': line_tokens[4],
                    'phonemes': line_tokens[5]
                }
                self.items.append(item)

        # clean text with basic stuff -- multiple spaces, case sensitivity and punctuation
        for idx in range(len(self.items)):
            item_text = self.items[idx]['text']
            item_phon = self.items[idx]['phonemes'] 
            if not hp.use_punctuation: 
                item_text = text.remove_punctuation(item_text)
                item_phon = text.remove_punctuation(item_phon)
            if not hp.case_sensitive: 
                item_text = text.to_lower(item_text)
            if hp.remove_multiple_wspaces: 
                item_text = text.remove_odd_whitespaces(item_text)
                item_phon = text.remove_odd_whitespaces(item_phon)
            self.items[idx]['text'] = item_text
            self.items[idx]['phonemes'] = item_phon

        # convert text into squence of character ids
        for idx in range(len(self.items)):
            self.items[idx]['phonemes'] = text.to_sequence(self.items[idx]['phonemes'], use_phonemes=True)
            self.items[idx]['text'] = text.to_sequence(self.items[idx]['text'], use_phonemes=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.load_item(index)

    def load_item(self, index, load_waveform=False):
        item = self.items[index]
        audio_path = item['audio']
        if load_waveform: 
            full_audio_path = os.path.join(self.root_dir, audio_path)
            audio_data = audio.load(full_audio_path)
        else: audio_data = self.load_mel(audio_path, item['spectrogram'])
        return (item['phonemes'] if hp.use_phonemes else item['text'], audio_data)

    def load_mel(self, audio_path, spectrogram_path):
        if hp.cache_spectrograms:
            full_spec_path = os.path.join(self.root_dir, spectrogram_path)
            melspec = np.load(full_spec_path)
        else:
            full_audio_path = os.path.join(self.root_dir, audio_path)
            audio_data = audio.load(full_audio_path)
            melspec = audio.mel_spectrogram(audio_data)
        assert np.shape(melspec)[0] == hp.num_mels, (
                f'Mel dimension mismatch: given {np.shape(melspec)[0]}, expected {hp.num_mels}')
        if hp.normalize_spectrogram:
            melspec = audio.normalize_spectrogram(melspec)
        return melspec

    @staticmethod
    def create_meta_file(dataset_name, dataset_root_dir, output_metafile_name, audio_sample_rate, num_fft_freqs, spectrograms=True, phonemes=True):
        """Create metafile and spectrograms or phonemized utterances.
        
        Format details:
            Every line of the metadata file contains info about one dataset item .
            The line has following format 
                'id|speaker_id|audio_file_path|spectrogram_file_path|text|phonemized_text'
            And the following must hold
                'audio_file_path' can be empty if loading just spectrograms
                'spectrogram_file_path' can be empty if loading raw waveforms
                'text' should be carefully normalized and should contain interpunciton
                'phonemized_text' can be empty if loading just raw text  
        """

        # save current sample rate and fft freqs hyperparameters, as we may process dataset with different sample rate
        old_sample_rate = hp.sample_rate
        hp.sample_rate = audio_sample_rate
        old_fft_freqs = hp.num_fft
        hp.num_fft = num_fft_freqs

        # load metafiles, an item is a list like: [text, audiopath, speaker_id]
        items = loaders.get_loader_by_name(dataset_name)(dataset_root_dir)

        # build dictionary for translation to IPA, see utils.text for details
        if phonemes:
            texts = [i[0] for i in items]
            phoneme_dict = text.build_phoneme_dict(texts)

        # prepare directory which will store spectrograms
        if spectrograms:
            spec_dir = os.path.join(dataset_root_dir, 'spectrograms')
            if not os.path.exists(spec_dir): os.makedirs(spec_dir)

        # iterate through items and build the meta-file
        metafile_path = os.path.join(dataset_root_dir, output_metafile_name)
        with open(metafile_path, 'w', encoding='utf-8') as f:
            Logger.progress(0, prefix='Building metafile:')
            for i in range(len(items)):
                if i < 86576: continue
                raw_text, audio_path, speaker = items[i]
                phonemized_text = text.to_phoneme(raw_text, False, phoneme_dict) if phonemes else ""
                if spectrograms:  
                    spec_name = f'{str(i).zfill(6)}.npy'                 
                    spec_path = os.path.join('spectrograms', spec_name)
                    full_spec_path = os.path.join(spec_dir, spec_name)
                    if os.path.exists(full_spec_path): continue
                    full_audio_path = os.path.join(dataset_root_dir, audio_path)
                    audio_data = audio.load(full_audio_path)
                    melspec = audio.mel_spectrogram(audio_data)
                    np.save(full_spec_path, melspec)
                else: 
                    spec_path = ""
                print(f'{str(i).zfill(6)}|{speaker}|{audio_path}|{spec_path}|{raw_text}|{phonemized_text}', file=f)
                Logger.progress((i + 1) / len(items), prefix='Building metafile:')
        
        # restore the original sample rate and fft freq values
        hp.sample_rate = old_sample_rate
        hp.num_fft = old_fft_freqs


class TextToSpeechCollate():
    """Text to speech dataset collate function.
    
    1) zero-pad utterances and spectrograms in batches
    2) sort them by utterance lengths (because of torch packed sequence)
    3) provide vector of lengths

    Keyword arguments:
        meta_file (string): Meta-file of the dataset.
        root_dir (string): Root Directory of the dataset.
    """

    def __call__(self, batch):
        
        batch_size = len(batch)

        # get lengths
        utterance_lengths, spectrogram_lengths = [], []
        max_frames = 0
        for u, s in batch:
            utterance_lengths.append(len(u))
            spectrogram_lengths.append(len(s[0]))
            if spectrogram_lengths[-1] > max_frames:
                max_frames = spectrogram_lengths[-1] 

        utterance_lengths = torch.LongTensor(utterance_lengths)
        sorted_utterance_lengths, sorted_idxs = torch.sort(utterance_lengths, descending=True)
        spectrogram_lengths = torch.LongTensor(spectrogram_lengths)[sorted_idxs]

        # zero-pad utterances, spectrograms
        padded_utterances = torch.zeros(batch_size, sorted_utterance_lengths[0], dtype=torch.long)
        padded_spectrograms = torch.zeros(batch_size, hp.num_mels, max_frames, dtype=torch.float)
        padded_stop_tokens = torch.zeros(batch_size, max_frames, dtype=torch.float)
        for i, idx in enumerate(sorted_idxs):
            u, s = batch[idx]
            padded_utterances[i, :len(u)] = torch.LongTensor(u)
            padded_spectrograms[i, :, :s[0].size] = torch.FloatTensor(s)
            padded_stop_tokens[i, s[0].size-1:] = 1

        return sorted_utterance_lengths, padded_utterances, padded_spectrograms, padded_stop_tokens, spectrogram_lengths

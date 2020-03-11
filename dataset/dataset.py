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
        
    Arguments:
        dataset_root_dir (string): Root Directory of the dataset.
    Keyword arguments:
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
        self.dev = TextToSpeechDataset(val_full_path, dataset_root_dir, self.train.unique_speakers)  
        assert len(self.dev.unique_speakers) == len(self.train.unique_speakers), (
                f'Validation set contains speakers which are not present in train set!')     
        
        # create test set
        if test_file:
            test_full_path = os.path.join(dataset_root_dir, test_file)
            if not os.path.exists(test_full_path):
                raise IOError(f'The test set meta-file not found, given: {test_full_path}')
            self.test = TextToSpeechDataset(test_full_path, dataset_root_dir, self.train.unique_speakers)
            assert len(self.test.unique_speakers) == len(self.train.unique_speakers), (
                f'Test set contains speakers which are not present in test set!')


class TextToSpeechDataset(torch.utils.data.Dataset):
    """Text to speech dataset.
    
        1) Load dataset metadata/data.
        2) Perform cleaning operations on the loaded utterances (phonemized).
        3) Compute mel-spectrograms or linear spectrograms (cached).
        4) Convert text into sequences of indices.

    Metadata format:
        The meta-file of a dataset (and corresponding spectrograms and phonemized utterances) can be 
        created by running the static TextToSpeechDataset.create_meta_file method!
        See the method for details about the format of the meta-file.
        
    Arguments:
        meta_file (string): Meta-file of the dataset.
        dataset_root_dir (string): Root Directory of the dataset.
    Keyword arguments:
        known_unique_speakers (list of string, default []): List of speaker UIDs (for initialization).
    """

    def __init__(self, meta_file, dataset_root_dir, known_unique_speakers=[]):
        random.seed(1234)
        self.root_dir = dataset_root_dir

        # read meta-file: id|speaker|language|audio_file_path|mel_spectrogram_path|linear_spectrogram_path|text|phonemized_text

        self.unique_speakers = known_unique_speakers.copy()
        unique_speakers_set = set(self.unique_speakers)
        self.items = []
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_tokens = line[:-1].split('|')
                item = {
                    'id': line_tokens[0],
                    'speaker': line_tokens[1],
                    'language': line_tokens[2],
                    'audio': line_tokens[3],
                    'spectrogram': line_tokens[4],
                    'linear_spectrogram': line_tokens[5],
                    'text': line_tokens[6],
                    'phonemes': line_tokens[7]
                }
                if item['language'] in hp.languages:
                    if line_tokens[1] not in unique_speakers_set:
                        unique_speakers_set.add(line_tokens[1])
                        self.unique_speakers.append(line_tokens[1])
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

        # convert text into a sequence of character IDs, convert language and speaker names to IDs
        for idx in range(len(self.items)):
            self.items[idx]['phonemes'] = text.to_sequence(self.items[idx]['phonemes'], use_phonemes=True)
            self.items[idx]['text'] = text.to_sequence(self.items[idx]['text'], use_phonemes=False)
            self.items[idx]['speaker'] = self.unique_speakers.index(self.items[idx]['speaker'])
            self.items[idx]['language'] = hp.languages.index(self.items[idx]['language'])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        audio_path = item['audio']
        mel_spec = self.load_spectrogram(audio_path, item['spectrogram'], hp.normalize_spectrogram, True)
        lin_spec = self.load_spectrogram(audio_path, item['linear_spectrogram'], hp.normalize_spectrogram, False) if hp.predict_linear else None
        return (item['speaker'], item['language'], item['phonemes'] if hp.use_phonemes else item['text'], mel_spec, lin_spec)

    def load_spectrogram(self, audio_path, spectrogram_path, normalize, is_mel):
        """Load a mel or linear spectrogram from file or compute from scratch if needed.
        
        Arguments:
            audio_path (string): Path to the audio from which will (possibly) be the spectrogram computed.
            spectrogram_path (string): Path to the spectrogram file which will be loaded (possibly).
            normalize (boolean): If True, the spectrogram is normalized (per channel, extract mean and divide by std).
            is_mel (boolean): If True, the mel spectrogram is loaded or computed, otherwise returns a linear spectrogram.
        """

        # load or compute spectrogram
        if hp.cache_spectrograms:
            full_spec_path = os.path.join(self.root_dir, spectrogram_path)
            spectrogram = np.load(full_spec_path)
        else:
            full_audio_path = os.path.join(self.root_dir, audio_path)
            audio_data = audio.load(full_audio_path)
            spectrogram = audio.spectrogram(audio_data, is_mel)

        # check spectrogram dimensions
        expected_dimension = hp.num_mels if is_mel else hp.num_fft // 2 + 1
        assert np.shape(spectrogram)[0] == expected_dimension, (
                f'Spectrogram dimensions mismatch: given {np.shape(spectrogram)[0]}, expected {expected_dimension}')
        
        # normalize if desired
        if normalize:
            spectrogram = audio.normalize_spectrogram(spectrogram, is_mel)

        return spectrogram

    def get_normalization_constants(self, is_mel):
        """Compute per-channel mean and variance of the data contained in this collection."""
        mean = 0.0
        std = 0.0
        for item in self.items:
            path = item['spectrogram'] if is_mel else item['linear_spectrogram']
            spectrogram = self.load_spectrogram(item['audio'], path, False, is_mel)
            mean += np.mean(spectrogram, axis=1, keepdims=True)
            std += np.std(spectrogram, axis=1, keepdims=True)
        mean /= len(self.items)
        std /= len(self.items)
        return mean, std

    def get_num_speakers(self):
        """Get number of unique speakers in the dataset."""
        speakers = set()
        for idx in range(len(self.items)):
            speakers.add(self.items[idx]['speaker'])
        return len(speakers)

    def get_num_languages(self):
        """Get number of unique languages in the dataset."""
        languages = set()
        for idx in range(len(self.items)):
            languages.add(self.items[idx]['language'])
        return len(languages)

    @staticmethod
    def create_meta_file(dataset_name, dataset_root_dir, output_metafile_name, audio_sample_rate, num_fft_freqs, spectrograms=True, phonemes=True):
        """Create the meta-file and spectrograms (mel and linear, optionally) or phonemized utterances (optionally).
        
        Format details:
            Every line of the metadata file contains info about one dataset item.
            The line has following format 
                'id|speaker|language|audio_file_path|mel_spectrogram_path|linear_spectrogram_path|text|phonemized_text'
            And the following must hold
                'audio_file_path' can be empty if loading just spectrograms
                'text' should be carefully normalized and should contain interpunction
                'phonemized_text' can be empty if loading just raw text  
        
        Arguments:
            dataset_name (string): Name of the dataset, loaders.py should contain a function for loading with a corresponding name.
            dataset_root_dir (string): Root directory from which is the dataset build and to which are spectrograms and the meta-file saved..
            output_metafile_name (string): Name of the output meta-file.
            audio_sample_rate (int): Sample rate of audios, used if spectrograms is set True.
            num_fft_freqs (int): Number of frequency bands used during spectrogram computation, used if spectrograms is set True.
        Keyword arguments:
            spectrograms (boolean, default True): If true, spetrograms (both mel and linear) are computed and saved.
            phonemes (boolean, default True): If true, phonemized variants of utterances are computed and saved.
        """

        # save current sample rate and fft freqs hyperparameters, as we may process dataset with different sample rate
        if spectrograms:
            old_sample_rate = hp.sample_rate
            hp.sample_rate = audio_sample_rate
            old_fft_freqs = hp.num_fft
            hp.num_fft = num_fft_freqs

        # load metafiles, an item is a list like: [text, audiopath, speaker_id, language_code]
        items = loaders.get_loader_by_name(dataset_name)(dataset_root_dir)

        # build dictionaries for translation to IPA from source languages, see utils.text for details
        if phonemes:
            text_lang_pairs = [(i[0], hp.languages[0] if i[3] == "" else i[3]) for i in items]
            phoneme_dicts = text.build_phoneme_dicts(text_lang_pairs)

        # prepare directories which will store spectrograms
        if spectrograms:
            spectrogram_dirs = [os.path.join(dataset_root_dir, 'spectrograms'), 
                                os.path.join(dataset_root_dir, 'linear_spectrograms')]
            for x in spectrogram_dirs:
                if not os.path.exists(x): os.makedirs(x)

        # iterate through items and build the meta-file
        metafile_path = os.path.join(dataset_root_dir, output_metafile_name)
        with open(metafile_path, 'w', encoding='utf-8') as f:
            Logger.progress(0, prefix='Building metafile:')
            for i in range(len(items)):
                raw_text, audio_path, speaker, language = items[i]   
                if language == "": language = hp.languages[0]
                phonemized_text = text.to_phoneme(raw_text, False, language, phoneme_dicts[language]) if phonemes else ""     
                spectrogram_paths = "|"
                if spectrograms:
                    spec_name = f'{str(i).zfill(6)}.npy'                 
                    audio_data = audio.load(os.path.join(dataset_root_dir, audio_path))
                    np.save(os.path.join(spectrogram_dirs[0], spec_name), audio.spectrogram(audio_data, True))
                    np.save(os.path.join(spectrogram_dirs[1], spec_name), audio.spectrogram(audio_data, False))
                    spectrogram_paths = os.path.join('spectrograms', spec_name) + '|' + os.path.join('linear_spectrograms', spec_name)
                print(f'{str(i).zfill(6)}|{speaker}|{language}|{audio_path}|{spectrogram_paths}|{raw_text}|{phonemized_text}', file=f)
                Logger.progress((i + 1) / len(items), prefix='Building metafile:')
        
        # restore the original sample rate and fft freq values
        if spectrograms:
            hp.sample_rate = old_sample_rate
            hp.num_fft = old_fft_freqs


class TextToSpeechCollate():
    """Collate function for TextToSpeechDataset.
    
    Arguments:
        sort_by_text_length (boolean): If True, returns batch ordered by utterance lengths (suitable for RNNs).
    """

    def __init__(self, sort_by_text_length):
        self.sort_by_text_length = sort_by_text_length

    def __call__(self, batch):
        batch_size = len(batch)

        # iterate batch, get lengths of spectrograms and uttrances
        utterance_lengths, spectrogram_lengths = [], []
        speakers = []
        languages = []
        max_frames = 0
        for s, l, u, a, _ in batch:
            speakers.append(s)
            languages.append(l)
            utterance_lengths.append(len(u))
            spectrogram_lengths.append(len(a[0]))
            if spectrogram_lengths[-1] > max_frames:
                max_frames = spectrogram_lengths[-1] 

        # convert collected lists to tensors
        utterance_lengths = torch.LongTensor(utterance_lengths)
        spectrogram_lengths = torch.LongTensor(spectrogram_lengths)
        speakers = None if not hp.multi_speaker else torch.LongTensor(speakers)
        languages = None if not hp.multi_language else torch.LongTensor(languages)

        if self.sort_by_text_length:
            utterance_lengths, sorted_idxs = torch.sort(utterance_lengths, descending=True)
            # permute other tensors according to sorted_idx
            spectrogram_lengths = spectrogram_lengths[sorted_idxs]
            if speakers is not None: speakers = speakers[sorted_idxs]
            if languages is not None: 
                languages = languages[sorted_idxs]
                # convert a vector of language indices into a vector of one-hots (used as weight vectors for accent control)
                one_hots = torch.zeros(languages.size(0), languages.size(1), hp.language_number).zero_()
                languages = one_hot.scatter_(2, languages.data, 1)
        else:
            sorted_idxs = range(batch_size)
        
        # zero-pad utterances, spectrograms
        utterances = torch.zeros(batch_size, max(utterance_lengths), dtype=torch.long)
        mel_spectrograms = torch.zeros(batch_size, hp.num_mels, max_frames, dtype=torch.float)
        lin_spectrograms = torch.zeros(batch_size, hp.num_fft // 2 + 1, max_frames, dtype=torch.float) if hp.predict_linear else None
        stop_tokens = torch.zeros(batch_size, max_frames, dtype=torch.float)

        # fill tensors
        for i, idx in enumerate(sorted_idxs):
            _, _, u, a, b = batch[idx]
            utterances[i, :len(u)] = torch.LongTensor(u)
            mel_spectrograms[i, :, :a[0].size] = torch.FloatTensor(a)
            if hp.predict_linear:
                lin_spectrograms[i, :, :b[0].size] = torch.FloatTensor(b) 
            stop_tokens[i, a[0].size - hp.stop_frames:] = 1

        return utterances, utterance_lengths, mel_spectrograms, lin_spectrograms, spectrogram_lengths, stop_tokens, speakers, languages
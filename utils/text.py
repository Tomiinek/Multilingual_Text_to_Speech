import re 
import string 

from phonemizer.separator import Separator
from phonemizer.phonemize import phonemize
from training.params import Params as hp


_pad = '_'    # padding of the sequences to align text in batches to the same length
_eos = '~'    # character which marks the end of a sequnce, further characters are invalid
_unk = '@'    # symbols which are not in hp.characters and are present are substituted by this

_other_symbols = [_pad, _eos, _unk] + list(hp.punctuations_in) + list(hp.punctuations_out)
_char_to_id = {s: i for i, s in enumerate(_other_symbols + list(hp.characters))}
_id_to_char = {i: s for i, s in enumerate(_other_symbols + list(hp.characters))}
_phon_to_id = {s: i for i, s in enumerate(_other_symbols + list(hp.phonemes))}
_id_to_phon = {i: s for i, s in enumerate(_other_symbols + list(hp.phonemes))}


def to_phoneme(utterance):
    '''Convert graphemes of the utterance without new line to phonemes.'''
    clear_utterance = remove_punctuation(utterance)
    if not hp.use_punctuation: return _phonemize(clear_utterance)[:-1]
    else:
        # TODO: phonemizing word by word is very slow, instead, we should build a dictionary
        #       of all words in the dataset and phonemize the dictionary 
        clear_words = clear_utterance.split()
        phonemes = [_phonemize(w)[:-1] for w in clear_words]
        in_word = False
        punctuation_seen = False
        utterance_phonemes = ""
        clear_offset = word_idx = 0
        for idx, char in enumerate(utterance):
            # encountered non-punctuation char
            if idx - clear_offset < len(clear_utterance) and char == clear_utterance[idx - clear_offset]:
                if not in_word:
                    if char in string.whitespace: 
                        punctuation_seen = False
                        continue
                    in_word = True
                    utterance_phonemes += (' ' if idx != 0 and not punctuation_seen else '') + phonemes[word_idx]
                    word_idx += 1 
                else: 
                    if char in string.whitespace: in_word = False 
                punctuation_seen = False           
            # this should be punctuation
            else:
                clear_offset += 1
                if in_word and char in hp.punctuations_in: continue
                utterance_phonemes += (' ' if not in_word and not punctuation_seen else '') + char
                punctuation_seen = True
        return utterance_phonemes


def _phonemize(text):
    seperators = Separator(word=' ', phone='')
    phonemes = phonemize(text, separator=seperators, backend='espeak', language=hp.language)
    phonemes.replace('\n', ' ', 1)            
    return phonemes


def to_lower(text):
    '''Convert uppercase text into lowercase.'''
    return text.lower()


def remove_odd_whitespaces(text):
    '''Remove multiple and trailing/leading whitespaces.'''
    return ' '.join(text.split())


def remove_punctuation(text):
    '''Remove punctuation from text.'''
    punct_re = '[' + hp.punctuations_in + hp.punctuations_out + ']'
    return re.sub(punct_re.replace('-', '\-'), '', text)


def to_sequence(text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.'''
    transform_dict = _phon_to_id if hp.use_phonemes else _char_to_id
    sequence = [_unk if c not in transform_dict else transform_dict[c] for c in text]
    sequence.append(_eos)
    return sequence


def to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    transform_dict = _id_to_phon if hp.use_phonemes else _id_to_char
    result = ''
    for symbol_id in sequence:
        if symbol_id == _eos: break
        if symbol_id in transform_dict:
            s = transform_dict[symbol_id]
            result += s
    return result
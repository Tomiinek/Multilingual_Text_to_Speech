import re 
import string 

from phonemizer.separator import Separator
from phonemizer.phonemize import phonemize
import epitran
from params.params import Params as hp
from utils.logging import Logger


_pad = '_'    # a dummy character for padding sequences to align text in batches to the same length
_eos = '~'    # character which marks the end of a sequnce, further characters are invalid
_unk = '@'    # symbols which are not in hp.characters and are present are substituted by this


def _other_symbols():
    return [_pad, _eos, _unk] + list(hp.punctuations_in) + list(hp.punctuations_out)


def build_phoneme_dicts(text_lang_pairs):
    """Create dictionaries (possibly more languages) of words (from a list of texts) with IPA equivalents."""
    dictionaries = {}
    Logger.progress(0 / len(text_lang_pairs), prefix='Building phoneme dictionary:')
    for i, (t, l) in enumerate(text_lang_pairs):
        if not (l in dictionaries):
            dictionaries[l] = {}
        clear_words = remove_punctuation(t).split()
        for w in clear_words:
            if w in dictionaries[l]: continue
            dictionaries[l][w] = _phonemize(w, l)[:-1]
        Logger.progress((i+1) / len(text_lang_pairs), prefix='Building phoneme dictionary:')
    return dictionaries
    

def to_phoneme(text, ignore_punctuation, language, phoneme_dictionary=None):
    """Convert graphemes of the utterance without new line to phonemes.
    
    Arguments:
        text (string): The text to be translated into IPA.
        ignore_punctuation (bool): Set to False if the punctuation should be preserved.
        language (default hp.language): language code (e.g. en-us)
    Keyword argumnets:
        phoneme_dictionary (default None): A language specific dictionary of words with IPA equivalents, 
            used to speed up the translation which preserves punctuation (because the used phonemizer
            cannot handle punctuation properly, so we need to do it word by word).
    """
    
    clear_text = remove_punctuation(text)
    if ignore_punctuation: 
        return _phonemize(clear_text)[:-1]
    
    # phonemize words of the input text
    clear_words = clear_text.split()
    if not phoneme_dictionary: phoneme_dictionary = {}
    phonemes = []
    for w in clear_words:
        phonemes.append(phoneme_dictionary[w] if w in phoneme_dictionary else _phonemize(w, language)[:-1])

    # add punctuation to match the punctuation in the input 
    in_word = False
    punctuation_seen = False
    text_phonemes = ""
    clear_offset = word_idx = 0
    
    for idx, char in enumerate(text):
        # encountered non-punctuation char
        if idx - clear_offset < len(clear_text) and char == clear_text[idx - clear_offset]:
            if not in_word:
                if char in string.whitespace: 
                    punctuation_seen = False
                    continue    
                in_word = True
                text_phonemes += (' ' if idx != 0 and not punctuation_seen else '') + phonemes[word_idx]
                word_idx += 1 
            else: 
                if char in string.whitespace: in_word = False 
            punctuation_seen = False           
        # this should be punctuation
        else:
            clear_offset += 1
            if in_word and char in hp.punctuations_in: continue
            text_phonemes += (' ' if not in_word and not punctuation_seen else '') + char
            punctuation_seen = True

    return text_phonemes


def _phonemize(text, language):
    try:
        seperators = Separator(word=' ', phone='')
        phonemes = phonemize(text, separator=seperators, backend='espeak', language=language)           
    except RuntimeError:
        epi = epitran.Epitran(language)
        phonemes = epi.transliterate(text, normpunc=True)
    phonemes.replace('\n', ' ', 1)   
    return phonemes


def to_lower(text):
    """Convert uppercase text into lowercase."""
    return text.lower()


def remove_odd_whitespaces(text):
    """Remove multiple and trailing/leading whitespaces."""
    return ' '.join(text.split())


def remove_punctuation(text):
    """Remove punctuation from text."""
    punct_re = '[' + hp.punctuations_out + hp.punctuations_in + ']'
    return re.sub(punct_re.replace('-', '\-'), '', text)


def to_sequence(text, use_phonemes=False):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text."""
    transform_dict = {s: i for i, s in enumerate(_other_symbols() + list(hp.phonemes if use_phonemes else hp.characters))}
    sequence = [transform_dict[_unk] if c not in transform_dict else transform_dict[c] for c in text]
    sequence.append(transform_dict[_eos])
    return sequence


def to_text(sequence, use_phonemes=False):
    """Converts a sequence of IDs back to a string"""
    transform_dict = {i: s for i, s in enumerate(_other_symbols() + list(hp.phonemes if use_phonemes else hp.characters))}
    result = ''
    for symbol_id in sequence:
        if symbol_id in transform_dict:
            s = transform_dict[symbol_id]
            if s == _eos: break
            result += s
    return result
import json


class Params:

    @staticmethod
    def load(json_path):
        with open(json_path) as f:
            params = json.load(f)
            for k, v in params.items(): setattr(Params, k, v)

    @staticmethod
    def save(json_path):
        with open(json_path, 'w') as f:
            members = [attr for attr in dir(Params) if not callable(getattr(Params, attr)) and not attr.startswith("__")]
            json.dump({ k: Params.__dict__[k] for k in members }, f, indent=4)

    # DATASET:
    
    dataset = 'ljspeech'
    cache_spectrograms = True
    cache_phonemes = True
    sort_by_length = False

    # TEXT:

    language = 'en-gb'          # espeak format: phonemize --help
    use_punctuation = True      # punctuations_{in, out} are valid only if True
    use_phonemes = False        # phonemes are valind only if True
    case_sensitive = True
    remove_multiple_wspaces = False 
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
    # all phonemes of IPA: 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧ ɚ˞ɫ'
    phonemes = 'pbtdkɡfvθðszʃʒhmnŋn̩ll̩rjwʔɪeœɒʌʊiᵻːaɔuəɑɜˌˈ '
    punctuations_out = '"(),.:;?!'
    punctuations_in  = '\'-'
    
    # AUDIO:

    sample_rate = 22050 
    stft_window_ms = 50
    stft_shift_ms = 12.5
    num_fft = 2048
    num_mels = 80

    griffin_lim_iters = 50
    griffin_lim_power = 1.5 

    reference_spectrogram_db = 20

    normalize_spectrogram = True
    normalize_symetric = True
    normalize_scaling = 4
    normalize_minimal_db = -100

    use_preemphasis = True
    preemphasis = 0.97
    
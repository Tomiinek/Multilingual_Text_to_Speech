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

    @staticmethod
    def symbols_count():
        symbols_count = len(Params.characters)
        if Params.use_phonemes: symbols_count = len(Params.phonemes)
        if Params.use_punctuation: symbols_count += len(Params.punctuations_out) + len(Params.punctuations_in)
        return symbols_count

    version = "1.0"

    # TRAINING:
    epochs = 10 
    batch_size = 2
    learning_rate = 1e-3
    learning_rate_decay = 0.93
    weight_decay = 1e-6
    max_input_length = 500
    max_output_length = 5000
    cudnn_enabled = False
    gradient_clipping = 5

    # MODEL:

    embedding_dimension = 512
    encoder_dimension = 512
    encoder_blocks = 3
    encoder_kernel_size = 5
    prenet_dimension = 256
    prenet_layers = 2
    attention_dimension = 128
    attention_kernel_size = 31
    attention_location_dimension = 1024
    decoder_dimension = 1024
    postnet_dimension = 512
    postnet_blocks = 5
    postnet_kernel_size = 5
    dropout = 0.5
    zoneout_hidden = 0.1
    zoneout_cell = 0.1

    # DATASET:
    
    cache_spectrograms = True

    # TEXT:

    language = 'en-gb'          # espeak format: phonemize --help
    use_punctuation = True      # punctuations_{in, out} are valid only if True
    use_phonemes = False        # phonemes are valind only if True
    case_sensitive = True
    remove_multiple_wspaces = False 
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
    # all phonemes of IPA: 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧ ɚ˞ɫ'
    phonemes = 'ɹɐpbtdkɡfvθðszʃʒhmnŋn̩ll̩rjwʔɪeœɒʌʊiᵻːaɔuəɑɜˌˈ '
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

    
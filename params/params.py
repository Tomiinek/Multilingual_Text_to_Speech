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
    epochs = 600
    batch_size = 56
    learning_rate = 5e-2
    learning_rate_decay = 0.985
    learning_rate_decay_start = 35000
    weight_decay = 1e-6
    max_output_length = 5000
    cudnn_enabled = True
    gradient_clipping = 5
    guided_attention_loss = True
    guided_attention_toleration = 0.2
    guided_attention_gain = 1.0
    constant_teacher_forcing = True
    teacher_forcing = 1.0 
    teacher_forcing_steps = 100000
    teacher_forcing_start_steps = 50000
    checkpoint_each_epochs = 5

    # MODEL:

    embedding_dimension = 512
    encoder_dimension = 512
    encoder_blocks = 3
    encoder_kernel_size = 5
    prenet_dimension = 256
    prenet_layers = 2
    attention_type = "location_sensitive"   # one of: location_sensitive, forward, forward_transition_agent
    attention_dimension = 128
    attention_kernel_size = 31
    attention_location_dimension = 32
    decoder_dimension = 1024
    decoder_regularization = 'dropout'
    zoneout_hidden = 0.1
    zoneout_cell = 0.1
    dropout_hidden = 0.1
    postnet_dimension = 512
    postnet_blocks = 5
    postnet_kernel_size = 5
    dropout = 0.5   

    # DATASET:
    
    dataset = "ljspeech"        # one of: ljspeech, vctk, my_blizzard
    cache_spectrograms = True

    # TEXT:

    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
    case_sensitive = True
    remove_multiple_wspaces = False 

    use_punctuation = True      # punctuations_{in, out} are valid only if True
    punctuations_out = '"(),.:;?!'
    punctuations_in  = '\'-'

    use_phonemes = False        # phonemes are valid only if True
    language = 'en-gb'          # espeak format: phonemize --help
    # all phonemes of IPA: 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧ ɚ˞ɫ'
    phonemes = 'ɹɐpbtdkɡfvθðszʃʒhmnŋn̩ll̩rjwʔɪeœɒʌʊiᵻːaɔuəɑɜˌˈ '
    
    # AUDIO:

    # ljspeech    - 22050, 2048
    # vctk        - 48000, 2400
    # my_blizzard - 44100, 2250
    sample_rate = 22050 
    num_fft = 2048
    num_mels = 80

    stft_window_ms = 50
    stft_shift_ms = 12.5
    reference_spectrogram_db = 20
    
    griffin_lim_iters = 50
    griffin_lim_power = 1.5 

    normalize_spectrogram = True 

    use_preemphasis = True
    preemphasis = 0.97

    
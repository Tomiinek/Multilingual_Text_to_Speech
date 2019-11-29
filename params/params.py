import json


class Params:

    @staticmethod
    def load_state_dict(d):
        for k, v in d.items(): setattr(Params, k, v)

    @staticmethod
    def state_dict():
        members = [attr for attr in dir(Params) if not callable(getattr(Params, attr)) and not attr.startswith("__")]
        return { k: Params.__dict__[k] for k in members }

    @staticmethod
    def load(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
            Params.load_state_dict(params)

    @staticmethod
    def save(json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            d = Params.state_dict()
            json.dump(d, f, indent=4)

    @staticmethod
    def symbols_count():
        symbols_count = len(Params.characters)
        if Params.use_phonemes: symbols_count = len(Params.phonemes)
        if Params.use_punctuation: symbols_count += len(Params.punctuations_out) + len(Params.punctuations_in)
        return symbols_count

    version = "1.0"

    # TRAINING:
    epochs = 300
    batch_size = 56
    learning_rate = 2e-3
    learning_rate_decay = 0.5
    learning_rate_decay_start = 15000
    learning_rate_decay_each = 15000
    weight_decay = 1e-6
    max_output_length = 5000
    gradient_clipping = 0.25
    guided_attention_loss = True
    guided_attention_steps = 15000
    guided_attention_toleration = 0.25
    guided_attention_gain = 1.015
    constant_teacher_forcing = True
    teacher_forcing = 1.0 
    teacher_forcing_steps = 100000
    teacher_forcing_start_steps = 50000
    checkpoint_each_epochs = 10

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

    predict_linear = True
    cbhg_bank_kernels = 8
    cbhg_bank_dimension = 128
    cbhg_projection_kernel_size = 3
    cbhg_projection_dimension = 256
    cbhg_highway_dimension = 128
    cbhg_rnn_dim = 128
    cbhg_dropout = 0.0

    multi_speaker = False
    multi_language = False
    embedding_type = "simple"
    speaker_embedding_dimension = 64
    language_embedding_dimension = 8
    speaker_number = 0
    language_number = 0


    # DATASET:
    
    dataset = "ljspeech"        # one of: ljspeech, vctk, my_blizzard, mailabs
    cache_spectrograms = True
    languages = ['en-us']       # espeak format: phonemize --help

    # TEXT:

    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
    case_sensitive = True
    remove_multiple_wspaces = True 

    use_punctuation = True      # punctuations_{in, out} are valid only if True
    punctuations_out = '、。，"(),.:;¿?¡!\\'
    punctuations_in  = '’\'-'

    # all phonemes of IPA: 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧ ɚ˞ɫ'
    use_phonemes = False   # phonemes are valid only if True
    phonemes = 'ɹɐpbtdkɡfvθðszʃʒhmnŋlrwjeəɪɒuːɛiaʌʊɑɜɔx '
   
    # AUDIO:

    # ljspeech    - 22050, 2048
    # vctk        - 48000, 2400
    # my_blizzard - 44100, 2250
    sample_rate = 22050 
    num_fft = 1102
    num_mels = 80

    stft_window_ms = 50
    stft_shift_ms = 12.5
    
    griffin_lim_iters = 50
    griffin_lim_power = 1.5 

    normalize_spectrogram = True 

    use_preemphasis = True
    preemphasis = 0.97
import numpy as np
import scipy

import librosa
import librosa.feature
import librosa.effects
import soundfile as sf

from params.params import Params as hp


def load(path):
    """Load a sound file into numpy array."""
    data, sample_rate = sf.read(path)
    assert hp.sample_rate == sample_rate, (
        f'Sample rate do not match: given {hp.sample_rate}, expected {sample_rate}')
    return data


def save(data, path):
    """Save numpy array as sound file."""
    sf.write(path, data, samplerate=hp.sample_rate)


def ms_to_frames(ms):
    """Convert milliseconds into number of frames."""
    return int(hp.sample_rate * ms / 1000)


def trim_silence(data, window_ms, hop_ms, top_db=50, margin_ms=0):
    """Trim leading and trailing silence from an audio signal."""
    wf = ms_to_frames(window_ms)
    hf = ms_to_frames(hop_ms)
    mf = ms_to_frames(margin_ms)
    return librosa.effects.trim(data[mf:-mf], top_db=top_db, frame_length=wf, hop_length=hf)
  

def duration(data):
    """Return duration of an audio signal in seconds."""
    return librosa.get_duration(data)


def endpoint(data, silence_ms, top_db=-40):
    """Return endpoint (frame index) of an audio signal."""
    silence_frames = ms_to_frames(silence_ms)
    hop_frames = int(silence_frames / 4)
    threshold = db_to_amplitude(top_db)
    for x in range(hop_frames, len(data) - silence_frames, hop_length):
        if np.max(data[x:(x + silence_frames)]) < threshold: return x + hop_length
    return len(data)


def amplitude_to_db(x):
    """Convert amplitude to decibels."""
    return librosa.amplitude_to_db(x, top_db=None)


def db_to_amplitude(x):
    """Convert decibels to amplitude."""
    return librosa.db_to_amplitude(x)


def preemphasis(y):
    """Preemphasize the signal.

      Should be used together with Griffin-Lim and not used together with WaveNet.

    See: https://github.com/Rayhane-mamah/Tacotron-2/issues/157
         https://github.com/Rayhane-mamah/Tacotron-2/issues/160
    """
    # y[n] = x[n] - perc * x[n-1]
    return scipy.signal.lfilter([1, -hp.preemphasis], [1], y)


def deemphasis(y):
    """Deemphasize the signal."""
    # y[n] + perc * y[n-1] = x[n] 
    return scipy.signal.lfilter([1], [1, -hp.preemphasis], y)


def spectrogram(y, mel=False):
    """Convert waveform to log-magnitude spectrogram."""
    if hp.use_preemphasis: y = preemphasis(y)
    wf = ms_to_frames(hp.stft_window_ms)
    hf = ms_to_frames(hp.stft_shift_ms)
    S = np.abs(librosa.stft(y, n_fft=hp.num_fft , hop_length=hf, win_length=wf))
    if mel: S = librosa.feature.melspectrogram(S=S, sr=hp.sample_rate, n_mels=hp.num_mels)
    return amplitude_to_db(S) - hp.reference_spectrogram_db


def mel_spectrogram(y):
    """Convert waveform to log-mel-spectrogram."""
    return spectrogram(y, True)


def inverse_spectrogram(s, mel=False):
    """Convert log-magnitude spectrogram to waveform."""
    S = db_to_amplitude(s + hp.reference_spectrogram_db)
    wf = ms_to_frames(hp.stft_window_ms)
    hf = ms_to_frames(hp.stft_shift_ms)
    if mel: S = librosa.feature.inverse.mel_to_stft(S, power=1, sr=hp.sample_rate, n_fft=hp.num_fft)
    y = librosa.griffinlim(S ** hp.griffin_lim_power, n_iter=hp.griffin_lim_iters, hop_length=hf, win_length=wf)
    if hp.use_preemphasis: y = deemphasis(y)
    return y


def inverse_mel_spectrogram(s):
    """Convert log-mel-spectrogram to waveform."""
    return inverse_spectrogram(s, True)


def normalize_spectrogram(S):
    """Normalize log-magnitude spectrogram."""
    # One should consider other setup:
    # https://github.com/keithito/tacotron/issues/98
    # https://github.com/Rayhane-mamah/Tacotron-2/issues/18#issuecomment-382637788
    assert S.max() <= 0
    return (S - hp.normalize_mean) / hp.normalize_variance


def denormalize_spectrogram(S):
    """Denormalize log-magnitude spectrogram."""
    return S * hp.normalize_variance + hp.normalize_mean
import numpy as np
import scipy
from fastdtw import fastdtw
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
    if mf != 0: data = data[mf:-mf]
    return librosa.effects.trim(data, top_db=top_db, frame_length=wf, hop_length=hf)
  

def duration(data):
    """Return duration of an audio signal in seconds."""
    return librosa.get_duration(data, sr=hp.sample_rate)


def amplitude_to_db(x):
    """Convert amplitude to decibels."""
    return librosa.amplitude_to_db(x, ref=np.max, top_db=None)


def db_to_amplitude(x):
    """Convert decibels to amplitude."""
    return librosa.db_to_amplitude(x)


def preemphasis(y):
    """Preemphasize the signal."""
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
    S = np.abs(librosa.stft(y, n_fft=hp.num_fft, hop_length=hf, win_length=wf))
    if mel: S = librosa.feature.melspectrogram(S=S, sr=hp.sample_rate, n_mels=hp.num_mels)
    return amplitude_to_db(S)


def mel_spectrogram(y):
    """Convert waveform to log-mel-spectrogram."""
    return spectrogram(y, True)


def linear_to_mel(S):
    """Convert linear to mel spectrogram (this does not return the same spec. as mel_spec. method due to the db->amplitude conversion)."""
    S = db_to_amplitude(S)
    S = librosa.feature.melspectrogram(S=S, sr=hp.sample_rate, n_mels=hp.num_mels)
    return amplitude_to_db(S)


def inverse_spectrogram(s, mel=False):
    """Convert log-magnitude spectrogram to waveform."""
    S = db_to_amplitude(s)
    wf = ms_to_frames(hp.stft_window_ms)
    hf = ms_to_frames(hp.stft_shift_ms)
    if mel: S = librosa.feature.inverse.mel_to_stft(S, power=1, sr=hp.sample_rate, n_fft=hp.num_fft)
    y = librosa.griffinlim(S ** hp.griffin_lim_power, n_iter=hp.griffin_lim_iters, hop_length=hf, win_length=wf)
    if hp.use_preemphasis: y = deemphasis(y)
    y /= max(y)
    return y


def inverse_mel_spectrogram(s):
    """Convert log-mel-spectrogram to waveform."""
    return inverse_spectrogram(s, True)


def normalize_spectrogram(S, is_mel):
    """Normalize log-magnitude spectrogram."""
    if is_mel: return (S - hp.mel_normalize_mean) / hp.mel_normalize_variance
    else:      return (S - hp.lin_normalize_mean) / hp.lin_normalize_variance


def denormalize_spectrogram(S, is_mel):
    """Denormalize log-magnitude spectrogram."""
    if is_mel: return S * hp.mel_normalize_variance + hp.mel_normalize_mean
    else:      return S * hp.lin_normalize_variance + hp.lin_normalize_mean


def get_spectrogram_mfcc(S):
    """Compute MFCCs of a mel spectrogram."""
    return librosa.feature.mfcc(n_mfcc=hp.num_mfcc, S=(S/10))


def get_mfcc(y):
    """Compute MFCCs of a waveform."""
    return get_mfcc(audio.mel_spectrogram(y))


def mel_cepstral_distorision(S1, S2, mode):
    """Compute Mel Cepstral Distorsion between two mel spectrograms.

    Arguments:
        S1 and S2 -- mel spectrograms
        mode -- 'cut' to cut off frames of longer seq.
                'stretch' to stretch linearly the shorter seq.
                'dtw' to compute DTW with minimal possible MCD
    """

    def mcd(s1, s2):
        diff = s1 - s2
        return np.average(np.sqrt(np.sum(diff*diff, axis=0)))

    x, y = get_spectrogram_mfcc(S1)[1:], get_spectrogram_mfcc(S2)[1:]

    if mode == 'cut':
        if y.shape[1] > x.shape[1]: y = y[:,:x.shape[1]]
        if x.shape[1] > y.shape[1]: x = x[:,:y.shape[1]]

    elif mode == 'stretch':
        if x.shape[1] > y.shape[1]:
            m = x.shape[1]
            y = np.array([y[:, i * y.shape[1]//m] for i in range(m)]).T
        else:
            m = y.shape[1]
            x = np.array([x[:, i * x.shape[1]//m] for i in range(m)]).T

    elif mode == 'dtw':       
        x, y = x.T, y.T
        _, path = fastdtw(x, y, dist=mcd)     
        pathx, pathy = map(list,zip(*path))    
        x, y = x[pathx].T, y[pathy].T

    return mcd(x, y)
import random

import librosa.display
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import audio, text
from params.params import Params as hp


class Logger:
    @staticmethod
    def initialize(logdir, flush_seconds):
        Logger._sw = SummaryWriter(log_dir=logdir, flush_secs=flush_seconds)

    @staticmethod
    def progress(progress, prefix='', length=70):
        progress *= 100
        step = 100/length
        filled, reminder = int(progress // step), progress % step
        loading_bar = filled * '█'
        loading_bar += '░' if reminder < step / 3 else '▒' if reminder < step * 2/3 else '▓'
        loading_bar += max(0, length - filled) * '░' if progress < 100 else ''
        print(f'\r{prefix} {loading_bar} {progress:.1f}%', end=('' if progress < 100 else '\n'), flush=True)

    @staticmethod
    def training(train_step, losses, gradient, learning_rate, duration):
        """Log batch training."""  
        total_loss = sum(losses.values())
        Logger._sw.add_scalar(f'Loss/train_total', total_loss, train_step)
        for n, l in losses.items():
            Logger._sw.add_scalar(f'Loss/train_{n}', l, train_step)  
        Logger._sw.add_scalar("GradientNorm/train", gradient, train_step)
        Logger._sw.add_scalar("LearningRate/train", learning_rate, train_step)
        Logger._sw.add_scalar("Duration/train", duration, train_step)

    @staticmethod
    def evaluation(log_name, epoch, losses, source, target, prediction, target_len, source_len, stop_target, stop_prediction, alignment):
        """Log evaluation results."""

        # log loss functions
        total_loss = sum(losses.values())
        Logger._sw.add_scalar(f'Loss/{log_name}_total', total_loss, epoch)
        for n, l in losses.items():
            Logger._sw.add_scalar(f'Loss/{log_name}_{n}', l, epoch) 

        # show random output - spectrogram, stop token, alignment and audio
        idx = random.randint(0, alignment.size(0) - 1)
        # log spectrograms
        predicted_spec = prediction[idx].data.cpu().numpy()[:, :target_len[idx]]
        target_spec = target[idx].data.cpu().numpy()[:, :target_len[idx]]
        if hp.normalize_spectrogram:
            predicted_spec = audio.denormalize_spectrogram(predicted_spec, not hp.predict_linear)
            target_spec = audio.denormalize_spectrogram(target_spec, not hp.predict_linear)
        waveform = audio.inverse_spectrogram(predicted_spec, not hp.predict_linear)
        Logger._sw.add_audio(f"Audio/{log_name}", waveform, epoch, sample_rate=hp.sample_rate)
        Logger._sw.add_figure(f"Mel_predicted/{log_name}", Logger._plot_spectrogram(predicted_spec), epoch)
        Logger._sw.add_figure(f"Mel_target/{log_name}", Logger._plot_spectrogram(target_spec), epoch)     
        # log alignment
        alignment = alignment[idx].data.cpu().numpy().T
        alignment = alignment[:source_len[idx], :target_len[idx]]
        Logger._sw.add_figure(f"Alignment/{log_name}", Logger._plot_alignment(alignment), epoch)          
        # log source text
        utterance = text.to_text(source[idx].data.cpu().numpy()[:source_len[idx]], hp.use_phonemes)
        Logger._sw.add_text(f"Text/{log_name}", utterance, epoch) 
        # log stop tokens
        Logger._sw.add_figure(f"Stop/{log_name}", Logger._plot_stop_tokens(stop_target[idx].data.cpu().numpy(), stop_prediction[idx].data.cpu().numpy()), epoch) 

    @staticmethod
    def _plot_spectrogram(s):
        fig = plt.figure(figsize=(16, 4))
        librosa.display.specshow(s + hp.reference_spectrogram_db, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        return fig

    @staticmethod
    def _plot_alignment(alignment):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        cax = ax.imshow(alignment, origin='lower', aspect='auto', interpolation='none')
        fig.colorbar(cax, ax=ax)
        plt.ylabel('Input index')
        plt.xlabel('Decoder step')
        plt.tight_layout() 
        return fig

    @staticmethod
    def _plot_stop_tokens(target, prediciton):
        fig = plt.figure(figsize=(14, 4))
        ax = fig.add_subplot(111)
        ax.scatter(range(len(target)), target, alpha=0.5, color='blue', marker='+', s=1, label='target')
        ax.scatter(range(len(prediciton)), prediciton, alpha=0.5, color='red', marker='.', s=1, label='predicted')
        plt.xlabel("Frames (Blue target, Red predicted)")
        plt.ylabel("Stop token probability")
        plt.tight_layout()
        return fig

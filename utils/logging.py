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
        Logger._sw.add_scalar(f'Train/loss_total', total_loss, train_step)
        for n, l in losses.items():
            Logger._sw.add_scalar(f'Train/loss_{n}', l, train_step)  
        Logger._sw.add_scalar("Train/gradient_norm", gradient, train_step)
        Logger._sw.add_scalar("Train/learning_rate", learning_rate, train_step)
        Logger._sw.add_scalar("Train/duration", duration, train_step)

    @staticmethod
    def evaluation(eval_step, losses, mcd, source_len, target_len, source, target, prediction_forced, prediction, stop_prediction, stop_target, alignment):
        """Log evaluation results."""

        # Log losses
        total_loss = sum(losses.values())
        Logger._sw.add_scalar(f'Eval/loss_total', total_loss, eval_step)
        for n, l in losses.items():
            Logger._sw.add_scalar(f'Eval/loss_{n}', l, eval_step) 

        # Show random sample: spectrogram, stop token probability, alignment and audio
        idx = random.randint(0, alignment.size(0) - 1)
        predicted_spec = prediction[idx, :, :target_len[idx]].data.cpu().numpy()
        f_predicted_spec = prediction_forced[idx, :, :target_len[idx]].data.cpu().numpy()
        target_spec = target[idx, :, :target_len[idx]].data.cpu().numpy()  

        # Log spectrograms
        if hp.normalize_spectrogram:
            predicted_spec = audio.denormalize_spectrogram(predicted_spec, not hp.predict_linear)
            f_predicted_spec = audio.denormalize_spectrogram(f_predicted_spec, not hp.predict_linear)
            target_spec = audio.denormalize_spectrogram(target_spec, not hp.predict_linear)
        Logger._sw.add_figure(f"Predicted/generated", Logger._plot_spectrogram(predicted_spec), eval_step)
        Logger._sw.add_figure(f"Predicted/forced", Logger._plot_spectrogram(f_predicted_spec), eval_step)
        Logger._sw.add_figure(f"Target/eval", Logger._plot_spectrogram(target_spec), eval_step) 
        # Log audio
        waveform = audio.inverse_spectrogram(predicted_spec, not hp.predict_linear)
        Logger._sw.add_audio(f"Audio/generated", waveform, eval_step, sample_rate=hp.sample_rate)  
        waveform = audio.inverse_spectrogram(f_predicted_spec, not hp.predict_linear)
        Logger._sw.add_audio(f"Audio/forced", waveform, eval_step, sample_rate=hp.sample_rate)              
        # Log alignment
        alignment = alignment[idx, :target_len[idx], :source_len[idx]].data.cpu().numpy().T
        Logger._sw.add_figure(f"Alignment/eval", Logger._plot_alignment(alignment), eval_step)                
        # Log source text
        utterance = text.to_text(source[idx].data.cpu().numpy()[:source_len[idx]], hp.use_phonemes)
        Logger._sw.add_text(f"Text/eval", utterance, eval_step)      
        # Log stop tokens
        Logger._sw.add_figure(f"Stop/eval", Logger._plot_stop_tokens(stop_target[idx].data.cpu().numpy(), stop_prediction[idx].data.cpu().numpy()), eval_step) 
        # Log mel cepstral distorsion
        Logger._sw.add_scalar(f'Eval/mcd', mcd, eval_step)


    @staticmethod
    def _plot_spectrogram(s):
        fig = plt.figure(figsize=(16, 4))
        hf = int(hp.sample_rate * hp.stft_shift_ms / 1000)
        librosa.display.specshow(s, sr=hp.sample_rate, hop_length=hf, x_axis='time', y_axis='mel', cmap='magma')
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

    @staticmethod
    def _plot_mfcc(mfcc):
        fig = plt.figure(figsize=(16, 4))
        librosa.display.specshow(mfcc, x_axis='time', cmap='magma')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()
        return fig

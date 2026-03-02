import torch
import torch.nn.functional as F
import torchaudio


class AudioProcessor:
    def __init__(self, config):
        self.config = config

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            center=False,
            power=2.0
        )

        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=config.n_fft // 2 + 1,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate
        )

        self.griffin = torchaudio.transforms.GriffinLim(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            power=2.0,
            n_iter=64   # Number of Griffin-Lim iterations
        )

    def process_audio(self, waveform):
        """
        waveform: Tensor [T] or [1, T]
        returns: log-mel [n_mels, frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        max_len = int(self.config.sample_rate * self.config.max_duration)

        if waveform.shape[-1] > max_len:
            waveform = waveform[..., :max_len]
        else:
            waveform = F.pad(
                waveform,
                (0, max_len - waveform.shape[-1])
            )

        mel = self.mel(waveform)
        log_mel = torch.log(mel + 1e-9)

        return log_mel.squeeze(0)

    def reconstruct_audio(self, log_mel):
        """
        log_mel: [n_mels, frames]
        returns: waveform [T]
        """
        mel = torch.exp(log_mel).cpu()

        linear_spec = self.inverse_mel(mel)
        audio = self.griffin(linear_spec)

        return audio

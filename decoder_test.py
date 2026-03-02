import os
os.environ["WANDB_SYMLINK"] = "false"  # Windows fix

import torch
import torchaudio

from config import ModelConfig
from model import VocalSoundTransformer
from audio_utils import AudioProcessor

DATA_DIR = r"C:/Desktop/datasets/audio_16_VocalSound"
import os
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
SAMPLE_FILE = r"C:/Desktop/test_audio/f0003_0_laughter.wav"

print("Using file:", SAMPLE_FILE)


def main():
    device = torch.device("cpu")
    print("Using device:", device)

    config = ModelConfig()

    model = VocalSoundTransformer(config).to(device)
    ckpt_path = "checkpoints/best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("Model weights loaded")

    processor = AudioProcessor(config)

    # real audio loading
    waveform, sr = torchaudio.load(SAMPLE_FILE)
    if sr != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, config.sample_rate)

    mel_input = processor.process_audio(waveform).unsqueeze(0).to(device)  # [1, n_mels, T]

    label_tensor = torch.tensor([0], device=device)  # adjust class if needed
    with torch.no_grad():
        mel_recon, _ = model(mel_input, label_tensor)

    audio = processor.reconstruct_audio(mel_recon.squeeze(0))

    out_path = "reconstructed.wav"
    torchaudio.save(out_path, audio.unsqueeze(0), sample_rate=config.sample_rate)
    print(f"Saved reconstructed audio: {out_path}")


if __name__ == "__main__":
    main()

import torch

def generate_vocal_sound(model, label, config, device):
    model.eval()
    T = int(config.max_duration * config.sample_rate / config.hop_length)
    print(config.max_duration, config.sample_rate, config.hop_length)

    label_tensor = torch.tensor([label], device=device)

    with torch.no_grad():
        dummy_mel = torch.zeros((1, config.n_mels, T), device=device)
        mel_recon, _ = model(dummy_mel, label_tensor)

    return mel_recon.squeeze(0)  # [n_mels, T]

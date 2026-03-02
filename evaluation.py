import torch
import numpy as np
import gc

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def reduce(values):
    v = np.array(values)
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max())
    }


class EvaluationMetrics:
    # Comprehensive metrics for vocal sound generation

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def signal_to_noise_ratio(self, original, reconstructed):
        noise = original - reconstructed
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean(noise ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        return snr.item()

    def peak_signal_to_noise_ratio(self, original, reconstructed):
        mse = torch.mean((original - reconstructed) ** 2)
        max_val = torch.max(torch.abs(original))
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-10))
        return psnr.item()

    def spectral_convergence(self, original, reconstructed):
        numerator = torch.norm(original - reconstructed, p='fro')
        denominator = torch.norm(original, p='fro')
        return (numerator / (denominator + 1e-10)).item()

    def log_spectral_distance(self, original, reconstructed):
        log_original = torch.log(torch.abs(original) + 1e-10)
        log_reconstructed = torch.log(torch.abs(reconstructed) + 1e-10)
        lsd = torch.sqrt(torch.mean((log_original - log_reconstructed) ** 2))
        return lsd.item()

    def mel_cepstral_distortion(self, original, reconstructed):
        original_mfcc = self._compute_mfcc(original)
        reconstructed_mfcc = self._compute_mfcc(reconstructed)
        diff = original_mfcc - reconstructed_mfcc
        mcd = (10.0 / np.log(10)) * torch.sqrt(2 * torch.sum(diff ** 2, dim=0)).mean()
        return mcd.item()

    def _compute_mfcc(self, mel_spec):
        mel_spec = mel_spec.float()

        # log_mel = torch.log(torch.abs(mel_spec) + 1e-9)
        # DCT-II via FFT trick
        N = mel_spec.size(0)

        # Create mirrored signal
        v = torch.cat([mel_spec, mel_spec.flip(dims=[0])], dim=0)
        V = torch.fft.fft(v, dim=0)
        # Take real part and scale
        mfcc = torch.real(V[:N])
        # Keep first 13 coefficients
        return mfcc[:13]

    def codebook_perplexity(self, codes):
        codes_flat = codes.flatten()
        unique_codes, counts = torch.unique(codes_flat, return_counts=True)
        probabilities = counts.float() / len(codes_flat)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
        perplexity = torch.exp(entropy)
        return perplexity.item(), len(unique_codes)

    def codebook_usage(self, codes, codebook_size):
        unique_codes = torch.unique(codes.flatten())
        usage = (len(unique_codes) / codebook_size) * 100
        return usage.item()

    def frechet_audio_distance_simple(self, original_features, generated_features):
        mu_orig = torch.mean(original_features, dim=0)
        mu_gen = torch.mean(generated_features, dim=0)
        sigma_orig = torch.cov(original_features.T)
        sigma_gen = torch.cov(generated_features.T)
        diff = mu_orig - mu_gen
        covmean = self._matrix_sqrt(sigma_orig @ sigma_gen)
        fad = torch.sum(diff ** 2) + torch.trace(sigma_orig + sigma_gen - 2 * covmean)
        return fad.item()

    def _matrix_sqrt(self, matrix):
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        return eigenvectors @ torch.diag(torch.sqrt(eigenvalues.clamp(min=0))) @ eigenvectors.T

    def compute_all_metrics(self, original_mel, reconstructed_mel, codes_list, codebook_size):
        metrics = {
            'SNR (dB)': self.signal_to_noise_ratio(original_mel, reconstructed_mel),
            'PSNR (dB)': self.peak_signal_to_noise_ratio(original_mel, reconstructed_mel),
            'Spectral Convergence': self.spectral_convergence(original_mel, reconstructed_mel),
            'Log-Spectral Distance': self.log_spectral_distance(original_mel, reconstructed_mel),
            'MCD (dB)': self.mel_cepstral_distortion(original_mel, reconstructed_mel),
        }

        for i, codes in enumerate(codes_list):
            perplexity, unique = self.codebook_perplexity(codes)
            usage = self.codebook_usage(codes, codebook_size)
            metrics[f'Codebook_{i}_Perplexity'] = perplexity
            metrics[f'Codebook_{i}_Unique_Codes'] = unique
            metrics[f'Codebook_{i}_Usage (%)'] = usage

        return metrics


def evaluate_model(model, dataloader, device, config, log_to_wandb=False):
    model.eval()
    evaluator = EvaluationMetrics(sample_rate=config.sample_rate)

    all_metrics = {}
    class_metrics = {}

    with torch.no_grad():
        for batch in dataloader:
            mel = batch["mel"].to(device)
            label = batch["label"].to(device)

            recon, vq_loss = model(mel, label)

            # per sample 
            for i in range(mel.size(0)):
                metrics = evaluator.compute_all_metrics(
                    mel[i],
                    recon[i],
                    codes_list=[],  # TODO: expose VQ codes later
                    codebook_size=config.codebook_size
                )

                metrics["Reconstruction_Loss"] = torch.mean(
                    (mel[i] - recon[i]) ** 2
                ).item()
                metrics["VQ_Loss"] = vq_loss.item()

                cls = label[i].item()

                for k, v in metrics.items():
                    all_metrics.setdefault(k, []).append(v)
                    class_metrics.setdefault(cls, {}).setdefault(k, []).append(v)

    final_metrics = {k: reduce(v) for k, v in all_metrics.items()}
    final_class_metrics = {
        cls: {k: reduce(v) for k, v in m.items()}
        for cls, m in class_metrics.items()
    }

    if log_to_wandb:
        if not _WANDB_AVAILABLE:
            raise RuntimeError("wandb not installed")
        wandb.log({f"eval/{k}": v["mean"] for k, v in final_metrics.items()})

    gc.collect()

    return final_metrics, final_class_metrics



def print_metrics_table(metrics, class_metrics=None, label_names=None):
    print("\n" + "=" * 80)
    print(" OVERALL EVALUATION METRICS")
    print("=" * 80)

    quality_metrics = [
        'SNR (dB)', 'PSNR (dB)', 'MCD (dB)',
        'Spectral Convergence', 'Log-Spectral Distance'
    ]

    print("\n Audio Quality Metrics:")
    print(f"{'Metric':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)

    for key in quality_metrics:
        if key in metrics:
            m = metrics[key]
            print(f"{key:<30} {m['mean']:>11.4f} {m['std']:>11.4f} {m['min']:>11.4f} {m['max']:>11.4f}")

    print("\n Loss Metrics:")
    for key in ['Reconstruction_Loss', 'VQ_Loss']:
        if key in metrics:
            m = metrics[key]
            print(f"{key:<30} {m['mean']:>11.6f} {m['std']:>11.6f} {m['min']:>11.6f} {m['max']:>11.6f}")

    if class_metrics and label_names:
        print("\n" + "=" * 80)
        print(" PER-CLASS METRICS")
        print("=" * 80)
        for cls, m in class_metrics.items():
            name = label_names[cls] if label_names else str(cls)
            print(f"\n Class {cls} — {name.upper()}")
            print(f"{'Metric':<30} {'Mean':<12} {'Std':<12}")
            print("-" * 60)
            for key in ['SNR (dB)', 'PSNR (dB)', 'MCD (dB)', 'Spectral Convergence', 'Reconstruction_Loss']:
                if key in m:
                    print(f"{key:<30} {m[key]['mean']:>11.4f} {m[key]['std']:>11.4f}")
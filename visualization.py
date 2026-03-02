import torch
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def visualize_results(
    mel_original,
    mel_reconstructed,
    audio_processor,
    config,
    log_to_wandb=False,
    step=None,
    show_plots=True
):

    if log_to_wandb and not _WANDB_AVAILABLE:
        raise RuntimeError("wandb is not installed but log_to_wandb=True")

    mel_original = mel_original.detach().cpu()
    mel_reconstructed = mel_reconstructed.detach().cpu()

    diff = torch.abs(mel_original - mel_reconstructed)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # Original mel
    axes[0, 0].imshow(
        mel_original.numpy(),
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )
    axes[0, 0].set_title("Original Mel Spectrogram")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Mel Frequency")

    # Reconstructed mel
    axes[0, 1].imshow(
        mel_reconstructed.numpy(),
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )
    axes[0, 1].set_title("Reconstructed Mel Spectrogram")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Mel Frequency")

    # Absolute difference
    axes[1, 0].imshow(
        diff.numpy(),
        aspect="auto",
        origin="lower",
        cmap="hot"
    )
    axes[1, 0].set_title("Absolute Difference")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Mel Frequency")

    # Histogram of differences
    axes[1, 1].hist(
        diff.flatten().numpy(),
        bins=50
    )
    axes[1, 1].set_title("Distribution of Absolute Differences")
    axes[1, 1].set_xlabel("Absolute Difference")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()

    if log_to_wandb:
        wandb.log(
            {
                "mel/original": wandb.Image(axes[0, 0]),
                "mel/reconstructed": wandb.Image(axes[0, 1]),
                "mel/difference": wandb.Image(axes[1, 0]),
                "mel/diff_histogram": wandb.Image(axes[1, 1]),
            },
            step=step
        )

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    audio_orig = audio_processor.reconstruct_audio(mel_original)
    audio_recon = audio_processor.reconstruct_audio(mel_reconstructed)

    if log_to_wandb:
        wandb.log(
            {
                "audio/original": wandb.Audio(
                    audio_orig.numpy(),
                    sample_rate=config.sample_rate,
                    caption="Original"
                ),
                "audio/reconstructed": wandb.Audio(
                    audio_recon.numpy(),
                    sample_rate=config.sample_rate,
                    caption="Reconstructed"
                ),
            },
            step=step
        )
    else:
        try:
            from IPython.display import Audio, display
            print("\nOriginal Audio:")
            display(Audio(audio_orig.numpy(), rate=config.sample_rate))
            print("\nReconstructed Audio:")
            display(Audio(audio_recon.numpy(), rate=config.sample_rate))
        except ImportError:
            pass

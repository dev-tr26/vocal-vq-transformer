#  VocalSound Transformer

A deep learning project for **vocal sound reconstruction and generation** using a Transformer-based autoencoder with Vector Quantization (VQ). Trained on the [VocalSound dataset](https://github.com/YuanGongND/vocalsound), the model learns to encode, compress, and reconstruct six categories of human vocal sounds from mel-spectrograms.

---

## Project Structure

```
├── config.py           # All hyperparameters and model configuration
├── model.py            # VocalSoundTransformer architecture (Transformer + VQ)
├── audio_utils.py      # Audio processing: mel-spectrogram & Griffin-Lim reconstruction
├── dataset.py          # VocalSoundDataset & data loading utilities
├── train.py            # Training loop with validation and checkpointing
├── evaluation.py       # Comprehensive audio quality metrics
├── generation.py       # Inference: generate vocal sounds from class labels
├── decoder_test.py     # Test reconstruction on a real audio file
├── data_cleaning.py    # Fix JSON dataset file paths for local setup
└── checkpoints/        # Saved model checkpoints (auto-created during training)
```

---

##  Model Architecture

**`VocalSoundTransformer`** (`model.py`) is a Transformer-based autoencoder:

- **Input Projection** — Linear layer maps mel-spectrogram bins → hidden dimension
- **Class Conditioning** — Learned class embeddings added to the sequence
- **Positional Encoding** — Sinusoidal positional encodings
- **Transformer Encoder** — Multi-head self-attention (configurable layers/heads)
- **Residual Vector Quantization (RVQ)** — Multiple VQ codebooks applied sequentially on residuals for discrete latent representation
- **Output Projection** — Linear layer maps hidden dim → mel-spectrogram bins

```
Mel Input → Linear → Class Embed + Pos Enc → Transformer → RVQ → Transformer → Linear → Mel Recon
```

### Supported Vocal Sound Classes

| Index | Class             |
|-------|-------------------|
| 0     | Laughter          |
| 1     | Sigh              |
| 2     | Cough             |
| 3     | Throat Clearing   |
| 4     | Sneeze            |
| 5     | Sniff             |

---

## Configuration

All settings are centralized in `config.py` via the `ModelConfig` dataclass:

| Category   | Parameter        | Default  | Description                        |
|------------|------------------|----------|------------------------------------|
| Audio      | `sample_rate`    | 16000    | Audio sample rate (Hz)             |
| Audio      | `n_mels`         | 80       | Number of mel filterbanks          |
| Audio      | `n_fft`          | 1024     | FFT window size                    |
| Audio      | `hop_length`     | 256      | STFT hop length                    |
| Audio      | `max_duration`   | 10.0     | Max audio clip length (seconds)    |
| Model      | `hidden_dim`     | 512      | Transformer hidden dimension       |
| Model      | `num_layers`     | 6        | Number of Transformer encoder layers |
| Model      | `num_heads`      | 8        | Number of attention heads          |
| Model      | `ff_dim`         | 2048     | Feed-forward dimension             |
| Model      | `dropout`        | 0.1      | Dropout rate                       |
| VQ         | `num_codebooks`  | 2        | Number of RVQ codebooks            |
| VQ         | `codebook_size`  | 1024     | Entries per codebook               |
| Training   | `batch_size`     | 8        | Training batch size                |
| Training   | `learning_rate`  | 1e-4     | AdamW learning rate                |
| Training   | `num_epochs`     | 20       | Total training epochs              |
| Training   | `gradient_clip`  | 1.0      | Gradient clipping norm             |

---

### 1. Install Dependencies

```bash
pip install torch torchaudio numpy pandas wandb tqdm
```

### 2. Prepare the Dataset

Download the [VocalSound dataset](https://github.com/YuanGongND/vocalsound) and update the paths in `data_cleaning.py`:

```python
json_dir  = Path("path/to/datafiles")
audio_base = Path("path/to/audio_16k")
```

Then run the path-fixing script:

```bash
python data_cleaning.py
```

This updates the `.json` split files (`tr.json`, `val.json`, `te.json`) to use your local audio paths.

### 3. Train the Model

```python
from config import ModelConfig
from model import VocalSoundTransformer
from dataset import VocalSoundDataset
from train import train_model
from torch.utils.data import DataLoader
import torch

config = ModelConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = VocalSoundDataset("train", config, data_dir="path/to/VocalSound")
val_ds   = VocalSoundDataset("val",   config, data_dir="path/to/VocalSound")

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=config.batch_size)

model = VocalSoundTransformer(config).to(device)
train_model(model, train_loader, val_loader, config, device)
```

Checkpoints are saved to `checkpoints/` after each epoch. The best model (lowest validation loss) is saved as `checkpoints/best_model.pt`.

### 4. Generate a Vocal Sound

```python
from generation import generate_vocal_sound
from audio_utils import AudioProcessor
import torchaudio, torch

# label: 0=laughter, 1=sigh, 2=cough, 3=throat_clearing, 4=sneeze, 5=sniff
mel = generate_vocal_sound(model, label=0, config=config, device=device)
audio = AudioProcessor(config).reconstruct_audio(mel)
torchaudio.save("output.wav", audio.unsqueeze(0), sample_rate=config.sample_rate)
```

### 5. Test Reconstruction on a Real File

```bash
python decoder_test.py
```

Edit the `SAMPLE_FILE` path inside `decoder_test.py` to point to your `.wav` file. The reconstructed audio is saved as `reconstructed.wav`.

---

## Evaluation

The `evaluation.py` module computes a comprehensive set of audio quality metrics:

| Metric                    | Description                                          |
|---------------------------|------------------------------------------------------|
| **SNR (dB)**              | Signal-to-Noise Ratio                                |
| **PSNR (dB)**             | Peak Signal-to-Noise Ratio                           |
| **MCD (dB)**              | Mel-Cepstral Distortion                              |
| **Spectral Convergence**  | Frobenius norm ratio of spectral error               |
| **Log-Spectral Distance** | RMS distance in log-spectral domain                  |
| **Codebook Perplexity**   | Diversity measure per VQ codebook                    |
| **Codebook Usage (%)**    | Percentage of codebook entries actively used         |
| **Reconstruction Loss**   | MSE between input and reconstructed mel-spectrogram  |
| **VQ Loss**               | Vector quantization commitment loss                  |

Metrics are reported globally and broken down **per vocal sound class**. Results can optionally be logged to [Weights & Biases](https://wandb.ai/).

---

##  Experiment Tracking

Set `use_wandb: True` in `ModelConfig` to enable W&B logging. Training logs include per-epoch train/val reconstruction loss, VQ loss, and learning rate. Evaluation logs include all quality metrics per class.

---

##  Key Design Choices

- **Residual Vector Quantization (RVQ)** — Multiple codebooks applied on successive residuals allow richer discrete representations without increasing codebook size.
- **Class Conditioning** — Learned class embeddings injected at every sequence position guide the model to generate class-specific reconstructions.
- **Griffin-Lim Vocoder** — Used for phase reconstruction when converting mel-spectrograms back to waveforms (64 iterations).
- **Dual Transformer Pass** — The encoder runs before *and* after VQ quantization, letting the model refine quantized representations.

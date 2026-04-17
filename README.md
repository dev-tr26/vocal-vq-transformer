#  VocalSound Transformer

- A deep learning project for **vocal sound reconstruction and generation** using a Transformer-based autoencoder with Vector Quantization (VQ).  It is just a small experiment .

- Trained on the [VocalSound dataset](https://github.com/YuanGongND/vocalsound), the model learns to encode, compress, and reconstruct six categories of human vocal sounds from mel-spectrograms.

---

##  Model Architecture

**`VocalSoundTransformer`** is a Transformer-based autoencoder:

- **Input Projection** — Linear layer maps mel-spectrogram bins → hidden dimension
- **Class Conditioning** — Learned class embeddings added to the sequence
- **Positional Encoding** — Sinusoidal positional encodings
- **Transformer Encoder** — Multi-head self-attention (configurable layers/heads)
- **Residual Vector Quantization (RVQ)** — Multiple VQ codebooks applied sequentially on residuals for discrete latent representation
- **Output Projection** — Linear layer maps hidden dim -> mel-spectrogram bins

```
Mel Input -> Linear -> Class Embed + Pos Enc -> Transformer -> RVQ -> Transformer -> Linear -> Mel Recon
```

### Supported Vocal Sound Classes

0. Laughter
1. Sigh  
2. Cough 
3. Throat Clearing
4. Sniff 
5. Sneeze

---


- Dataset [VocalSound dataset](https://github.com/YuanGongND/vocalsound)

- split files (`tr.json`, `val.json`, `te.json`) to use local audio paths.

- Checkpoints are saved to `checkpoints/` after each epoch. The best model (lowest validation loss) is saved as `checkpoints/best_model.pt`.





## Results

Final evaluation metrics after training (global + per-class breakdown).

---

##  Overall Evaluation (All Classes)

| Metric                    | Value   |
| ------------------------- | ------- |
| **Reconstruction Loss**   | 52.59   |
| **VQ Loss**               | 1.9427  |
| **PSNR (dB)**             | 8.62    |
| **SNR (dB)**              | 3.38    |
| **Log-Spectral Distance** | 1.129   |
| **Spectral Convergence**  | 0.720   |
| **MCD (dB)**              | 5826.74 |

---

# Per-Class Metrics

---

## Class 0 — Laughter

| Metric                | Mean    | Std     |
| --------------------- | ------- | ------- |
| Reconstruction Loss   | 56.71   | 22.26   |
| VQ Loss               | 1.9427  | ~0      |
| PSNR (dB)             | 8.14    | 2.03    |
| SNR (dB)              | 2.35    | 3.18    |
| Log-Spectral Distance | 1.257   | 0.257   |
| Spectral Convergence  | 0.823   | 0.371   |
| MCD (dB)              | 6210.08 | 1594.75 |

---

## Class 1 — Sigh

| Metric                | Mean    | Std     |
| --------------------- | ------- | ------- |
| Reconstruction Loss   | 44.94   | 21.93   |
| VQ Loss               | 1.9427  | ~0      |
| PSNR (dB)             | 9.27    | 2.20    |
| SNR (dB)              | 3.79    | 2.84    |
| Log-Spectral Distance | 1.133   | 0.255   |
| Spectral Convergence  | 0.686   | 0.277   |
| MCD (dB)              | 5336.18 | 1629.12 |

---

## Class 2 — Cough

| Metric                | Mean    | Std     |
| --------------------- | ------- | ------- |
| Reconstruction Loss   | 53.50   | 21.19   |
| VQ Loss               | 1.9427  | ~0      |
| PSNR (dB)             | 8.21    | 2.03    |
| SNR (dB)              | 2.79    | 3.03    |
| Log-Spectral Distance | 1.135   | 0.225   |
| Spectral Convergence  | 0.776   | 0.331   |
| MCD (dB)              | 5859.02 | 1548.31 |

---

## Class 3 — Throat Clearing

| Metric                | Mean    | Std     |
| --------------------- | ------- | ------- |
| Reconstruction Loss   | 58.17   | 22.04   |
| VQ Loss               | 1.9427  | ~0      |
| PSNR (dB)             | 8.29    | 1.76    |
| SNR (dB)              | 3.34    | 2.52    |
| Log-Spectral Distance | 1.113   | 0.203   |
| Spectral Convergence  | 0.715   | 0.272   |
| MCD (dB)              | 6172.04 | 1611.32 |

---

## Class 4 — Sneeze

| Metric                | Mean    | Std     |
| --------------------- | ------- | ------- |
| Reconstruction Loss   | 50.76   | 22.42   |
| VQ Loss               | 1.9427  | ~0      |
| PSNR (dB)             | 8.99    | 1.85    |
| SNR (dB)              | 4.31    | 2.28    |
| Log-Spectral Distance | 0.990   | 0.217   |
| Spectral Convergence  | 0.634   | 0.217   |
| MCD (dB)              | 5624.20 | 1666.05 |

---

## Class 5 — Sniff

| Metric                | Mean    | Std     |
| --------------------- | ------- | ------- |
| Reconstruction Loss   | 51.46   | 23.52   |
| VQ Loss               | 1.9427  | ~0      |
| PSNR (dB)             | 8.83    | 1.98    |
| SNR (dB)              | 3.70    | 2.63    |
| Log-Spectral Distance | 1.145   | 0.250   |
| Spectral Convergence  | 0.689   | 0.277   |
| MCD (dB)              | 5757.78 | 1724.64 |

---



##  Key Design Choices

- **Residual Vector Quantization (RVQ)** — Multiple codebooks applied on successive residuals allow richer discrete representations without increasing codebook size.
- **Class Conditioning** — Learned class embeddings injected at every sequence position guide the model to generate class-specific reconstructions.
- **Griffin-Lim Vocoder** — Used for phase reconstruction when converting mel-spectrograms back to waveforms (64 iterations).
- **Dual Transformer Pass** — The encoder runs before *and* after VQ quantization, letting the model refine quantized representations.


## Limitations

---

### Mel-Cepstral Distortion (MCD)

- Important: The reported MCD values are currently computed incorrectly and should not be interpreted as physically meaningful. 

### Codebook Usage Not Logged

Although VQ loss is reported, the following are not currently tracked:

- Codebook usage percentage
- Codebook perplexity
- Number of active embeddings

Without these metrics, codebook health and latent utilization cannot be fully assessed.

### Metric Interpretation

- Reconstruction loss is computed in mel-spectrogram space (MSE).
- PSNR and SNR are waveform-domain metrics.
- Log-Spectral Distance and Spectral Convergence operate in the frequency domain.

Because different metrics operate in different domains (mel vs waveform vs spectrum), they may not correlate perfectly with perceived audio quality.

--- 

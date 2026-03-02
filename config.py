from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Audio
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    max_duration: float = 10.0

    # Model
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 1000

    # VQ
    num_codebooks: int = 2
    codebook_size: int = 256

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 20
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Classes
    num_classes: int = 6


    # Logging
    use_wandb: bool = True

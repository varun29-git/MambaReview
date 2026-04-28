from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    seq_len: int = 256
    vocab_size: int = 4096
    max_steps: int = 5000
    lr: float = 1e-3
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0


@dataclass(frozen=True)
class Mamba1Config:
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


@dataclass(frozen=True)
class Mamba2Config:
    # This is intentionally a closer parameter match to the Mamba-1 baseline.
    # The old d_model=256 setup made Mamba-2 much smaller, which muddied the comparison.
    d_model: int = 288
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 64
    ngroups: int = 1
    tie_embeddings: bool = True


@dataclass(frozen=True)
class Mamba3SISOConfig:
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    tie_embeddings: bool = True


TRAIN_CONFIG = TrainConfig()

MODEL_CONFIGS = {
    "mamba1": Mamba1Config(),
    "mamba2": Mamba2Config(),
    "mamba3_siso": Mamba3SISOConfig(),
}


MODEL_DISPLAY_NAMES = {
    "mamba1": "Vanilla Mamba",
    "mamba2": "Mamba-2",
    "mamba3_siso": "Mamba-3 SISO",
}

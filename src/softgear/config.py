"""Configuration dataclasses for SoftGear."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 10
    hidden_dim: int = 128
    num_heads: int = 8
    ffn_dim: int = 512
    num_gears: int = 7
    dropout: float = 0.1


@dataclass
class DataConfig:
    path: str = "data/sudoku-extreme"
    batch_size: int = 64
    num_workers: int = 4
    max_samples: int | None = None


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    hardening: str = "gradual"
    lr_decay: float = 0.5
    binary_factor: float = 0.4
    ema_alphas: list[float] = field(default_factory=list)
    patience: int = 5
    gradient_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"


@dataclass
class WandbConfig:
    project: str | None = None
    entity: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    wandb: WandbConfig = field(default_factory=WandbConfig)

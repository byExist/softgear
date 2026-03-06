"""Configuration dataclasses for SoftGear."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    hidden_dim: int
    num_heads: int
    ffn_dim: int
    num_gears: int
    dropout: float
    identity_init: bool
    scale: float


@dataclass
class DataConfig:
    path: str
    batch_size: int
    num_workers: int
    max_samples: int | None
    curriculum: bool


@dataclass
class TrainingConfig:
    lr: float
    optimizer: str
    weight_decay: float
    hardening: str
    lr_decay: float
    binary_factor: float
    ema_alpha: float
    patience: int
    min_delta: float
    gradient_clip: float
    max_total_steps: int | None
    checkpoint_dir: str


@dataclass
class WandbConfig:
    project: str | None
    entity: str | None


@dataclass
class Config:
    task: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    seed: int
    wandb: WandbConfig

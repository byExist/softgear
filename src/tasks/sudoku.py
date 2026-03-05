from __future__ import annotations

from collections.abc import Callable

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from src.models.analyzer import Analyzer
from src.models.chain import Chain
from src.models.gear import Gear

SEQ_LEN = 81


class SudokuEncoder(nn.Module):
    """Token + positional embedding for 9x9 Sudoku boards."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(SEQ_LEN, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.arange(x.size(-1), device=x.device)
        return self.token_emb(x) + self.pos_emb(pos)


def build_sudoku_model(cfg: DictConfig) -> Analyzer:
    """Build Analyzer with empty chain."""
    encoder = SudokuEncoder(cfg.vocab_size, cfg.hidden_dim)
    decoder = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
    chain = Chain()
    return Analyzer(encoder, decoder, chain, cfg.hidden_dim)


def make_gear_factory(cfg: DictConfig) -> Callable[[int], Gear]:
    """Factory that creates a single-layer Gear for a given phase index.

    Phase 0: normal init (first gear learns from scratch).
    Phase 1+: identity init (new gear preserves existing block behavior).
    """

    def factory(phase_idx: int) -> Gear:
        return Gear(
            cfg.hidden_dim,
            cfg.num_heads,
            cfg.ffn_dim,
            cfg.dropout,
            identity_init=(phase_idx > 0),
        )

    return factory


def mount_all_gears(model: Analyzer, cfg: DictConfig) -> None:
    """Mount all gears at once (for loading trained models)."""
    factory = make_gear_factory(cfg)
    for i in range(cfg.num_gears):
        model.chain.mount(factory(i))

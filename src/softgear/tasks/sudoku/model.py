from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from softgear.config import ModelConfig
from softgear.models.analyzer import Analyzer
from softgear.models.chain import Chain
from softgear.models.gear import Gear

class SudokuEncoder(nn.Module):
    """Token + positional embedding for Sudoku boards."""

    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.arange(x.size(-1), device=x.device)
        return self.token_emb(x) + self.pos_emb(pos)


def build_sudoku_model(cfg: ModelConfig) -> Analyzer:
    """Build Analyzer with empty chain."""
    encoder = SudokuEncoder(cfg.vocab_size, cfg.seq_len, cfg.hidden_dim)
    decoder = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
    chain = Chain()
    return Analyzer(encoder, decoder, chain, cfg.hidden_dim)


def make_gear_factory(cfg: ModelConfig) -> Callable[[int], Gear]:
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
            identity_init=(cfg.identity_init and phase_idx > 0),
        )

    return factory


def sudoku_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy over all cells: (B, seq, vocab) vs (B, seq)."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def sudoku_predict(logits: Tensor) -> Tensor:
    """Argmax over vocab dimension: (B, seq, vocab) -> (B, seq)."""
    return logits.argmax(dim=-1)


def mount_all_gears(model: Analyzer, cfg: ModelConfig) -> None:
    """Mount all gears at once (for loading trained models)."""
    factory = make_gear_factory(cfg)
    for i in range(cfg.num_gears):
        model.chain.mount(factory(i))

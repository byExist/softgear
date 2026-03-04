import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from src.models.gear import Gear
from src.models.gear_chain import GearChain
from src.models.softgear import SoftGearModel

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


def build_sudoku_model(cfg: DictConfig) -> SoftGearModel:
    """Build a SoftGearModel configured for Sudoku."""
    encoder = SudokuEncoder(cfg.vocab_size, cfg.hidden_dim)
    decoder = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
    gear_chain = GearChain(
        [
            Gear(r, cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim, cfg.dropout)
            for r in cfg.gear_resolutions
        ]
    )
    return SoftGearModel(encoder, decoder, gear_chain, cfg.hidden_dim)

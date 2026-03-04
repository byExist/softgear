import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from src.models.base import BaseModel, ModelOutput
from src.models.gear import Gear
from src.models.gear_chain import GearChain

SEQ_LEN = 81


class SoftGearModel(BaseModel):
    """Full SoftGear model: embedding + gear chain + output head."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embedding = nn.Embedding(SEQ_LEN, cfg.hidden_dim)
        self.gear_chain = GearChain(
            [
                Gear(size, cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim, cfg.dropout)
                for size in cfg.gear_sizes
            ]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.output_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x: Tensor) -> ModelOutput:
        pos = torch.arange(x.size(-1), device=x.device)
        h = self.embedding(x) + self.pos_embedding(pos)
        final, round_outputs = self.gear_chain(h)
        logits = self.output_head(self.norm(final))
        intermediate_logits = [
            self.output_head(self.norm(r)) for r in round_outputs[:-1]
        ]
        return ModelOutput(logits=logits, intermediate_logits=intermediate_logits)

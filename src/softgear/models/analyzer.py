from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from softgear.models.chain import Chain


@dataclass
class ModelOutput:
    logits: Tensor


class Analyzer(nn.Module):
    """Complete differential analyzer: encoder -> chain -> decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        chain: Chain,
        hidden_dim: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.chain = chain
        self.norm = nn.LayerNorm(hidden_dim)
        self.decoder = decoder

    def forward(self, x: Tensor) -> ModelOutput:
        h = self.encoder(x)
        h = self.chain(h)
        logits = self.decoder(self.norm(h))
        return ModelOutput(logits=logits)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

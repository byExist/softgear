from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from src.models.chain import Chain


@dataclass
class ModelOutput:
    logits: Tensor
    intermediate_logits: list[Tensor] | None = None
    halt_round: int | None = None


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
        final, round_outputs = self.chain(h)
        logits = self.decoder(self.norm(final))
        intermediate_logits = [
            self.decoder(self.norm(r)) for r in round_outputs[:-1]
        ]
        halt_round = (
            len(round_outputs)
            if not self.training and len(round_outputs) < self.chain.num_repeats
            else None
        )
        return ModelOutput(
            logits=logits,
            intermediate_logits=intermediate_logits,
            halt_round=halt_round,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

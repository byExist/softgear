from torch import Tensor, nn

from src.models.base import ModelOutput
from src.models.gear_chain import GearChain


class SoftGearModel(nn.Module):
    """Generic SoftGear model: encoder -> gear chain -> decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        gear_chain: GearChain,
        hidden_dim: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.gear_chain = gear_chain
        self.norm = nn.LayerNorm(hidden_dim)
        self.decoder = decoder

    def forward(self, x: Tensor) -> ModelOutput:
        h = self.encoder(x)
        final, round_outputs = self.gear_chain(h)
        logits = self.decoder(self.norm(final))
        intermediate_logits = [
            self.decoder(self.norm(r)) for r in round_outputs[:-1]
        ]
        return ModelOutput(logits=logits, intermediate_logits=intermediate_logits)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

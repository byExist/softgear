from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class ModelOutput:
    logits: Tensor
    intermediate_logits: list[Tensor] | None = None
    halt_round: int | None = None


class BaseModel(nn.Module):
    """Common interface for SoftGear models."""

    def forward(self, x: Tensor) -> ModelOutput:
        raise NotImplementedError

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

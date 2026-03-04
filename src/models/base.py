from dataclasses import dataclass

from torch import Tensor


@dataclass
class ModelOutput:
    logits: Tensor
    intermediate_logits: list[Tensor] | None = None
    halt_round: int | None = None

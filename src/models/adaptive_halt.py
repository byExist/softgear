from collections.abc import Sequence

from torch import Tensor, nn

from src.models.gear import Gear
from src.models.gear_chain import GearChain


class AdaptiveHalt(nn.Module):
    """RD-VLA style adaptive halting based on output change magnitude.

    Halts when ||y_k - y_{k-1}||^2 < delta.
    Active only during inference (eval mode).
    """

    def __init__(self, delta: float = 0.01):
        super().__init__()
        self.delta = delta

    def should_halt(self, prev_output: Tensor, curr_output: Tensor) -> bool:
        diff = (curr_output - prev_output).pow(2).mean()
        return bool(diff.item() < self.delta)


class GearChainWithHalt(GearChain):
    """GearChain that supports adaptive halting during inference."""

    def __init__(self, gears: Sequence[Gear], delta: float = 0.01):
        super().__init__(gears)
        self.halt = AdaptiveHalt(delta)

    def forward(self, h: Tensor) -> tuple[Tensor, list[Tensor]]:
        round_outputs: list[Tensor] = []
        for round_idx in range(self.depth):
            for gear_idx in range(round_idx + 1):
                h = self.gears[gear_idx](h)
            round_outputs.append(h)

            # Adaptive halt: inference only, need at least 2 rounds
            if not self.training and round_idx > 0:
                if self.halt.should_halt(round_outputs[-2], round_outputs[-1]):
                    break

        return h, round_outputs

from __future__ import annotations

from torch import Tensor, nn

from src.models.gear import Gear


class Chain(nn.Module):
    """Weight-tied block: K gears (layers) repeated N times.

    Block = [G1 -> G2 -> ... -> GK]
    Forward: Block(Block(...Block(h)...))  (num_repeats times)

    Each gear has independent weights within the block.
    The block as a whole is weight-tied across iterations.
    """

    def __init__(self, num_repeats: int, halt_delta: float | None = None):
        super().__init__()
        self.gears = nn.ModuleList()
        self.num_repeats = num_repeats
        self.halt_delta = halt_delta

    def mount(self, gear: Gear) -> None:
        """Add a new gear (layer) to the block."""
        self.gears.append(gear)

    def forward(self, h: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Apply the block num_repeats times.

        Returns:
            final_output: output of the last iteration
            iteration_outputs: list of outputs from each iteration (for deep supervision)
        """
        iteration_outputs: list[Tensor] = []
        for i in range(self.num_repeats):
            for gear in self.gears:
                h = gear(h)
            iteration_outputs.append(h)

            if (
                self.halt_delta is not None
                and not self.training
                and i > 0
            ):
                diff = (iteration_outputs[-1] - iteration_outputs[-2]).pow(2).mean()
                if diff.item() < self.halt_delta:
                    break

        return h, iteration_outputs

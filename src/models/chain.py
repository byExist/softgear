from __future__ import annotations

from torch import Tensor, nn

from src.models.gear import Gear


class Chain(nn.Module):
    """Gear chain: sequential stack of gears (layers).

    Gears are mounted progressively during training.
    Forward pass simply runs each gear in order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gears = nn.ModuleList()

    def mount(self, gear: Gear) -> None:
        """Add a new gear (layer) to the chain."""
        self.gears.append(gear)

    def forward(self, h: Tensor) -> Tensor:
        for gear in self.gears:
            h = gear(h)
        return h

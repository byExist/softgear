from collections.abc import Sequence

from torch import Tensor, nn

from src.models.gear import Gear


class GearChain(nn.Module):
    """Chain of gears connected via the reentry pattern.

    Reentry execution (D=4 example):
        Round 1: G1(h)
        Round 2: G2(G1(h))
        Round 3: G3(G2(G1(h)))
        Round 4: G4(G3(G2(G1(h))))

    Each gear has independent weights (no weight-tying).
    Within a gear: residual connections (standard Transformer).
    Between gears: direct pass (no residual).
    """

    def __init__(self, gears: Sequence[Gear]):
        super().__init__()
        self.gears = nn.ModuleList(gears)
        self.depth = len(gears)

    def forward(self, h: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Run the reentry pattern.

        Returns:
            final_output: output of the last round
            round_outputs: list of outputs from each round (for deep supervision)
        """
        round_outputs: list[Tensor] = []
        for round_idx in range(self.depth):
            for gear_idx in range(round_idx + 1):
                h = self.gears[gear_idx](h)
            round_outputs.append(h)
        return h, round_outputs

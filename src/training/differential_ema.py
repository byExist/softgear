from __future__ import annotations

from torch import Tensor

from src.models.softgear import SoftGearModel


class DifferentialEMA:
    """Per-gear EMA with different decay rates.

    Smaller gears (run more often) get lower alpha for faster adaptation.
    Larger gears (run less often) get higher alpha for stability.
    """

    def __init__(self, model: SoftGearModel, alphas: list[float]):
        self.model = model
        gears = list(model.gear_chain.gears)
        if len(alphas) != len(gears):
            raise ValueError(
                f"alphas length ({len(alphas)}) must match gear count ({len(gears)})"
            )
        self.alphas = alphas

        # Build mapping: param_name -> alpha
        self._param_alpha: dict[str, float] = {}
        for gear_idx, gear in enumerate(gears):
            prefix = f"gear_chain.gears.{gear_idx}."
            for name, _ in gear.named_parameters():
                full_name = prefix + name
                self._param_alpha[full_name] = alphas[gear_idx]

        # Initialize shadow params as copies of current params
        self._shadow: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if name in self._param_alpha:
                self._shadow[name] = param.data.clone()

        self._backup: dict[str, Tensor] = {}

    def update(self) -> None:
        """Update shadow parameters: shadow = alpha * shadow + (1 - alpha) * param."""
        for name, param in self.model.named_parameters():
            if name in self._shadow:
                alpha = self._param_alpha[name]
                self._shadow[name].mul_(alpha).add_(param.data, alpha=1.0 - alpha)

    def apply_shadow(self) -> None:
        """Replace model parameters with shadow parameters for evaluation."""
        self._backup.clear()
        for name, param in self.model.named_parameters():
            if name in self._shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self._shadow[name])

    def restore(self) -> None:
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {"shadow": {k: v.clone() for k, v in self._shadow.items()}}

    def load_state_dict(self, state: dict[str, dict[str, Tensor]]) -> None:
        for k, v in state["shadow"].items():
            self._shadow[k] = v.clone()

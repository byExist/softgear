from __future__ import annotations

from torch import Tensor

from softgear.models.analyzer import Analyzer


class DifferentialEMA:
    """Per-gear EMA that supports independent decay rates per gear.

    Currently used with uniform alpha across all gears.
    The per-gear design allows future experimentation with
    differentiated decay rates.

    Gears are discovered lazily via _sync_gears() since they are
    mounted progressively during training.
    """

    def __init__(self, model: Analyzer, alphas: list[float]):
        self.model = model
        self.alphas = alphas
        self._shadow: dict[str, Tensor] = {}
        self._param_alpha: dict[str, float] = {}
        self._registered = 0
        self._backup: dict[str, Tensor] = {}

    def _sync_gears(self) -> None:
        """Discover newly mounted gears and register their parameters."""
        gears = list(self.model.chain.gears)
        for i in range(self._registered, len(gears)):
            if i >= len(self.alphas):
                raise ValueError(f"No alpha for gear {i}")
            prefix = f"chain.gears.{i}."
            for name, param in gears[i].named_parameters():
                full = prefix + name
                self._param_alpha[full] = self.alphas[i]
                self._shadow[full] = param.data.clone()
        self._registered = len(gears)

    def update(self) -> None:
        """Update shadow parameters: shadow = alpha * shadow + (1 - alpha) * param."""
        self._sync_gears()
        for name, param in self.model.named_parameters():
            if name in self._shadow:
                alpha = self._param_alpha[name]
                self._shadow[name].mul_(alpha).add_(param.data, alpha=1.0 - alpha)

    def reset_shadows(self) -> None:
        """Reset shadow parameters to current model parameters.

        Call after restoring model to a previous state (e.g. phase-best)
        so that EMA shadows match the restored weights.
        """
        for name, param in self.model.named_parameters():
            if name in self._shadow:
                self._shadow[name] = param.data.clone()

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

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch.optim import Optimizer

from src.models.analyzer import Analyzer
from src.models.gear import Gear


class ProgressiveDepthScheduler:
    """Progressively mount gears during training.

    Phase 1: mount G1, train with lr
    Phase 2: mount G2 with lr, decay G1 lr by lr_decay
    ...
    Phase D: all gears mounted
    """

    def __init__(
        self,
        model: Analyzer,
        optimizer: Optimizer,
        gear_factory: Callable[[int], Gear],
        num_gears: int,
        base_lr: float = 3e-4,
        lr_decay: float = 0.5,
        patience: int = 5,
    ):
        self.model = model
        self.optimizer = optimizer
        self._gear_factory = gear_factory
        self._num_gears = num_gears
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.patience = patience
        self._phase = 0
        self._val_losses: list[float] = []

    @property
    def max_rounds(self) -> int:
        return self._num_gears

    def current_phase(self) -> int:
        return self._phase

    def advance_phase(self) -> None:
        """Create and mount the next gear, decay lr of existing gears."""
        if self._phase >= self._num_gears:
            return

        # Decay lr of existing gear param groups
        for group in self.optimizer.param_groups:
            if group.get("is_gear", False):
                group["lr"] *= self.lr_decay

        # Create and mount new gear
        new_gear = self._gear_factory(self._phase)
        new_gear.to(next(self.model.parameters()).device)
        self.model.chain.mount(new_gear)

        # Add new gear params to optimizer
        self.optimizer.add_param_group(
            {"params": list(new_gear.parameters()), "lr": self.base_lr, "is_gear": True}
        )

        self._phase += 1
        self._val_losses.clear()

    def should_advance(self, val_loss: float) -> bool:
        """Advance when val_loss hasn't improved for `patience` epochs.

        Tracks the best val_loss seen in this phase.  If `patience`
        epochs pass without beating it, the gear has stopped improving
        and we should mount the next one (or finish training).
        """
        self._val_losses.append(val_loss)
        if len(self._val_losses) <= self.patience:
            return False

        best_idx = min(range(len(self._val_losses)), key=self._val_losses.__getitem__)
        epochs_since_best = len(self._val_losses) - 1 - best_idx
        return epochs_since_best >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "val_losses": list(self._val_losses),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        target_phase = state["phase"]
        # Replay phase advances to mount gears and restore optimizer state
        while self._phase < target_phase:
            self.advance_phase()
        self._val_losses = list(state["val_losses"])

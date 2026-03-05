from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch.optim import Optimizer

from softgear.models.analyzer import Analyzer
from softgear.models.gear import Gear


class ProgressiveDepthScheduler:
    """Progressively mount gears during training.

    Phase 1: mount G1, train with lr
    Phase 2: mount G2 with lr, decay G1 lr by lr_decay
    ...
    Phase D: all gears mounted

    Hardening strategies:
        gradual:      older gears get multiplicatively decayed lr
        none:         all gears keep the same lr (no protection)
        freeze:       existing gears get lr=0
        binary:       existing gears get binary_factor * base_lr
        from_scratch: all gears mounted at once, single phase
    """

    STRATEGIES = {"gradual", "none", "freeze", "binary", "from_scratch"}

    def __init__(
        self,
        model: Analyzer,
        optimizer: Optimizer,
        gear_factory: Callable[[int], Gear],
        num_gears: int,
        base_lr: float = 3e-4,
        lr_decay: float = 0.5,
        patience: int = 5,
        hardening: str = "gradual",
        binary_factor: float = 0.4,
    ):
        if hardening not in self.STRATEGIES:
            raise ValueError(f"Unknown hardening strategy: {hardening!r}")
        self.model = model
        self.optimizer = optimizer
        self._gear_factory = gear_factory
        self._num_gears = num_gears
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.patience = patience
        self.hardening = hardening
        self.binary_factor = binary_factor
        self._phase = 0
        self._val_losses: list[float] = []

    @property
    def max_rounds(self) -> int:
        if self.hardening == "from_scratch":
            return 1
        return self._num_gears

    def current_phase(self) -> int:
        return self._phase

    def advance_phase(self) -> None:
        """Create and mount the next gear, apply hardening to existing gears."""
        if self._phase >= self._num_gears:
            return

        if self.hardening == "from_scratch":
            self._mount_all()
            return

        # Apply hardening strategy to existing gear param groups
        self._apply_hardening()

        # Create and mount new gear
        self._mount_one()

    def _apply_hardening(self) -> None:
        for group in self.optimizer.param_groups:
            if not group.get("is_gear", False):
                continue
            if self.hardening == "gradual":
                group["lr"] *= self.lr_decay
            elif self.hardening == "freeze":
                group["lr"] = 0.0
            elif self.hardening == "binary":
                group["lr"] = self.base_lr * self.binary_factor
            # hardening == "none": keep lr unchanged

    def _mount_one(self) -> None:
        new_gear = self._gear_factory(self._phase)
        new_gear.to(next(self.model.parameters()).device)
        self.model.chain.mount(new_gear)
        self.optimizer.add_param_group(
            {"params": list(new_gear.parameters()), "lr": self.base_lr, "is_gear": True}
        )
        self._phase += 1
        self._val_losses.clear()

    def _mount_all(self) -> None:
        """Mount all gears at once (from-scratch baseline)."""
        while self._phase < self._num_gears:
            new_gear = self._gear_factory(self._phase)
            new_gear.to(next(self.model.parameters()).device)
            self.model.chain.mount(new_gear)
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

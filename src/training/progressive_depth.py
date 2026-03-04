from __future__ import annotations

from typing import Any

from torch.optim import Optimizer

from src.models.softgear import SoftGearModel


class ProgressiveDepthScheduler:
    """Progressively activate gears during training.

    Phase 1: only G1 trains, active_depth=1
    Phase 2: G1(lr*decay) + G2(lr), active_depth=2
    ...
    Phase D: all gears active
    """

    def __init__(
        self,
        model: SoftGearModel,
        optimizer: Optimizer,
        base_lr: float = 3e-4,
        lr_decay: float = 0.5,
        advance_threshold: float = 0.001,
        patience: int = 5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.advance_threshold = advance_threshold
        self.patience = patience
        self._phase = 0
        self._val_losses: list[float] = []
        self._max_depth = model.gear_chain.depth

        # Freeze all gears initially
        for gear in model.gear_chain.gears:
            for param in gear.parameters():
                param.requires_grad = False

        # Set active_depth to 0 rounds (no forward until first advance)
        model.gear_chain.active_depth = 0

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def current_phase(self) -> int:
        return self._phase

    def advance_phase(self) -> None:
        """Activate the next gear and decay lr of previous gears."""
        if self._phase >= self._max_depth:
            return

        # Decay lr of existing param groups (skip non-gear groups like embedding)
        for group in self.optimizer.param_groups:
            if group.get("is_gear", False):
                group["lr"] *= self.lr_decay

        # Unfreeze the new gear
        new_gear = self.model.gear_chain.gears[self._phase]
        for param in new_gear.parameters():
            param.requires_grad = True

        # Add new gear params to optimizer
        self.optimizer.add_param_group(
            {"params": list(new_gear.parameters()), "lr": self.base_lr, "is_gear": True}
        )

        self._phase += 1
        self.model.gear_chain.active_depth = self._phase
        self._val_losses.clear()

    def should_advance(self, val_loss: float) -> bool:
        """Check if validation loss has plateaued using moving average."""
        if self._phase >= self._max_depth:
            return False

        self._val_losses.append(val_loss)
        if len(self._val_losses) < self.patience * 2:
            return False

        recent = self._val_losses[-self.patience :]
        previous = self._val_losses[-self.patience * 2 : -self.patience]
        improvement = sum(previous) / len(previous) - sum(recent) / len(recent)
        return improvement < self.advance_threshold

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "val_losses": list(self._val_losses),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        target_phase = state["phase"]
        # Replay phase advances to restore optimizer state
        while self._phase < target_phase:
            self.advance_phase()
        self._val_losses = list(state["val_losses"])

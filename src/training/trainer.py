from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.softgear import SoftGearModel
from src.training.deep_supervision import DeepSupervisionLoss
from src.training.differential_ema import DifferentialEMA
from src.training.progressive_depth import ProgressiveDepthScheduler

log = logging.getLogger(__name__)


class SoftGearTrainer:
    """Training loop integrating progressive depth, deep supervision, and differential EMA."""

    def __init__(
        self,
        cfg: DictConfig,
        model: SoftGearModel,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: torch.device | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cpu")

        self.model.to(self.device)

        tcfg = cfg.training

        # Non-gear params (embedding, norm, output_head)
        non_gear_params = [
            p
            for n, p in model.named_parameters()
            if not n.startswith("gear_chain.gears.")
        ]
        self.optimizer = AdamW(
            non_gear_params,
            lr=tcfg.lr,
            weight_decay=tcfg.weight_decay,
        )

        self.loss_fn = DeepSupervisionLoss(nn.CrossEntropyLoss(), alpha=tcfg.alpha)
        self.progressive = ProgressiveDepthScheduler(
            model,
            self.optimizer,
            base_lr=tcfg.lr,
            lr_decay=tcfg.lr_decay,
            advance_threshold=tcfg.advance_threshold,
        )
        self.ema = DifferentialEMA(model, list(tcfg.ema_alphas))

        self.gradient_clip = tcfg.gradient_clip
        self.max_epochs_per_phase = tcfg.max_epochs_per_phase

        # Resume state (set by load_checkpoint)
        self._resume_phase = 0
        self._resume_epoch = 0
        self._best_val_loss = float("inf")

    def train(self, checkpoint_dir: str | Path | None = None) -> None:
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = self._best_val_loss
        depth = self.progressive.max_depth

        for phase in range(1, depth + 1):
            if phase <= self._resume_phase:
                # Already restored by load_checkpoint; skip advance
                if phase < self._resume_phase:
                    continue
                # phase == _resume_phase: start from _resume_epoch
                epoch_start = self._resume_epoch
            else:
                self.progressive.advance_phase()
                epoch_start = 0

            log.info("Phase %d/%d started (epoch_start=%d)", phase, depth, epoch_start)

            for epoch in range(epoch_start, self.max_epochs_per_phase):
                train_loss = self._train_epoch()
                val_loss = self._validate()
                self.ema.update()

                log.info(
                    "Phase %d Epoch %d: train_loss=%.4f val_loss=%.4f",
                    phase,
                    epoch,
                    train_loss,
                    val_loss,
                )

                if checkpoint_dir is not None:
                    self.save_checkpoint(
                        checkpoint_dir / "latest.pt",
                        phase=phase,
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(
                            checkpoint_dir / "best.pt",
                            phase=phase,
                            epoch=epoch,
                            best_val_loss=best_val_loss,
                        )
                        log.info("New best val_loss=%.4f saved", best_val_loss)

                if self.progressive.should_advance(val_loss):
                    log.info("Phase %d converged, advancing", phase)
                    break

        # Reset resume state
        self._resume_phase = 0
        self._resume_epoch = 0

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in self.train_loader:
            inputs: Tensor = batch[0].to(self.device)
            targets: Tensor = batch[1].to(self.device)

            output = self.model(inputs)
            loss = self.loss_fn(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()  # type: ignore[reportUnknownMemberType]

            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)

    def _validate(self) -> float:
        self.model.eval()
        self.ema.apply_shadow()

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs: Tensor = batch[0].to(self.device)
                targets: Tensor = batch[1].to(self.device)

                output = self.model(inputs)
                loss = self.loss_fn(output, targets)
                total_loss += loss.item()
                count += 1

        self.ema.restore()
        return total_loss / max(count, 1)

    def save_checkpoint(
        self,
        path: str | Path,
        phase: int = 0,
        epoch: int = 0,
        best_val_loss: float = float("inf"),
    ) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "progressive_state": self.progressive.state_dict(),
                "ema_state": self.ema.state_dict(),  # type: ignore[reportUnknownArgumentType]
                "phase": phase,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            },  # type: ignore[reportUnknownArgumentType]
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        # Restore progressive first (replays phase advances to match param groups)
        self.progressive.load_state_dict(ckpt["progressive_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore[reportUnknownMemberType]
        self.ema.load_state_dict(ckpt["ema_state"])  # type: ignore[reportUnknownMemberType]

        # Set resume state for train()
        self._resume_phase = ckpt.get("phase", 0)
        self._resume_epoch = ckpt.get("epoch", 0) + 1  # resume from next epoch
        self._best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log.info(
            "Resumed from phase %d epoch %d (best_val_loss=%.4f)",
            self._resume_phase,
            self._resume_epoch,
            self._best_val_loss,
        )

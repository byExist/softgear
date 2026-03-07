from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dataclasses

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from softgear.config import Config
from softgear.models.analyzer import Analyzer
from softgear.models.gear import Gear
from softgear.training.differential_ema import DifferentialEMA
from softgear.training.progressive_depth import ProgressiveDepthScheduler

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

LossFn = Callable[[Tensor, Tensor], Tensor]
PredictFn = Callable[[Tensor], Tensor]
MetricsFn = Callable[[Tensor, Tensor, Tensor], dict[str, float]]


class Trainer:
    """Task-independent training loop with progressive depth and differential EMA."""

    def __init__(
        self,
        cfg: Config,
        model: Analyzer,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        gear_factory: Callable[[int], Gear],
        loss_fn: LossFn,
        predict_fn: PredictFn,
        metrics_fn: MetricsFn,
        device: torch.device | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.predict_fn = predict_fn
        self.metrics_fn = metrics_fn
        self.device = device or torch.device("cpu")

        self.model.to(self.device)

        tcfg = cfg.training

        # Non-gear params (embedding, norm, output_head)
        non_gear_params = [
            p for n, p in model.named_parameters() if not n.startswith("chain.gears.")
        ]
        self.optimizer = AdamW(
            non_gear_params,
            lr=tcfg.lr,
            weight_decay=tcfg.weight_decay,
        )

        self.progressive = ProgressiveDepthScheduler(
            model,
            self.optimizer,
            gear_factory,
            cfg.model.num_gears,
            base_lr=tcfg.lr,
            lr_decay=tcfg.lr_decay,
            patience=tcfg.patience,
            hardening=tcfg.hardening,
            binary_factor=tcfg.binary_factor,
            min_delta=tcfg.min_delta,
        )
        ema_alphas = [tcfg.ema_alpha] * cfg.model.num_gears
        self.ema = DifferentialEMA(model, ema_alphas)

        self.gradient_clip = tcfg.gradient_clip
        self._step_limit = tcfg.max_total_steps
        self._global_step = 0

        # Resume state (set by load_checkpoint)
        self._resume_phase = 0
        self._resume_epoch = 0
        self._best_val_loss = float("inf")

        # wandb (disabled when wandb not installed or project is null)
        self._wandb_run = None
        if wandb is not None and cfg.wandb.project:
            self._wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=dataclasses.asdict(cfg),
                resume="allow",
            )

    def train(self, checkpoint_dir: str | Path | None = None) -> None:
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = self._best_val_loss
        num_phases = self.progressive.max_rounds

        for phase in range(1, num_phases + 1):
            if phase <= self._resume_phase:
                # Already restored by load_checkpoint; skip advance
                if phase < self._resume_phase:
                    continue
                # phase == _resume_phase: start from _resume_epoch
                epoch_start = self._resume_epoch
            else:
                self.progressive.advance_phase()
                epoch_start = 0

            log.info(
                "Phase %d/%d started (epoch_start=%d)", phase, num_phases, epoch_start
            )

            # Track per-phase best for restore-on-advance
            phase_best_val_loss = float("inf")
            phase_best_model_state: dict[str, Tensor] | None = None

            for epoch in itertools.count(epoch_start):
                train_loss, grad_norms = self._train_epoch()
                val_loss, val_metrics = self._validate()
                self.ema.update()

                metrics_str = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                grad_str = " ".join(
                    f"{k}={v:.4f}" for k, v in sorted(grad_norms.items())
                )
                log.info(
                    "Phase %d Epoch %d: train_loss=%.4f val_loss=%.4f %s %s",
                    phase,
                    epoch,
                    train_loss,
                    val_loss,
                    metrics_str,
                    grad_str,
                )

                if val_loss < phase_best_val_loss:
                    phase_best_val_loss = val_loss
                    phase_best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }

                if self._wandb_run is not None:
                    self._wandb_run.log(
                        {
                            "phase": phase,
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            **val_metrics,
                            **grad_norms,
                        }
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

                if self._should_stop(val_loss):
                    # Restore to phase-best state before advancing
                    if phase_best_model_state is not None:
                        self.model.load_state_dict(phase_best_model_state)
                        self.ema.reset_shadows()
                        log.info(
                            "Restored to phase %d best (val_loss=%.4f)",
                            phase,
                            phase_best_val_loss,
                        )
                    if self._step_limit_reached():
                        log.info(
                            "Step limit reached (%d steps), training complete",
                            self._global_step,
                        )
                        break
                    if phase < num_phases:
                        log.info("Phase %d converged, advancing", phase)
                    else:
                        log.info("Phase %d converged, training complete", phase)
                    break
            else:
                # epoch loop completed without break; continue to next phase
                continue
            # step limit reached: break outer phase loop too
            if self._step_limit_reached():
                break

        # Reset resume state
        self._resume_phase = 0
        self._resume_epoch = 0

        if self._wandb_run is not None:
            self._wandb_run.finish()

    def _should_stop(self, val_loss: float) -> bool:
        if self.progressive.check_convergence(val_loss):
            return True
        if self._step_limit_reached():
            return True
        return False

    def _step_limit_reached(self) -> bool:
        return self._step_limit is not None and self._global_step >= self._step_limit

    def _collect_gradient_norms(self) -> dict[str, float]:
        """Collect L2 gradient norms per component (before clipping)."""
        norms: dict[str, float] = {}

        for i, gear in enumerate(self.model.chain.gears):
            total = 0.0
            for p in gear.parameters():
                if p.grad is not None:
                    total += p.grad.data.norm(2).item() ** 2  # type: ignore[reportUnknownMemberType]
            norms[f"grad_norm/gear_{i}"] = total**0.5

        non_gear_total = 0.0
        for n, p in self.model.named_parameters():
            if not n.startswith("chain.gears.") and p.grad is not None:
                non_gear_total += p.grad.data.norm(2).item() ** 2  # type: ignore[reportUnknownMemberType]
        norms["grad_norm/non_gear"] = non_gear_total**0.5

        return norms

    def _train_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        count = 0
        grad_norm_sums: dict[str, float] = {}

        for batch in self.train_loader:
            inputs: Tensor = batch[0].to(self.device)
            targets: Tensor = batch[1].to(self.device)

            output = self.model(inputs)
            loss = self.loss_fn(output.logits, targets)

            self.optimizer.zero_grad()
            loss.backward()  # type: ignore[reportUnknownMemberType]

            step_norms = self._collect_gradient_norms()
            for k, v in step_norms.items():
                grad_norm_sums[k] = grad_norm_sums.get(k, 0.0) + v

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()  # type: ignore[reportUnknownMemberType]

            total_loss += loss.item()
            count += 1
            self._global_step += 1

        avg_norms = {k: v / max(count, 1) for k, v in grad_norm_sums.items()}
        return total_loss / max(count, 1), avg_norms

    def _validate(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        self.ema.apply_shadow()

        total_loss = 0.0
        count = 0
        all_preds: list[Tensor] = []
        all_targets: list[Tensor] = []
        all_inputs: list[Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs: Tensor = batch[0].to(self.device)
                targets: Tensor = batch[1].to(self.device)

                output = self.model(inputs)
                loss = self.loss_fn(output.logits, targets)
                total_loss += loss.item()
                count += 1

                preds = self.predict_fn(output.logits)
                all_preds.append(preds)
                all_targets.append(targets)
                all_inputs.append(inputs)

        self.ema.restore()
        avg_loss = total_loss / max(count, 1)
        metrics = self.metrics_fn(
            torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_inputs)
        )
        return avg_loss, metrics

    def save_checkpoint(
        self,
        path: str | Path,
        phase: int = 0,
        epoch: int = 0,
        best_val_loss: float = float("inf"),
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": dataclasses.asdict(self.cfg),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "progressive_state": self.progressive.state_dict(),
                "ema_state": self.ema.state_dict(),  # type: ignore[reportUnknownArgumentType]
                "phase": phase,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "global_step": self._global_step,
            },  # type: ignore[reportUnknownArgumentType]
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Progressive first: mounts gears so model structure matches checkpoint
        self.progressive.load_state_dict(ckpt["progressive_state"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore[reportUnknownMemberType]
        self.ema.load_state_dict(ckpt["ema_state"])  # type: ignore[reportUnknownMemberType]

        # Set resume state for train()
        self._resume_phase = ckpt.get("phase", 0)
        self._resume_epoch = ckpt.get("epoch", 0) + 1  # resume from next epoch
        self._best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self._global_step = ckpt.get("global_step", 0)
        log.info(
            "Resumed from phase %d epoch %d step %d (best_val_loss=%.4f)",
            self._resume_phase,
            self._resume_epoch,
            self._global_step,
            self._best_val_loss,
        )

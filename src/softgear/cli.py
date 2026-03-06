"""SoftGear CLI — typer-based entry point for training and evaluation."""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal, Optional

import numpy as np
import torch
import typer

app = typer.Typer(help="SoftGear: Learnable Asymmetric Gear Chains")
log = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.command()
def train(
    # task
    task: Annotated[str, typer.Option(help="Task name")],
    # experiment (independent variables)
    hardening: Annotated[
        Literal["gradual", "none", "freeze", "binary", "from_scratch"],
        typer.Option(
            help="Hardening strategy: gradual|none|freeze|binary|from_scratch"
        ),
    ] = "gradual",
    identity_init: Annotated[
        bool,
        typer.Option(
            "--identity-init/--no-identity-init",
            help="Identity init for new gears (disable for ablation)",
        ),
    ] = True,
    scale: Annotated[
        float,
        typer.Option(
            help="Resolution scale: gear i gets round(scale^i) layers (1.0=uniform)"
        ),
    ] = 1.0,
    num_gears: Annotated[int, typer.Option(help="Number of progressive gears")] = 7,
    lr_decay: Annotated[
        float, typer.Option(help="LR decay per phase (gradual hardening)")
    ] = 0.5,
    binary_factor: Annotated[
        float, typer.Option(help="Binary hardening factor (binary hardening)")
    ] = 0.4,
    curriculum: Annotated[
        bool,
        typer.Option(
            "--curriculum/--no-curriculum",
            help="Sort training data by difficulty (easy first)",
        ),
    ] = False,
    # training
    lr: Annotated[float, typer.Option(help="Learning rate")] = 3e-4,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 64,
    patience: Annotated[int, typer.Option(help="Epochs before advancing phase")] = 5,
    min_delta: Annotated[
        float, typer.Option(help="Minimum loss improvement to reset patience")
    ] = 1e-4,
    weight_decay: Annotated[float, typer.Option(help="Weight decay")] = 0.01,
    ema_alpha: Annotated[
        float, typer.Option(help="EMA decay rate (uniform for all gears)")
    ] = 0.995,
    gradient_clip: Annotated[float, typer.Option(help="Gradient clip norm")] = 1.0,
    max_total_steps: Annotated[
        Optional[int], typer.Option(help="Max total training steps (None=unlimited)")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    # model
    hidden_dim: Annotated[int, typer.Option(help="Hidden dimension")] = 128,
    num_heads: Annotated[int, typer.Option(help="Attention heads")] = 4,
    ffn_dim: Annotated[int, typer.Option(help="FFN dimension")] = 512,
    dropout: Annotated[float, typer.Option(help="Dropout rate")] = 0.1,
    # data & io
    data_path: Annotated[
        Optional[Path], typer.Option(help="Dataset directory (default: task-specific)")
    ] = None,
    max_samples: Annotated[
        Optional[int], typer.Option(help="Limit dataset size")
    ] = None,
    checkpoint_dir: Annotated[Path, typer.Option(help="Checkpoint directory")] = Path(
        "checkpoints"
    ),
    resume: Annotated[
        Optional[Path], typer.Option(help="Checkpoint to resume from")
    ] = None,
    # logging
    wandb_project: Annotated[
        Optional[str], typer.Option(help="W&B project name")
    ] = None,
    wandb_entity: Annotated[Optional[str], typer.Option(help="W&B entity name")] = None,
) -> None:
    """Train a SoftGear model with progressive depth."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    from softgear.config import Config, TrainingConfig, WandbConfig
    from softgear.tasks.registry import get_task
    from softgear.training.trainer import Trainer
    from softgear.utils.device import get_device

    t = get_task(task)

    model_cfg = replace(
        t.model_defaults,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_gears=num_gears,
        dropout=dropout,
        identity_init=identity_init,
        scale=scale,
    )
    data_cfg = replace(
        t.data_defaults,
        batch_size=batch_size,
        max_samples=max_samples,
        curriculum=curriculum,
        **({"path": str(data_path)} if data_path else {}),
    )

    cfg = Config(
        task=task,
        model=model_cfg,
        data=data_cfg,
        training=TrainingConfig(
            lr=lr,
            optimizer="adamw",
            weight_decay=weight_decay,
            hardening=hardening,
            lr_decay=lr_decay,
            binary_factor=binary_factor,
            patience=patience,
            min_delta=min_delta,
            gradient_clip=gradient_clip,
            ema_alpha=ema_alpha,
            max_total_steps=max_total_steps,
            checkpoint_dir=str(checkpoint_dir),
        ),
        seed=seed,
        wandb=WandbConfig(project=wandb_project, entity=wandb_entity),
    )

    _seed_everything(cfg.seed)
    log.info("Config: %s", cfg)

    device = get_device()
    model = t.build_model(cfg.model)
    log.info("Model parameters: %d", model.parameter_count())

    train_loader, val_loader = t.build_loaders(cfg.data)
    log.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    gear_factory = t.make_gear_factory(cfg.model)
    trainer = Trainer(
        cfg,
        model,
        train_loader,
        val_loader,
        gear_factory=gear_factory,
        loss_fn=t.loss_fn,
        predict_fn=t.predict_fn,
        metrics_fn=t.metrics_fn,
        device=device,
    )

    if resume:
        trainer.load_checkpoint(resume)

    trainer.train(checkpoint_dir=str(checkpoint_dir))


@app.command()
def download(
    task: Annotated[str, typer.Argument(help="Task name")],
    data_path: Annotated[
        Optional[Path], typer.Option(help="Download directory (default: task-specific)")
    ] = None,
) -> None:
    """Download dataset for a task."""
    from softgear.tasks.registry import get_task

    t = get_task(task)
    if t.download is None:
        typer.echo(f"Task {task!r} has no download function", err=True)
        raise typer.Exit(1)

    dest = str(data_path) if data_path else t.data_defaults.path
    t.download(dest)


@app.command("eval")
def evaluate(
    checkpoint: Annotated[Path, typer.Argument(help="Checkpoint file path")],
    data_path: Annotated[
        Optional[Path], typer.Option(help="Dataset directory (default: task-specific)")
    ] = None,
    max_samples: Annotated[
        Optional[int], typer.Option(help="Limit dataset size")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 64,
) -> None:
    """Evaluate a trained SoftGear model."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    from softgear.config import ModelConfig
    from softgear.tasks.registry import get_task
    from softgear.utils.device import get_device

    device = get_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    task_name = ckpt["config"].get("task", "sudoku9")
    t = get_task(task_name)

    model_cfg = ModelConfig(**ckpt["config"]["model"])
    data_cfg = replace(
        t.data_defaults,
        batch_size=batch_size,
        max_samples=max_samples,
        **({"path": str(data_path)} if data_path else {}),
    )

    model = t.build_model(model_cfg)
    t.mount_all_gears(model, model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, val_loader = t.build_loaders(data_cfg)

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_inputs: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            output = model(inputs)
            preds = t.predict_fn(output.logits)
            all_preds.append(preds)
            all_targets.append(targets)
            all_inputs.append(inputs)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    inputs = torch.cat(all_inputs)

    metrics = t.metrics_fn(preds, targets, inputs)
    for name, value in metrics.items():
        log.info("%-20s %.4f", name, value)


if __name__ == "__main__":
    app()

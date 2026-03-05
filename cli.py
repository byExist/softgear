"""SoftGear CLI — typer-based entry point for training and evaluation."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Annotated, Optional

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
    # common overrides
    resume: Annotated[Optional[Path], typer.Option(help="Checkpoint to resume from")] = None,
    max_samples: Annotated[Optional[int], typer.Option(help="Limit dataset size")] = None,
    num_gears: Annotated[int, typer.Option(help="Number of progressive gears")] = 7,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 64,
    hardening: Annotated[str, typer.Option(help="Hardening strategy: gradual|none|freeze|binary|from_scratch")] = "gradual",
    patience: Annotated[int, typer.Option(help="Epochs before advancing phase")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    wandb_project: Annotated[Optional[str], typer.Option(help="W&B project name")] = None,
    checkpoint_dir: Annotated[Path, typer.Option(help="Checkpoint directory")] = Path("checkpoints"),
    # rarely changed
    data_path: Annotated[Path, typer.Option(help="Dataset directory")] = Path("data/sudoku-extreme"),
    hidden_dim: Annotated[int, typer.Option(help="Hidden dimension")] = 128,
    num_heads: Annotated[int, typer.Option(help="Attention heads")] = 8,
    ffn_dim: Annotated[int, typer.Option(help="FFN dimension")] = 512,
    dropout: Annotated[float, typer.Option(help="Dropout rate")] = 0.1,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 3e-4,
    weight_decay: Annotated[float, typer.Option(help="Weight decay")] = 0.01,
    lr_decay: Annotated[float, typer.Option(help="LR decay per phase")] = 0.5,
    binary_factor: Annotated[float, typer.Option(help="Binary hardening factor")] = 0.4,
    gradient_clip: Annotated[float, typer.Option(help="Gradient clip norm")] = 1.0,
) -> None:
    """Train a SoftGear model with progressive depth."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from softgear.config import Config, DataConfig, ModelConfig, TrainingConfig, WandbConfig
    from softgear.tasks.sudoku.data import build_sudoku_loaders
    from softgear.tasks.sudoku.metrics import sudoku_accuracy
    from softgear.tasks.sudoku.model import build_sudoku_model, make_gear_factory
    from softgear.training.trainer import Trainer
    from softgear.utils.device import get_device

    cfg = Config(
        model=ModelConfig(
            hidden_dim=hidden_dim, num_heads=num_heads,
            ffn_dim=ffn_dim, num_gears=num_gears, dropout=dropout,
        ),
        data=DataConfig(
            path=str(data_path), batch_size=batch_size, max_samples=max_samples,
        ),
        training=TrainingConfig(
            lr=lr, weight_decay=weight_decay, hardening=hardening,
            lr_decay=lr_decay, binary_factor=binary_factor,
            patience=patience, gradient_clip=gradient_clip,
            checkpoint_dir=str(checkpoint_dir),
        ),
        seed=seed,
        wandb=WandbConfig(project=wandb_project),
    )

    _seed_everything(cfg.seed)
    log.info("Config: %s", cfg)

    device = get_device()
    model = build_sudoku_model(cfg.model)
    log.info("Model parameters: %d", model.parameter_count())

    train_loader, val_loader = build_sudoku_loaders(cfg.data)
    log.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    gear_factory = make_gear_factory(cfg.model)
    trainer = Trainer(
        cfg, model, train_loader, val_loader,
        gear_factory=gear_factory, metrics_fn=sudoku_accuracy, device=device,
    )

    if resume:
        trainer.load_checkpoint(resume)

    trainer.train(checkpoint_dir=str(checkpoint_dir))


@app.command("eval")
def evaluate(
    checkpoint: Annotated[Path, typer.Argument(help="Checkpoint file path")],
    max_samples: Annotated[Optional[int], typer.Option(help="Limit dataset size")] = None,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 64,
    data_path: Annotated[Path, typer.Option(help="Dataset directory")] = Path("data/sudoku-extreme"),
    hidden_dim: Annotated[int, typer.Option(help="Hidden dimension")] = 128,
    num_heads: Annotated[int, typer.Option(help="Attention heads")] = 8,
    ffn_dim: Annotated[int, typer.Option(help="FFN dimension")] = 512,
    num_gears: Annotated[int, typer.Option(help="Number of gears")] = 7,
    dropout: Annotated[float, typer.Option(help="Dropout rate")] = 0.1,
) -> None:
    """Evaluate a trained SoftGear model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from softgear.config import Config, DataConfig, ModelConfig
    from softgear.tasks.sudoku.data import build_sudoku_loaders
    from softgear.tasks.sudoku.metrics import sudoku_accuracy
    from softgear.tasks.sudoku.model import build_sudoku_model, mount_all_gears
    from softgear.utils.device import get_device

    cfg = Config(
        model=ModelConfig(
            hidden_dim=hidden_dim, num_heads=num_heads,
            ffn_dim=ffn_dim, num_gears=num_gears, dropout=dropout,
        ),
        data=DataConfig(
            path=str(data_path), batch_size=batch_size, max_samples=max_samples,
        ),
    )

    device = get_device()

    model = build_sudoku_model(cfg.model)
    mount_all_gears(model, cfg.model)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, val_loader = build_sudoku_loaders(cfg.data)

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_inputs: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            output = model(inputs)
            preds = output.logits.argmax(dim=-1)
            all_preds.append(preds)
            all_targets.append(targets)
            all_inputs.append(inputs)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    inputs = torch.cat(all_inputs)

    metrics = sudoku_accuracy(preds, targets, inputs)
    log.info("Cell accuracy:   %.4f", metrics["cell_accuracy"])
    log.info("Blank accuracy:  %.4f", metrics["blank_accuracy"])
    log.info("Puzzle accuracy: %.4f", metrics["puzzle_accuracy"])


if __name__ == "__main__":
    app()

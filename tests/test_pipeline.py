import tempfile

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.models.softgear import SoftGearModel
from src.training.deep_supervision import DeepSupervisionLoss
from src.training.differential_ema import DifferentialEMA
from src.training.progressive_depth import ProgressiveDepthScheduler
from src.training.trainer import SoftGearTrainer
from tests.test_softgear import make_cfg

BATCH = 4
SEQ_LEN = 16
VOCAB_SIZE = 11


def _make_dummy_loader(n_samples: int = 16) -> DataLoader[tuple[torch.Tensor, ...]]:
    inputs = torch.randint(0, VOCAB_SIZE, (n_samples, SEQ_LEN))
    targets = torch.randint(0, VOCAB_SIZE, (n_samples, SEQ_LEN))
    return DataLoader(TensorDataset(inputs, targets), batch_size=BATCH)


def test_full_pipeline_dummy():
    """Full pipeline: model -> loss -> backward -> optimizer -> EMA (no errors)."""
    cfg = make_cfg(gear_sizes=[1, 2])
    model = SoftGearModel(cfg)

    non_gear_params = [
        p for n, p in model.named_parameters() if not n.startswith("gear_chain.gears.")
    ]
    optimizer = AdamW(non_gear_params, lr=1e-3)
    loss_fn = DeepSupervisionLoss(nn.CrossEntropyLoss(), alpha=0.3)
    progressive = ProgressiveDepthScheduler(model, optimizer, base_lr=1e-3)
    ema = DifferentialEMA(model, [0.99, 0.999])

    progressive.advance_phase()  # Phase 1

    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    # Forward + backward + step
    output = model(x)
    loss = loss_fn(output, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # type: ignore[reportUnknownMemberType]
    ema.update()

    # EMA apply/restore
    ema.apply_shadow()
    with torch.no_grad():
        model(x)
    ema.restore()


def test_phase_transition_end_to_end():
    """Phase 1 -> Phase 2 transition with training steps."""
    cfg = make_cfg(gear_sizes=[1, 2, 3])
    model = SoftGearModel(cfg)

    non_gear_params = [
        p for n, p in model.named_parameters() if not n.startswith("gear_chain.gears.")
    ]
    optimizer = AdamW(non_gear_params, lr=1e-3)
    loss_fn = DeepSupervisionLoss(nn.CrossEntropyLoss(), alpha=0.3)
    progressive = ProgressiveDepthScheduler(model, optimizer, base_lr=1e-3)
    ema = DifferentialEMA(model, [0.99, 0.995, 0.999])

    for phase in range(1, 4):
        progressive.advance_phase()
        assert model.gear_chain.active_depth == phase

        x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

        model.train()
        output = model(x)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # type: ignore[reportUnknownMemberType]
        ema.update()


def test_trainer_runs():
    """SoftGearTrainer runs without errors on dummy data."""
    model_cfg = make_cfg(gear_sizes=[1, 2])
    full_cfg = OmegaConf.create(
        {
            "model": OmegaConf.to_container(model_cfg),
            "training": {
                "lr": 1e-3,
                "weight_decay": 0.01,
                "alpha": 0.3,
                "lr_decay": 0.5,
                "ema_alphas": [0.99, 0.999],
                "advance_threshold": -1.0,  # always advance after 2 epochs
                "patience": 1,  # minimal patience for fast test
                "gradient_clip": 1.0,
                "max_epochs_per_phase": 2,
            },
        }
    )

    model = SoftGearModel(model_cfg)
    loader = _make_dummy_loader()

    trainer = SoftGearTrainer(full_cfg, model, loader, loader)
    trainer.train()


def test_checkpoint_save_load():
    """Checkpoint save/load preserves trainer state."""
    model_cfg = make_cfg(gear_sizes=[1, 2])
    full_cfg = OmegaConf.create(
        {
            "model": OmegaConf.to_container(model_cfg),
            "training": {
                "lr": 1e-3,
                "weight_decay": 0.01,
                "alpha": 0.3,
                "lr_decay": 0.5,
                "ema_alphas": [0.99, 0.999],
                "advance_threshold": 0.001,
                "patience": 5,
                "gradient_clip": 1.0,
                "max_epochs_per_phase": 2,
            },
        }
    )

    model = SoftGearModel(model_cfg)
    loader = _make_dummy_loader()
    trainer = SoftGearTrainer(full_cfg, model, loader, loader)

    # Advance to phase 2 and do some training
    trainer.progressive.advance_phase()
    trainer.progressive.advance_phase()

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        trainer.save_checkpoint(f.name, phase=2, epoch=1, best_val_loss=0.5)

        # Load into fresh trainer
        model2 = SoftGearModel(model_cfg)
        trainer2 = SoftGearTrainer(full_cfg, model2, loader, loader)
        trainer2.load_checkpoint(f.name)

        assert trainer2.progressive.current_phase() == 2
        assert model2.gear_chain.active_depth == 2


def test_auto_save_creates_checkpoints():
    """train() with checkpoint_dir creates latest.pt and best.pt."""
    model_cfg = make_cfg(gear_sizes=[1])
    full_cfg = OmegaConf.create(
        {
            "model": OmegaConf.to_container(model_cfg),
            "training": {
                "lr": 1e-3,
                "weight_decay": 0.01,
                "alpha": 0.0,
                "lr_decay": 0.5,
                "ema_alphas": [0.99],
                "advance_threshold": -1.0,
                "patience": 1,
                "gradient_clip": 1.0,
                "max_epochs_per_phase": 1,
            },
        }
    )

    model = SoftGearModel(model_cfg)
    loader = _make_dummy_loader()
    trainer = SoftGearTrainer(full_cfg, model, loader, loader)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(checkpoint_dir=tmpdir)

        from pathlib import Path

        ckpt_dir = Path(tmpdir)
        assert (ckpt_dir / "latest.pt").exists()
        assert (ckpt_dir / "best.pt").exists()

        ckpt = torch.load(ckpt_dir / "latest.pt", weights_only=False)
        assert "phase" in ckpt
        assert "epoch" in ckpt
        assert "best_val_loss" in ckpt

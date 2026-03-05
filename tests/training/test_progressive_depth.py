"""Tests for ProgressiveDepthScheduler."""

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from softgear.models.analyzer import Analyzer
from softgear.models.chain import Chain
from softgear.models.gear import Gear
from softgear.training.progressive_depth import ProgressiveDepthScheduler

D, HEADS, FFN = 32, 4, 64
VOCAB = 5
NUM_GEARS = 3


def _gear_factory(phase_idx: int) -> Gear:
    return Gear(D, HEADS, FFN, dropout=0.0)


def _make_model() -> Analyzer:
    encoder = nn.Embedding(VOCAB, D)
    decoder = nn.Linear(D, VOCAB)
    return Analyzer(encoder, decoder, Chain(), hidden_dim=D)


def _make_scheduler(
    model: Analyzer | None = None,
    hardening: str = "gradual",
    patience: int = 3,
) -> tuple[Analyzer, AdamW, ProgressiveDepthScheduler]:
    model = model or _make_model()
    non_gear_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("chain.gears.")
    ]
    optimizer = AdamW(non_gear_params, lr=1e-3)
    scheduler = ProgressiveDepthScheduler(
        model, optimizer, _gear_factory, NUM_GEARS,
        base_lr=1e-3, lr_decay=0.5, patience=patience,
        hardening=hardening,
    )
    return model, optimizer, scheduler


def test_advance_mounts_gear():
    model, _, scheduler = _make_scheduler()
    assert len(model.chain.gears) == 0
    scheduler.advance_phase()
    assert len(model.chain.gears) == 1
    scheduler.advance_phase()
    assert len(model.chain.gears) == 2


def test_max_rounds():
    _, _, scheduler = _make_scheduler(hardening="gradual")
    assert scheduler.max_rounds == NUM_GEARS

    _, _, scheduler = _make_scheduler(hardening="from_scratch")
    assert scheduler.max_rounds == 1


def test_from_scratch_mounts_all():
    model, _, scheduler = _make_scheduler(hardening="from_scratch")
    scheduler.advance_phase()
    assert len(model.chain.gears) == NUM_GEARS


def test_gradual_decays_lr():
    _, optimizer, scheduler = _make_scheduler(hardening="gradual")
    scheduler.advance_phase()  # phase 1: gear0 at base_lr
    scheduler.advance_phase()  # phase 2: gear0 decayed, gear1 at base_lr

    gear_groups = [g for g in optimizer.param_groups if g.get("is_gear")]
    assert gear_groups[0]["lr"] == 1e-3 * 0.5
    assert gear_groups[1]["lr"] == 1e-3


def test_freeze_zeros_lr():
    _, optimizer, scheduler = _make_scheduler(hardening="freeze")
    scheduler.advance_phase()
    scheduler.advance_phase()

    gear_groups = [g for g in optimizer.param_groups if g.get("is_gear")]
    assert gear_groups[0]["lr"] == 0.0
    assert gear_groups[1]["lr"] == 1e-3


def test_binary_sets_factor():
    _, optimizer, scheduler = _make_scheduler(hardening="binary")
    scheduler.advance_phase()
    scheduler.advance_phase()

    gear_groups = [g for g in optimizer.param_groups if g.get("is_gear")]
    assert gear_groups[0]["lr"] == 1e-3 * 0.4
    assert gear_groups[1]["lr"] == 1e-3


def test_none_keeps_lr():
    _, optimizer, scheduler = _make_scheduler(hardening="none")
    scheduler.advance_phase()
    scheduler.advance_phase()

    gear_groups = [g for g in optimizer.param_groups if g.get("is_gear")]
    assert gear_groups[0]["lr"] == 1e-3
    assert gear_groups[1]["lr"] == 1e-3


def test_invalid_hardening():
    with pytest.raises(ValueError, match="Unknown hardening"):
        _make_scheduler(hardening="invalid")


def test_should_advance_respects_patience():
    _, _, scheduler = _make_scheduler(patience=3)
    scheduler.advance_phase()

    # Improving losses: should not advance
    for loss in [1.0, 0.9, 0.8]:
        assert not scheduler.should_advance(loss)

    # Plateau: patience=3 epochs without improvement
    scheduler.should_advance(0.85)
    scheduler.should_advance(0.85)
    assert scheduler.should_advance(0.85) is True


def test_state_dict_roundtrip():
    _, _, scheduler = _make_scheduler()
    scheduler.advance_phase()
    scheduler.advance_phase()
    scheduler.should_advance(1.0)

    state = scheduler.state_dict()
    assert state["phase"] == 2
    assert len(state["val_losses"]) == 1

    # Restore into fresh scheduler
    model2, _, scheduler2 = _make_scheduler()
    scheduler2.load_state_dict(state)
    assert scheduler2.current_phase() == 2
    assert len(model2.chain.gears) == 2

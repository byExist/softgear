import torch
from torch.optim import AdamW

from src.models.softgear import SoftGearModel
from src.training.progressive_depth import ProgressiveDepthScheduler
from tests.test_softgear import make_cfg

BATCH = 2
SEQ_LEN = 16
VOCAB_SIZE = 11
BASE_LR = 1e-3


def _make_scheduler(
    gear_sizes: list[int] | None = None,
) -> tuple[SoftGearModel, ProgressiveDepthScheduler]:
    if gear_sizes is None:
        gear_sizes = [1, 2, 3, 4]
    cfg = make_cfg(gear_sizes=gear_sizes)
    model = SoftGearModel(cfg)
    # Optimizer with non-gear params (embedding, norm, output_head)
    non_gear_params = [
        p for n, p in model.named_parameters() if not n.startswith("gear_chain.gears.")
    ]
    optimizer = AdamW(non_gear_params, lr=BASE_LR)
    scheduler = ProgressiveDepthScheduler(
        model, optimizer, base_lr=BASE_LR, lr_decay=0.5
    )
    return model, scheduler


def test_phase_activation():
    """Phase k should have first k gears with requires_grad=True."""
    model, scheduler = _make_scheduler()

    for phase in range(1, 5):
        scheduler.advance_phase()
        assert scheduler.current_phase() == phase

        for i, gear in enumerate(model.gear_chain.gears):
            for param in gear.parameters():
                if i < phase:
                    assert param.requires_grad, (
                        f"Gear {i} should be active in phase {phase}"
                    )
                else:
                    assert not param.requires_grad, (
                        f"Gear {i} should be frozen in phase {phase}"
                    )


def test_lr_decay_on_advance():
    """Previous gears' lr should be multiplied by lr_decay on phase advance."""
    _, scheduler = _make_scheduler([1, 2, 3])

    scheduler.advance_phase()  # Phase 1: G1 at base_lr
    gear_groups = [g for g in scheduler.optimizer.param_groups if g.get("is_gear")]
    assert len(gear_groups) == 1
    assert gear_groups[0]["lr"] == BASE_LR

    scheduler.advance_phase()  # Phase 2: G1 at base_lr*0.5, G2 at base_lr
    gear_groups = [g for g in scheduler.optimizer.param_groups if g.get("is_gear")]
    assert len(gear_groups) == 2
    assert abs(gear_groups[0]["lr"] - BASE_LR * 0.5) < 1e-10
    assert gear_groups[1]["lr"] == BASE_LR

    scheduler.advance_phase()  # Phase 3: G1 at 0.25, G2 at 0.5, G3 at 1.0
    gear_groups = [g for g in scheduler.optimizer.param_groups if g.get("is_gear")]
    assert len(gear_groups) == 3
    assert abs(gear_groups[0]["lr"] - BASE_LR * 0.25) < 1e-10
    assert abs(gear_groups[1]["lr"] - BASE_LR * 0.5) < 1e-10
    assert gear_groups[2]["lr"] == BASE_LR


def test_gradient_isolation():
    """Inactive gears should receive no gradients."""
    model, scheduler = _make_scheduler()
    scheduler.advance_phase()  # Phase 1: only G1 active

    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    output = model(x)
    loss = torch.nn.CrossEntropyLoss()(
        output.logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    loss.backward()

    # G1 should have gradients
    for param in model.gear_chain.gears[0].parameters():
        assert param.grad is not None

    # G2, G3, G4 should have NO gradients
    for i in range(1, 4):
        for param in model.gear_chain.gears[i].parameters():
            assert param.grad is None, f"Gear {i} should have no gradient in phase 1"


def test_should_advance():
    """should_advance returns True when moving average improvement is below threshold."""
    _, scheduler = _make_scheduler()
    scheduler.advance_phase()

    # Need patience*2 (default 10) data points before any decision
    # Feed 5 decreasing losses (previous window)
    for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
        assert not scheduler.should_advance(loss)

    # Feed 5 more clearly lower losses (recent window) → big improvement → no advance
    for loss in [0.2, 0.2, 0.2, 0.2]:
        assert not scheduler.should_advance(loss)
    assert not scheduler.should_advance(0.2)  # previous avg=0.8, recent avg=0.2

    # Reset with a new scheduler for plateau test
    _, scheduler2 = _make_scheduler()
    scheduler2.advance_phase()

    # Feed 10 nearly identical losses → plateau → should advance
    for loss in [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
        scheduler2.should_advance(loss)
    assert scheduler2.should_advance(0.5)  # previous avg ≈ recent avg → advance

    # Reset: worsening val_loss should NOT advance (overfitting scenario)
    _, scheduler3 = _make_scheduler()
    scheduler3.advance_phase()

    # Feed losses that get worse then plateau at a bad value
    for loss in [1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5]:
        scheduler3.should_advance(loss)
    assert not scheduler3.should_advance(1.5)  # worsened → don't advance


def test_active_depth():
    """GearChain should only run k rounds in phase k."""
    model, scheduler = _make_scheduler()

    for phase in range(1, 5):
        scheduler.advance_phase()
        assert model.gear_chain.active_depth == phase

        x = torch.randn(BATCH, SEQ_LEN, 64)  # hidden_dim=64
        _, round_outputs = model.gear_chain(x)
        assert len(round_outputs) == phase


def test_no_advance_past_max():
    """advance_phase should be a no-op when all gears are active."""
    _, scheduler = _make_scheduler([1, 2])

    scheduler.advance_phase()
    scheduler.advance_phase()
    assert scheduler.current_phase() == 2

    scheduler.advance_phase()  # should be no-op
    assert scheduler.current_phase() == 2

import torch

from src.models.softgear import SoftGearModel
from src.training.differential_ema import DifferentialEMA
from tests.test_softgear import make_cfg

BATCH = 2
SEQ_LEN = 16


def _make_ema(alphas: list[float] | None = None) -> tuple[SoftGearModel, DifferentialEMA]:
    cfg = make_cfg(gear_sizes=[1, 2])
    model = SoftGearModel(cfg)
    if alphas is None:
        alphas = [0.99, 0.999]
    ema = DifferentialEMA(model, alphas)
    return model, ema



def test_different_alphas_per_gear():
    """Gears with different alphas should have different shadow convergence rates."""
    model, ema = _make_ema([0.5, 0.99])

    # Perturb model params so shadow diverges
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))

    # Single update
    ema.update()

    # Apply shadow and measure how far each gear moved from perturbed params
    perturbed = {n: p.data.clone() for n, p in model.named_parameters()}
    ema.apply_shadow()

    # Gear 0 (alpha=0.5) should have moved MORE toward original (lower alpha = faster adapt)
    # Gear 1 (alpha=0.99) should have moved LESS (higher alpha = slower adapt)
    gear0_diff = sum(
        (perturbed[n] - p.data).abs().sum().item()
        for n, p in model.named_parameters()
        if n.startswith("gear_chain.gears.0.")
    )
    gear1_diff = sum(
        (perturbed[n] - p.data).abs().sum().item()
        for n, p in model.named_parameters()
        if n.startswith("gear_chain.gears.1.")
    )

    # Normalize by parameter count
    gear0_count = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith("gear_chain.gears.0.")
    )
    gear1_count = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith("gear_chain.gears.1.")
    )

    # Lower alpha (gear 0) → shadow moves faster toward current params → smaller diff
    assert gear0_diff / gear0_count < gear1_diff / gear1_count

    ema.restore()


def test_shadow_convergence():
    """After many updates with constant params, shadow should match params."""
    model, ema = _make_ema([0.9, 0.9])

    original = {n: p.data.clone() for n, p in model.named_parameters()}

    for _ in range(200):
        ema.update()

    ema.apply_shadow()

    for name, param in model.named_parameters():
        if name.startswith("gear_chain.gears."):
            assert torch.allclose(param.data, original[name], atol=1e-5), (
                f"Shadow did not converge for {name}"
            )

    ema.restore()


def test_apply_and_restore():
    """apply_shadow should change params, restore should revert them."""
    model, ema = _make_ema()

    # Perturb params so shadow differs
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    originals = {n: p.data.clone() for n, p in model.named_parameters()}

    ema.apply_shadow()

    # At least some gear params should change
    any_changed = any(
        not torch.equal(p.data, originals[n])
        for n, p in model.named_parameters()
        if n.startswith("gear_chain.gears.")
    )
    assert any_changed, "apply_shadow did not change any parameter"

    ema.restore()

    for name, param in model.named_parameters():
        assert torch.equal(param.data, originals[name]), f"restore failed for {name}"


def test_state_dict_roundtrip():
    """state_dict save/load must preserve shadow state."""
    model, ema = _make_ema()
    ema.update()
    ema.update()

    state = ema.state_dict()

    model2, ema2 = _make_ema()
    ema2.load_state_dict(state)

    # Both EMAs should produce the same shadow params when applied
    ema.apply_shadow()
    params1 = {n: p.data.clone() for n, p in model.named_parameters()}
    ema.restore()

    ema2.apply_shadow()
    params2 = {n: p.data.clone() for n, p in model2.named_parameters()}
    ema2.restore()

    for key in params1:
        if key.startswith("gear_chain.gears."):
            assert torch.equal(params1[key], params2[key]), f"Mismatch for {key}"

"""Tests for DifferentialEMA."""

import torch
from torch import nn

from softgear.models.analyzer import Analyzer
from softgear.models.chain import Chain
from softgear.models.gear import Gear
from softgear.training.differential_ema import DifferentialEMA

D, HEADS, FFN = 32, 4, 64
VOCAB = 5


def _gear_factory(phase_idx: int) -> Gear:
    return Gear(D, HEADS, FFN, dropout=0.0)


def _make_model() -> Analyzer:
    encoder = nn.Embedding(VOCAB, D)
    decoder = nn.Linear(D, VOCAB)
    return Analyzer(encoder, decoder, Chain(), hidden_dim=D)


@torch.no_grad()
def test_apply_changes_params():
    model = _make_model()
    model.chain.mount(_gear_factory(0))
    ema = DifferentialEMA(model, alphas=[0.99])

    # Change params and update EMA so shadow diverges
    for _ in range(3):
        for p in model.chain.gears[0].parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        ema.update()

    params_before = {n: p.data.clone() for n, p in model.named_parameters()}
    ema.apply_shadow()

    # At least some gear params should differ after apply
    any_changed = any(
        not torch.equal(p.data, params_before[n])
        for n, p in model.named_parameters()
        if n.startswith("chain.gears.")
    )
    assert any_changed
    ema.restore()


@torch.no_grad()
def test_restore_recovers_original():
    model = _make_model()
    model.chain.mount(_gear_factory(0))
    ema = DifferentialEMA(model, alphas=[0.99])

    for _ in range(3):
        for p in model.chain.gears[0].parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        ema.update()

    original_params = {n: p.data.clone() for n, p in model.named_parameters()}
    ema.apply_shadow()
    ema.restore()

    for n, p in model.named_parameters():
        torch.testing.assert_close(p.data, original_params[n])


@torch.no_grad()
def test_state_dict_roundtrip():
    model = _make_model()
    model.chain.mount(_gear_factory(0))
    ema = DifferentialEMA(model, alphas=[0.9])
    ema.update()

    state = ema.state_dict()
    assert "shadow" in state

    # Restore into fresh EMA and verify apply produces same result
    ema2 = DifferentialEMA(model, alphas=[0.9])
    ema2.load_state_dict(state)

    ema.apply_shadow()
    params1 = {n: p.data.clone() for n, p in model.named_parameters()}
    ema.restore()

    ema2.apply_shadow()
    params2 = {n: p.data.clone() for n, p in model.named_parameters()}
    ema2.restore()

    for n in params1:
        torch.testing.assert_close(params1[n], params2[n])

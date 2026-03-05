"""Tests for Gear."""

import torch

from softgear.models.gear import Gear

B, SEQ, D, HEADS, FFN = 2, 16, 32, 4, 64


def _make_gear(identity_init: bool = False) -> Gear:
    return Gear(D, HEADS, FFN, dropout=0.0, identity_init=identity_init)


def test_forward_shape():
    gear = _make_gear()
    x = torch.randn(B, SEQ, D)
    with torch.no_grad():
        assert gear(x).shape == (B, SEQ, D)


def test_identity_init_preserves_input():
    gear = _make_gear(identity_init=True)
    gear.eval()
    x = torch.randn(B, SEQ, D)
    with torch.no_grad():
        torch.testing.assert_close(gear(x), x, atol=1e-5, rtol=1e-5)


def test_identity_init_zeros_output_projections():
    gear = _make_gear(identity_init=True)
    assert gear.self_attn.out_proj.weight.abs().sum() == 0
    assert gear.self_attn.out_proj.bias.abs().sum() == 0
    assert gear.linear2.weight.abs().sum() == 0
    assert gear.linear2.bias.abs().sum() == 0

"""Tests for Chain."""

import torch

from softgear.models.chain import Chain
from softgear.models.gear import Gear

B, SEQ, D, HEADS, FFN = 2, 16, 32, 4, 64


def _make_gear() -> Gear:
    return Gear(D, HEADS, FFN, dropout=0.0)


def test_empty_is_identity():
    chain = Chain()
    x = torch.randn(B, SEQ, D)
    torch.testing.assert_close(chain(x), x)


def test_mount_increases_gear_count():
    chain = Chain()
    assert len(chain.gears) == 0
    chain.mount(_make_gear())
    assert len(chain.gears) == 1
    chain.mount(_make_gear())
    assert len(chain.gears) == 2


def test_forward_shape_after_mount():
    chain = Chain()
    chain.mount(_make_gear())
    chain.mount(_make_gear())
    x = torch.randn(B, SEQ, D)
    assert chain(x).shape == (B, SEQ, D)

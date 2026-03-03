from collections.abc import Callable

import pytest
import torch

from src.models.gear import Gear, GearLayer

HIDDEN_DIM = 64
NUM_HEADS = 4
FFN_DIM = 256
BATCH = 2
SEQ_LEN = 16

MakeGearFixture = Callable[[int], Gear]


@pytest.fixture
def make_gear() -> MakeGearFixture:
    def _make(num_layers: int) -> Gear:
        return Gear(num_layers, HIDDEN_DIM, NUM_HEADS, FFN_DIM, dropout=0.0)

    return _make


def test_gear_output_shape(make_gear: MakeGearFixture):
    """Input and output dimensions must match (hidden_dim preserved)."""
    gear = make_gear(2)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    out = gear(x)
    assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)


def test_gear_different_sizes(make_gear: MakeGearFixture):
    """Gears with different layer counts produce same-shaped outputs."""
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    for num_layers in [1, 2, 3, 4]:
        gear = make_gear(num_layers)
        out = gear(x)
        assert out.shape == x.shape, f"num_layers={num_layers}"


def test_gear_gradient_flow(make_gear: MakeGearFixture):
    """Gradients must flow through all layers."""
    gear = make_gear(3)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
    out = gear(x)
    out.sum().backward()

    assert x.grad is not None
    for name, param in gear.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_gear_parameter_count(make_gear: MakeGearFixture):
    """Parameter count scales linearly with num_layers."""
    g1 = make_gear(1)
    g2 = make_gear(2)
    g4 = make_gear(4)

    p1 = sum(p.numel() for p in g1.parameters())
    p2 = sum(p.numel() for p in g2.parameters())
    p4 = sum(p.numel() for p in g4.parameters())

    assert p2 == p1 * 2
    assert p4 == p1 * 4


def test_gear_layer_count(make_gear: MakeGearFixture):
    """Gear must contain exactly num_layers GearLayers."""
    for n in [1, 2, 3, 4]:
        gear = make_gear(n)
        assert len(gear.layers) == n
        for layer in gear.layers:
            assert isinstance(layer, GearLayer)

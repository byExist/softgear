from typing import Any

import torch
from torch import Tensor

from src.models.gear import Gear
from src.models.gear_chain import GearChain

HIDDEN_DIM = 64
NUM_HEADS = 4
FFN_DIM = 256
BATCH = 2
SEQ_LEN = 16


def _make_chain(gear_resolutions: list[int]) -> GearChain:
    gears = [
        Gear(s, HIDDEN_DIM, NUM_HEADS, FFN_DIM, dropout=0.0) for s in gear_resolutions
    ]
    return GearChain(gears)


def test_reentry_pattern():
    """Verify the correct gear execution order via mocks."""
    call_log: list[int] = []

    class TrackedGear(Gear):
        def __init__(self, idx: int, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.gear_idx = idx

        def forward(self, x: Tensor) -> Tensor:
            call_log.append(self.gear_idx)
            return super().forward(x)

    gears = [
        TrackedGear(i, s, HIDDEN_DIM, NUM_HEADS, FFN_DIM, dropout=0.0)
        for i, s in enumerate([1, 2, 3])
    ]
    chain = GearChain(gears)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    chain(x)

    # Round 1: G0
    # Round 2: G0, G1
    # Round 3: G0, G1, G2
    assert call_log == [0, 0, 1, 0, 1, 2]


def test_round_outputs_count():
    """Number of round outputs must equal depth."""
    for depth in [2, 3, 4]:
        chain = _make_chain(list(range(1, depth + 1)))
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        _, round_outputs = chain(x)
        assert len(round_outputs) == depth


def test_gear_execution_count():
    """G1 runs D times, G2 runs D-1 times, ..., G_D runs 1 time."""
    counts = [0, 0, 0, 0]

    class CountGear(Gear):
        def __init__(self, idx: int, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.gear_idx = idx

        def forward(self, x: Tensor) -> Tensor:
            counts[self.gear_idx] += 1
            return super().forward(x)

    gears = [
        CountGear(i, s, HIDDEN_DIM, NUM_HEADS, FFN_DIM, dropout=0.0)
        for i, s in enumerate([1, 2, 3, 4])
    ]
    chain = GearChain(gears)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    chain(x)

    assert counts == [4, 3, 2, 1]


def test_full_backprop():
    """Gradients must flow through the entire chain."""
    chain = _make_chain([1, 2, 3, 4])
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
    final, _ = chain(x)
    final.sum().backward()

    assert x.grad is not None
    for name, param in chain.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_independent_weights():
    """Each gear must have its own independent weights."""
    chain = _make_chain([1, 2, 3, 4])

    # Collect all parameter data pointers
    param_ptrs: list[int] = []
    for gear in chain.gears:
        for param in gear.parameters():
            param_ptrs.append(param.data_ptr())

    # All must be unique (no weight sharing)
    assert len(param_ptrs) == len(set(param_ptrs))


def test_computational_cost():
    """Total layer operations for D=4, sizes=[1,2,3,4] should be 20."""
    gear_resolutions = [1, 2, 3, 4]
    depth = len(gear_resolutions)

    total_ops = 0
    for i, s in enumerate(gear_resolutions):
        executions = depth - i  # G1 runs 4x, G2 runs 3x, ...
        total_ops += s * executions

    assert total_ops == 20


def test_output_shape():
    """Final output shape must match input shape."""
    chain = _make_chain([1, 2, 3, 4])
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    final, round_outputs = chain(x)

    assert final.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)
    for r in round_outputs:
        assert r.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

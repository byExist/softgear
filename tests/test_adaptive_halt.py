import torch

from src.models.adaptive_halt import AdaptiveHalt, GearChainWithHalt
from src.models.gear import Gear

HIDDEN_DIM = 64
NUM_HEADS = 4
FFN_DIM = 256
BATCH = 2
SEQ_LEN = 16


def test_halt_when_converged():
    """Should halt when outputs are nearly identical."""
    halt = AdaptiveHalt(delta=0.1)
    a = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    b = a + 1e-4  # tiny difference
    assert halt.should_halt(a, b) is True


def test_no_halt_when_diverging():
    """Should not halt when outputs differ significantly."""
    halt = AdaptiveHalt(delta=0.01)
    a = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    b = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)  # completely different
    assert halt.should_halt(a, b) is False


def test_training_mode_no_halt():
    """In training mode, all rounds must execute (no early stopping)."""
    gears = [
        Gear(size, HIDDEN_DIM, NUM_HEADS, FFN_DIM, dropout=0.0) for size in [1, 2, 3, 4]
    ]
    chain = GearChainWithHalt(gears, delta=1e10)  # very large delta = always halt

    chain.train()
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    _, round_outputs = chain(x)

    # All 4 rounds must run in training mode
    assert len(round_outputs) == 4


def test_eval_mode_early_halt():
    """In eval mode with large delta, should halt early."""
    gears = [
        Gear(size, HIDDEN_DIM, NUM_HEADS, FFN_DIM, dropout=0.0) for size in [1, 2, 3, 4]
    ]
    chain = GearChainWithHalt(gears, delta=1e10)  # very large delta

    chain.eval()
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    with torch.no_grad():
        _, round_outputs = chain(x)

    # Should halt before all 4 rounds
    assert len(round_outputs) < 4


def test_delta_sensitivity():
    """Smaller delta should allow more rounds than larger delta."""
    gears_small = [
        Gear(size, HIDDEN_DIM, NUM_HEADS, FFN_DIM, dropout=0.0) for size in [1, 2, 3, 4]
    ]
    gears_large = [
        Gear(size, HIDDEN_DIM, NUM_HEADS, FFN_DIM, dropout=0.0) for size in [1, 2, 3, 4]
    ]

    # Copy weights so both chains behave identically
    for g_s, g_l in zip(gears_small, gears_large):
        g_l.load_state_dict(g_s.state_dict())

    chain_small_delta = GearChainWithHalt(gears_small, delta=1e-10)
    chain_large_delta = GearChainWithHalt(gears_large, delta=1e10)

    chain_small_delta.eval()
    chain_large_delta.eval()

    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    with torch.no_grad():
        _, rounds_small = chain_small_delta(x)
        _, rounds_large = chain_large_delta(x)

    # Small delta: more rounds (harder to halt)
    # Large delta: fewer rounds (easy to halt)
    assert len(rounds_small) >= len(rounds_large)


def test_halt_preserves_gradient_in_training():
    """Training mode must allow full gradient flow even with halt module."""
    gears = [
        Gear(size, HIDDEN_DIM, NUM_HEADS, FFN_DIM, dropout=0.0) for size in [1, 2, 3]
    ]
    chain = GearChainWithHalt(gears, delta=0.01)
    chain.train()

    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
    final, _ = chain(x)
    final.sum().backward()

    assert x.grad is not None
    for name, param in chain.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

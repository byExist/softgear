import torch
from torch import nn

from src.models.base import ModelOutput
from src.models.softgear import SoftGearModel
from src.training.deep_supervision import DeepSupervisionLoss
from tests.test_softgear import make_cfg

BATCH = 2
SEQ_LEN = 16
VOCAB_SIZE = 11


def _make_model_output() -> ModelOutput:
    logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE)
    intermediate_logits = [
        torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE) for _ in range(3)
    ]
    return ModelOutput(logits=logits, intermediate_logits=intermediate_logits)


def _make_targets() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


def test_loss_includes_all_rounds():
    """Total loss must exceed main-only loss when alpha > 0."""
    loss_fn = DeepSupervisionLoss(nn.CrossEntropyLoss(), alpha=0.3)
    output = _make_model_output()
    targets = _make_targets()

    total_loss = loss_fn(output, targets)
    main_only = nn.CrossEntropyLoss()(
        output.logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    assert total_loss > main_only


def test_alpha_zero_equals_main_loss():
    """alpha=0 should yield the same loss as plain CrossEntropyLoss."""
    loss_fn = DeepSupervisionLoss(nn.CrossEntropyLoss(), alpha=0.0)
    output = _make_model_output()
    targets = _make_targets()

    total_loss = loss_fn(output, targets)
    main_only = nn.CrossEntropyLoss()(
        output.logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    assert torch.allclose(total_loss, main_only)


def test_no_intermediates():
    """Works when intermediate_logits is None or empty."""
    loss_fn = DeepSupervisionLoss(nn.CrossEntropyLoss(), alpha=0.3)
    targets = _make_targets()

    candidates: list[list[torch.Tensor] | None] = [None, []]
    for intermediates in candidates:
        output = ModelOutput(
            logits=torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE),
            intermediate_logits=intermediates,
        )
        loss = loss_fn(output, targets)
        assert loss.ndim == 0  # scalar


def test_gradient_to_all_gears():
    """Backward must flow gradients to all gear parameters."""
    model = SoftGearModel(make_cfg())
    loss_fn = DeepSupervisionLoss(nn.CrossEntropyLoss(), alpha=0.3)

    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    output = model(x)
    loss = loss_fn(output, targets)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

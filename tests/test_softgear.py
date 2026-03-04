from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.base import ModelOutput
from src.models.sudoku_model import build_sudoku_model


def make_cfg(**overrides: Any) -> DictConfig:
    defaults: dict[str, Any] = dict(
        vocab_size=11,
        hidden_dim=64,
        num_heads=4,
        ffn_dim=256,
        gear_resolutions=[1, 2, 3, 4],
        dropout=0.0,
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


BATCH = 2
SEQ_LEN = 81


def test_model_output_interface():
    """Model must return a ModelOutput dataclass."""
    model = build_sudoku_model(make_cfg())
    x = torch.randint(0, 11, (BATCH, SEQ_LEN))
    out = model(x)

    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (BATCH, SEQ_LEN, 11)
    assert out.intermediate_logits is not None
    # 4 gears = 4 rounds, intermediate = first 3
    assert len(out.intermediate_logits) == 3
    for il in out.intermediate_logits:
        assert il.shape == (BATCH, SEQ_LEN, 11)


def test_parameter_count():
    """Parameter count must be deterministic for a given config."""
    cfg = make_cfg()
    m1 = build_sudoku_model(cfg)
    m2 = build_sudoku_model(cfg)
    assert m1.parameter_count() == m2.parameter_count()
    assert m1.parameter_count() > 0


def test_config_driven():
    """Model works with various gear_resolutions configurations."""
    for sizes in [[1, 2], [1, 2, 3, 4], [2, 4, 8]]:
        cfg = make_cfg(gear_resolutions=sizes)
        model = build_sudoku_model(cfg)
        x = torch.randint(0, 11, (BATCH, SEQ_LEN))
        out = model(x)
        assert out.logits.shape == (BATCH, SEQ_LEN, 11)
        assert len(out.intermediate_logits) == len(sizes) - 1


def test_gradient_flow():
    """Gradients must flow from output to embedding."""
    model = build_sudoku_model(make_cfg())
    x = torch.randint(0, 11, (BATCH, SEQ_LEN))
    out = model(x)
    out.logits.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_intermediate_logits_differ():
    """Intermediate logits from different rounds should not be identical."""
    model = build_sudoku_model(make_cfg())
    x = torch.randint(0, 11, (BATCH, SEQ_LEN))
    out = model(x)

    # At least some intermediate logits should differ from each other
    if len(out.intermediate_logits) >= 2:
        assert not torch.allclose(
            out.intermediate_logits[0], out.intermediate_logits[1]
        )

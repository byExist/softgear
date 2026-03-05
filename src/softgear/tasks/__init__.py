"""Task registry for SoftGear experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader

from softgear.config import DataConfig, ModelConfig
from softgear.models.analyzer import Analyzer
from softgear.models.gear import Gear


@dataclass(frozen=True)
class TaskSpec:
    build_model: Callable[[ModelConfig], Analyzer]
    make_gear_factory: Callable[[ModelConfig], Callable[[int], Gear]]
    build_loaders: Callable[[DataConfig], tuple[DataLoader[Any], DataLoader[Any]]]
    metrics_fn: Callable[[Tensor, Tensor, Tensor], dict[str, float]]
    mount_all_gears: Callable[[Analyzer, ModelConfig], None]


def get_task(name: str) -> TaskSpec:
    """Lazy-load a task spec by name."""
    if name == "sudoku":
        from softgear.tasks.sudoku import spec
        return spec
    raise ValueError(f"Unknown task: {name!r}. Available: sudoku")

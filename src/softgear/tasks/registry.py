"""Task registry for SoftGear experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any  # used in DataLoader type

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
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    predict_fn: Callable[[Tensor], Tensor]
    metrics_fn: Callable[[Tensor, Tensor, Tensor], dict[str, float]]
    mount_all_gears: Callable[[Analyzer, ModelConfig], None]
    model_defaults: ModelConfig
    data_defaults: DataConfig
    download: Callable[[str], None] | None = None


_TASKS = ("sudoku4", "sudoku9")


def get_task(name: str) -> TaskSpec:
    """Lazy-load a task spec by name."""
    if name == "sudoku4":
        from softgear.tasks.sudoku.spec import sudoku4
        return sudoku4
    if name == "sudoku9":
        from softgear.tasks.sudoku.spec import sudoku9
        return sudoku9
    raise ValueError(f"Unknown task: {name!r}. Available: {', '.join(_TASKS)}")

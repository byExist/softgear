"""Sudoku task specifications for different board sizes."""

from softgear.config import DataConfig, ModelConfig
from softgear.tasks.registry import TaskSpec
from softgear.tasks.sudoku.data import build_sudoku_loaders
from softgear.tasks.sudoku.download import download_sudoku4, download_sudoku9
from softgear.tasks.sudoku.metrics import sudoku_accuracy
from softgear.tasks.sudoku.model import (
    build_sudoku_model, make_gear_factory, mount_all_gears,
    sudoku_loss, sudoku_predict,
)

sudoku4 = TaskSpec(
    build_model=build_sudoku_model,
    make_gear_factory=make_gear_factory,
    build_loaders=build_sudoku_loaders,
    loss_fn=sudoku_loss,
    predict_fn=sudoku_predict,
    metrics_fn=sudoku_accuracy,
    mount_all_gears=mount_all_gears,
    model_defaults=ModelConfig(vocab_size=5, seq_len=16),
    data_defaults=DataConfig(path="data/sudoku-4x4"),
    download=download_sudoku4,
)

sudoku9 = TaskSpec(
    build_model=build_sudoku_model,
    make_gear_factory=make_gear_factory,
    build_loaders=build_sudoku_loaders,
    loss_fn=sudoku_loss,
    predict_fn=sudoku_predict,
    metrics_fn=sudoku_accuracy,
    mount_all_gears=mount_all_gears,
    model_defaults=ModelConfig(vocab_size=10, seq_len=81),
    data_defaults=DataConfig(path="data/sudoku-extreme"),
    download=download_sudoku9,
)

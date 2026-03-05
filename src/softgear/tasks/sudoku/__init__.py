from softgear.tasks import TaskSpec
from softgear.tasks.sudoku.data import build_sudoku_loaders
from softgear.tasks.sudoku.metrics import sudoku_accuracy
from softgear.tasks.sudoku.model import build_sudoku_model, make_gear_factory, mount_all_gears

spec = TaskSpec(
    build_model=build_sudoku_model,
    make_gear_factory=make_gear_factory,
    build_loaders=build_sudoku_loaders,
    metrics_fn=sudoku_accuracy,
    mount_all_gears=mount_all_gears,
)

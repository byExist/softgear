"""Sudoku-specific evaluation metrics."""

from torch import Tensor


def sudoku_accuracy(
    predictions: Tensor,
    targets: Tensor,
    inputs: Tensor,
) -> dict[str, float]:
    """Compute sudoku accuracy metrics.

    Args:
        predictions: (B, 81) argmax predictions.
        targets: (B, 81) ground truth solutions.
        inputs: (B, 81) original puzzles (0 = blank cell).

    Returns:
        cell_accuracy: Accuracy over all cells.
        blank_accuracy: Accuracy over blank cells only (input == 0).
        puzzle_accuracy: Fraction of fully correct puzzles.
    """
    correct = predictions == targets

    # Cell accuracy: all cells
    cell_accuracy = correct.float().mean().item()

    # Blank accuracy: only cells where input was 0
    blank_mask = inputs == 0
    if blank_mask.any():
        blank_accuracy = correct[blank_mask].float().mean().item()
    else:
        blank_accuracy = 1.0

    # Puzzle accuracy: all 81 cells correct
    puzzle_accuracy = correct.all(dim=1).float().mean().item()

    return {
        "cell_accuracy": cell_accuracy,
        "blank_accuracy": blank_accuracy,
        "puzzle_accuracy": puzzle_accuracy,
    }

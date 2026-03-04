"""Tests for Sudoku evaluation metrics."""

import torch

from src.evaluation.sudoku_metrics import sudoku_accuracy


class TestSudokuAccuracy:
    def test_perfect_prediction(self) -> None:
        targets = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        inputs = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
        preds = targets.clone()

        metrics = sudoku_accuracy(preds, targets, inputs)
        assert metrics["cell_accuracy"] == 1.0
        assert metrics["blank_accuracy"] == 1.0
        assert metrics["puzzle_accuracy"] == 1.0

    def test_blank_accuracy(self) -> None:
        # Input: cells 0,1 are blank, cells 2,3 are given
        inputs = torch.tensor([[0, 0, 3, 4]])
        targets = torch.tensor([[1, 2, 3, 4]])
        # Predict: cell 0 correct, cell 1 wrong, given cells correct
        preds = torch.tensor([[1, 9, 3, 4]])

        metrics = sudoku_accuracy(preds, targets, inputs)
        assert metrics["blank_accuracy"] == 0.5  # 1/2 blank cells correct
        assert metrics["cell_accuracy"] == 0.75  # 3/4 total cells correct

    def test_puzzle_accuracy_one_wrong(self) -> None:
        # Two puzzles: first all correct, second has one wrong
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        inputs = torch.tensor([[0, 0, 0], [0, 0, 0]])
        preds = torch.tensor([[1, 2, 3], [4, 5, 9]])  # second puzzle has 1 error

        metrics = sudoku_accuracy(preds, targets, inputs)
        assert metrics["puzzle_accuracy"] == 0.5  # 1/2 puzzles fully correct

    def test_no_blanks(self) -> None:
        # All cells are given (no blanks)
        inputs = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([[1, 2, 3]])
        preds = torch.tensor([[1, 2, 3]])

        metrics = sudoku_accuracy(preds, targets, inputs)
        assert metrics["blank_accuracy"] == 1.0

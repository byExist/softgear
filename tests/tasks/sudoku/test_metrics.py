"""Tests for sudoku accuracy metrics."""

import torch

from softgear.tasks.sudoku.metrics import sudoku_accuracy


def test_perfect_prediction():
    preds = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[1, 2, 3, 4]])
    inputs = torch.tensor([[0, 0, 3, 4]])
    metrics = sudoku_accuracy(preds, targets, inputs)
    assert metrics["cell_accuracy"] == 1.0
    assert metrics["blank_accuracy"] == 1.0
    assert metrics["puzzle_accuracy"] == 1.0


def test_all_wrong():
    preds = torch.tensor([[9, 9, 9, 9]])
    targets = torch.tensor([[1, 2, 3, 4]])
    inputs = torch.tensor([[0, 0, 0, 0]])
    metrics = sudoku_accuracy(preds, targets, inputs)
    assert metrics["cell_accuracy"] == 0.0
    assert metrics["blank_accuracy"] == 0.0
    assert metrics["puzzle_accuracy"] == 0.0


def test_blank_only():
    preds = torch.tensor([[1, 2, 9, 9]])
    targets = torch.tensor([[1, 2, 3, 4]])
    inputs = torch.tensor([[1, 2, 0, 0]])  # blanks at positions 2, 3
    metrics = sudoku_accuracy(preds, targets, inputs)
    assert metrics["cell_accuracy"] == 0.5
    assert metrics["blank_accuracy"] == 0.0
    assert metrics["puzzle_accuracy"] == 0.0


def test_no_blanks():
    preds = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[1, 2, 3, 4]])
    inputs = torch.tensor([[1, 2, 3, 4]])  # no blanks
    metrics = sudoku_accuracy(preds, targets, inputs)
    assert metrics["blank_accuracy"] == 1.0

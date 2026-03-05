"""Tests for sudoku loss and predict."""

import torch

from softgear.tasks.sudoku.model import sudoku_loss, sudoku_predict


def test_loss_runs():
    logits = torch.randn(2, 16, 5)
    targets = torch.randint(0, 5, (2, 16))
    loss = sudoku_loss(logits, targets)
    assert loss.shape == ()
    assert loss.item() > 0


def test_predict_argmax():
    logits = torch.zeros(2, 4, 5)
    logits[0, 0, 3] = 10.0  # prediction should be 3
    logits[0, 1, 1] = 10.0  # prediction should be 1
    preds = sudoku_predict(logits)
    assert preds[0, 0] == 3
    assert preds[0, 1] == 1

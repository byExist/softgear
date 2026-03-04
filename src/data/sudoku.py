"""Sudoku-Extreme dataset and data loader utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

SEQ_LEN = 81


def _tokenize(s: str) -> np.ndarray:
    """Convert 81-char sudoku string to int8 array. '.' -> 0, '1'-'9' -> 1-9."""
    return np.array([0 if c == "." else int(c) for c in s], dtype=np.int8)


class SudokuDataset(Dataset[tuple[Tensor, Tensor]]):
    """Sudoku-Extreme dataset. Each item is (puzzle, solution) of shape (81,)."""

    def __init__(self, csv_path: str | Path, max_samples: int | None = None):
        self.puzzles: np.ndarray  # (N, 81) int8
        self.solutions: np.ndarray  # (N, 81) int8

        puzzles: list[np.ndarray] = []
        solutions: list[np.ndarray] = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break
                puzzles.append(_tokenize(row["question"]))
                solutions.append(_tokenize(row["answer"]))

        self.puzzles = np.stack(puzzles)
        self.solutions = np.stack(solutions)

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return (
            torch.from_numpy(self.puzzles[idx].astype(np.int64)),  # type: ignore[reportUnknownMemberType]
            torch.from_numpy(self.solutions[idx].astype(np.int64)),  # type: ignore[reportUnknownMemberType]
        )


def build_sudoku_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Build train and validation DataLoaders from config."""
    data_dir = Path(cfg.data.path)
    max_samples = getattr(cfg.data, "max_samples", None)

    train_dataset = SudokuDataset(data_dir / "train.csv", max_samples=max_samples)
    val_dataset = SudokuDataset(data_dir / "test.csv", max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

"""Tests for Sudoku dataset and data loaders."""

import csv
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.data.sudoku import SudokuDataset, build_sudoku_loaders


@pytest.fixture()
def csv_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with small train/test CSV files."""
    rows = [
        {
            "source": "test",
            "question": "5...27..9..41......1..5.3...92.6.8...5......66..7..29.8...7...2.......8...9..36..",
            "answer": "583427169974136528216859374792364851351298746648715293865971432137642985429583617",
            "rating": "18",
        },
        {
            "source": "test",
            "question": ".2.1..9.4....2.6....3.94..5..497..5..1...24.7....4..6...6.............8..9..1.7.6",
            "answer": "827165934459328671163794825384976152615832497972541368236487519741659283598213746",
            "rating": "95",
        },
    ]
    for name in ["train.csv", "test.csv"]:
        with open(tmp_path / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "question", "answer", "rating"])
            writer.writeheader()
            writer.writerows(rows)
    return tmp_path


class TestSudokuDataset:
    def test_length(self, csv_dir: Path) -> None:
        ds = SudokuDataset(csv_dir / "train.csv")
        assert len(ds) == 2

    def test_item_shape(self, csv_dir: Path) -> None:
        ds = SudokuDataset(csv_dir / "train.csv")
        puzzle, solution = ds[0]
        assert puzzle.shape == (81,)
        assert solution.shape == (81,)
        assert puzzle.dtype == torch.long
        assert solution.dtype == torch.long

    def test_tokenization(self, csv_dir: Path) -> None:
        ds = SudokuDataset(csv_dir / "train.csv")
        puzzle, solution = ds[0]
        # '5' -> 5, '.' -> 0
        assert puzzle[0].item() == 5
        assert puzzle[1].item() == 0
        # Solution has no blanks
        assert (solution > 0).all()
        assert (solution <= 9).all()

    def test_vocab_range(self, csv_dir: Path) -> None:
        ds = SudokuDataset(csv_dir / "train.csv")
        for i in range(len(ds)):
            puzzle, solution = ds[i]
            assert puzzle.min() >= 0
            assert puzzle.max() <= 9
            assert solution.min() >= 1
            assert solution.max() <= 9

    def test_max_samples(self, csv_dir: Path) -> None:
        ds = SudokuDataset(csv_dir / "train.csv", max_samples=1)
        assert len(ds) == 1


class TestBuildSudokuLoaders:
    def test_dataloader_batch(self, csv_dir: Path) -> None:
        cfg = OmegaConf.create({
            "data": {
                "path": str(csv_dir),
                "batch_size": 2,
                "num_workers": 0,
                "max_samples": None,
            }
        })
        train_loader, _ = build_sudoku_loaders(cfg)
        batch = next(iter(train_loader))
        puzzles, solutions = batch
        assert puzzles.shape == (2, 81)
        assert solutions.shape == (2, 81)

    def test_train_val_split(self, csv_dir: Path) -> None:
        cfg = OmegaConf.create({
            "data": {
                "path": str(csv_dir),
                "batch_size": 2,
                "num_workers": 0,
                "max_samples": None,
            }
        })
        train_loader, val_loader = build_sudoku_loaders(cfg)
        assert len(train_loader.dataset) == 2  # type: ignore[arg-type]
        assert len(val_loader.dataset) == 2  # type: ignore[arg-type]

"""Tests for SudokuDataset curriculum sorting."""

from pathlib import Path

from softgear.tasks.sudoku.data import SudokuDataset


def test_sort_by_difficulty_monotonic(tmp_path: Path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "question,answer\n"
        "1234,1234\n"  # 0 blanks
        "....,1234\n"  # 4 blanks
        ".2.4,1234\n"  # 2 blanks
        "..34,1234\n"  # 2 blanks
    )
    ds = SudokuDataset(csv_path)
    ds.sort_by_difficulty()
    blank_counts = (ds.puzzles == 0).sum(axis=1)
    assert all(blank_counts[i] <= blank_counts[i + 1] for i in range(len(blank_counts) - 1))

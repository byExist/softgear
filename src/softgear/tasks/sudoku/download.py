"""Download functions for Sudoku datasets."""

from __future__ import annotations

import csv
import random
import urllib.request
from pathlib import Path


def download_sudoku4(data_dir: str) -> None:
    """Download 4x4 Sudoku dataset from GitHub."""
    raw_url = "https://raw.githubusercontent.com/Black-Phoenix/4x4-Sudoku-Dataset/master/4x4_sudoku_unique_puzzles.csv"
    out = Path(data_dir)
    train_path = out / "train.csv"
    test_path = out / "test.csv"

    if train_path.exists() and test_path.exists():
        print("Already downloaded. Remove train.csv/test.csv to re-download.")
        return

    out.mkdir(parents=True, exist_ok=True)

    print("Downloading 4x4 sudoku dataset...")
    tmp = out / "raw.csv"
    urllib.request.urlretrieve(raw_url, tmp)

    print("Converting format and splitting 90/10...")
    random.seed(42)
    rows: list[tuple[str, str]] = []
    with open(tmp, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["Puzzle"].replace("0", ".")
            a = row["Solution"].replace("0", ".")
            rows.append((q, a))

    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    train, test = rows[:split], rows[split:]

    for path, data in [(train_path, train), (test_path, test)]:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "answer"])
            w.writerows(data)

    tmp.unlink()
    print(f"{len(train)} train / {len(test)} test puzzles")
    print("Done.")


def download_sudoku9(data_dir: str) -> None:
    """Download Sudoku-Extreme dataset from Hugging Face."""
    from huggingface_hub import snapshot_download  # type: ignore[reportUnknownVariableType]

    out = Path(data_dir)
    train_path = out / "train.csv"
    test_path = out / "test.csv"

    if train_path.exists() and test_path.exists():
        print("Already downloaded. Remove train.csv/test.csv to re-download.")
        return

    print("Downloading sudoku-extreme from Hugging Face...")
    snapshot_download(
        "sapientinc/sudoku-extreme",
        local_dir=str(out),
        repo_type="dataset",
    )

    train_lines = sum(1 for _ in open(train_path)) - 1
    test_lines = sum(1 for _ in open(test_path)) - 1
    print(f"{train_lines} train / {test_lines} test puzzles")
    print("Done.")

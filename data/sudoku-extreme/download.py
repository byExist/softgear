#!/usr/bin/env python3
"""Download Sudoku-Extreme dataset from Hugging Face.

Source: https://huggingface.co/datasets/sapientinc/sudoku-extreme
Format: question,answer (81-char strings, '.' = blank)
~3.8M train / ~423K test puzzles, rated by tdoku backtrack count.
"""

import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
train_path = here / "train.csv"
test_path = here / "test.csv"

if train_path.exists() and test_path.exists():
    print("Already downloaded. Remove train.csv/test.csv to re-download.")
    raise SystemExit(0)

print("Downloading sudoku-extreme from Hugging Face...")
subprocess.run(
    ["huggingface-cli", "download", "sapientinc/sudoku-extreme",
     "--local-dir", str(here), "--repo-type", "dataset"],
    check=True,
)

train_lines = sum(1 for _ in open(train_path)) - 1
test_lines = sum(1 for _ in open(test_path)) - 1
print(f"{train_lines} train / {test_lines} test puzzles")
print("Done.")

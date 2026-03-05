#!/usr/bin/env python3
"""Download 4x4 Sudoku dataset from GitHub and convert to project format.

Source: https://github.com/Black-Phoenix/4x4-Sudoku-Dataset
Original: CSV with 0=blank, columns Puzzle/Solution (1M puzzles)
Output:   CSV with .=blank, columns question/answer, 90/10 train/test split.
"""

import csv
import random
import urllib.request
from pathlib import Path

RAW_URL = "https://raw.githubusercontent.com/Black-Phoenix/4x4-Sudoku-Dataset/master/4x4_sudoku_unique_puzzles.csv"

here = Path(__file__).resolve().parent
train_path = here / "train.csv"
test_path = here / "test.csv"

if train_path.exists() and test_path.exists():
    print("Already downloaded. Remove train.csv/test.csv to re-download.")
    raise SystemExit(0)

print("Downloading 4x4 sudoku dataset...")
tmp = here / "raw.csv"
urllib.request.urlretrieve(RAW_URL, tmp)

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

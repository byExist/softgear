"""End-to-end smoke tests for pretrain.py and evaluate.py.

Skipped when Sudoku data is not available locally.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

DATA_DIR = Path("data/sudoku-extreme")
SKIP_MSG = "Sudoku data not available (data/sudoku-extreme/train.csv)"

requires_data = pytest.mark.skipif(
    not (DATA_DIR / "train.csv").exists(),
    reason=SKIP_MSG,
)


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.slow
@requires_data
def test_pretrain_smoke():
    """pretrain.py completes without error and creates checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run([
            "pretrain.py",
            "model.hidden_dim=64",
            "model.num_heads=4",
            "model.ffn_dim=256",
            "model.gear_sizes=[1,2]",
            "model.adaptive_halt.enabled=false",
            f"training.checkpoint_dir={tmpdir}",
            "training.max_epochs_per_phase=1",
            "training.ema_alphas=[0.99,0.999]",
            "data.max_samples=64",
            "wandb.project=null",
        ])

        assert result.returncode == 0, (
            f"pretrain.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert (Path(tmpdir) / "best.pt").exists(), "best.pt not created"
        assert (Path(tmpdir) / "latest.pt").exists(), "latest.pt not created"


@pytest.mark.slow
@requires_data
def test_evaluate_smoke():
    """evaluate.py loads checkpoint and prints puzzle_accuracy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train first
        train_result = _run([
            "pretrain.py",
            "model.hidden_dim=64",
            "model.num_heads=4",
            "model.ffn_dim=256",
            "model.gear_sizes=[1,2]",
            "model.adaptive_halt.enabled=false",
            f"training.checkpoint_dir={tmpdir}",
            "training.max_epochs_per_phase=1",
            "training.ema_alphas=[0.99,0.999]",
            "data.max_samples=64",
            "wandb.project=null",
        ])
        assert train_result.returncode == 0, (
            f"pretrain.py failed:\nstdout: {train_result.stdout}\nstderr: {train_result.stderr}"
        )

        # Evaluate
        ckpt_path = Path(tmpdir) / "best.pt"
        eval_result = _run([
            "evaluate.py",
            "model.hidden_dim=64",
            "model.num_heads=4",
            "model.ffn_dim=256",
            "model.gear_sizes=[1,2]",
            "model.adaptive_halt.enabled=false",
            f"+checkpoint={ckpt_path}",
            "data.max_samples=64",
            "wandb.project=null",
        ])

        assert eval_result.returncode == 0, (
            f"evaluate.py failed:\nstdout: {eval_result.stdout}\nstderr: {eval_result.stderr}"
        )
        combined = (eval_result.stdout + eval_result.stderr).lower()
        assert "puzzle accuracy" in combined or "puzzle_accuracy" in combined, (
            f"puzzle_accuracy not in output:\nstdout: {eval_result.stdout}\nstderr: {eval_result.stderr}"
        )

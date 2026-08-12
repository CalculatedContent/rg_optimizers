from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.monitor import (
    format_monitor_snapshot,
    load_monitor_frames,
)
from rg_nanogpt_one_head.spectral import (
    SPECTRAL_METRICS,
    summarize_spectral_frame,
)


def test_rand_distance_is_a_first_class_summary_metric() -> None:
    assert "rand_distance" in SPECTRAL_METRICS
    frame = pd.DataFrame(
        {
            "alpha": [2.0, 2.4, 2.8],
            "D": [0.05, 0.10, 0.15],
            "ERG_gap": [1.0, 2.0, 3.0],
            "num_traps": [0.0, 1.0, 2.0],
            "rand_distance": [0.20, 0.30, 0.40],
        }
    )

    summary = summarize_spectral_frame(
        frame,
        step=10,
        tokens_seen=80,
        epoch=0.5,
    )

    assert summary["rand_distance_n"] == 3
    assert summary["rand_distance_mean"] == pytest.approx(0.30)
    assert summary["rand_distance_median"] == pytest.approx(0.30)
    assert summary["rand_distance_min"] == pytest.approx(0.20)
    assert summary["rand_distance_max"] == pytest.approx(0.40)


def test_live_monitor_displays_latest_layer_rand_distance(tmp_path) -> None:
    run_dir = tmp_path / "results" / "muon" / "seed_1337"
    spectral_dir = run_dir / "spectral"
    spectral_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "step": 10,
                "epoch": 0.1,
                "primary_lr": 0.01,
                "val_loss": 6.5,
                "val_accuracy": 0.15,
            },
            {
                "step": 20,
                "epoch": 0.2,
                "primary_lr": 0.009,
                "val_loss": 6.2,
                "val_accuracy": 0.17,
            },
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)

    rows = []
    for step, epoch, offset in ((10, 0.1, 0.0), (20, 0.2, 0.1)):
        for matrix_name, alpha in (("L00_W_Q", 2.0), ("L00_W_K", 2.4)):
            rows.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "matrix_name": matrix_name,
                    "alpha": alpha + offset,
                    "D": 0.05 + offset,
                    "rand_distance": 0.25 + offset,
                    "ERG_gap": 3.0 + offset,
                    "num_traps": 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(
        spectral_dir / "layers.csv",
        index=False,
    )

    metrics, layers = load_monitor_frames(run_dir)
    text = format_monitor_snapshot(
        run_dir,
        metrics,
        layers,
        recent=2,
    )

    assert "step=20" in text
    assert "rand_distance" in text
    assert "RAND_DISTANCE:" in text
    assert "0.3500" in text
    assert "L00_W_Q" in text
    assert "L00_W_K" in text


def test_live_monitor_reports_incompatible_randomized_output(tmp_path) -> None:
    run_dir = tmp_path / "results" / "muon" / "seed_1337"
    spectral_dir = run_dir / "spectral"
    spectral_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "step": 1,
                "epoch": 0.1,
                "matrix_name": "L00_W_Q",
                "alpha": 2.0,
                "D": 0.1,
                "ERG_gap": 0.0,
                "num_traps": 0.0,
            }
        ]
    ).to_csv(spectral_dir / "layers.csv", index=False)

    _, layers = load_monitor_frames(run_dir)
    text = format_monitor_snapshot(
        run_dir,
        pd.DataFrame(),
        layers,
    )

    assert "INCOMPATIBLE SPECTRAL OUTPUT" in text
    assert "rand_distance" in text
    assert "randomize=True" in text

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.validation_report import (
    all_validation_evaluations,
    format_summary,
    normalize_metrics,
    resolve_metrics_path,
    validation_by_epoch,
)


def sample_metrics(max_epoch: float = 10.112) -> pd.DataFrame:
    epochs = list(np.arange(0.0, 10.0001, 0.25))
    if max_epoch > 10.0:
        epochs.append(float(max_epoch))
    return pd.DataFrame(
        [
            {
                "step": index * 250,
                "epoch": epoch,
                "primary_lr": 0.002 if epoch >= 1.0 else 0.02,
                "train_accuracy": min(0.20, 0.01 + 0.018 * epoch),
                "val_accuracy": min(0.19, 0.008 + 0.017 * epoch),
                "train_loss": 10.0 - 0.4 * epoch,
                "val_loss": 10.1 - 0.39 * epoch,
            }
            for index, epoch in enumerate(epochs)
        ]
    )


def test_integer_epoch_report_uses_full_current_run() -> None:
    report = validation_by_epoch(sample_metrics(), interval=1.0)
    assert report["TARGET_EPOCH"].tolist() == [float(value) for value in range(11)]
    assert report.iloc[-1]["ACTUAL_EPOCH"] == pytest.approx(10.0)
    assert report["IS_CURRENT"].eq(0).all()


def test_report_can_append_latest_partial_epoch() -> None:
    report = validation_by_epoch(
        sample_metrics(), interval=1.0, include_current=True
    )
    current = report.iloc[-1]
    assert len(report) == 12
    assert current["ACTUAL_EPOCH"] == pytest.approx(10.112)
    assert current["IS_CURRENT"] == 1


def test_quarter_epoch_report_includes_train_columns() -> None:
    report = validation_by_epoch(
        sample_metrics(max_epoch=2.0),
        interval=0.25,
        end_epoch=2.0,
        include_train=True,
    )
    assert len(report) == 9
    assert "TRAIN_ACC_%" in report.columns
    assert "TRAIN_LOSS" in report.columns
    assert report["EPOCH_ERROR"].abs().max() == pytest.approx(0.0)


def test_all_evaluations_preserves_complete_rows() -> None:
    metrics = sample_metrics(max_epoch=2.0)
    report = all_validation_evaluations(metrics, include_train=True)
    assert len(report) == len(metrics)
    assert "TRAIN_ACC_%" in report.columns
    assert "VAL_ACC_%" in report.columns


def test_normalization_keeps_last_duplicate_step() -> None:
    metrics = sample_metrics(max_epoch=2.0)
    duplicate = metrics.iloc[[3]].copy()
    duplicate["val_accuracy"] = 0.5
    normalized = normalize_metrics(
        pd.concat([metrics, duplicate], ignore_index=True)
    )
    selected = normalized[normalized["step"] == int(duplicate.iloc[0]["step"])]
    assert len(selected) == 1
    assert float(selected.iloc[0]["val_accuracy"]) == pytest.approx(0.5)


def test_missing_validation_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="val_loss"):
        normalize_metrics(sample_metrics().drop(columns=["val_loss"]))


def test_metrics_path_supports_any_optimizer(tmp_path) -> None:
    path = resolve_metrics_path(
        metrics_csv=None,
        run_dir=None,
        results_root=tmp_path,
        optimizer="muon_clip",
        seed=9011,
        device="cpu",
    )
    assert path == tmp_path / "muon_clip" / "seed_9011" / "metrics.csv"


def test_summary_reports_current_and_global_best() -> None:
    metrics = sample_metrics(max_epoch=2.0)
    metrics.loc[metrics["epoch"] == 7.25, "val_accuracy"] = 0.25
    text = format_summary(metrics)
    assert "CURRENT" in text
    assert "BEST VALIDATION ACCURACY" in text
    assert "epoch=7.2500" in text
    assert "val_acc=25.00%" in text

"""Shared live-monitoring tables for the MNIST TraceLogRG experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RUNS = ("AdamW baseline", "AdamW + TraceLogRG")
LAYERS = ("fc1", "fc2", "fc3")


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame is not None and not frame.empty:
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def layer_names(values: pd.Series) -> pd.Series:
    return values.astype(str).str.split(".").str[-1]


def validate_weightwatcher(frame: pd.DataFrame) -> None:
    """Require successful alpha and ERG-gap rows to come directly from WeightWatcher."""
    ok = frame.loc[frame["status"].eq("ok")]
    if ok.empty:
        return
    required = {"alpha", "ERG_gap", "alpha_source", "ERG_gap_source"}
    missing = required - set(ok.columns)
    if missing:
        raise RuntimeError(f"WeightWatcher output is missing {sorted(missing)}")
    if not ok["alpha_source"].astype(str).eq("WeightWatcher").all():
        raise RuntimeError("alpha was not sourced directly from WeightWatcher")
    if not ok["ERG_gap_source"].astype(str).eq("WeightWatcher").all():
        raise RuntimeError("ERG_gap was not sourced directly from WeightWatcher")
    if ok[["alpha", "ERG_gap"]].isna().any().any():
        raise RuntimeError("WeightWatcher returned a missing alpha or ERG_gap")


def monitor_performance(
    performance: pd.DataFrame,
    state: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Add online overfitting indicators using only epochs seen so far."""
    rows = []
    for _, row in performance.iterrows():
        run = str(row["run"])
        record = state[run]
        test_loss = float(row["test_loss"])
        test_acc = float(row["test_acc"])
        if test_loss < record["best_loss"]:
            record["best_loss"] = test_loss
            record["best_loss_epoch"] = int(row["epoch"])
        if test_acc > record["peak_acc"]:
            record["peak_acc"] = test_acc
            record["peak_acc_epoch"] = int(row["epoch"])
        out = dict(row)
        out.update(
            best_test_loss_so_far=record["best_loss"],
            best_test_loss_epoch=record["best_loss_epoch"],
            test_loss_rebound=test_loss - record["best_loss"],
            peak_test_acc_so_far=record["peak_acc"],
            peak_test_acc_epoch=record["peak_acc_epoch"],
            peak_to_current_test_acc_drop=record["peak_acc"] - test_acc,
            loss_generalization_gap=float(row["test_loss"]) - float(row["train_loss"]),
            accuracy_generalization_gap=float(row["train_acc"]) - float(row["test_acc"]),
        )
        rows.append(out)
    return pd.DataFrame(rows)


def correction_table(
    steps: pd.DataFrame,
    expected_steps: int,
    next_supports: dict[str, int],
) -> pd.DataFrame:
    """Summarize correction magnitude, firing, and coverage for every layer."""
    rows = []
    for layer in LAYERS:
        parameter = f"{layer}.weight"
        group = (
            steps.loc[steps["parameter"].astype(str).str.endswith(parameter)].copy()
            if steps is not None and not steps.empty
            else pd.DataFrame()
        )
        if group.empty:
            rows.append(
                dict(
                    layer=layer,
                    support_used=np.nan,
                    next_epoch_support=next_supports.get(parameter, np.nan),
                    opportunities=0,
                    coverage=0.0,
                    fired=0,
                    fired_fraction=0.0,
                    negative_drift_fraction=np.nan,
                    mean_correction_ratio_all_steps=0.0,
                    mean_correction_ratio_when_fired=np.nan,
                    max_correction_ratio=0.0,
                    geometry_failures=0,
                    selected_drift_residual=np.nan,
                )
            )
            continue

        for column in (
            "correction_ratio",
            "base_trace_log_drift",
            "corrected_trace_log_drift",
            "retained_rank",
        ):
            if column not in group:
                group[column] = np.nan
            group[column] = pd.to_numeric(group[column], errors="coerce")

        fired = group["status"].eq("ok")
        finite_drift = group["base_trace_log_drift"].notna()
        ratio_all = group["correction_ratio"].fillna(0.0)
        ratio_fired = group.loc[fired, "correction_ratio"]
        base = group.loc[fired, "base_trace_log_drift"].abs().sum()
        corrected = group.loc[fired, "corrected_trace_log_drift"].abs().sum()

        rows.append(
            dict(
                layer=layer,
                support_used=group["retained_rank"].dropna().median(),
                next_epoch_support=next_supports.get(parameter, np.nan),
                opportunities=len(group),
                coverage=group["global_step"].nunique() / expected_steps,
                fired=int(fired.sum()),
                fired_fraction=float(fired.mean()),
                negative_drift_fraction=(
                    float(group.loc[finite_drift, "base_trace_log_drift"].lt(0).mean())
                    if finite_drift.any()
                    else np.nan
                ),
                mean_correction_ratio_all_steps=float(ratio_all.mean()),
                mean_correction_ratio_when_fired=(
                    float(ratio_fired.mean()) if len(ratio_fired) else np.nan
                ),
                max_correction_ratio=float(ratio_all.max()),
                geometry_failures=int(group["status"].eq("geometry_failed").sum()),
                selected_drift_residual=(
                    float(corrected / base) if base > 0 else np.nan
                ),
            )
        )
    return pd.DataFrame(rows)

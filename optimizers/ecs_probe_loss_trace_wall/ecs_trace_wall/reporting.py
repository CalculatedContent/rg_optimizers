"""Metric rows, confidence intervals, correction summaries, and audits."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .config import ExperimentConfig
from .optimizer import CorrectionRecord


def performance_row(
    *,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    train_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
    learning_rate: float,
    parameter_norm: float,
    epoch_train_time_sec: float,
    correction_attempts: int,
    correction_accepts: int,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "train_loss": float(train_metrics["loss"]),
        "test_loss": float(test_metrics["loss"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "train_perplexity": float(train_metrics["perplexity"]),
        "test_perplexity": float(test_metrics["perplexity"]),
        "accuracy_generalization_gap": float(
            train_metrics["accuracy"] - test_metrics["accuracy"]
        ),
        "loss_generalization_gap": float(
            test_metrics["loss"] - train_metrics["loss"]
        ),
        "learning_rate": float(learning_rate),
        "parameter_l2_norm": float(parameter_norm),
        "epoch_train_time_sec": float(epoch_train_time_sec),
        "correction_attempts": int(correction_attempts),
        "correction_accepts": int(correction_accepts),
    }


def flatten_correction_record(
    record: CorrectionRecord,
    *,
    seed: int,
    epoch: int,
    draw_cycle_start: Optional[int],
    draw_cycle_end: Optional[int],
) -> list[dict[str, Any]]:
    if not record.attempted:
        return []
    common = {
        "arm": "trace_wall",
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(record.global_step),
        "applied": bool(record.applied),
        "reason": record.reason,
        "probe_examples": int(record.probe_examples),
        "probe_cycle_start": draw_cycle_start,
        "probe_cycle_end": draw_cycle_end,
        "probe_loss_before": record.probe_loss_before,
        "probe_loss_after": record.probe_loss_after,
        "probe_loss_change": record.probe_loss_change,
        "directional_derivative": record.directional_derivative,
        "line_search_scale": record.line_search_scale,
        "line_search_iterations": record.line_search_iterations,
    }
    return [{**common, **layer.to_dict()} for layer in record.layers]


def student_t_summary(
    frame: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    metrics: Sequence[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    long = frame.melt(
        id_vars=list(id_columns),
        value_vars=[metric for metric in metrics if metric in frame.columns],
        var_name="metric",
        value_name="value",
    )
    long = long[np.isfinite(pd.to_numeric(long["value"], errors="coerce"))]
    rows: list[dict[str, Any]] = []
    group_columns = [*id_columns, "metric"]
    for keys, group in long.groupby(group_columns, sort=True, dropna=False):
        values = group["value"].to_numpy(dtype=float)
        n = int(values.size)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem = float(std / math.sqrt(n)) if n > 0 else float("nan")
        critical = (
            float(student_t.ppf(0.5 + confidence / 2.0, df=n - 1))
            if n > 1
            else 0.0
        )
        half = float(critical * sem)
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci_half_width": half,
                "ci_low": mean - half,
                "ci_high": mean + half,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def correction_summary(corrections: pd.DataFrame) -> pd.DataFrame:
    if corrections.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (layer, seed), group in corrections.groupby(["parameter_name", "seed"]):
        attempts = int(group["global_step"].nunique())
        accepted = int(group.loc[group["applied"], "global_step"].nunique())
        rows.append(
            {
                "parameter_name": layer,
                "seed": int(seed),
                "attempts": attempts,
                "accepted": accepted,
                "acceptance_rate": accepted / attempts if attempts else float("nan"),
                "mean_probe_loss_change": float(group["probe_loss_change"].mean()),
                "mean_line_search_scale": float(group["line_search_scale"].mean()),
                "mean_ecs_rank": float(group["ecs_rank"].mean()),
                "mean_correction_to_base_step_ratio": float(
                    group["correction_to_base_step_ratio"].mean()
                ),
                "max_projection_identity_error": float(
                    group["projection_identity_error"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_pairing(
    performance: pd.DataFrame,
    spectral: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    expected_seeds = set(config.seeds)
    expected_epochs = set(range(config.epochs + 1))
    expected_arms = {"baseline", "trace_wall"}
    if set(performance["seed"].astype(int)) != expected_seeds:
        raise RuntimeError("performance seed grid is incomplete")
    if set(performance["epoch"].astype(int)) != expected_epochs:
        raise RuntimeError("performance epoch grid is incomplete")
    if set(performance["arm"]) != expected_arms:
        raise RuntimeError("paired arms are incomplete")
    for seed in expected_seeds:
        for arm in expected_arms:
            observed = set(
                performance.loc[
                    (performance["seed"] == seed) & (performance["arm"] == arm),
                    "epoch",
                ].astype(int)
            )
            if observed != expected_epochs:
                raise RuntimeError((seed, arm, sorted(observed)))
    expected_layers = {
        name.removesuffix(".weight").split(".")[-1]
        for name in config.trace_wall.parameter_names
    }
    for seed in expected_seeds:
        for arm in expected_arms:
            for epoch in expected_epochs:
                observed = set(
                    spectral.loc[
                        (spectral["seed"] == seed)
                        & (spectral["arm"] == arm)
                        & (spectral["epoch"] == epoch),
                        "layer",
                    ]
                )
                if observed != expected_layers:
                    raise RuntimeError((seed, arm, epoch, observed))

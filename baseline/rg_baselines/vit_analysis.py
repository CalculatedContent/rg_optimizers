"""Statistically correct analysis for the CIFAR-10 small-ViT baselines.

The unit of replication is a complete training run. Layers and WeightWatcher
fits are repeated measurements inside a run; they are never treated as
independent replicates when constructing uncertainty intervals.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .statistics import student_t_critical_95
from .vit_cifar10 import DEFAULT_VIT_SEEDS

OPTIMIZER_ORDER = ("sgd_momentum", "adamw", "muon")
OPTIMIZER_LABELS = {
    "sgd_momentum": "SGD + Nesterov",
    "adamw": "AdamW",
    "muon": "Muon + auxiliary AdamW",
}
OPTIMIZER_COLORS = {
    "sgd_momentum": "#0072B2",
    "adamw": "#D55E00",
    "muon": "#009E73",
}
MATRIX_TYPE_COLORS = {
    "W_Q": "#0072B2",
    "W_K": "#E69F00",
    "W_V": "#009E73",
    "W_O": "#D55E00",
    "W_MLP_IN": "#CC79A7",
    "W_MLP_OUT": "#56B4E9",
}


def _mean_ci95(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    mean = float(array.mean())
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    std = float(array.std(ddof=1))
    sem = std / math.sqrt(n)
    half = student_t_critical_95(n) * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_half_width": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def summarize_across_seeds(
    frame: pd.DataFrame,
    *,
    value: str,
    group_columns: Sequence[str],
    expected_seeds: Sequence[int] = DEFAULT_VIT_SEEDS,
) -> pd.DataFrame:
    """Summarize exactly one observation per seed in every requested group."""

    required = {*group_columns, "seed", value}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")
    expected = tuple(int(seed) for seed in expected_seeds)
    if len(set(expected)) != len(expected):
        raise ValueError("expected_seeds must be unique")

    subset = frame[[*group_columns, "seed", value]].copy()
    subset[value] = pd.to_numeric(subset[value], errors="coerce")
    rows: list[dict[str, object]] = []
    for group_key, group in subset.groupby(list(group_columns), sort=True):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        duplicates = group.groupby("seed").size()
        if (duplicates > 1).any():
            raise RuntimeError(
                "pseudo-replication detected: a group contains more than one "
                "observation for at least one seed"
            )
        observed = tuple(sorted(group["seed"].astype(int).unique()))
        if observed != tuple(sorted(expected)):
            raise RuntimeError(
                f"incomplete seed set for group {key_tuple}: {observed} != {expected}"
            )
        row = dict(zip(group_columns, key_tuple, strict=True))
        row.update(_mean_ci95(group[value]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_performance(
    history: pd.DataFrame,
    metric: str,
    *,
    expected_seeds: Sequence[int] = DEFAULT_VIT_SEEDS,
) -> pd.DataFrame:
    return summarize_across_seeds(
        history,
        value=metric,
        group_columns=("optimizer", "epoch"),
        expected_seeds=expected_seeds,
    )


def summarize_layer_metric(
    spectral: pd.DataFrame,
    metric: str,
    *,
    expected_seeds: Sequence[int] = DEFAULT_VIT_SEEDS,
) -> pd.DataFrame:
    """Return one CI per physical matrix, never one CI per matrix type pool."""

    return summarize_across_seeds(
        spectral,
        value=metric,
        group_columns=(
            "optimizer",
            "matrix_name",
            "matrix_type",
            "block",
            "epoch",
        ),
        expected_seeds=expected_seeds,
    )


def validation_selected_rows(history: pd.DataFrame) -> pd.DataFrame:
    """Select one row per optimizer/seed using validation loss only."""

    required = {"optimizer", "seed", "epoch", "val_loss"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"missing columns for checkpoint selection: {sorted(missing)}")
    rows = []
    for (_, _), run in history.groupby(["optimizer", "seed"], sort=True):
        candidate = run.dropna(subset=["val_loss"]).sort_values(
            ["val_loss", "epoch"], ascending=[True, True]
        )
        if candidate.empty:
            raise RuntimeError("run has no finite validation loss")
        row = candidate.iloc[0].copy()
        row["checkpoint"] = "validation_selected"
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def final_rows(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _), run in history.groupby(["optimizer", "seed"], sort=True):
        row = run.sort_values("epoch").iloc[-1].copy()
        row["checkpoint"] = "final"
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def terminal_summary(
    history: pd.DataFrame,
    *,
    metrics: Sequence[str] = (
        "test_loss",
        "test_accuracy",
        "val_loss",
        "val_accuracy",
    ),
    expected_seeds: Sequence[int] = DEFAULT_VIT_SEEDS,
) -> pd.DataFrame:
    selected = pd.concat(
        [final_rows(history), validation_selected_rows(history)],
        ignore_index=True,
    )
    rows = []
    for metric in metrics:
        summary = summarize_across_seeds(
            selected,
            value=metric,
            group_columns=("optimizer", "checkpoint"),
            expected_seeds=expected_seeds,
        )
        summary.insert(2, "metric", metric)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def load_vit_results(
    run_root: str | Path,
    *,
    optimizers: Sequence[str] = OPTIMIZER_ORDER,
    seeds: Sequence[int] = DEFAULT_VIT_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    histories: list[pd.DataFrame] = []
    spectral_frames: list[pd.DataFrame] = []
    root = Path(run_root)
    for optimizer in optimizers:
        for seed in seeds:
            run_dir = root / optimizer / f"seed_{int(seed)}"
            history_path = run_dir / "history.csv"
            spectral_path = run_dir / "weightwatcher_by_epoch_layer.csv"
            completion_path = run_dir / "run_complete.json"
            if not completion_path.is_file():
                raise FileNotFoundError(f"incomplete ViT run: {run_dir}")
            history = pd.read_csv(history_path)
            history["optimizer"] = optimizer
            history["seed"] = int(seed)
            histories.append(history)
            layer = pd.read_csv(spectral_path)
            layer["optimizer"] = optimizer
            layer["seed"] = int(seed)
            spectral_frames.append(layer)
    return (
        pd.concat(histories, ignore_index=True, sort=False),
        pd.concat(spectral_frames, ignore_index=True, sort=False),
    )


def plot_performance_metric(
    history: pd.DataFrame,
    metric: str,
    *,
    output: str | Path | None = None,
    expected_seeds: Sequence[int] = DEFAULT_VIT_SEEDS,
):
    summary = summarize_performance(
        history, metric, expected_seeds=expected_seeds
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for optimizer in OPTIMIZER_ORDER:
        raw = history[history["optimizer"].eq(optimizer)]
        for _, seed_frame in raw.groupby("seed"):
            axis.plot(
                seed_frame["epoch"],
                seed_frame[metric],
                color=OPTIMIZER_COLORS[optimizer],
                alpha=0.18,
                linewidth=0.8,
            )
        curve = summary[summary["optimizer"].eq(optimizer)]
        if curve.empty:
            continue
        axis.plot(
            curve["epoch"],
            curve["mean"],
            color=OPTIMIZER_COLORS[optimizer],
            linewidth=2.0,
            label=OPTIMIZER_LABELS[optimizer],
        )
        axis.fill_between(
            curve["epoch"],
            curve["ci95_low"],
            curve["ci95_high"],
            color=OPTIMIZER_COLORS[optimizer],
            alpha=0.15,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(f"{metric}: mean and 95% Student-t CI across runs")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=170, bbox_inches="tight")
    return figure, axis, summary


def plot_layer_metric(
    spectral: pd.DataFrame,
    *,
    optimizer: str,
    metric: str,
    output_dir: str | Path | None = None,
    expected_seeds: Sequence[int] = DEFAULT_VIT_SEEDS,
) -> list[Path]:
    """Plot one figure per block so every band has exactly three run replicates."""

    summary = summarize_layer_metric(
        spectral, metric, expected_seeds=expected_seeds
    )
    subset = summary[summary["optimizer"].eq(optimizer)]
    paths: list[Path] = []
    for block, block_frame in subset.groupby("block", sort=True):
        figure, axis = plt.subplots(figsize=(10, 5.5))
        for matrix_type, curve in block_frame.groupby("matrix_type", sort=False):
            curve = curve.sort_values("epoch")
            color = MATRIX_TYPE_COLORS.get(matrix_type)
            axis.plot(
                curve["epoch"],
                curve["mean"],
                linewidth=2.0,
                label=matrix_type,
                color=color,
            )
            axis.fill_between(
                curve["epoch"],
                curve["ci95_low"],
                curve["ci95_high"],
                alpha=0.14,
                color=color,
            )
        if metric == "alpha":
            axis.axhline(2.0, linestyle="--", linewidth=1.0, color="black")
        if metric == "ERG_gap":
            axis.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric)
        axis.set_title(
            f"{OPTIMIZER_LABELS[optimizer]} block {int(block)}: {metric}"
        )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncol=2)
        figure.tight_layout()
        if output_dir is not None:
            path = (
                Path(output_dir)
                / f"{optimizer}_block_{int(block):02d}_{metric}.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=170, bbox_inches="tight")
            paths.append(path)
    return paths

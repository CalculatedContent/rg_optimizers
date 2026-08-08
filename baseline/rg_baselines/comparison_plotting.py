"""Color-consistent plots for persisted MNIST optimizer comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PERFORMANCE_PLOTS: tuple[tuple[str, str, str, float | None, bool], ...] = (
    ("test_accuracy", "Test accuracy (%)", "01_test_accuracy_95ci.png", None, True),
    (
        "validation_accuracy",
        "Validation accuracy (%)",
        "02_validation_accuracy_95ci.png",
        None,
        True,
    ),
    ("train_accuracy", "Train accuracy (%)", "03_train_accuracy_95ci.png", None, True),
    ("test_loss", "Test cross-entropy", "04_test_loss_95ci.png", None, False),
    (
        "validation_loss",
        "Validation cross-entropy",
        "05_validation_loss_95ci.png",
        None,
        False,
    ),
    ("train_loss", "Train cross-entropy", "06_train_loss_95ci.png", None, False),
    (
        "test_perplexity",
        "Test classification perplexity, exp(CE)",
        "07_test_perplexity_95ci.png",
        None,
        False,
    ),
    (
        "validation_perplexity",
        "Validation classification perplexity, exp(CE)",
        "08_validation_perplexity_95ci.png",
        None,
        False,
    ),
    (
        "test_accuracy_gap",
        "Train - test accuracy (percentage points)",
        "09_test_accuracy_gap_95ci.png",
        0.0,
        True,
    ),
    (
        "validation_accuracy_gap",
        "Train - validation accuracy (percentage points)",
        "10_validation_accuracy_gap_95ci.png",
        0.0,
        True,
    ),
    (
        "test_loss_gap",
        "Test - train cross-entropy",
        "11_test_loss_gap_95ci.png",
        0.0,
        False,
    ),
    (
        "validation_loss_gap",
        "Validation - train cross-entropy",
        "12_validation_loss_gap_95ci.png",
        0.0,
        False,
    ),
    (
        "primary_lr",
        "Primary learning rate",
        "13_primary_learning_rate.png",
        None,
        False,
    ),
)

SPECTRAL_PLOTS: tuple[tuple[str, str, str, float | None], ...] = (
    ("alpha", "WeightWatcher alpha", "20_weightwatcher_alpha_95ci.png", 2.0),
    ("num_traps", "Randomized-MP correlation traps", "21_num_traps_95ci.png", 0.0),
    (
        "detX_num",
        "WeightWatcher detX retained count",
        "22_weightwatcher_detx_num_95ci.png",
        None,
    ),
    (
        "num_pl_spikes",
        "WeightWatcher power-law spike count",
        "23_weightwatcher_num_pl_spikes_95ci.png",
        None,
    ),
    ("ERG_gap", "WeightWatcher ERG gap", "24_erg_gap_95ci.png", 0.0),
    ("m_midpoint", "Midpoint retained rank", "25_midpoint_rank_95ci.png", None),
    (
        "trace_log_midpoint_per_eval",
        "Midpoint trace-log per retained eigenvalue",
        "26_trace_log_midpoint_per_eval_95ci.png",
        0.0,
    ),
    ("stable_rank", "Stable rank", "27_stable_rank_95ci.png", None),
)


def _finish(fig, path: Path, *, show: bool) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_performance_metric(
    performance: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    path: Path,
    seeds: Sequence[int],
    optimizer_order: Sequence[str],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    horizontal_reference: float | None = None,
    percent: bool = False,
    show: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for optimizer in optimizer_order:
        color = colors[optimizer]
        raw = performance.loc[performance["optimizer"].eq(optimizer)]
        for seed in seeds:
            rows = raw.loc[
                raw["seed"].astype(int).eq(int(seed))
            ].sort_values("epoch")
            values = rows[metric].astype(float).to_numpy()
            if percent:
                values = 100.0 * values
            ax.plot(
                rows["epoch"],
                values,
                color=color,
                linewidth=0.9,
                alpha=0.18,
            )

        rows = summary.loc[
            summary["optimizer"].eq(optimizer)
            & summary["metric"].eq(metric)
        ].sort_values("epoch")
        if rows.empty:
            continue
        x = rows["epoch"].astype(float).to_numpy()
        mean = rows["mean"].astype(float).to_numpy()
        low = rows["ci_low"].astype(float).to_numpy()
        high = rows["ci_high"].astype(float).to_numpy()
        half = rows["ci_half_width"].astype(float).to_numpy()
        if percent:
            mean, low, high, half = [
                100.0 * values for values in (mean, low, high, half)
            ]
        ax.plot(x, mean, color=color, linewidth=2.6, label=labels[optimizer])
        ax.fill_between(x, low, high, color=color, alpha=0.14)
        stride = max(1, len(x) // 15)
        ax.errorbar(
            x[::stride],
            mean[::stride],
            yerr=half[::stride],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.5,
            alpha=0.8,
        )

    if horizontal_reference is not None:
        value = 100.0 * horizontal_reference if percent else horizontal_reference
        ax.axhline(
            value,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
        )
    ax.set(
        xlabel="Epoch",
        ylabel=ylabel,
        title=f"{ylabel} across optimizer baselines",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return _finish(fig, path, show=show)


def plot_spectral_metric(
    spectral: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    path: Path,
    seeds: Sequence[int],
    optimizer_order: Sequence[str],
    layer_order: Sequence[str],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    horizontal_reference: float | None = None,
    show: bool = True,
) -> Path:
    fig, axes = plt.subplots(
        1, len(layer_order), figsize=(16.0, 4.8), sharex=True
    )
    for ax, layer in zip(axes, layer_order, strict=True):
        for optimizer in optimizer_order:
            color = colors[optimizer]
            raw = spectral.loc[
                spectral["optimizer"].eq(optimizer)
                & spectral["layer"].astype(str).eq(str(layer))
            ]
            for seed in seeds:
                rows = raw.loc[
                    raw["seed"].astype(int).eq(int(seed))
                ].sort_values("epoch")
                ax.plot(
                    rows["epoch"],
                    rows[metric],
                    color=color,
                    linewidth=0.8,
                    alpha=0.15,
                )

            rows = summary.loc[
                summary["optimizer"].eq(optimizer)
                & summary["layer"].astype(str).eq(str(layer))
                & summary["metric"].eq(metric)
            ].sort_values("epoch")
            if rows.empty:
                continue
            x = rows["epoch"].astype(float).to_numpy()
            mean = rows["mean"].astype(float).to_numpy()
            low = rows["ci_low"].astype(float).to_numpy()
            high = rows["ci_high"].astype(float).to_numpy()
            ax.plot(x, mean, color=color, linewidth=2.2, label=labels[optimizer])
            ax.fill_between(x, low, high, color=color, alpha=0.12)

        if horizontal_reference is not None:
            ax.axhline(
                horizontal_reference,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.6,
            )
        ax.set(title=str(layer).upper(), xlabel="Epoch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(f"{ylabel} across optimizer baselines", y=1.02)
    return _finish(fig, path, show=show)


def plot_all_comparisons(
    *,
    performance: pd.DataFrame,
    performance_summary: pd.DataFrame,
    spectral: pd.DataFrame,
    spectral_summary: pd.DataFrame,
    output_dir: str | Path,
    seeds: Sequence[int],
    optimizer_order: Sequence[str],
    layer_order: Sequence[str],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    show: bool = True,
) -> tuple[Path, ...]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for metric, ylabel, filename, reference, percent in PERFORMANCE_PLOTS:
        if metric not in performance.columns:
            continue
        if not performance[metric].notna().any():
            continue
        paths.append(
            plot_performance_metric(
                performance,
                performance_summary,
                metric=metric,
                ylabel=ylabel,
                path=output / filename,
                seeds=seeds,
                optimizer_order=optimizer_order,
                labels=labels,
                colors=colors,
                horizontal_reference=reference,
                percent=percent,
                show=show,
            )
        )
    for metric, ylabel, filename, reference in SPECTRAL_PLOTS:
        if metric not in spectral.columns:
            continue
        paths.append(
            plot_spectral_metric(
                spectral,
                spectral_summary,
                metric=metric,
                ylabel=ylabel,
                path=output / filename,
                seeds=seeds,
                optimizer_order=optimizer_order,
                layer_order=layer_order,
                labels=labels,
                colors=colors,
                horizontal_reference=reference,
                show=show,
            )
        )
    return tuple(paths)

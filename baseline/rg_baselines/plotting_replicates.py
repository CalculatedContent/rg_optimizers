"""Replicate-aware plots with consistent colors and 95% Student-t error bars."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting_style import (
    CI_BAND_ALPHA,
    ERROR_CAP_SIZE,
    LAYER_COLORS,
    LAYER_MARKERS,
    MEAN_LINE_WIDTH,
    METRIC_COLORS,
    PERFORMANCE_COLORS,
    SEED_LINE_WIDTH,
    SEED_TRACE_ALPHA,
)
from .replicates import BaselineReplicateResult


def _save(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")


def _valid_spectral(result: BaselineReplicateResult) -> pd.DataFrame:
    return result.spectral_metrics[
        result.spectral_metrics["status"].astype(str).eq("ok")
    ].copy()


def _summary_for(
    summary: pd.DataFrame,
    *,
    metric: str,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    selected = summary[summary["metric"].astype(str).eq(metric)].copy()
    if layer is not None:
        selected = selected[selected["layer"].astype(str).eq(layer)]
    return selected.sort_values("epoch")


def _plot_seed_traces_and_ci(
    ax: plt.Axes,
    *,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    color: str,
    label: str,
    marker: str = "o",
    show_seed_traces: bool = True,
) -> None:
    """Plot individual runs faintly and the mean with a Student-t CI band."""

    raw_metric = raw[["seed", "epoch", metric]].copy()
    raw_metric[metric] = pd.to_numeric(raw_metric[metric], errors="coerce")
    raw_metric = raw_metric.dropna(subset=[metric]).sort_values(["seed", "epoch"])
    if show_seed_traces:
        for _, seed_frame in raw_metric.groupby("seed"):
            ax.plot(
                seed_frame["epoch"],
                seed_frame[metric],
                color=color,
                alpha=SEED_TRACE_ALPHA,
                linewidth=SEED_LINE_WIDTH,
                zorder=1,
            )

    aggregate = _summary_for(summary, metric=metric)
    if aggregate.empty:
        raise RuntimeError(f"No replicate summary rows were found for {metric!r}.")
    x = aggregate["epoch"].to_numpy(dtype=float)
    mean = aggregate["mean"].to_numpy(dtype=float)
    low = aggregate["ci_low"].to_numpy(dtype=float)
    high = aggregate["ci_high"].to_numpy(dtype=float)
    half = aggregate["ci_half_width"].to_numpy(dtype=float)

    finite_ci = np.isfinite(low) & np.isfinite(high)
    if finite_ci.any():
        ax.fill_between(
            x[finite_ci],
            low[finite_ci],
            high[finite_ci],
            color=color,
            alpha=CI_BAND_ALPHA,
            linewidth=0.0,
            zorder=2,
        )
    yerr = np.where(np.isfinite(half), half, 0.0)
    ax.errorbar(
        x,
        mean,
        yerr=yerr,
        color=color,
        marker=marker,
        markersize=4.5,
        linewidth=MEAN_LINE_WIDTH,
        capsize=ERROR_CAP_SIZE,
        elinewidth=1.0,
        label=label,
        zorder=3,
    )


def _plot_layer_metric(
    ax: plt.Axes,
    *,
    result: BaselineReplicateResult,
    metric: str,
    ylabel: str,
    title: str,
    reference: Optional[float] = None,
    log_scale: bool = False,
) -> None:
    raw = _valid_spectral(result)
    for layer in ("fc1", "fc2", "fc3"):
        layer_raw = raw[raw["layer"].astype(str).eq(layer)]
        layer_summary = result.spectral_summary[
            result.spectral_summary["layer"].astype(str).eq(layer)
        ]
        _plot_seed_traces_and_ci(
            ax,
            raw=layer_raw,
            summary=layer_summary,
            metric=metric,
            color=LAYER_COLORS[layer],
            marker=LAYER_MARKERS[layer],
            label=layer.upper(),
        )
    if reference is not None:
        ax.axhline(reference, color="#333333", linestyle="--", linewidth=1.4)
    if log_scale:
        ax.set_yscale("log")
    ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)


def _ci_caption(result: BaselineReplicateResult) -> str:
    return (
        f"mean with two-sided 95% Student-t CI across "
        f"{result.replicate_count} independent seeds"
    )


def plot_0_loss_and_accuracy_with_ci(
    result: BaselineReplicateResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Required plot 0: train/test loss and train/test accuracy per epoch."""

    raw = result.performance
    summary = result.performance_summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    _plot_seed_traces_and_ci(
        axes[0],
        raw=raw,
        summary=summary,
        metric="train_loss",
        color=PERFORMANCE_COLORS["train"],
        marker="o",
        label="Train loss",
    )
    _plot_seed_traces_and_ci(
        axes[0],
        raw=raw,
        summary=summary,
        metric="test_loss",
        color=PERFORMANCE_COLORS["test"],
        marker="s",
        label="Test loss",
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Cross-entropy loss",
        title="Full train and test loss",
    )

    _plot_seed_traces_and_ci(
        axes[1],
        raw=raw,
        summary=summary,
        metric="train_accuracy",
        color=PERFORMANCE_COLORS["train"],
        marker="o",
        label="Train accuracy",
    )
    _plot_seed_traces_and_ci(
        axes[1],
        raw=raw,
        summary=summary,
        metric="test_accuracy",
        color=PERFORMANCE_COLORS["test"],
        marker="s",
        label="Test accuracy",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Accuracy",
        title="Full train and test accuracy",
    )
    axes[1].set_ylim(0.0, 1.01)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"{result.optimizer_label} — {_ci_caption(result)}")
    fig.tight_layout()
    _save(
        fig,
        Path(output_dir) / "0_train_test_loss_accuracy_95ci.png"
        if output_dir
        else None,
    )
    if show:
        plt.show()
    return fig


def plot_0b_test_accuracy_focus_with_ci(
    result: BaselineReplicateResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Dedicated test-accuracy plot so the primary outcome is never hidden."""

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    _plot_seed_traces_and_ci(
        ax,
        raw=result.performance,
        summary=result.performance_summary,
        metric="test_accuracy",
        color=PERFORMANCE_COLORS["test"],
        marker="s",
        label="Test accuracy",
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Test accuracy",
        title=f"{result.optimizer_label}: test accuracy per epoch",
    )
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.25)
    ax.legend(title=_ci_caption(result))
    fig.tight_layout()
    _save(
        fig,
        Path(output_dir) / "0b_test_accuracy_95ci.png"
        if output_dir
        else None,
    )
    if show:
        plt.show()
    return fig


def plot_1_layerwise_alpha_with_ci(
    result: BaselineReplicateResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Required plot 1: layerwise WeightWatcher alpha at every epoch."""

    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_layer_metric(
        ax,
        result=result,
        metric="alpha",
        ylabel="WeightWatcher alpha",
        title=f"Layerwise WeightWatcher alpha — {result.optimizer_label}",
        reference=2.0,
    )
    ax.text(
        0.01,
        0.02,
        _ci_caption(result),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    fig.tight_layout()
    _save(
        fig,
        Path(output_dir) / "1_layerwise_weightwatcher_alpha_95ci.png"
        if output_dir
        else None,
    )
    if show:
        plt.show()
    return fig


def plot_2_original_erg_boundaries_with_ci(
    result: BaselineReplicateResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Required plot 2: original full-M WeightWatcher boundaries and gap."""

    items = (
        ("detX_num", "WeightWatcher detX retained count", "detX_num", None),
        ("num_pl_spikes", "WeightWatcher PL retained count", "num_pl_spikes", None),
        ("ERG_gap", "Full-M ERG gap", "detX_num - num_pl_spikes", 0.0),
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for ax, (metric, title, ylabel, reference) in zip(axes, items):
        _plot_layer_metric(
            ax,
            result=result,
            metric=metric,
            ylabel=ylabel,
            title=title,
            reference=reference,
        )
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"{result.optimizer_label} — {_ci_caption(result)}")
    fig.tight_layout()
    _save(
        fig,
        Path(output_dir) / "2_original_weightwatcher_boundaries_95ci.png"
        if output_dir
        else None,
    )
    if show:
        plt.show()
    return fig


def plot_3_midpoint_and_trace_log_with_ci(
    result: BaselineReplicateResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Required plot 3: midpoint rank plus total and per-mode trace-log."""

    items = (
        (
            "m_midpoint",
            "Original midpoint retained rank",
            "m_midpoint",
            None,
        ),
        (
            "trace_log_midpoint_per_eval",
            "Midpoint trace-log per retained eigenvalue",
            "mean log rescaled eigenvalue",
            0.0,
        ),
        (
            "trace_log_midpoint_total",
            "Midpoint trace-log total",
            "sum log rescaled eigenvalue",
            0.0,
        ),
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for ax, (metric, title, ylabel, reference) in zip(axes, items):
        _plot_layer_metric(
            ax,
            result=result,
            metric=metric,
            ylabel=ylabel,
            title=title,
            reference=reference,
        )
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"{result.optimizer_label} — {_ci_caption(result)}")
    fig.tight_layout()
    _save(
        fig,
        Path(output_dir) / "3_midpoint_rank_and_trace_log_95ci.png"
        if output_dir
        else None,
    )
    if show:
        plt.show()
    return fig


def plot_4_effective_rank_and_energy_with_ci(
    result: BaselineReplicateResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Additional effective-rank and retained-energy diagnostics."""

    items = (
        ("stable_rank", "Stable rank", None),
        ("participation_ratio", "Participation-ratio effective rank", None),
        ("entropy_effective_rank", "Entropy effective rank", None),
        ("midpoint_energy_fraction", "Midpoint retained-energy fraction", None),
        ("boundary_overlap_ratio", "PL/detX boundary overlap", None),
        ("top1_energy_fraction", "Largest-mode energy fraction", None),
    )
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    for ax, (metric, title, reference) in zip(axes.ravel(), items):
        _plot_layer_metric(
            ax,
            result=result,
            metric=metric,
            ylabel=metric,
            title=title,
            reference=reference,
        )
    fig.suptitle(f"Additional spectral diagnostics — {result.optimizer_label}")
    fig.tight_layout()
    _save(
        fig,
        Path(output_dir) / "4_effective_rank_and_energy_95ci.png"
        if output_dir
        else None,
    )
    if show:
        plt.show()
    return fig


def plot_5_optimizer_diagnostics_with_ci(
    result: BaselineReplicateResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Gradient, parameter-norm, and timing diagnostics across seeds."""

    raw = result.performance
    summary = result.performance_summary
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    for metric, label, marker in (
        ("mean_gradient_norm_before_clip", "Mean gradient norm", "o"),
        ("max_gradient_norm_before_clip", "Maximum gradient norm", "s"),
    ):
        _plot_seed_traces_and_ci(
            axes[0],
            raw=raw,
            summary=summary,
            metric=metric,
            color=METRIC_COLORS[metric],
            marker=marker,
            label=label,
        )
    axes[0].set(ylabel="Gradient norm", title="Gradient norm before clipping")

    _plot_seed_traces_and_ci(
        axes[1],
        raw=raw,
        summary=summary,
        metric="parameter_l2_norm",
        color=METRIC_COLORS["parameter_l2_norm"],
        marker="o",
        label="Parameter L2 norm",
    )
    axes[1].set(ylabel="Parameter L2 norm", title="Whole-model parameter norm")

    for metric, label, marker in (
        ("train_time_sec", "Training", "o"),
        ("evaluation_time_sec", "Evaluation", "s"),
        ("weightwatcher_time_sec", "WeightWatcher", "^"),
    ):
        _plot_seed_traces_and_ci(
            axes[2],
            raw=raw,
            summary=summary,
            metric=metric,
            color=METRIC_COLORS[metric],
            marker=marker,
            label=label,
        )
    axes[2].set(xlabel="Epoch", ylabel="Seconds", title="Epoch timing")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"{result.optimizer_label} — {_ci_caption(result)}")
    fig.tight_layout()
    _save(
        fig,
        Path(output_dir) / "5_optimizer_diagnostics_95ci.png"
        if output_dir
        else None,
    )
    if show:
        plt.show()
    return fig


def plot_6_spectral_scale_and_conditioning_with_ci(
    result: BaselineReplicateResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Normalization, scale, and conditioning audits for every layer."""

    items = (
        ("normalized_lambda_max", "Largest rescaled eigenvalue", False),
        (
            "normalized_lambda_midpoint_cut",
            "Rescaled eigenvalue at midpoint boundary",
            False,
        ),
        ("geometric_mean_midpoint", "Midpoint geometric mean", False),
        ("eigenvalue_condition_number", "ESD condition number", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for ax, (metric, title, log_scale) in zip(axes.ravel(), items):
        _plot_layer_metric(
            ax,
            result=result,
            metric=metric,
            ylabel=metric,
            title=title,
            log_scale=log_scale,
        )
    fig.suptitle(f"Spectral scale and conditioning — {result.optimizer_label}")
    fig.tight_layout()
    _save(
        fig,
        Path(output_dir) / "6_spectral_scale_and_conditioning_95ci.png"
        if output_dir
        else None,
    )
    if show:
        plt.show()
    return fig


def plot_all_replicates(
    result: BaselineReplicateResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Create every required and supplemental aggregate plot."""

    functions = (
        plot_0_loss_and_accuracy_with_ci,
        plot_0b_test_accuracy_focus_with_ci,
        plot_1_layerwise_alpha_with_ci,
        plot_2_original_erg_boundaries_with_ci,
        plot_3_midpoint_and_trace_log_with_ci,
        plot_4_effective_rank_and_energy_with_ci,
        plot_5_optimizer_diagnostics_with_ci,
        plot_6_spectral_scale_and_conditioning_with_ci,
    )
    return [
        function(result, output_dir=output_dir, show=show)
        for function in functions
    ]

"""Correlation-trap plots for single and replicated baseline runs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from .plotting_replicates import _ci_caption, _plot_layer_metric, _save
from .results import BaselineResult


def plot_7_layerwise_num_traps_with_ci(
    result,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Plot randomized-MP correlation-trap counts for FC1, FC2, and FC3."""

    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_layer_metric(
        ax,
        result=result,
        metric="num_traps",
        ylabel="Correlation traps (num_traps)",
        title=(
            "Layerwise WeightWatcher correlation traps "
            f"— {result.optimizer_label}"
        ),
        reference=0.0,
    )
    ax.set_ylim(bottom=-0.05)
    ax.text(
        0.01,
        0.98,
        "WeightWatcher analyze(randomize=True): randomized MP outlier count",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    ax.text(0.01, 0.02, _ci_caption(result), transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    _save(
        fig,
        Path(output_dir) / "7_layerwise_weightwatcher_num_traps_95ci.png"
        if output_dir
        else None,
    )
    if show:
        plt.show()
    return fig


def plot_layerwise_num_traps(
    result: BaselineResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
):
    """Legacy single-run correlation-trap plot."""

    frame = result.spectral_metrics[
        result.spectral_metrics.status.astype(str).eq("ok")
    ].sort_values(["layer", "epoch"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for layer, group in frame.groupby("layer"):
        ax.plot(group.epoch, group.num_traps, marker="o", label=str(layer).upper())
    ax.axhline(0, linestyle="--")
    ax.set(
        xlabel="Epoch",
        ylabel="Correlation traps (num_traps)",
        title=f"Layerwise WeightWatcher correlation traps — {result.config.optimizer_label}",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output_dir:
        destination = Path(output_dir) / "6_layerwise_weightwatcher_num_traps.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig

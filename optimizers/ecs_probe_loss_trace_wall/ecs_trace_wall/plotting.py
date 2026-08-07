"""Fixed-color paired plots for the ECS probe-loss TraceWall experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import PairedExperimentResult

COLORS = {
    "baseline": "#0072B2",  # blue
    "trace_wall": "#009E73",  # bluish green
    "probe_before": "#D55E00",  # vermillion
    "probe_after": "#009E73",
}
LINESTYLES = {"train": "--", "test": "-"}


def _save(fig: plt.Figure, output_dir: Optional[Path], filename: str) -> None:
    fig.tight_layout()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, dpi=170, bbox_inches="tight")


def _summary_slice(
    summary: pd.DataFrame,
    *,
    metric: str,
    arm: str,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    mask = summary["metric"].eq(metric) & summary["arm"].eq(arm)
    if layer is not None:
        mask &= summary["layer"].eq(layer)
    return summary.loc[mask].sort_values("epoch")


def plot_loss_and_accuracy(
    result: PairedExperimentResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
) -> list[plt.Figure]:
    destination = Path(output_dir) if output_dir is not None else None
    figures: list[plt.Figure] = []
    for quantity in ("loss", "accuracy"):
        fig, axis = plt.subplots(figsize=(9.5, 5.5))
        for arm in ("baseline", "trace_wall"):
            for split in ("train", "test"):
                frame = _summary_slice(
                    result.performance_summary,
                    metric=f"{split}_{quantity}",
                    arm=arm,
                )
                axis.plot(
                    frame["epoch"],
                    frame["mean"],
                    color=COLORS[arm],
                    linestyle=LINESTYLES[split],
                    linewidth=2.2,
                    label=f"{arm.replace('_', ' ').title()} — {split}",
                )
                axis.fill_between(
                    frame["epoch"],
                    frame["ci_low"],
                    frame["ci_high"],
                    color=COLORS[arm],
                    alpha=0.14,
                )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy" if quantity == "loss" else "Accuracy")
        axis.set_title(
            f"{result.config.optimizer.name.replace('_', ' ').title()}: "
            f"baseline versus ECS probe-loss TraceWall"
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
        _save(fig, destination, f"01_{quantity}_95ci.png")
        figures.append(fig)

    if show:
        plt.show()
    return figures


def plot_ecs_ranks(
    result: PairedExperimentResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    destination = Path(output_dir) if output_dir is not None else None
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    layer_styles = {"fc1": "-", "fc2": "--", "fc3": ":"}
    for arm in ("baseline", "trace_wall"):
        for layer in sorted(result.spectral["layer"].unique()):
            frame = _summary_slice(
                result.spectral_summary,
                metric="ecs_rank",
                arm=arm,
                layer=layer,
            )
            axis.plot(
                frame["epoch"],
                frame["mean"],
                color=COLORS[arm],
                linestyle=layer_styles.get(layer, "-"),
                linewidth=2.0,
                label=f"{arm.replace('_', ' ').title()} — {layer.upper()}",
            )
            axis.fill_between(
                frame["epoch"],
                frame["ci_low"],
                frame["ci_high"],
                color=COLORS[arm],
                alpha=0.10,
            )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Self-consistent ECS rank")
    axis.set_title("ECS support continues to be recomputed as training evolves")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    _save(fig, destination, "02_ecs_rank_95ci.png")
    if show:
        plt.show()
    return fig


def plot_weightwatcher_alpha(
    result: PairedExperimentResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    if "alpha" not in result.spectral.columns or not np.isfinite(
        result.spectral["alpha"].to_numpy(dtype=float)
    ).any():
        return None
    destination = Path(output_dir) if output_dir is not None else None
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    layer_styles = {"fc1": "-", "fc2": "--", "fc3": ":"}
    for arm in ("baseline", "trace_wall"):
        for layer in sorted(result.spectral["layer"].unique()):
            frame = _summary_slice(
                result.spectral_summary,
                metric="alpha",
                arm=arm,
                layer=layer,
            )
            if frame.empty:
                continue
            axis.plot(
                frame["epoch"],
                frame["mean"],
                color=COLORS[arm],
                linestyle=layer_styles.get(layer, "-"),
                linewidth=2.0,
                label=f"{arm.replace('_', ' ').title()} — {layer.upper()}",
            )
            axis.fill_between(
                frame["epoch"],
                frame["ci_low"],
                frame["ci_high"],
                color=COLORS[arm],
                alpha=0.10,
            )
    axis.axhline(2.0, color="black", linewidth=1.0, alpha=0.55)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("WeightWatcher alpha")
    axis.set_title("Layerwise WeightWatcher alpha")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    _save(fig, destination, "03_weightwatcher_alpha_95ci.png")
    if show:
        plt.show()
    return fig


def plot_probe_corrections(
    result: PairedExperimentResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
) -> list[plt.Figure]:
    if result.corrections.empty:
        return []
    destination = Path(output_dir) if output_dir is not None else None
    figures: list[plt.Figure] = []

    unique = result.corrections.drop_duplicates(["seed", "global_step"])
    grouped = unique.groupby("global_step", sort=True)
    summary = grouped.agg(
        probe_loss_before=("probe_loss_before", "mean"),
        probe_loss_after=("probe_loss_after", "mean"),
        line_search_scale=("line_search_scale", "mean"),
        acceptance=("applied", "mean"),
    ).reset_index()

    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.plot(
        summary["global_step"],
        summary["probe_loss_before"],
        color=COLORS["probe_before"],
        linewidth=2.0,
        label="ECS-truncated probe loss before",
    )
    axis.plot(
        summary["global_step"],
        summary["probe_loss_after"],
        color=COLORS["probe_after"],
        linewidth=2.0,
        label="ECS-truncated probe loss after",
    )
    axis.set_xlabel("Global training step")
    axis.set_ylabel("Rotating-probe cross-entropy")
    axis.set_title("Accepted correction directly minimizes the current ECS probe loss")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, destination, "04_probe_loss_before_after.png")
    figures.append(fig)

    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    for layer in sorted(result.corrections["parameter_name"].unique()):
        frame = (
            result.corrections.loc[
                result.corrections["parameter_name"].eq(layer)
            ]
            .groupby("global_step", as_index=False)
            .agg(
                correction_ratio=("correction_to_base_step_ratio", "mean"),
                ecs_rank=("ecs_rank", "mean"),
            )
        )
        axis.plot(
            frame["global_step"],
            frame["correction_ratio"],
            linewidth=1.9,
            label=layer,
        )
    axis.set_xlabel("Global training step")
    axis.set_ylabel("Accepted correction norm / base-step norm")
    axis.set_title("TraceWall task component added to each selected matrix update")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, destination, "05_correction_to_base_step_ratio.png")
    figures.append(fig)

    if show:
        plt.show()
    return figures


def plot_all(
    result: PairedExperimentResult,
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = True,
) -> list[plt.Figure]:
    figures = plot_loss_and_accuracy(result, output_dir=output_dir, show=False)
    figures.append(plot_ecs_ranks(result, output_dir=output_dir, show=False))
    alpha = plot_weightwatcher_alpha(result, output_dir=output_dir, show=False)
    if alpha is not None:
        figures.append(alpha)
    figures.extend(plot_probe_corrections(result, output_dir=output_dir, show=False))
    if show:
        plt.show()
    return figures

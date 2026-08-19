"""Publication-style plots for long-horizon and tangent-RG experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LAYER_COLORS = {
    "fc1.weight": "#0072B2",
    "fc2.weight": "#D55E00",
    "fc3.weight": "#009E73",
}


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _finish(fig: Any, output: str | Path | None) -> Any:
    fig.tight_layout()
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight")
    return fig


def plot_metric_with_seed_ci(
    frame: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    reference: float | None = None,
    output: str | Path | None = None,
) -> Any:
    """Faint seed trajectories with mean and 95% Student-t bands by layer."""

    from .reporting import seed_confidence_intervals

    required = {"optimizer", "seed", "step", "layer", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"plot input is missing columns: {sorted(missing)}")
    plt = _pyplot()
    optimizers = list(dict.fromkeys(frame["optimizer"].astype(str)))
    fig, axes = plt.subplots(
        len(optimizers), 1, figsize=(9.0, 3.2 * len(optimizers)), squeeze=False
    )
    for axis, optimizer in zip(axes[:, 0], optimizers):
        subset = frame[frame["optimizer"] == optimizer]
        for layer, layer_frame in subset.groupby("layer"):
            color = LAYER_COLORS.get(str(layer), "#666666")
            for _, seed_frame in layer_frame.groupby("seed"):
                ordered = seed_frame.sort_values("step")
                axis.plot(
                    ordered["step"], ordered[metric], color=color, alpha=0.18, lw=0.9
                )
        ci = seed_confidence_intervals(
            subset,
            group_columns=("optimizer", "step", "layer"),
            metrics=(metric,),
        )
        ci = ci[ci["metric"] == metric]
        for layer, layer_ci in ci.groupby("layer"):
            ordered = layer_ci.sort_values("step")
            x = ordered["step"].to_numpy(dtype=float)
            color = LAYER_COLORS.get(str(layer), "#666666")
            axis.plot(x, ordered["mean"], color=color, lw=2.0, label=str(layer))
            axis.fill_between(
                x,
                ordered["ci_low"].to_numpy(dtype=float),
                ordered["ci_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.18,
            )
        if reference is not None:
            axis.axhline(float(reference), color="black", ls="--", lw=1.0)
        axis.set_xscale("symlog", linthresh=1.0)
        axis.set_title(str(optimizer))
        axis.set_xlabel("optimizer step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, fontsize=8)
    return _finish(fig, output)


def plot_alpha_and_trace(
    frame: pd.DataFrame,
    *,
    output_prefix: str | Path | None = None,
) -> tuple[Any, Any]:
    """Required fixed-point plots with alpha-two and trace-log-zero references."""

    prefix = Path(output_prefix) if output_prefix is not None else None
    alpha = plot_metric_with_seed_ci(
        frame,
        metric="alpha",
        ylabel="power-law alpha",
        reference=2.0,
        output=(prefix.with_name(prefix.name + "_alpha.png") if prefix else None),
    )
    trace = plot_metric_with_seed_ci(
        frame,
        metric="trace_log_per_eval",
        ylabel="trace-log per retained mode",
        reference=0.0,
        output=(prefix.with_name(prefix.name + "_trace_log.png") if prefix else None),
    )
    return alpha, trace


def plot_pdf_ccdf(
    values: Any,
    *,
    fit_row: dict[str, Any] | pd.Series | None = None,
    title: str = "",
    output: str | Path | None = None,
) -> Any:
    """Log-log PDF and CCDF with the package-selected fit window marked."""

    from .powerlaw_fit import empirical_ccdf, positive_values

    sample = positive_values(values)
    x, ccdf = empirical_ccdf(sample)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    bins = np.geomspace(sample[0], sample[-1], min(50, max(8, sample.size // 3)))
    axes[0].hist(sample, bins=bins, density=True, histtype="step", lw=1.8)
    axes[1].step(x, ccdf, where="post", lw=1.8)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        if fit_row is not None and np.isfinite(float(fit_row.get("xmin", np.nan))):
            axis.axvline(float(fit_row["xmin"]), color="#D55E00", ls="--", label="xmin")
            xmax = float(fit_row.get("xmax", np.nan))
            if np.isfinite(xmax):
                axis.axvline(xmax, color="#009E73", ls=":", label="tail max")
            axis.legend(fontsize=8)
    axes[0].set(xlabel="value", ylabel="density", title=f"{title} PDF")
    axes[1].set(xlabel="value", ylabel="P(X >= x)", title=f"{title} CCDF")
    return _finish(fig, output)


def plot_method_stability(
    frame: pd.DataFrame,
    *,
    output: str | Path | None = None,
) -> Any:
    """Compare alpha, KS distance, and tail size across methods and nulls."""

    required = {"operator_kind", "alpha", "ks_D", "n_tail"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"method table is missing columns: {sorted(missing)}")
    summary = frame.groupby("operator_kind", sort=True).agg(
        alpha=("alpha", "mean"), ks_D=("ks_D", "mean"), n_tail=("n_tail", "mean")
    )
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for axis, metric, label in zip(
        axes, ("alpha", "ks_D", "n_tail"), ("alpha", "KS D", "tail count")
    ):
        summary[metric].plot.bar(ax=axis, color="#0072B2")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    return _finish(fig, output)

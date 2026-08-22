#!/usr/bin/env python3
"""Build a complete, static report from the reduced Jacobian run.

No Jacobians are computed here. The program joins the saved Jacobian spectra
and fits to the baseline WeightWatcher and performance tables, then writes all
comparison figures, analysis-ready CSVs, and a browsable HTML index.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import logging
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUITE = "mnist_mlp3_tangent_rg_v1_muonclip_short100_10seed"
OPTIMIZERS = ("muonclip_rms", "adamw")
LAYERS = ("fc1.weight", "fc2.weight", "fc3.weight")
COLORS = {"muonclip_rms": "#0072B2", "adamw": "#D55E00"}
OPTIMIZER_LABELS = {"muonclip_rms": "MuonClip-RMS", "adamw": "AdamW"}
METHOD_LABELS = {
    "centered_log_singular_radial_pullback": "Centered log-singular radial",
    "ecs_grassmann_cartan_cover_full_row_shell_pullback": "ECS full-row shell",
    "ecs_grassmann_cartan_cover_detx_shell_pullback": "ECS detX shell",
    "gap_aware_projector_detx_shell_pullback": "Gap-aware projector (detX)",
    "trace_free_log_gram_detx_shell_pullback": "Trace-free log Gram (detX)",
    "gram_ridge_resolvent_detx_shell_zratio_0p50_pullback": "Gram ridge resolvent (detX, z=0.5 boundary)",
    "feshbach_trace_free_log_detx_shell_pullback": "Feshbach trace-free log (detX)",
}
METHOD_STYLES = {
    "centered_log_singular_radial_pullback": ("-", "o"),
    "ecs_grassmann_cartan_cover_full_row_shell_pullback": ("--", "s"),
    "ecs_grassmann_cartan_cover_detx_shell_pullback": (":", "^"),
    "gap_aware_projector_detx_shell_pullback": ("-.", "D"),
    "trace_free_log_gram_detx_shell_pullback": ((0, (5, 1)), "P"),
    "gram_ridge_resolvent_detx_shell_zratio_0p50_pullback": (
        (0, (3, 1, 1, 1)), "X"
    ),
    "feshbach_trace_free_log_detx_shell_pullback": ((0, (1, 1)), "v"),
}
ECS_METHODS = (
    "ecs_grassmann_cartan_cover_full_row_shell_pullback",
    "ecs_grassmann_cartan_cover_detx_shell_pullback",
)
EXTENDED_ECS_METHODS = (
    "gap_aware_projector_detx_shell_pullback",
    "trace_free_log_gram_detx_shell_pullback",
    "gram_ridge_resolvent_detx_shell_zratio_0p50_pullback",
    "feshbach_trace_free_log_detx_shell_pullback",
)
EXPECTED_METHODS_BY_LAYER = {
    "fc1.weight": ("centered_log_singular_radial_pullback", *ECS_METHODS, *EXTENDED_ECS_METHODS),
    "fc2.weight": ("centered_log_singular_radial_pullback", *ECS_METHODS, *EXTENDED_ECS_METHODS),
    "fc3.weight": ("centered_log_singular_radial_pullback",),
}
QUOTIENT_LABELS = {
    "midpoint_ecs_control": "Midpoint ECS control",
    "gram_ridge": "Gram diagonal subtraction",
    "feshbach_downfolding": "Anchor-frozen Feshbach",
}
QUOTIENT_PROFILE_LABELS = {
    "midpoint": "no counterterm",
    "tau_fraction_0p25": "τ=0.25 λboundary",
    "tau_fraction_0p50": "τ=0.50 λboundary",
    "tau_fraction_0p75": "τ=0.75 λboundary",
    "ridge_ratio_1em2": "ρ=0.01 anchor scale",
}
FLOW_LABELS = {
    "two_checkpoint_generalized_gram_radial": "Generalized-Gram radial rate",
    "two_checkpoint_aligned_transfer_core": "Aligned transfer-core rate",
    "two_checkpoint_ecs_topk_grassmann": "ECS top-k Grassmann rate",
    "two_checkpoint_relative_polar_tilt": "Relative-polar tilt rate",
    "two_checkpoint_radial_quotient_observed_secant": "Observed radial quotient secant",
    "two_checkpoint_radial_jacobian_prediction": "Local radial Jacobian prediction",
}


def configure_logging(report_root: Path) -> logging.Logger:
    report_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("short100_jacobian_report")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    for stream in (sys.stdout, report_root / "report.log"):
        handler = (
            logging.StreamHandler(stream)
            if hasattr(stream, "write")
            else logging.FileHandler(stream, mode="w")
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def require_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required data table is missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"required data table is empty: {path}")
    return frame


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def save_figure(
    fig: plt.Figure, path: Path, *, bottom: float = 0.0, top: float = 1.0
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, bottom, 1.0, top))
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def shared_legend(fig: plt.Figure, axes: Iterable[plt.Axes], *, columns: int = 3) -> None:
    unique: dict[str, object] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    fig.legend(
        unique.values(), unique.keys(), loc="lower center", ncol=columns,
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005),
    )


def curve_style(optimizer: str, method: str) -> dict[str, object]:
    linestyle, marker = METHOD_STYLES.get(method, ("-", "o"))
    return {
        "color": COLORS[optimizer],
        "linestyle": linestyle,
        "marker": marker,
        "markersize": 4.0,
        "linewidth": 1.9,
    }


def plot_jacobian_metric(
    fits: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    reference: float | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), sharex=True)
    for axis, layer in zip(axes, LAYERS):
        subset = fits[fits["layer"].astype(str).eq(layer)]
        for (optimizer, method), curve in subset.groupby(["optimizer", "method"]):
            curve = curve.sort_values("epoch")
            axis.plot(
                curve["epoch"], curve[metric],
                label=f"{OPTIMIZER_LABELS.get(optimizer, optimizer)} — "
                f"{METHOD_LABELS.get(method, method)}",
                **curve_style(str(optimizer), str(method)),
            )
        if reference is not None:
            axis.axhline(reference, color="black", linestyle=(0, (1, 2)), linewidth=1.2)
        axis.set(title=layer.replace(".weight", ""), xlabel="epoch", ylabel=ylabel)
        available = [
            METHOD_LABELS.get(method, method)
            for method in EXPECTED_METHODS_BY_LAYER[layer]
            if subset["method"].astype(str).eq(method).any()
        ]
        axis.text(
            0.01, 0.02, "present: " + ", ".join(available),
            transform=axis.transAxes, fontsize=6.5, color="#555555",
            ha="left", va="bottom",
        )
        axis.grid(True, alpha=0.25)
    shared_legend(fig, axes, columns=3)
    fig.suptitle(title)
    return save_figure(fig, path, bottom=0.16, top=0.94)


def plot_ecs_quotient_comparison(fits: pd.DataFrame, path: Path) -> Path:
    """Plot ECS-only alpha for FC1/FC2 while exposing coincident fits.

    Lines remain at the exact analysis epochs. Marker locations receive a tiny
    horizontal display-only offset, so equal full-row and detX values remain
    visible instead of one marker painting over the other.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharex=True)
    marker_offsets = {ECS_METHODS[0]: -0.45, ECS_METHODS[1]: 0.45}
    for axis, layer in zip(axes, ("fc1.weight", "fc2.weight")):
        subset = fits[
            fits["layer"].astype(str).eq(layer)
            & fits["method"].astype(str).isin(ECS_METHODS)
        ]
        for (optimizer, method), curve in subset.groupby(["optimizer", "method"]):
            curve = curve.sort_values("epoch")
            linestyle, marker = METHOD_STYLES[str(method)]
            label = (
                f"{OPTIMIZER_LABELS.get(str(optimizer), str(optimizer))} — "
                f"{METHOD_LABELS[str(method)]}"
            )
            x = pd.to_numeric(curve["epoch"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(curve["alpha"], errors="coerce").to_numpy(float)
            axis.plot(
                x, y, color=COLORS[str(optimizer)], linestyle=linestyle,
                linewidth=1.8, label=label,
            )
            axis.scatter(
                x + marker_offsets[str(method)], y, color=COLORS[str(optimizer)],
                marker=marker, s=28, zorder=4,
            )
        axis.axhline(2.0, color="black", linestyle=(0, (1, 2)), linewidth=1.2)
        axis.set(
            title=layer.replace(".weight", ""), xlabel="epoch",
            ylabel="ECS Jacobian energy alpha",
        )
        axis.grid(True, alpha=0.25)
    shared_legend(fig, axes, columns=2)
    fig.suptitle(
        "ECS quotient comparison — lines use exact epochs; markers offset ±0.45 for visibility"
    )
    return save_figure(fig, path, bottom=0.18, top=0.92)


def plot_single_optimizer_jacobians(
    fits: pd.DataFrame, optimizer: str, path: Path
) -> Path:
    """Make a readable optimizer-specific view of every local map Jacobian."""
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), sharex=True)
    subset_optimizer = fits[fits["optimizer"].astype(str).eq(optimizer)]
    method_order = [
        method for method in METHOD_LABELS
        if subset_optimizer["method"].astype(str).eq(method).any()
    ]
    for axis, layer in zip(axes, LAYERS):
        subset = subset_optimizer[subset_optimizer["layer"].astype(str).eq(layer)]
        for index, method in enumerate(method_order):
            curve = subset[subset["method"].astype(str).eq(method)].sort_values("epoch")
            if curve.empty:
                continue
            linestyle, marker = METHOD_STYLES.get(method, ("-", "o"))
            axis.plot(
                curve["epoch"], curve["alpha"], color=palette(index % 10),
                linestyle=linestyle, marker=marker, markersize=3.5,
                linewidth=1.6, label=METHOD_LABELS.get(method, method),
            )
        axis.axhline(2.0, color="black", linestyle=(0, (1, 2)), linewidth=1.1)
        axis.set(title=layer.replace(".weight", ""), xlabel="epoch", ylabel="Jacobian energy alpha")
        axis.grid(True, alpha=0.25)
    shared_legend(fig, axes, columns=4)
    fig.suptitle(f"{OPTIMIZER_LABELS[optimizer]} — single-checkpoint quotient-map Jacobians")
    return save_figure(fig, path, bottom=0.18, top=0.93)


def plot_weight_quotients_by_optimizer(
    quotient_fits: pd.DataFrame, optimizer: str, path: Path
) -> Path:
    frame = quotient_fits[
        quotient_fits["optimizer"].astype(str).eq(optimizer)
        & quotient_fits["layer"].astype(str).isin(("fc1.weight", "fc2.weight"))
    ].copy()
    frame = frame[bool_series(frame["fit_ok"])]
    profile_colors = {
        "midpoint": "#4C78A8",
        "tau_fraction_0p25": "#F2CF5B",
        "tau_fraction_0p50": "#F58518",
        "tau_fraction_0p75": "#B279A2",
        "ridge_ratio_1em2": "#54A24B",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), sharex=True)
    variants = ("raw", "clip_xmax")
    for row, variant in enumerate(variants):
        for column, layer in enumerate(("fc1.weight", "fc2.weight")):
            axis = axes[row, column]
            subset = frame[
                frame["layer"].astype(str).eq(layer)
                & frame["fit_variant"].astype(str).eq(variant)
            ]
            for (method, profile), curve in subset.groupby(["method", "profile_id"]):
                curve = curve.sort_values("epoch")
                axis.plot(
                    curve["epoch"], curve["alpha"],
                    color=profile_colors.get(str(profile), "#777777"),
                    marker="s" if variant == "clip_xmax" else "o",
                    markersize=3.5, linewidth=1.7,
                    label=(
                        f"{QUOTIENT_LABELS.get(str(method), str(method))}, "
                        f"{QUOTIENT_PROFILE_LABELS.get(str(profile), str(profile))}"
                    ),
                )
            axis.axhline(2.0, color="black", linestyle=(0, (1, 2)), linewidth=1.1)
            axis.set(
                title=f"{layer.replace('.weight', '')} — {variant}",
                xlabel="epoch", ylabel="WeightWatcher alpha of W′",
            )
            axis.grid(True, alpha=0.25)
    shared_legend(fig, axes.ravel(), columns=3)
    fig.suptitle(f"{OPTIMIZER_LABELS[optimizer]} — transformed weight quotient representatives")
    return save_figure(fig, path, bottom=0.14, top=0.94)


def plot_checkpoint_flows_by_optimizer(
    flow_fits: pd.DataFrame, optimizer: str, path: Path
) -> Path:
    primary = flow_fits[
        flow_fits["optimizer"].astype(str).eq(optimizer)
        & flow_fits["spectrum_kind"].astype(str).eq("energy_derived_from_amplitude")
        & pd.to_numeric(flow_fits["clip_top_k"], errors="coerce").eq(0)
    ]
    palette = plt.get_cmap("Dark2")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), sharex=True)
    for axis, layer in zip(axes, ("fc1.weight", "fc2.weight")):
        subset = primary[primary["layer"].astype(str).eq(layer)]
        for index, (method, curve) in enumerate(subset.groupby("method")):
            curve = curve.sort_values("epoch_end")
            axis.plot(
                curve["epoch_end"], curve["alpha"], color=palette(index),
                marker="o", markersize=3.5, linewidth=1.7,
                label=FLOW_LABELS.get(str(method), str(method)),
            )
        axis.axhline(2.0, color="black", linestyle=(0, (1, 2)), linewidth=1.1)
        axis.set(title=layer.replace(".weight", ""), xlabel="ending epoch", ylabel="finite-flow energy alpha")
        axis.grid(True, alpha=0.25)
    shared_legend(fig, axes, columns=2)
    fig.suptitle(f"{OPTIMIZER_LABELS[optimizer]} — between-checkpoint RG-flow observables (not Jacobians)")
    return save_figure(fig, path, bottom=0.18, top=0.92)


def plot_jacobian_transport_by_optimizer(
    transport: pd.DataFrame, optimizer: str, path: Path
) -> Path:
    """Compare finite quotient flow to the local map-Jacobian prediction."""
    frame = transport[transport["optimizer"].astype(str).eq(optimizer)]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), sharex=True)
    for column, layer in enumerate(("fc1.weight", "fc2.weight")):
        curve = frame[frame["layer"].astype(str).eq(layer)].sort_values("epoch_end")
        axes[0, column].plot(
            curve["epoch_end"], curve["relative_linearization_error"],
            color=COLORS[optimizer], marker="o", linewidth=1.8,
        )
        axes[1, column].plot(
            curve["epoch_end"], curve["cosine_observed_vs_jacobian"],
            color=COLORS[optimizer], marker="s", linewidth=1.8,
        )
        axes[0, column].set(
            title=layer.replace(".weight", ""),
            ylabel="relative linearization error",
        )
        axes[1, column].set(xlabel="ending epoch", ylabel="observed/JVP cosine")
        axes[1, column].axhline(1.0, color="black", linestyle=(0, (1, 2)), linewidth=1.0)
        for axis in axes[:, column]:
            axis.grid(True, alpha=0.25)
    fig.suptitle(
        f"{OPTIMIZER_LABELS[optimizer]} — observed radial quotient flow versus local Jacobian"
    )
    return save_figure(fig, path, top=0.94)


def build_method_coverage(primary: pd.DataFrame) -> pd.DataFrame:
    """Return an explicit expected-versus-observed method inventory."""
    rows = []
    for optimizer in OPTIMIZERS:
        for layer in LAYERS:
            layer_rows = primary[
                primary["optimizer"].astype(str).eq(optimizer)
                & primary["layer"].astype(str).eq(layer)
            ]
            expected = EXPECTED_METHODS_BY_LAYER[layer]
            for method in expected:
                observed = layer_rows[layer_rows["method"].astype(str).eq(method)]
                epochs = sorted(
                    pd.to_numeric(observed["epoch"], errors="coerce")
                    .dropna().astype(int).unique()
                )
                rows.append({
                    "optimizer": optimizer,
                    "layer": layer,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "expected_epoch_count": 10,
                    "observed_epoch_count": len(epochs),
                    "observed_epochs": ",".join(map(str, epochs)),
                    "coverage_status": "complete" if len(epochs) == 10 else "INCOMPLETE",
                })
    return pd.DataFrame(rows)


def plot_fit_quality(fits: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 8.2), sharex=True)
    for column, layer in enumerate(LAYERS):
        subset = fits[fits["layer"].astype(str).eq(layer)]
        for (optimizer, method), curve in subset.groupby(["optimizer", "method"]):
            curve = curve.sort_values("epoch")
            style = curve_style(str(optimizer), str(method))
            label = (
                f"{OPTIMIZER_LABELS.get(optimizer, optimizer)} — "
                f"{METHOD_LABELS.get(method, method)}"
            )
            axes[0, column].plot(curve["epoch"], curve["ks_D"], label=label, **style)
            axes[1, column].plot(
                curve["epoch"], curve["tail_decades"], label=label, **style
            )
        axes[0, column].set(title=layer.replace(".weight", ""), ylabel="KS D")
        axes[1, column].set(xlabel="epoch", ylabel="tail decades")
        for axis in axes[:, column]:
            axis.grid(True, alpha=0.25)
    shared_legend(fig, axes.ravel(), columns=3)
    fig.suptitle("Jacobian power-law fit quality")
    return save_figure(fig, path, bottom=0.12, top=0.95)


def plot_weightwatcher(weightwatcher: pd.DataFrame, path: Path) -> Path:
    frame = weightwatcher.copy()
    frame["fit_ok_bool"] = bool_series(frame["fit_ok"])
    frame = frame[
        frame["layer"].astype(str).isin(LAYERS)
        & frame["fit_variant"].astype(str).isin({"raw", "clip_xmax"})
        & frame["fit_ok_bool"]
        & pd.to_numeric(frame["epoch"], errors="coerce").gt(0)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), sharex=True)
    variant_style = {"raw": ("--", "o"), "clip_xmax": ("-", "s")}
    for axis, layer in zip(axes, LAYERS):
        subset = frame[frame["layer"].astype(str).eq(layer)]
        for (optimizer, variant), curve in subset.groupby(["optimizer", "fit_variant"]):
            curve = curve.sort_values("epoch")
            linestyle, marker = variant_style[str(variant)]
            axis.plot(
                curve["epoch"], curve["alpha"],
                color=COLORS[str(optimizer)], linestyle=linestyle, marker=marker,
                markersize=3.5, linewidth=1.8,
                label=f"{OPTIMIZER_LABELS[str(optimizer)]} — {variant}",
            )
        axis.axhline(2.0, color="black", linestyle=(0, (1, 2)), linewidth=1.2)
        axis.set(title=layer.replace(".weight", ""), xlabel="epoch", ylabel="WW alpha")
        axis.grid(True, alpha=0.25)
    shared_legend(fig, axes, columns=4)
    fig.suptitle("WeightWatcher controls: raw and fix_fingers=clip_xmax")
    return save_figure(fig, path, bottom=0.13, top=0.94)


def plot_performance(performance: pd.DataFrame, path: Path) -> Path:
    metrics = (
        ("train_accuracy", "Train accuracy", True),
        ("test_accuracy", "Test accuracy", True),
        ("train_loss", "Train loss", False),
        ("test_loss", "Test loss", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)
    for axis, (metric, title, bounded) in zip(axes.ravel(), metrics):
        for optimizer, curve in performance.groupby("optimizer"):
            curve = curve.sort_values("epoch")
            axis.plot(
                curve["epoch"], curve[metric], color=COLORS[str(optimizer)],
                linewidth=2.0, label=OPTIMIZER_LABELS[str(optimizer)],
            )
        axis.set(title=title, xlabel="epoch", ylabel=metric.replace("_", " "))
        if bounded:
            values = pd.to_numeric(performance[metric], errors="coerce")
            low = max(0.0, float(values.min()) - 0.01)
            high = min(1.001, float(values.max()) + 0.005)
            axis.set_ylim(low, high)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("MuonClip-RMS versus AdamW performance — seed 101")
    return save_figure(fig, path)


def empirical_ccdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    sample = np.sort(np.asarray(tuple(values), dtype=float))
    sample = sample[np.isfinite(sample) & (sample > 0)]
    return sample, np.arange(sample.size, 0, -1, dtype=float) / sample.size


def plot_spectral_galleries(spectra: pd.DataFrame, figure_root: Path) -> list[Path]:
    outputs: list[Path] = []
    available_epochs = sorted(pd.to_numeric(spectra["epoch"], errors="coerce").dropna().astype(int).unique())
    requested = [epoch for epoch in (10, 50, 100) if epoch in available_epochs]
    for optimizer in OPTIMIZERS:
        for layer in LAYERS:
            fig, axes = plt.subplots(1, len(requested), figsize=(5.2 * len(requested), 4.5))
            axes = np.atleast_1d(axes)
            subset = spectra[
                spectra["optimizer"].astype(str).eq(optimizer)
                & spectra["layer"].astype(str).eq(layer)
            ]
            for axis, epoch in zip(axes, requested):
                state = subset[pd.to_numeric(subset["epoch"], errors="coerce").eq(epoch)]
                for method, modes in state.groupby("method"):
                    x, y = empirical_ccdf(modes["gram_eigenvalue"])
                    linestyle, marker = METHOD_STYLES.get(str(method), ("-", "o"))
                    axis.step(
                        x, y, where="post", linestyle=linestyle, linewidth=1.7,
                        label=METHOD_LABELS.get(str(method), str(method)),
                    )
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.set(title=f"epoch {epoch}", xlabel="Jacobian Gram eigenvalue", ylabel="CCDF")
                axis.grid(True, alpha=0.25)
                axis.legend(fontsize=7, frameon=False)
            fig.suptitle(f"{OPTIMIZER_LABELS[optimizer]} — {layer} spectral evolution")
            output = figure_root / "spectral_galleries" / f"{optimizer}_{layer.replace('.', '_')}.png"
            outputs.append(save_figure(fig, output))
    return outputs


def plot_alpha_performance_relationship(
    fits: pd.DataFrame, performance: pd.DataFrame, path: Path
) -> Path:
    joined = fits.merge(
        performance[["optimizer", "seed", "epoch", "test_accuracy", "test_loss"]],
        on=["optimizer", "seed", "epoch"], how="inner", validate="many_to_one",
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for axis, layer in zip(axes, LAYERS):
        subset = joined[joined["layer"].astype(str).eq(layer)]
        for (optimizer, method), points in subset.groupby(["optimizer", "method"]):
            linestyle, marker = METHOD_STYLES.get(str(method), ("-", "o"))
            axis.scatter(
                points["alpha"], points["test_accuracy"],
                color=COLORS[str(optimizer)], marker=marker, s=35, alpha=0.8,
                label=f"{OPTIMIZER_LABELS[str(optimizer)]} — "
                f"{METHOD_LABELS.get(str(method), str(method))}",
            )
        axis.axvline(2.0, color="black", linestyle=(0, (1, 2)), linewidth=1.2)
        axis.set(title=layer.replace(".weight", ""), xlabel="Jacobian energy alpha", ylabel="test accuracy")
        axis.grid(True, alpha=0.25)
    shared_legend(fig, axes, columns=3)
    fig.suptitle("Jacobian alpha versus held-out test accuracy")
    return save_figure(fig, path, bottom=0.16, top=0.94)


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def build_html(
    report_root: Path,
    figures: list[Path],
    tables: list[Path],
    primary: pd.DataFrame,
    coverage: pd.DataFrame,
    inventory: dict[str, int],
) -> Path:
    final = primary.sort_values("epoch").groupby(
        ["optimizer", "layer", "method"], as_index=False
    ).tail(1)
    final_table = final[
        ["optimizer", "epoch", "layer", "method", "alpha", "ks_D", "n_tail", "tail_decades", "fit_ok"]
    ].to_html(index=False, float_format=lambda value: f"{value:.5g}")
    coverage_table = coverage.to_html(index=False)
    figure_html = "\n".join(
        f'<section><h2>{escape(path.stem.replace("_", " ").title())}</h2>'
        f'<a href="{escape(relative(path, report_root))}">'
        f'<img src="{escape(relative(path, report_root))}" loading="lazy"></a></section>'
        for path in figures
    )
    table_html = "\n".join(
        f'<li><a href="{escape(relative(path, report_root))}">{escape(path.name)}</a></li>'
        for path in tables
    )
    inventory_html = "".join(
        f"<li><strong>{escape(name)}</strong>: {count:,}</li>"
        for name, count in inventory.items()
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Short100 quotient and RG-flow report</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;max-width:1500px;color:#202124}}
img{{max-width:100%;height:auto;border:1px solid #ddd}} section{{margin:34px 0}}
table{{border-collapse:collapse;font-size:13px}} th,td{{border:1px solid #ddd;padding:5px 8px}} th{{background:#f2f4f7}}
code{{background:#f3f3f3;padding:2px 4px}} .note{{background:#eef6ff;padding:14px;border-left:4px solid #0072B2}}
</style></head><body>
<h1>Short100 weight quotient, RG-flow, and Jacobian analysis</h1>
<p class="note"><strong>Scope.</strong> MuonClip-RMS versus AdamW, seed 101,
epochs 10,20,…,100. The centered log-singular radial Jacobian is evaluated on
FC1, FC2, and FC3. Both exact ECS quotient covers are evaluated on FC1 and FC2.
FC3 is intentionally radial-only because its 10-row output geometry is a
small-rank diagnostic rather than the large-layer ECS comparison.</p>
<h2>What each curve means</h2>
<ul>
<li><strong>Centered log-singular radial:</strong> the scale-quotiented radial
response in centered log singular-value coordinates. Its plotted spectrum is
the squared Jacobian singular-amplitude spectrum, and α is the power-law fit to
that energy spectrum.</li>
<li><strong>ECS full-row shell:</strong> the Grassmann/Cartan quotient cover whose
outer rank uses the full numerical row shell.</li>
<li><strong>ECS detX shell:</strong> the same quotient construction with the outer
rank restricted by the checkpoint's audited detX/ECS boundary.</li>
</ul>
<p>ECS fits use one physical retained-core amplitude for each uniformly repeated
shell group. The omitted coordinate copies have identical values: compressing
them leaves the empirical spectral shape, α, xmin, and KS distance unchanged,
while avoiding a fictitiously large independent sample size.</p>
<p><strong>How to read coincident curves.</strong> Full-row and detX ECS fits can be
exactly equal. In the all-method figure one line may cover another. The dedicated
ECS figure keeps lines at the true epochs but offsets full-row and detX markers by
−0.45 and +0.45 epoch solely for visibility. The CSV retains the exact epochs.</p>
<h2>Coverage audit</h2>
<p>Every expected method should have ten observations. Any row marked
<code>INCOMPLETE</code> means computation is genuinely missing; visual overlap is
not classified as missing.</p>{coverage_table}
<h2>Scientific questions and what is actually identified</h2>
<h3>Case 1 — heavy tails on a weight quotient representative</h3>
<p>For the state-level experiment, the strictly identifiable quotient is the
two-sided orthogonal orbit
<code>[W] = {{U W Vᵀ : U∈O(m), V∈O(n)}}</code>. Its invariants are the singular
values. We choose the rectangular-diagonal canonical section and then apply a
declared nuisance-removal hypothesis before running WeightWatcher on the
materialized matrix <code>W′</code>, both raw and with
<code>fix_fingers=clip_xmax</code>. A heavy tail found here is therefore a heavy
tail of that declared quotient representative—not proof that it is the unique
unknown Muon time-orbit quotient.</p>
<ul>
<li><strong>Midpoint ECS control:</strong> retain the midpoint-selected singular
sector without a counterterm.</li>
<li><strong>Gram diagonal subtraction:</strong>
<code>X=W Wᵀ</code> (or <code>WᵀW</code> on the smaller side) is transformed by
<code>λᵢ↦max(λᵢ−τ,0)</code>, scanned at τ/λboundary = 0.25, 0.50, and
0.75. This is nonlinear because of the positive-part threshold; it is not
a global rescaling and can change the fitted spectral shape.</li>
<li><strong>Feshbach downfolding:</strong> an independent epoch-10 singular frame
freezes P and Q, then
<code>X_eff=A−B(C+ρI)⁻¹Bᵀ</code> with ρ=0.01 times the anchor boundary scale.
Freezing the anchor is essential: choosing P from the same X diagonalizes the
Gram matrix and makes B=0, reducing the construction to truncation.</li>
</ul>
<h3>Case 2a — flow between checkpoints</h3>
<p>Two checkpoints identify finite flow/secant observables: generalized-Gram
radial log rates, a Procrustes-aligned rectangular transfer core, top-k ECS
Grassmann principal-angle rates, and relative-polar tilt rates. Their squared
rate amplitudes are fitted for heavy tails. These show how the saved trajectory
moves in quotient coordinates. They are <strong>not</strong>
<code>Dβ(W)</code>, the Jacobian of the optimizer/RG vector field; identifying
that object requires perturbation trajectories or a fully specified training
map with batch and optimizer state.</p>
<p>The centered log-singular coordinate additionally supplies a direct local
transport test. For each pair the report compares the observed secant
<code>[R(W₁)−R(W₀)]/Δt</code> with the actual map-Jacobian prediction
<code>D R_W₀[W₁−W₀]/Δt</code>, reporting both spectra, relative error, and cosine.
This tests whether the saved step follows the local quotient-map linearization;
it still does not turn that map derivative into <code>Dβ(W)</code>.</p>
<h3>Case 2b — a Jacobian at one checkpoint</h3>
<p>The single-checkpoint tables contain exact analytic derivatives of explicitly
declared weight-only maps: centered log-singular radial, ECS Grassmann/Cartan,
gap-aware projector, trace-free log Gram, ridge-resolvent, and Feshbach
trace-free log. These are genuine Jacobians of those maps at W, but not the
training-dynamics Jacobian unless the declared map is separately calibrated to
the optimizer step.</p>
<p>The single-checkpoint Feshbach result has a mandatory caveat: in the
checkpoint's own SVD frame B=0, so shell-downfolding terms vanish at first
order. The report keeps this curve as an explicit collapse/control. Nontrivial
state-level Feshbach behavior comes from the independently frozen epoch-10
anchor, while nontrivial first-order Feshbach dynamics would require a frozen
frame not diagonalizing the evaluation checkpoint.</p>
<h2>Data inventory</h2><ul>{inventory_html}</ul>
<h2>Analysis-ready tables</h2><ul>{table_html}</ul>
<h2>Final checkpoint primary fits</h2>{final_table}
{figure_html}
</body></html>"""
    destination = report_root / "index.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root", type=Path,
        default=Path("/private/tmp/rg-mnist-mlp3-short100-jacobians-reduced"),
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=Path("/private/tmp/rg-mnist-mlp3-short100-runs"),
    )
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--seed", type=int, default=101)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    analysis_root = args.analysis_root.expanduser().resolve()
    report_root = (
        args.report_root.expanduser().resolve()
        if args.report_root else analysis_root / "report"
    )
    figure_root = report_root / "figures"
    table_root = report_root / "tables"
    logger = configure_logging(report_root)
    try:
        logger.info("Loading reduced Jacobian outputs from %s", analysis_root)
        fits = require_csv(analysis_root / "jacobian_powerlaw_fits.csv")
        spectra = require_csv(analysis_root / "jacobian_spectra.csv")
        operators = require_csv(analysis_root / "jacobian_operators.csv")
        quotient_fits = require_csv(
            analysis_root / "weight_quotient_weightwatcher_fits.csv"
        )
        quotient_spectra = require_csv(analysis_root / "weight_quotient_spectra.csv")
        quotient_operators = require_csv(analysis_root / "weight_quotient_operators.csv")
        flow_fits = require_csv(analysis_root / "two_checkpoint_flow_fits.csv")
        flow_spectra = require_csv(analysis_root / "two_checkpoint_flow_spectra.csv")
        flow_operators = require_csv(analysis_root / "two_checkpoint_flow_operators.csv")
        flow_transport = require_csv(
            analysis_root / "two_checkpoint_jacobian_transport.csv"
        )
        primary = fits[
            fits["spectrum_kind"].astype(str).eq("energy_derived_from_amplitude")
            & pd.to_numeric(fits["clip_top_k"], errors="coerce").eq(0)
        ].copy()
        primary = primary[pd.to_numeric(primary["epoch"], errors="coerce").gt(0)]
        performance_frames = []
        weightwatcher_frames = []
        for optimizer in OPTIMIZERS:
            metrics = args.run_root / SUITE / optimizer / f"seed_{args.seed}" / "metrics"
            performance_frames.append(require_csv(metrics / "performance_by_analysis_epoch.csv"))
            weightwatcher_frames.append(require_csv(metrics / "weightwatcher_fits.csv"))
        performance = pd.concat(performance_frames, ignore_index=True, sort=False)
        weightwatcher = pd.concat(weightwatcher_frames, ignore_index=True, sort=False)
        for column in ("epoch", "alpha", "ks_D", "n_tail", "tail_decades"):
            if column in primary:
                primary[column] = pd.to_numeric(primary[column], errors="coerce")
        performance = performance[pd.to_numeric(performance["epoch"], errors="coerce").gt(0)]
        performance = performance.sort_values(["optimizer", "epoch"])

        table_root.mkdir(parents=True, exist_ok=True)
        coverage = build_method_coverage(primary)
        tables = [
            table_root / "jacobian_primary_energy_fits.csv",
            table_root / "jacobian_all_fits.csv",
            table_root / "jacobian_spectra_all_modes.csv",
            table_root / "jacobian_operator_metadata.csv",
            table_root / "weightwatcher_raw_and_clip_xmax.csv",
            table_root / "performance_train_test.csv",
            table_root / "jacobian_method_coverage.csv",
            table_root / "weight_quotient_weightwatcher_fits.csv",
            table_root / "weight_quotient_spectra.csv",
            table_root / "weight_quotient_operators.csv",
            table_root / "two_checkpoint_flow_fits.csv",
            table_root / "two_checkpoint_flow_spectra.csv",
            table_root / "two_checkpoint_flow_operators.csv",
            table_root / "two_checkpoint_jacobian_transport.csv",
        ]
        primary.to_csv(tables[0], index=False)
        fits.to_csv(tables[1], index=False)
        spectra.to_csv(tables[2], index=False)
        operators.to_csv(tables[3], index=False)
        weightwatcher.to_csv(tables[4], index=False)
        performance.to_csv(tables[5], index=False)
        coverage.to_csv(tables[6], index=False)
        quotient_fits.to_csv(tables[7], index=False)
        quotient_spectra.to_csv(tables[8], index=False)
        quotient_operators.to_csv(tables[9], index=False)
        flow_fits.to_csv(tables[10], index=False)
        flow_spectra.to_csv(tables[11], index=False)
        flow_operators.to_csv(tables[12], index=False)
        flow_transport.to_csv(tables[13], index=False)

        figures = [
            plot_jacobian_metric(
                primary, "alpha", "Jacobian energy alpha",
                "MuonClip-RMS versus AdamW Jacobian alpha",
                figure_root / "01_jacobian_alpha_comparison.png", reference=2.0,
            ),
            plot_ecs_quotient_comparison(
                primary, figure_root / "01b_ecs_fc1_fc2_alpha_comparison.png"
            ),
            plot_fit_quality(primary, figure_root / "02_jacobian_fit_quality.png"),
            plot_jacobian_metric(
                primary, "n_tail", "package-selected tail modes",
                "Jacobian PL tail support",
                figure_root / "03_jacobian_tail_support.png",
            ),
            plot_weightwatcher(
                weightwatcher, figure_root / "04_weightwatcher_raw_vs_clip_xmax.png"
            ),
            plot_performance(performance, figure_root / "05_performance_accuracy_loss.png"),
            plot_alpha_performance_relationship(
                primary, performance, figure_root / "06_alpha_vs_test_accuracy.png"
            ),
        ]
        for optimizer in OPTIMIZERS:
            figures.extend([
                plot_single_optimizer_jacobians(
                    primary, optimizer,
                    figure_root / f"optimizer_views/{optimizer}_single_checkpoint_jacobians.png",
                ),
                plot_weight_quotients_by_optimizer(
                    quotient_fits, optimizer,
                    figure_root / f"optimizer_views/{optimizer}_weight_quotients.png",
                ),
                plot_checkpoint_flows_by_optimizer(
                    flow_fits, optimizer,
                    figure_root / f"optimizer_views/{optimizer}_checkpoint_flows.png",
                ),
                plot_jacobian_transport_by_optimizer(
                    flow_transport, optimizer,
                    figure_root / f"optimizer_views/{optimizer}_jacobian_transport.png",
                ),
            ])
        figures.extend(plot_spectral_galleries(spectra, figure_root))
        inventory = {
            "primary Jacobian fit rows": len(primary),
            "all Jacobian fit rows": len(fits),
            "saved Jacobian spectral modes": len(spectra),
            "operator metadata rows": len(operators),
            "WeightWatcher control rows": len(weightwatcher),
            "performance rows": len(performance),
            "weight-quotient WeightWatcher rows": len(quotient_fits),
            "weight-quotient spectral modes": len(quotient_spectra),
            "two-checkpoint flow fit rows": len(flow_fits),
            "two-checkpoint flow spectral modes": len(flow_spectra),
            "Jacobian transport comparison rows": len(flow_transport),
            "figures": len(figures),
        }
        index = build_html(report_root, figures, tables, primary, coverage, inventory)
        (report_root / "report_manifest.json").write_text(
            json.dumps({
                "analysis_root": str(analysis_root),
                "report_root": str(report_root),
                "inventory": inventory,
                "figures": [str(path) for path in figures],
                "tables": [str(path) for path in tables],
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Report complete: %s", index)
        print(index, flush=True)
        return 0
    except Exception:
        logger.exception("Report generation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

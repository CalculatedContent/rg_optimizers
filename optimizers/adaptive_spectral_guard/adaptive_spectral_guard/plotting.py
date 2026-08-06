"""Plot helpers for AdaptiveSpectralGuard experiments."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


BASELINE = "AdamW baseline"
GUARD = "AdamW + AdaptiveSpectralGuard"
RUN_COLORS = {BASELINE: "#2563A6", GUARD: "#238B57"}
BLUE = {"fc1": "#9ECAE1", "fc2": "#4292C6", "fc3": "#08519C"}
GREEN = {"fc1": "#A1D99B", "fc2": "#41AB5D", "fc3": "#006D2C"}
MARKERS = {"fc1": "o", "fc2": "s", "fc3": "^"}


def _finish(ax, title, xlabel, ylabel, reference=None):
    if reference is not None:
        ax.axhline(
            reference,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label=f"reference = {reference:g}",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_performance(performance: pd.DataFrame) -> None:
    for metric, title, ylabel in (
        ("train_acc", "Training accuracy", "Training accuracy"),
        ("test_acc", "Test accuracy", "Test accuracy"),
        ("train_loss", "Training cross-entropy loss", "Training loss"),
        ("test_loss", "Test cross-entropy loss", "Test loss"),
    ):
        fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=135)
        for run in (BASELINE, GUARD):
            group = performance.loc[
                performance["run"].eq(run)
            ].sort_values("epoch")
            ax.plot(
                group["epoch"],
                group[metric],
                marker="o",
                markersize=4,
                linewidth=2.4,
                color=RUN_COLORS[run],
                label=run,
            )
        _finish(ax, title, "Epoch", ylabel)


def _prepare_ww(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.loc[history["status"].eq("ok")].copy()
    frame["layer"] = (
        frame["layer_name"].astype(str).str.split(".").str[-1]
    )
    return frame.loc[frame["layer"].isin(["fc1", "fc2", "fc3"])]


def plot_weightwatcher(
    history: pd.DataFrame,
    metric: str,
    *,
    reference: float,
    title: str,
    ylabel: str,
) -> None:
    frame = _prepare_ww(history)
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=135)
    for run in (BASELINE, GUARD):
        palette = BLUE if run == BASELINE else GREEN
        for layer in ("fc1", "fc2", "fc3"):
            group = frame.loc[
                frame["run"].eq(run) & frame["layer"].eq(layer)
            ].sort_values("epoch")
            ax.plot(
                group["epoch"],
                group[metric],
                marker=MARKERS[layer],
                markersize=4,
                linewidth=2.1,
                color=palette[layer],
                label=f"{run} — {layer.upper()}",
            )
    _finish(ax, title, "Epoch", ylabel, reference)

    for layer in ("fc1", "fc2", "fc3"):
        fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=135)
        for run in (BASELINE, GUARD):
            group = frame.loc[
                frame["run"].eq(run) & frame["layer"].eq(layer)
            ].sort_values("epoch")
            ax.plot(
                group["epoch"],
                group[metric],
                marker="o",
                markersize=4,
                linewidth=2.4,
                color=RUN_COLORS[run],
                label=run,
            )
        _finish(
            ax,
            f"{title}: {layer.upper()}",
            "Epoch",
            ylabel,
            reference,
        )


def plot_controller(controller: pd.DataFrame) -> None:
    if controller.empty:
        return
    frame = controller.copy()
    frame["layer"] = (
        frame["parameter"].astype(str)
        .str.replace(".weight", "", regex=False)
        .str.split(".")
        .str[-1]
    )
    regime_value = {"off": 0.0, "weak": 1.0, "strong": 2.0}
    frame["regime_value"] = frame["regime"].map(regime_value)

    for metric, title, ylabel in (
        ("effective_gain", "Adaptive gain by layer", "Effective gain"),
        ("confidence", "ECS confidence by layer", "Confidence"),
        ("task_throttle", "Task-conflict throttle by layer", "Throttle"),
        ("regime_value", "Controller regime by layer", "0=off, 1=weak, 2=strong"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=135)
        for layer in ("fc1", "fc2", "fc3"):
            group = frame.loc[
                frame["layer"].eq(layer)
            ].sort_values("epoch")
            ax.plot(
                group["epoch"],
                group[metric],
                marker=MARKERS[layer],
                linewidth=2.4,
                color=GREEN[layer],
                label=layer.upper(),
            )
        _finish(ax, title, "Epoch", ylabel)


def plot_corrections(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    frame = summary.copy()
    frame["layer"] = (
        frame["parameter"].astype(str)
        .str.replace(".weight", "", regex=False)
        .str.split(".")
        .str[-1]
    )
    for metric, title, ylabel in (
        (
            "mean_combined_correction_ratio",
            "Mean correction relative to AdamW",
            "Mean ||correction|| / ||AdamW step||",
        ),
        (
            "mean_volume_correction_ratio",
            "Mean trace-log volume correction",
            "Mean volume correction ratio",
        ),
        (
            "mean_shape_correction_ratio",
            "Mean beta-E shape correction",
            "Mean shape correction ratio",
        ),
        (
            "mean_task_conflict_ratio_pre",
            "Attempted task conflict before safeguard",
            "<grad, correction> / |<grad, AdamW step>|",
        ),
        (
            "mean_task_conflict_ratio_post",
            "Task conflict after safeguard",
            "<grad, correction> / |<grad, AdamW step>|",
        ),
        (
            "harmful_attempt_fraction",
            "Fraction of attempted corrections that oppose descent",
            "Harmful-attempt fraction",
        ),
    ):
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=135)
        for layer in ("fc1", "fc2", "fc3"):
            group = frame.loc[
                frame["layer"].eq(layer)
            ].sort_values("epoch")
            ax.plot(
                group["epoch"],
                group[metric],
                marker=MARKERS[layer],
                linewidth=2.4,
                color=GREEN[layer],
                label=layer.upper(),
            )
        _finish(ax, title, "Epoch", ylabel)


def plot_matched_convergence(
    performance: pd.DataFrame,
    weightwatcher: pd.DataFrame,
) -> None:
    """Compare generalization at matched training progress, not matched epoch."""

    pairs = (
        ("train_loss", "test_loss", "Test loss versus train loss"),
        ("train_acc", "test_acc", "Test accuracy versus train accuracy"),
    )
    for x, y, title in pairs:
        fig, ax = plt.subplots(figsize=(7.5, 6), dpi=135)
        for run in (BASELINE, GUARD):
            group = performance.loc[
                performance["run"].eq(run) & performance["epoch"].ge(1)
            ].sort_values("epoch")
            ax.plot(
                group[x],
                group[y],
                marker="o",
                markersize=4,
                linewidth=2.4,
                color=RUN_COLORS[run],
                label=run,
            )
        _finish(ax, title, x, y)

    ww = _prepare_ww(weightwatcher)
    perf = performance[["run", "epoch", "train_loss", "train_acc"]]
    merged = ww.merge(perf, on=["run", "epoch"], how="inner")
    for layer in ("fc1", "fc2", "fc3"):
        fig, ax = plt.subplots(figsize=(7.5, 6), dpi=135)
        for run in (BASELINE, GUARD):
            group = merged.loc[
                merged["run"].eq(run)
                & merged["layer"].eq(layer)
                & merged["epoch"].ge(1)
            ].sort_values("train_loss")
            ax.plot(
                group["train_loss"],
                group["alpha"],
                marker="o",
                markersize=4,
                linewidth=2.4,
                color=RUN_COLORS[run],
                label=run,
            )
        ax.axhline(2.0, color="black", linestyle="--", linewidth=1.4)
        _finish(
            ax,
            f"{layer.upper()} alpha at matched train loss",
            "Training loss",
            "WeightWatcher alpha",
        )

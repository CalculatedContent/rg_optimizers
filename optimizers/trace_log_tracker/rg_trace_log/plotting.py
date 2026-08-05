"""Plot helpers for the MNIST trace-log RG experiment."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_accuracy(performance):
    for metric, title in (("train_acc", "Train accuracy"), ("test_acc", "Test accuracy")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for run, group in performance.groupby("run"):
            ax.plot(group["epoch"], group[metric], marker="o", label=run)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.show()


def plot_weightwatcher_metric(history, column, title, ylabel, reference=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = history[history["status"] == "ok"]
    for (run, layer), group in valid.groupby(["run", "layer_name"]):
        ax.plot(group["epoch"], group[column], marker="o", label=f"{run}: {layer}")
    if reference is not None:
        ax.axhline(reference, linestyle="--", label=f"reference = {reference:g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    plt.show()


def plot_beta(history):
    reliable = history[
        (history["status"] == "ok")
        & history["scale_balance_reliable"].fillna(False)
    ]
    plot_weightwatcher_metric(
        reliable,
        "beta_E_midpoint",
        "Midpoint logarithmic-shell scale balance",
        "midpoint beta_E",
        reference=0.0,
    )
    plot_weightwatcher_metric(
        reliable,
        "shell_energy_rms_midpoint",
        "Full midpoint shell-energy imbalance",
        "log shell-energy RMS",
    )


def plot_correction_summary(summary):
    if summary.empty:
        print("No RG corrections were applied.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for parameter, group in summary.groupby("parameter"):
        ax.plot(
            group["epoch"],
            group["mean_correction_ratio"],
            marker="o",
            label=parameter,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mean ||RG correction|| / ||AdamW step||")
    ax.set_title("Trace-log RG correction size")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()

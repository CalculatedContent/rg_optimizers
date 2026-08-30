#!/usr/bin/env python3
"""Build a live HTML report from an incomplete or completed long run."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATRIX_TYPES = ("W_Q", "W_K", "W_V", "W_O", "W_MLP_IN", "W_MLP_OUT")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    for attempt in range(2):
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            if attempt:
                raise
            time.sleep(0.1)
    return pd.DataFrame()


def save_training_plot(metrics: pd.DataFrame, output: Path) -> None:
    frame = metrics.sort_values("step").copy()
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    axis = axes[0, 0]
    axis.plot(frame["epoch"], frame["train_loss"], label="train")
    axis.plot(frame["epoch"], frame["val_loss"], label="validation")
    axis.set(title="Loss", xlabel="Corpus-equivalent epoch", ylabel="NLL")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.plot(frame["epoch"], frame["val_perplexity"], color="#D55E00")
    axis.set(title="Validation perplexity", xlabel="Corpus-equivalent epoch")

    axis = axes[1, 0]
    for column, label in (
        ("primary_lr", "learning rate"),
        ("grad_norm_pre_clip", "gradient norm"),
        ("update_to_weight_ratio", "update / weight"),
    ):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            axis.plot(frame["epoch"], values, label=label)
    axis.set_yscale("log")
    axis.set(title="Optimization", xlabel="Corpus-equivalent epoch")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    axis.plot(frame["epoch"], frame["tokens_per_sec"], label="tokens / sec")
    memory = pd.to_numeric(
        frame.get("mps_driver_allocated_mb", pd.Series(dtype=float)),
        errors="coerce",
    )
    if len(memory) == len(frame) and np.isfinite(memory).any():
        twin = axis.twinx()
        twin.plot(frame["epoch"], memory, color="#CC79A7", label="MPS MiB")
        twin.set_ylabel("MPS driver MiB")
    axis.set(title="Throughput and memory", xlabel="Corpus-equivalent epoch")
    axis.set_ylabel("Tokens / sec")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_spectral_plot(layers: pd.DataFrame, metric: str, output: Path) -> None:
    frame = layers.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame["block"] = pd.to_numeric(frame["block"], errors="coerce")
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 6))
    for axis, matrix_type in zip(axes.flat, MATRIX_TYPES, strict=True):
        subset = frame[frame["matrix_type"] == matrix_type]
        for block, block_frame in subset.groupby("block", sort=True):
            block_index = int(block)
            axis.plot(
                block_frame["epoch"],
                block_frame[metric],
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                color=colors[block_index % len(colors)],
                label=f"block {block_index}",
            )
        if metric in {"alpha_raw", "alpha_clip_xmax"}:
            axis.axhline(2.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(matrix_type)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=6,
            frameon=False,
        )
    figure.suptitle(f"{metric}: each transformer block", y=0.998)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_qk_plot(frame: pd.DataFrame, output: Path) -> None:
    data = frame.sort_values("step").copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(data["step"], data["mean_max_logit"], label="mean max logit")
    axes[0].plot(data["step"], data["max_logit"], label="maximum logit")
    axes[0].axhline(
        float(data["threshold"].iloc[-1]),
        color="black",
        linestyle="--",
        label="clip threshold",
    )
    axes[0].legend(frameon=False)
    axes[0].set(title="QK logits", xlabel="Optimizer step")
    axes[1].plot(data["step"], data["active_fraction"], label="active fraction")
    axes[1].plot(data["step"], data["min_gamma"], label="minimum gamma")
    axes[1].set(title="QK clipping", xlabel="Optimizer step")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def build_report(experiment_root: Path, output: Path) -> Path:
    run_dir = experiment_root / "results" / "muon_clip" / "seed_20260830"
    metrics = read_csv(run_dir / "metrics.csv")
    layers = read_csv(run_dir / "spectral" / "layers.csv")
    qk = read_csv(run_dir / "muonclip_qk.csv")
    plots = output.parent / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    images: list[tuple[str, str]] = []
    if not metrics.empty:
        save_training_plot(metrics, plots / "training.png")
        images.append(("Training and validation", "plots/training.png"))
    if not layers.empty:
        for metric in (
            "alpha_raw",
            "alpha_clip_xmax",
            "ERG_gap",
            "rand_distance",
            "num_traps",
        ):
            if metric in layers:
                filename = f"{metric}.png"
                save_spectral_plot(layers, metric, plots / filename)
                images.append((f"WeightWatcher {metric}", f"plots/{filename}"))
    if not qk.empty:
        save_qk_plot(qk, plots / "qk_clip.png")
        images.append(("MuonClip QK diagnostics", "plots/qk_clip.png"))

    summary = "No evaluation row has been written yet."
    if not metrics.empty:
        row = metrics.sort_values("step").iloc[-1]
        summary = (
            f"Latest step: {int(row['step']):,}; epoch: {float(row['epoch']):.4f}; "
            f"train loss: {float(row['train_loss']):.4f}; "
            f"validation loss: {float(row['val_loss']):.4f}; "
            f"validation perplexity: {float(row['val_perplexity']):.2f}."
        )
    sections = "\n".join(
        f"<h2>{html.escape(title)}</h2><img src='{html.escape(source)}' alt='{html.escape(title)}'>"
        for title, source in images
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Large MuonClip nanoGPT live report</title>
<style>body{{max-width:1500px;margin:auto;padding:2rem;font-family:system-ui,sans-serif;color:#1f2937}}img{{display:block;width:100%;height:auto;border:1px solid #ddd;margin-bottom:2rem}}.note{{background:#f3f4f6;border-left:4px solid #555;padding:1rem}}</style></head>
<body><h1>Large MuonClip nanoGPT live report</h1>
<p class="note">Single seed 20260830; 6 blocks; 8 heads per block; width 384; context 512; 512M training tokens. This report is a non-destructive snapshot and may be regenerated while training continues.</p>
<p>{html.escape(summary)}</p>
{sections if sections else '<p>No plot-ready rows are available yet.</p>'}
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_report(args.experiment_root, args.output))


if __name__ == "__main__":
    main()

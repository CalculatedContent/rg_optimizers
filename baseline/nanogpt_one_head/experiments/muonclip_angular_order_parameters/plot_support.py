from __future__ import annotations

"""Plots and manifests for MuonClip angular quotient experiments."""

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finish_figure(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def metric_grid(
    *,
    method: str,
    metric: str,
    ylabel: str,
    records: list[dict[str, Any]],
    matrix_names: list[str],
    actual: pd.DataFrame,
    null_summary: pd.DataFrame | None,
    path: Path,
    reference: float | None = None,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    steps = np.asarray([record["step"] for record in records], dtype=float)
    for axis, matrix_name in zip(axes.ravel(), matrix_names):
        subset = actual[
            (actual["method"] == method)
            & (actual["matrix_name"] == matrix_name)
        ].sort_values("checkpoint_index")
        axis.plot(
            subset["step"],
            pd.to_numeric(subset.get(metric), errors="coerce"),
            "o-",
            label="actual",
        )
        if null_summary is not None:
            null = null_summary[
                (null_summary["method"] == method)
                & (null_summary["matrix_name"] == matrix_name)
            ].sort_values("checkpoint_index")
            if not null.empty:
                null = null.set_index("checkpoint_index").reindex(
                    range(len(records))
                )
                low = pd.to_numeric(null["q025"], errors="coerce").to_numpy(float)
                median = pd.to_numeric(
                    null["median"], errors="coerce"
                ).to_numpy(float)
                high = pd.to_numeric(null["q975"], errors="coerce").to_numpy(float)
                finite = np.isfinite(low) & np.isfinite(median) & np.isfinite(high)
                if finite.any():
                    axis.fill_between(
                        steps[finite], low[finite], high[finite], alpha=0.18,
                        label="matched null 95%",
                    )
                    axis.plot(
                        steps[finite], median[finite], linestyle=":",
                        linewidth=1.0, label="null median",
                    )
        if reference is not None:
            axis.axhline(
                reference,
                linestyle="--",
                linewidth=1.2,
                label=f"reference={reference:g}",
            )
        axis.set_title(matrix_name)
        axis.set_xlabel("training step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle(f"{method}: {ylabel}", fontsize=16)
    return finish_figure(fig, path)


def normalized_esd(matrix: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    values = singular**2
    values = values[np.isfinite(values) & (values > 1e-14)]
    return np.sort(values / np.mean(values)) if values.size else values


def esd_overlay(
    *,
    method: str,
    matrix_name: str,
    matrices: list[np.ndarray],
    records: list[dict[str, Any]],
    path: Path,
) -> Path | None:
    spectra = [normalized_esd(matrix) for matrix in matrices]
    positive = [values for values in spectra if values.size]
    if not positive:
        return None
    all_values = np.concatenate(positive)
    edges = np.geomspace(
        max(float(all_values.min()), 1e-8),
        float(all_values.max()) * 1.02,
        22,
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    fig, axis = plt.subplots(figsize=(8.8, 5.7))
    selected = {0, len(records) // 2, len(records) - 1}
    for index, (record, values) in enumerate(zip(records, spectra)):
        if not values.size:
            continue
        density, _ = np.histogram(values, bins=edges, density=True)
        mask = density > 0
        opacity = 0.35 + 0.65 * index / max(len(records) - 1, 1)
        label = f"step {record['step']}" if index in selected else None
        axis.loglog(
            centers[mask], density[mask], "o-", linewidth=1.0,
            alpha=opacity, label=label,
        )
    axis.set_xlabel("normalized Gram eigenvalue")
    axis.set_ylabel("density")
    axis.set_title(f"{method} / {matrix_name}: ESD flow")
    axis.grid(alpha=0.2, which="both")
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=8)
    return finish_figure(fig, path)


def contact_sheet(
    *,
    method: str,
    records: list[dict[str, Any]],
    matrix_names: list[str],
    metrics: pd.DataFrame,
    path: Path,
) -> Path:
    fig, axes = plt.subplots(
        len(records),
        len(matrix_names),
        figsize=(4.0 * len(matrix_names), 3.0 * len(records)),
        squeeze=False,
    )
    for row_index, record in enumerate(records):
        for column_index, matrix_name in enumerate(matrix_names):
            axis = axes[row_index, column_index]
            axis.axis("off")
            match = metrics[
                (metrics["method"] == method)
                & (metrics["checkpoint_index"] == row_index)
                & (metrics["matrix_name"] == matrix_name)
            ]
            image = None
            if not match.empty:
                text = str(match.iloc[0].get("native_esd_file", ""))
                image_path = Path(text) if text else None
                if image_path and image_path.is_file():
                    image = mpimg.imread(image_path)
            if image is None:
                axis.text(0.5, 0.5, "no ESD", ha="center", va="center")
            else:
                axis.imshow(image)
            if row_index == 0:
                axis.set_title(matrix_name, fontsize=9)
            if column_index == 0:
                axis.text(
                    -0.03,
                    0.5,
                    f"step {record['step']}",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=9,
                )
    fig.suptitle(f"Native WeightWatcher ESD trajectory — {method}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

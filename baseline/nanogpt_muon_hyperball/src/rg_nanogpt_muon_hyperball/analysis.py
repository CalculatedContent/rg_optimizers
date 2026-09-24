from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SUPPORTED_OPTIMIZERS
from .run_utils import run_directory, run_is_complete

OPTIMIZER_LABELS = {
    "muon": "Long Muon + auxiliary AdamW",
    "muon_hyperball": "Muon + HyperBall + auxiliary AdamW",
}
OPTIMIZER_COLORS = {
    "muon": "#009E73",
    "muon_hyperball": "#0072B2",
}
MATRIX_COLORS = {
    "W_Q": "#0072B2",
    "W_K": "#E69F00",
    "W_V": "#009E73",
    "W_O": "#D55E00",
    "W_MLP_IN": "#CC79A7",
    "W_MLP_OUT": "#56B4E9",
}

_T_975 = {
    1: 12.7062047364,
    2: 4.3026527297,
    3: 3.1824463053,
    4: 2.7764451052,
    5: 2.5705818356,
    6: 2.4469118511,
    7: 2.3646242510,
    8: 2.3060041352,
    9: 2.2621571629,
    10: 2.2281388520,
}


def mean_ci95(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
        }
    mean = float(array.mean())
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "sd": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
        }
    sd = float(array.std(ddof=1))
    sem = sd / math.sqrt(n)
    half = _T_975.get(n - 1, 1.9599639845) * sem
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_half_width": half,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _require_complete(results_root: str | Path, optimizer: str, seed: int) -> None:
    if not run_is_complete(results_root, optimizer, seed):
        raise FileNotFoundError(
            f"missing completed run for optimizer={optimizer} seed={seed}: "
            f"{run_directory(results_root, optimizer, seed)}"
        )


def _load_csvs(
    results_root: str | Path,
    relative_path: str,
    *,
    optimizers: Sequence[str],
    seeds: Sequence[int],
    require_complete: bool,
) -> pd.DataFrame:
    frames = []
    for optimizer in optimizers:
        for seed in seeds:
            if require_complete:
                _require_complete(results_root, optimizer, seed)
            path = run_directory(results_root, optimizer, seed) / relative_path
            if not path.is_file():
                if require_complete:
                    raise FileNotFoundError(path)
                continue
            frame = pd.read_csv(path)
            frame["optimizer"] = optimizer
            frame["optimizer_label"] = OPTIMIZER_LABELS.get(optimizer, optimizer)
            frame["seed"] = int(seed)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def run_status_table(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337,),
) -> pd.DataFrame:
    rows = []
    for optimizer in optimizers:
        for seed in seeds:
            run_dir = run_directory(results_root, optimizer, seed)
            path = run_dir / "run_complete.json"
            payload = json.loads(path.read_text()) if path.is_file() else {}
            rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS.get(optimizer, optimizer),
                    "seed": int(seed),
                    "complete": bool(payload.get("completed", False)),
                    "steps": payload.get("optimizer_steps", np.nan),
                    "final_test_loss": payload.get("final_test_loss", np.nan),
                    "final_test_accuracy": payload.get("final_test_accuracy", np.nan),
                    "run_dir": str(run_dir),
                }
            )
    return pd.DataFrame(rows)


def load_metrics(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337,),
    require_complete: bool = True,
) -> pd.DataFrame:
    frame = _load_csvs(
        results_root,
        "metrics.csv",
        optimizers=optimizers,
        seeds=seeds,
        require_complete=require_complete,
    )
    if frame.empty:
        return frame
    return frame.sort_values(["optimizer", "seed", "step"]).drop_duplicates(
        ["optimizer", "seed", "step"], keep="last"
    )


def load_epoch_metrics(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337,),
    require_complete: bool = True,
) -> pd.DataFrame:
    frame = _load_csvs(
        results_root,
        "epoch_metrics.csv",
        optimizers=optimizers,
        seeds=seeds,
        require_complete=require_complete,
    )
    if frame.empty:
        return frame
    return frame.sort_values(
        ["optimizer", "seed", "nominal_epoch"]
    ).drop_duplicates(["optimizer", "seed", "nominal_epoch"], keep="last")


def load_layer_metrics(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337,),
    require_complete: bool = True,
) -> pd.DataFrame:
    frame = _load_csvs(
        results_root,
        "spectral/layers.csv",
        optimizers=optimizers,
        seeds=seeds,
        require_complete=require_complete,
    )
    if frame.empty:
        return frame
    return frame.sort_values(
        ["optimizer", "seed", "epoch", "matrix_type"]
    ).drop_duplicates(
        ["optimizer", "seed", "step", "matrix_name"], keep="last"
    )


def load_spectral_summary(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337,),
    require_complete: bool = True,
) -> pd.DataFrame:
    frame = _load_csvs(
        results_root,
        "spectral/summary.csv",
        optimizers=optimizers,
        seeds=seeds,
        require_complete=require_complete,
    )
    if frame.empty:
        return frame
    return frame.sort_values(["optimizer", "seed", "epoch"]).drop_duplicates(
        ["optimizer", "seed", "step"], keep="last"
    )


def load_test_results(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337,),
) -> pd.DataFrame:
    rows = []
    for optimizer in optimizers:
        for seed in seeds:
            _require_complete(results_root, optimizer, seed)
            path = run_directory(results_root, optimizer, seed) / "test_results.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for checkpoint in ("final", "validation_selected"):
                values = payload[checkpoint]
                rows.append(
                    {
                        "optimizer": optimizer,
                        "optimizer_label": OPTIMIZER_LABELS[optimizer],
                        "seed": int(seed),
                        "checkpoint": checkpoint,
                        "step": int(values["step"]),
                        "test_loss": float(values["loss"]),
                        "test_perplexity": float(values["perplexity"]),
                        "test_accuracy": float(values["accuracy"]),
                        "test_bleu": float(values["bleu"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_by_epoch(
    frame: pd.DataFrame,
    metric: str,
    *,
    x: str = "nominal_epoch",
    group: Sequence[str] = ("optimizer",),
) -> pd.DataFrame:
    rows = []
    keys = [*group, x]
    subset = frame[[*keys, "seed", metric]].copy()
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    for values, group_frame in subset.groupby(keys, sort=True):
        values_tuple = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values_tuple, strict=True))
        row.update(mean_ci95(group_frame[metric]))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_epoch_metric(
    frame: pd.DataFrame,
    *,
    metric: str,
    x: str = "nominal_epoch",
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    title: str | None = None,
    output: str | Path | None = None,
):
    figure, axis = plt.subplots(figsize=(9, 5))
    for optimizer in optimizers:
        subset = frame[frame["optimizer"] == optimizer]
        if subset.empty:
            continue
        for _, seed_frame in subset.groupby("seed"):
            axis.plot(
                seed_frame[x],
                seed_frame[metric],
                color=OPTIMIZER_COLORS[optimizer],
                alpha=0.30,
                linewidth=1.0,
            )
        summary = summarize_by_epoch(subset, metric, x=x)
        axis.plot(
            summary[x],
            summary["mean"],
            color=OPTIMIZER_COLORS[optimizer],
            linewidth=2.0,
            label=OPTIMIZER_LABELS[optimizer],
        )
        if summary["ci95_lower"].notna().any():
            axis.fill_between(
                summary[x],
                summary["ci95_lower"],
                summary["ci95_upper"],
                color=OPTIMIZER_COLORS[optimizer],
                alpha=0.16,
            )
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(title or metric.replace("_", " ").title())
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=170, bbox_inches="tight")
    return figure, axis


def plot_layer_metric(
    frame: pd.DataFrame,
    *,
    optimizer: str,
    metric: str,
    title: str | None = None,
    output: str | Path | None = None,
):
    subset = frame[frame["optimizer"] == optimizer].copy()
    if subset.empty:
        raise ValueError(f"no layer data for optimizer={optimizer}")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for matrix_type, color in MATRIX_COLORS.items():
        matrix = subset[subset["matrix_type"] == matrix_type]
        if matrix.empty:
            continue
        summary = summarize_by_epoch(
            matrix, metric, x="epoch", group=("matrix_type",)
        )
        axis.plot(
            summary["epoch"],
            summary["mean"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=matrix_type,
        )
        if summary["ci95_lower"].notna().any():
            axis.fill_between(
                summary["epoch"],
                summary["ci95_lower"],
                summary["ci95_upper"],
                color=color,
                alpha=0.13,
            )
    if metric == "alpha":
        axis.axhline(2.0, linestyle="--", linewidth=1.0, label="alpha = 2")
    if metric == "ERG_gap":
        axis.axhline(0.0, linestyle="--", linewidth=1.0)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.set_title(title or f"{OPTIMIZER_LABELS[optimizer]} layer {metric}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=170, bbox_inches="tight")
    return figure, axis


def final_test_summary(test_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (optimizer, checkpoint), group in test_results.groupby(
        ["optimizer", "checkpoint"]
    ):
        for metric in (
            "test_loss",
            "test_perplexity",
            "test_accuracy",
            "test_bleu",
        ):
            rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS[optimizer],
                    "checkpoint": checkpoint,
                    "metric": metric,
                    **mean_ci95(group[metric]),
                }
            )
    return pd.DataFrame(rows)

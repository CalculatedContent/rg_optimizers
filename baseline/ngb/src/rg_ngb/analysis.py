from __future__ import annotations

from itertools import combinations
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
    "sgd_momentum": "SGD + Nesterov",
    "adamw": "AdamW",
    "muon": "Muon + auxiliary AdamW",
}
OPTIMIZER_COLORS = {
    "sgd_momentum": "#0072B2",
    "adamw": "#D55E00",
    "muon": "#009E73",
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
    11: 2.2009851601,
    12: 2.1788128297,
    13: 2.1603686565,
    14: 2.1447866879,
    15: 2.1314495456,
    16: 2.1199052992,
    17: 2.1098155778,
    18: 2.1009220402,
    19: 2.0930240544,
    20: 2.0859634473,
    21: 2.0796138447,
    22: 2.0738730679,
    23: 2.0686576104,
    24: 2.0638985616,
    25: 2.0595385528,
    26: 2.0555294386,
    27: 2.0518305165,
    28: 2.0484071418,
    29: 2.0452296421,
    30: 2.0422724563,
}


def mean_ci95(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {key: np.nan for key in ("mean", "sd", "sem", "ci95_half_width", "ci95_lower", "ci95_upper")} | {"n": 0}
    mean = float(array.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "sd": np.nan, "sem": np.nan, "ci95_half_width": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan}
    sd = float(array.std(ddof=1))
    sem = sd / math.sqrt(n)
    critical = _T_975.get(n - 1, 1.9599639845)
    half = critical * sem
    return {"n": n, "mean": mean, "sd": sd, "sem": sem, "ci95_half_width": half, "ci95_lower": mean - half, "ci95_upper": mean + half}


def completed_seed_sets(results_root: str | Path, optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS) -> dict[str, set[int]]:
    root = Path(results_root)
    result: dict[str, set[int]] = {}
    for optimizer in optimizers:
        seeds: set[int] = set()
        optimizer_root = root / optimizer
        if optimizer_root.is_dir():
            for directory in optimizer_root.glob("seed_*"):
                try:
                    seed = int(directory.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                if run_is_complete(root, optimizer, seed):
                    seeds.add(seed)
        result[optimizer] = seeds
    return result


def discover_matched_seeds(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    requested: Sequence[int] | None = None,
    minimum: int = 2,
) -> tuple[int, ...]:
    sets = completed_seed_sets(results_root, optimizers)
    matched = set.intersection(*(sets[optimizer] for optimizer in optimizers)) if optimizers else set()
    if requested is not None:
        requested_set = {int(seed) for seed in requested}
        missing = requested_set.difference(matched)
        if missing:
            raise FileNotFoundError(f"requested seeds are not complete for every optimizer: {sorted(missing)}")
        matched = requested_set
    result = tuple(sorted(matched))
    if len(result) < int(minimum):
        detail = {key: sorted(value) for key, value in sets.items()}
        raise FileNotFoundError(f"need at least {minimum} matched completed seeds; found {result}; per optimizer={detail}")
    return result


def run_status_table(results_root: str | Path, *, optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS, seeds: Sequence[int] | None = None) -> pd.DataFrame:
    sets = completed_seed_sets(results_root, optimizers)
    if seeds is None:
        seeds = sorted(set().union(*sets.values())) if sets else []
    rows = []
    for optimizer in optimizers:
        for seed in seeds:
            run_dir = run_directory(results_root, optimizer, int(seed))
            completion_path = run_dir / "run_complete.json"
            payload = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else {}
            rows.append({
                "optimizer": optimizer,
                "optimizer_label": OPTIMIZER_LABELS.get(optimizer, optimizer),
                "seed": int(seed),
                "complete": int(seed) in sets.get(optimizer, set()),
                "steps": payload.get("optimizer_steps", np.nan),
                "best_validation_loss": payload.get("best_validation_loss", np.nan),
                "final_test_loss": payload.get("final_test_loss", np.nan),
                "run_dir": str(run_dir),
            })
    return pd.DataFrame(rows)


def _load_csvs(results_root: str | Path, relative_path: str, *, optimizers: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for optimizer in optimizers:
        for seed in seeds:
            if not run_is_complete(results_root, optimizer, int(seed)):
                raise FileNotFoundError(f"incomplete run: optimizer={optimizer} seed={seed}")
            path = run_directory(results_root, optimizer, int(seed)) / relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame["optimizer"] = optimizer
            frame["optimizer_label"] = OPTIMIZER_LABELS.get(optimizer, optimizer)
            frame["seed"] = int(seed)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_metrics(results_root: str | Path, *, optimizers: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    frame = _load_csvs(results_root, "metrics.csv", optimizers=optimizers, seeds=seeds)
    return frame.sort_values(["optimizer", "seed", "step"]).drop_duplicates(["optimizer", "seed", "step"], keep="last")


def load_epoch_metrics(results_root: str | Path, *, optimizers: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    frame = _load_csvs(results_root, "epoch_metrics.csv", optimizers=optimizers, seeds=seeds)
    return frame.sort_values(["optimizer", "seed", "nominal_epoch"]).drop_duplicates(["optimizer", "seed", "nominal_epoch"], keep="last")


def load_layer_metrics(results_root: str | Path, *, optimizers: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    frame = _load_csvs(results_root, "spectral/layers.csv", optimizers=optimizers, seeds=seeds)
    return frame.sort_values(["optimizer", "seed", "epoch", "block", "matrix_type"]).drop_duplicates(["optimizer", "seed", "step", "matrix_name"], keep="last")


def load_spectral_summary(results_root: str | Path, *, optimizers: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    frame = _load_csvs(results_root, "spectral/summary.csv", optimizers=optimizers, seeds=seeds)
    return frame.sort_values(["optimizer", "seed", "epoch"]).drop_duplicates(["optimizer", "seed", "step"], keep="last")


def load_test_results(results_root: str | Path, *, optimizers: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for optimizer in optimizers:
        for seed in seeds:
            if not run_is_complete(results_root, optimizer, int(seed)):
                raise FileNotFoundError(f"incomplete run: optimizer={optimizer} seed={seed}")
            path = run_directory(results_root, optimizer, int(seed)) / "test_results.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for checkpoint in ("final", "validation_selected"):
                values = payload[checkpoint]
                rows.append({
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS[optimizer],
                    "seed": int(seed),
                    "checkpoint": checkpoint,
                    "step": int(values["step"]),
                    "test_loss": float(values["loss"]),
                    "test_accuracy": float(values["accuracy"]),
                    "test_bleu": float(values["bleu"]),
                })
    return pd.DataFrame(rows)


def summarize_by_epoch(frame: pd.DataFrame, metric: str, *, x: str = "nominal_epoch", group: Sequence[str] = ("optimizer",)) -> pd.DataFrame:
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


def final_test_summary(test_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (optimizer, checkpoint), group in test_results.groupby(["optimizer", "checkpoint"]):
        loss_stats = mean_ci95(group["test_loss"])
        for metric in ("test_loss", "test_accuracy", "test_bleu"):
            rows.append({
                "optimizer": optimizer,
                "optimizer_label": OPTIMIZER_LABELS[optimizer],
                "checkpoint": checkpoint,
                "metric": metric,
                "interval_method": "run_level_student_t",
                **mean_ci95(group[metric]),
            })
        rows.append({
            "optimizer": optimizer,
            "optimizer_label": OPTIMIZER_LABELS[optimizer],
            "checkpoint": checkpoint,
            "metric": "test_perplexity",
            "interval_method": "exp_of_loss_space_student_t_interval",
            "n": loss_stats["n"],
            "mean": math.exp(loss_stats["mean"]),
            "sd": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_lower": math.exp(loss_stats["ci95_lower"]),
            "ci95_upper": math.exp(loss_stats["ci95_upper"]),
        })
    return pd.DataFrame(rows)


def paired_optimizer_differences(test_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    optimizers = tuple(sorted(test_results["optimizer"].unique()))
    for checkpoint in ("final", "validation_selected"):
        checkpoint_frame = test_results[test_results["checkpoint"].eq(checkpoint)]
        for left, right in combinations(optimizers, 2):
            for metric in ("test_loss", "test_accuracy", "test_bleu"):
                a = checkpoint_frame[checkpoint_frame["optimizer"].eq(left)][["seed", metric]].rename(columns={metric: "left"})
                b = checkpoint_frame[checkpoint_frame["optimizer"].eq(right)][["seed", metric]].rename(columns={metric: "right"})
                paired = a.merge(b, on="seed", validate="one_to_one")
                stats = mean_ci95(paired["left"] - paired["right"])
                rows.append({
                    "checkpoint": checkpoint,
                    "contrast": f"{OPTIMIZER_LABELS[left]} - {OPTIMIZER_LABELS[right]}",
                    "left_optimizer": left,
                    "right_optimizer": right,
                    "metric": metric,
                    **stats,
                })
    return pd.DataFrame(rows)


def run_diagnostics(results_root: str | Path, *, optimizers: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for optimizer in optimizers:
        for seed in seeds:
            run_dir = run_directory(results_root, optimizer, int(seed))
            completion = json.loads((run_dir / "run_complete.json").read_text(encoding="utf-8"))
            metrics = pd.read_csv(run_dir / "metrics.csv").sort_values("step")
            final = metrics.iloc[-1]
            clipped = pd.to_numeric(metrics["gradient_clipped"], errors="coerce").dropna()
            ratios = pd.to_numeric(metrics["update_to_weight_ratio"], errors="coerce")
            rows.append({
                "optimizer": optimizer,
                "optimizer_label": OPTIMIZER_LABELS[optimizer],
                "seed": int(seed),
                "best_validation_step": int(completion["best_validation_step"]),
                "best_validation_loss": float(completion["best_validation_loss"]),
                "final_validation_loss": float(final["val_loss"]),
                "final_minus_best_validation_loss": float(final["val_loss"] - completion["best_validation_loss"]),
                "final_validation_accuracy": float(final["val_accuracy"]),
                "gradient_clipped_fraction": float(clipped.mean()) if not clipped.empty else np.nan,
                "max_update_to_weight_ratio": float(ratios.max(skipna=True)),
                "elapsed_seconds": float(completion["elapsed_seconds"]),
            })
    return pd.DataFrame(rows)


def diagnostic_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "best_validation_loss",
        "final_validation_loss",
        "final_minus_best_validation_loss",
        "final_validation_accuracy",
        "gradient_clipped_fraction",
        "max_update_to_weight_ratio",
        "elapsed_seconds",
    )
    rows = []
    for optimizer, group in diagnostics.groupby("optimizer"):
        for metric in metrics:
            rows.append({
                "optimizer": optimizer,
                "optimizer_label": OPTIMIZER_LABELS[optimizer],
                "metric": metric,
                **mean_ci95(group[metric]),
            })
    return pd.DataFrame(rows)


def plot_epoch_metric(frame: pd.DataFrame, *, metric: str, x: str = "nominal_epoch", optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS, title: str | None = None, output: str | Path | None = None):
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for optimizer in optimizers:
        subset = frame[frame["optimizer"].eq(optimizer)]
        if subset.empty:
            continue
        for _, seed_frame in subset.groupby("seed"):
            axis.plot(seed_frame[x], seed_frame[metric], color=OPTIMIZER_COLORS[optimizer], alpha=0.16, linewidth=0.8)
        summary = summarize_by_epoch(subset, metric, x=x)
        axis.plot(summary[x], summary["mean"], color=OPTIMIZER_COLORS[optimizer], linewidth=2.2, label=OPTIMIZER_LABELS[optimizer])
        axis.fill_between(summary[x], summary["ci95_lower"], summary["ci95_upper"], color=OPTIMIZER_COLORS[optimizer], alpha=0.14)
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(title or f"{metric}: run-level 95% Student-t CI")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    return figure, axis


def plot_layer_metric(frame: pd.DataFrame, *, optimizer: str, metric: str, output: str | Path | None = None):
    subset = frame[frame["optimizer"].eq(optimizer)].copy()
    if subset.empty:
        raise ValueError(f"no layer data for optimizer={optimizer}")
    blocks = tuple(sorted(int(value) for value in subset["block"].unique()))
    figure, axes = plt.subplots(len(blocks), 1, figsize=(11, max(5, 4 * len(blocks))), squeeze=False, sharex=True)
    for row, block in enumerate(blocks):
        axis = axes[row, 0]
        block_frame = subset[subset["block"].eq(block)]
        for matrix_type, color in MATRIX_COLORS.items():
            matrix = block_frame[block_frame["matrix_type"].eq(matrix_type)]
            if matrix.empty:
                continue
            summary = summarize_by_epoch(matrix, metric, x="epoch", group=("matrix_type",))
            axis.plot(summary["epoch"], summary["mean"], color=color, linewidth=2.0, marker="o", markersize=3, label=matrix_type)
            axis.fill_between(summary["epoch"], summary["ci95_lower"], summary["ci95_upper"], color=color, alpha=0.12)
        if metric == "alpha":
            axis.axhline(2.0, color="black", linestyle="--", linewidth=1.0, label="alpha = 2")
        if metric == "ERG_gap":
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_ylabel(metric)
        axis.set_title(f"{OPTIMIZER_LABELS[optimizer]} block {block}")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False, ncol=3)
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    return figure, axes

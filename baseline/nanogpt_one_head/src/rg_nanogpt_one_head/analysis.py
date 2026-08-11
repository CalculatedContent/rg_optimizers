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

_BLOCK_LINESTYLES = (
    "-",
    "--",
    "-.",
    ":",
)

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
    critical = _T_975.get(
        n - 1,
        1.9599639845,
    )
    half = critical * sem
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_half_width": half,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def _available_seed_dirs(
    results_root: str | Path,
    optimizer: str,
) -> tuple[int, ...]:
    root = Path(results_root) / str(optimizer)
    seeds: list[int] = []
    if not root.is_dir():
        return ()
    for path in root.glob("seed_*"):
        if not path.is_dir():
            continue
        try:
            seeds.append(
                int(path.name.removeprefix("seed_"))
            )
        except ValueError:
            continue
    return tuple(sorted(set(seeds)))


def discover_complete_seeds(
    results_root: str | Path,
    optimizer: str,
) -> tuple[int, ...]:
    return tuple(
        seed
        for seed in _available_seed_dirs(
            results_root,
            optimizer,
        )
        if run_is_complete(
            results_root,
            optimizer,
            seed,
        )
    )


def discover_matched_complete_seeds(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
) -> tuple[int, ...]:
    sets = [
        set(
            discover_complete_seeds(
                results_root,
                optimizer,
            )
        )
        for optimizer in optimizers
    ]
    if not sets:
        return ()
    return tuple(sorted(set.intersection(*sets)))


def run_status_table(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    selected = (
        tuple(int(seed) for seed in seeds)
        if seeds is not None
        else tuple(
            sorted(
                {
                    seed
                    for optimizer in optimizers
                    for seed in _available_seed_dirs(
                        results_root,
                        optimizer,
                    )
                }
            )
        )
    )
    rows = []
    for optimizer in optimizers:
        for seed in selected:
            run_dir = run_directory(
                results_root,
                optimizer,
                seed,
            )
            completion_path = (
                run_dir / "run_complete.json"
            )
            payload = (
                json.loads(
                    completion_path.read_text(
                        encoding="utf-8"
                    )
                )
                if completion_path.is_file()
                else {}
            )
            rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_label": (
                        OPTIMIZER_LABELS.get(
                            optimizer,
                            optimizer,
                        )
                    ),
                    "seed": int(seed),
                    "complete": run_is_complete(
                        results_root,
                        optimizer,
                        seed,
                    ),
                    "steps": payload.get(
                        "optimizer_steps",
                        np.nan,
                    ),
                    "best_validation_step": payload.get(
                        "best_validation_step",
                        np.nan,
                    ),
                    "best_validation_loss": payload.get(
                        "best_validation_loss",
                        np.nan,
                    ),
                    "final_test_loss": payload.get(
                        "final_test_loss",
                        np.nan,
                    ),
                    "final_test_accuracy": payload.get(
                        "final_test_accuracy",
                        np.nan,
                    ),
                    "run_dir": str(run_dir),
                }
            )
    return pd.DataFrame(rows)


def _require_complete(
    results_root: str | Path,
    optimizer: str,
    seed: int,
) -> None:
    if not run_is_complete(
        results_root,
        optimizer,
        seed,
    ):
        raise FileNotFoundError(
            "missing completed run for "
            f"optimizer={optimizer} seed={seed}: "
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
    frames: list[pd.DataFrame] = []
    for optimizer in optimizers:
        for seed in seeds:
            if require_complete:
                _require_complete(
                    results_root,
                    optimizer,
                    seed,
                )
            path = (
                run_directory(
                    results_root,
                    optimizer,
                    seed,
                )
                / relative_path
            )
            if not path.is_file():
                if require_complete:
                    raise FileNotFoundError(path)
                continue
            frame = pd.read_csv(path)
            frame["optimizer"] = optimizer
            frame["optimizer_label"] = (
                OPTIMIZER_LABELS.get(
                    optimizer,
                    optimizer,
                )
            )
            frame["seed"] = int(seed)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def load_metrics(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
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
    return (
        frame.sort_values(
            ["optimizer", "seed", "step"]
        )
        .drop_duplicates(
            ["optimizer", "seed", "step"],
            keep="last",
        )
    )


def load_epoch_metrics(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
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
    return (
        frame.sort_values(
            [
                "optimizer",
                "seed",
                "nominal_epoch",
            ]
        )
        .drop_duplicates(
            [
                "optimizer",
                "seed",
                "nominal_epoch",
            ],
            keep="last",
        )
    )


def load_layer_metrics(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
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
    return (
        frame.sort_values(
            [
                "optimizer",
                "seed",
                "epoch",
                "block",
                "matrix_type",
            ]
        )
        .drop_duplicates(
            [
                "optimizer",
                "seed",
                "step",
                "matrix_name",
            ],
            keep="last",
        )
    )


def load_spectral_summary(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
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
    return (
        frame.sort_values(
            ["optimizer", "seed", "epoch"]
        )
        .drop_duplicates(
            ["optimizer", "seed", "step"],
            keep="last",
        )
    )


def load_test_results(
    results_root: str | Path,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    seeds: Sequence[int] = (1337, 2027, 4099),
) -> pd.DataFrame:
    rows = []
    for optimizer in optimizers:
        for seed in seeds:
            _require_complete(
                results_root,
                optimizer,
                seed,
            )
            path = (
                run_directory(
                    results_root,
                    optimizer,
                    seed,
                )
                / "test_results.json"
            )
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
            for checkpoint in (
                "final",
                "validation_selected",
            ):
                values = payload[checkpoint]
                rows.append(
                    {
                        "optimizer": optimizer,
                        "optimizer_label": (
                            OPTIMIZER_LABELS[
                                optimizer
                            ]
                        ),
                        "seed": int(seed),
                        "checkpoint": checkpoint,
                        "step": int(values["step"]),
                        "test_loss": float(
                            values["loss"]
                        ),
                        "test_perplexity": float(
                            values["perplexity"]
                        ),
                        "test_accuracy": float(
                            values["accuracy"]
                        ),
                        "test_bleu": float(
                            values["bleu"]
                        ),
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
    subset = frame[
        [*keys, "seed", metric]
    ].copy()
    subset[metric] = pd.to_numeric(
        subset[metric],
        errors="coerce",
    )
    for values, group_frame in subset.groupby(
        keys,
        sort=True,
    ):
        values_tuple = (
            values
            if isinstance(values, tuple)
            else (values,)
        )
        row = dict(
            zip(
                keys,
                values_tuple,
                strict=True,
            )
        )
        row.update(
            mean_ci95(group_frame[metric])
        )
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
        subset = frame[
            frame["optimizer"] == optimizer
        ]
        if subset.empty:
            continue
        for _, seed_frame in subset.groupby(
            "seed"
        ):
            axis.plot(
                seed_frame[x],
                seed_frame[metric],
                color=OPTIMIZER_COLORS[
                    optimizer
                ],
                alpha=0.20,
                linewidth=0.9,
            )
        summary = summarize_by_epoch(
            subset,
            metric,
            x=x,
        )
        axis.plot(
            summary[x],
            summary["mean"],
            color=OPTIMIZER_COLORS[
                optimizer
            ],
            linewidth=2.0,
            label=OPTIMIZER_LABELS[
                optimizer
            ],
        )
        axis.fill_between(
            summary[x],
            summary["ci95_lower"],
            summary["ci95_upper"],
            color=OPTIMIZER_COLORS[
                optimizer
            ],
            alpha=0.16,
        )
    axis.set_xlabel(
        x.replace("_", " ").title()
    )
    axis.set_ylabel(
        metric.replace("_", " ").title()
    )
    axis.set_title(
        title
        or (
            f"{metric}: mean and 95% "
            "Student-t CI across seeds"
        )
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(
            output,
            dpi=170,
            bbox_inches="tight",
        )
    return figure, axis


def plot_layer_metric(
    frame: pd.DataFrame,
    *,
    optimizer: str,
    metric: str,
    title: str | None = None,
    output: str | Path | None = None,
):
    """Plot each block separately so blocks are never treated as replicates."""

    subset = frame[
        frame["optimizer"] == optimizer
    ].copy()
    if subset.empty:
        raise ValueError(
            f"no layer data for optimizer={optimizer}"
        )
    blocks = tuple(
        sorted(
            int(value)
            for value in subset["block"].unique()
        )
    )
    figure, axes = plt.subplots(
        1,
        len(blocks),
        figsize=(5.2 * len(blocks), 5.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, block in zip(
        axes[0],
        blocks,
        strict=True,
    ):
        block_frame = subset[
            subset["block"].astype(int)
            == int(block)
        ]
        for matrix_type in MATRIX_COLORS:
            matrix = block_frame[
                block_frame["matrix_type"]
                == matrix_type
            ]
            if matrix.empty:
                continue
            summary = summarize_by_epoch(
                matrix,
                metric,
                x="epoch",
                group=(
                    "block",
                    "matrix_type",
                    "matrix_name",
                ),
            )
            axis.plot(
                summary["epoch"],
                summary["mean"],
                color=MATRIX_COLORS[
                    matrix_type
                ],
                linewidth=2.0,
                marker="o",
                markersize=3,
                label=matrix_type,
            )
            axis.fill_between(
                summary["epoch"],
                summary["ci95_lower"],
                summary["ci95_upper"],
                color=MATRIX_COLORS[
                    matrix_type
                ],
                alpha=0.13,
            )
        if metric == "alpha":
            axis.axhline(
                2.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="alpha = 2",
            )
        if metric == "ERG_gap":
            axis.axhline(
                0.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
        axis.set_xlabel("Epoch")
        axis.set_title(f"Block {block}")
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel(metric)
    axes[0][-1].legend(
        frameon=False,
        ncol=2,
        fontsize=8,
    )
    figure.suptitle(
        title
        or (
            f"{OPTIMIZER_LABELS[optimizer]} "
            f"layer {metric}"
        )
    )
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(
            output,
            dpi=170,
            bbox_inches="tight",
        )
    return figure, axes


def plot_spectral_optimizer_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
    output: str | Path | None = None,
):
    return plot_epoch_metric(
        frame,
        metric=metric,
        x="epoch",
        optimizers=optimizers,
        title=(
            f"WeightWatcher {metric}: mean and "
            "95% Student-t CI"
        ),
        output=output,
    )


def _summary_row(
    *,
    optimizer: str,
    checkpoint: str,
    metric: str,
    stats: dict[str, float],
    interval_method: str,
) -> dict[str, object]:
    return {
        "optimizer": optimizer,
        "optimizer_label": (
            OPTIMIZER_LABELS[optimizer]
        ),
        "checkpoint": checkpoint,
        "metric": metric,
        "interval_method": interval_method,
        **stats,
    }


def final_test_summary(
    test_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize test metrics with a positive perplexity interval.

    Perplexity is reported as ``exp(mean loss)`` with the loss-space
    Student-t interval exponentiated. This avoids impossible negative
    perplexity bounds.
    """

    rows: list[dict[str, object]] = []
    for (
        optimizer,
        checkpoint,
    ), group in test_results.groupby(
        ["optimizer", "checkpoint"]
    ):
        loss_stats = mean_ci95(
            group["test_loss"]
        )
        rows.append(
            _summary_row(
                optimizer=optimizer,
                checkpoint=checkpoint,
                metric="test_loss",
                stats=loss_stats,
                interval_method=(
                    "arithmetic_student_t"
                ),
            )
        )
        ppl_stats = {
            "n": loss_stats["n"],
            "mean": math.exp(
                loss_stats["mean"]
            ),
            "sd": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_lower": math.exp(
                loss_stats["ci95_lower"]
            ),
            "ci95_upper": math.exp(
                loss_stats["ci95_upper"]
            ),
        }
        rows.append(
            _summary_row(
                optimizer=optimizer,
                checkpoint=checkpoint,
                metric="test_perplexity",
                stats=ppl_stats,
                interval_method=(
                    "exp_test_loss_student_t"
                ),
            )
        )
        for metric in (
            "test_accuracy",
            "test_bleu",
        ):
            rows.append(
                _summary_row(
                    optimizer=optimizer,
                    checkpoint=checkpoint,
                    metric=metric,
                    stats=mean_ci95(
                        group[metric]
                    ),
                    interval_method=(
                        "arithmetic_student_t"
                    ),
                )
            )
    return pd.DataFrame(rows)


def paired_test_differences(
    test_results: pd.DataFrame,
    *,
    optimizers: Sequence[str] = SUPPORTED_OPTIMIZERS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for checkpoint in sorted(
        test_results["checkpoint"].unique()
    ):
        selected = test_results[
            test_results["checkpoint"]
            == checkpoint
        ]
        for left, right in combinations(
            optimizers,
            2,
        ):
            for metric in (
                "test_loss",
                "test_accuracy",
                "test_bleu",
            ):
                left_frame = selected[
                    selected["optimizer"] == left
                ][["seed", metric]].rename(
                    columns={metric: "left"}
                )
                right_frame = selected[
                    selected["optimizer"] == right
                ][["seed", metric]].rename(
                    columns={metric: "right"}
                )
                paired = left_frame.merge(
                    right_frame,
                    on="seed",
                    validate="one_to_one",
                )
                stats = mean_ci95(
                    paired["left"]
                    - paired["right"]
                )
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "left_optimizer": left,
                        "right_optimizer": right,
                        "contrast": (
                            f"{OPTIMIZER_LABELS[left]} "
                            f"- {OPTIMIZER_LABELS[right]}"
                        ),
                        "metric": metric,
                        "interval_method": (
                            "paired_student_t"
                        ),
                        **stats,
                    }
                )
            loss_left = selected[
                selected["optimizer"] == left
            ][["seed", "test_loss"]].rename(
                columns={"test_loss": "left"}
            )
            loss_right = selected[
                selected["optimizer"] == right
            ][["seed", "test_loss"]].rename(
                columns={"test_loss": "right"}
            )
            paired_loss = loss_left.merge(
                loss_right,
                on="seed",
                validate="one_to_one",
            )
            log_ratio = mean_ci95(
                paired_loss["left"]
                - paired_loss["right"]
            )
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "left_optimizer": left,
                    "right_optimizer": right,
                    "contrast": (
                        f"{OPTIMIZER_LABELS[left]} "
                        f"/ {OPTIMIZER_LABELS[right]}"
                    ),
                    "metric": (
                        "test_perplexity_ratio"
                    ),
                    "interval_method": (
                        "exp_paired_loss_student_t"
                    ),
                    "n": log_ratio["n"],
                    "mean": math.exp(
                        log_ratio["mean"]
                    ),
                    "sd": np.nan,
                    "sem": np.nan,
                    "ci95_half_width": np.nan,
                    "ci95_lower": math.exp(
                        log_ratio["ci95_lower"]
                    ),
                    "ci95_upper": math.exp(
                        log_ratio["ci95_upper"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_diagnostics_table(
    metrics: pd.DataFrame,
    test_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_lookup = (
        test_results[
            test_results["checkpoint"]
            == "validation_selected"
        ]
        .set_index(["optimizer", "seed"])
    )
    final_lookup = (
        test_results[
            test_results["checkpoint"]
            == "final"
        ]
        .set_index(["optimizer", "seed"])
    )
    for (
        optimizer,
        seed,
    ), run in metrics.groupby(
        ["optimizer", "seed"],
        sort=True,
    ):
        ordered = run.sort_values("step")
        finite = ordered[
            np.isfinite(
                pd.to_numeric(
                    ordered["val_loss"],
                    errors="coerce",
                )
            )
        ]
        if finite.empty:
            continue
        best = finite.loc[
            finite["val_loss"].idxmin()
        ]
        final = finite.iloc[-1]
        key = (optimizer, int(seed))
        selected_test = selected_lookup.loc[key]
        final_test = final_lookup.loc[key]
        clip_values = pd.to_numeric(
            ordered.loc[
                ordered["step"].astype(int) > 0,
                "gradient_clipped",
            ],
            errors="coerce",
        )
        update_ratio = pd.to_numeric(
            ordered["update_to_weight_ratio"],
            errors="coerce",
        )
        rows.append(
            {
                "optimizer": optimizer,
                "optimizer_label": (
                    OPTIMIZER_LABELS[optimizer]
                ),
                "seed": int(seed),
                "best_validation_step": int(
                    selected_test["step"]
                ),
                "observed_best_validation_step": int(
                    best["step"]
                ),
                "best_validation_loss": float(
                    best["val_loss"]
                ),
                "final_validation_loss": float(
                    final["val_loss"]
                ),
                "final_minus_best_validation_loss": (
                    float(final["val_loss"])
                    - float(best["val_loss"])
                ),
                "best_validation_accuracy": float(
                    best["val_accuracy"]
                ),
                "final_validation_accuracy": float(
                    final["val_accuracy"]
                ),
                "selected_test_loss": float(
                    selected_test["test_loss"]
                ),
                "selected_test_accuracy": float(
                    selected_test[
                        "test_accuracy"
                    ]
                ),
                "final_test_loss": float(
                    final_test["test_loss"]
                ),
                "final_test_accuracy": float(
                    final_test["test_accuracy"]
                ),
                "max_update_to_weight_ratio": float(
                    update_ratio.max()
                ),
                "evaluation_snapshot_clip_fraction": (
                    float(clip_values.mean())
                    if clip_values.notna().any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_run_diagnostics(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    metrics = (
        "best_validation_loss",
        "final_validation_loss",
        "final_minus_best_validation_loss",
        "selected_test_loss",
        "selected_test_accuracy",
        "final_test_loss",
        "final_test_accuracy",
        "max_update_to_weight_ratio",
        "evaluation_snapshot_clip_fraction",
    )
    rows = []
    for optimizer, group in diagnostics.groupby(
        "optimizer",
        sort=True,
    ):
        for metric in metrics:
            rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_label": (
                        OPTIMIZER_LABELS[
                            optimizer
                        ]
                    ),
                    "metric": metric,
                    **mean_ci95(group[metric]),
                }
            )
    return pd.DataFrame(rows)

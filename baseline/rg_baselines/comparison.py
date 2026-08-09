"""Load, validate, compare, and persist the three MNIST reference baselines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .comparison_plotting import plot_all_comparisons
from .statistics import student_t_critical_95, summarize_numeric_metrics

OPTIMIZER_ORDER = ("sgd_momentum", "adamw", "sgd_momentum_muon")
OPTIMIZER_LABELS = {
    "sgd_momentum": "SGD + Nesterov momentum",
    "adamw": "AdamW",
    "sgd_momentum_muon": "Muon + auxiliary AdamW",
}
OPTIMIZER_COLORS = {
    "sgd_momentum": "#0072B2",
    "adamw": "#D55E00",
    "sgd_momentum_muon": "#009E73",
}
LAYER_ORDER = ("fc1", "fc2", "fc3")
REQUIRED_AGGREGATE_FILES = (
    "performance_by_epoch_and_seed.csv",
    "spectral_metrics_by_epoch_layer_and_seed.csv",
    "performance_summary_95ci.csv",
    "spectral_summary_95ci.csv",
    "replicate_manifest.json",
)
SHARED_CONFIG_KEYS = (
    "epochs",
    "batch_size",
    "validation_size",
    "split_seed",
    "num_workers",
    "grad_clip_norm",
    "train_eval_max_batches",
    "schedule",
    "checkpoint_every_epochs",
    "test_monitoring_only",
    "ww_min_evals",
    "ww_max_evals",
    "ww_svd_method",
    "ww_randomize",
    "strict_metrics",
    "save_epoch_checkpoints",
)
PERFORMANCE_METRICS = (
    "train_loss",
    "validation_loss",
    "test_loss",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "train_perplexity",
    "validation_perplexity",
    "test_perplexity",
    "validation_accuracy_gap",
    "test_accuracy_gap",
    "validation_loss_gap",
    "test_loss_gap",
    "primary_lr",
    "auxiliary_lr",
    "mean_gradient_norm_before_clip",
    "max_gradient_norm_before_clip",
    "parameter_l2_norm",
    "train_time_sec",
    "evaluation_time_sec",
    "weightwatcher_time_sec",
    "epoch_total_time_sec",
)
SPECTRAL_METRICS = (
    "alpha",
    "num_traps",
    "detX_num",
    "num_pl_spikes",
    "ERG_gap",
    "m_midpoint",
    "trace_log_midpoint_per_eval",
    "trace_log_midpoint_total",
    "stable_rank",
    "participation_ratio",
    "entropy_effective_rank",
    "boundary_overlap_ratio",
    "top1_energy_fraction",
    "pl_energy_fraction",
    "detx_energy_fraction",
    "midpoint_energy_fraction",
    "geometric_mean_midpoint",
    "normalized_lambda_max",
    "normalized_lambda_midpoint_cut",
    "eigenvalue_condition_number",
)
VALIDATION_ACCURACY_THRESHOLDS = (0.90, 0.95, 0.97, 0.98)
PAIR_ORDER = (
    ("adamw", "sgd_momentum"),
    ("sgd_momentum_muon", "sgd_momentum"),
    ("sgd_momentum_muon", "adamw"),
)
PAIR_METRICS = (
    "test_accuracy",
    "test_loss",
    "validation_accuracy",
    "validation_loss",
    "train_accuracy",
    "train_loss",
    "test_accuracy_gap",
    "test_loss_gap",
)


@dataclass(frozen=True)
class BaselineComparisonResult:
    run_root: Path
    output_dir: Path
    seeds: tuple[int, ...]
    epochs: int
    manifests: dict[str, dict[str, Any]]
    checkpoint_inventory: pd.DataFrame
    performance: pd.DataFrame
    spectral_metrics: pd.DataFrame
    performance_summary: pd.DataFrame
    spectral_summary: pd.DataFrame
    terminal_by_seed: pd.DataFrame
    terminal_summary: pd.DataFrame
    final_epoch_summary: pd.DataFrame
    convergence_by_seed: pd.DataFrame
    convergence_summary: pd.DataFrame
    paired_terminal_differences: pd.DataFrame
    plot_paths: tuple[Path, ...]
    expected_outputs: tuple[Path, ...]

    @property
    def paired_final_differences(self) -> pd.DataFrame:
        """Backward-compatible alias for historical notebooks."""

        return self.paired_terminal_differences


def _required_seed_paths(seed_dir: Path, epochs: int) -> list[Path]:
    paths = [
        seed_dir / "final_state.pt",
        seed_dir / "checkpoint_latest.pt",
        seed_dir / "checkpoint_best.pt",
        seed_dir / "run_complete.json",
        seed_dir / "test_results.json",
        seed_dir / "manifest.json",
        seed_dir / "config.json",
        seed_dir / "performance_by_epoch.csv",
        seed_dir / "spectral_metrics_by_epoch_and_layer.csv",
        seed_dir / "esd_history.npz",
    ]
    paths.extend(
        seed_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        for epoch in range(1, epochs + 1)
    )
    return paths


def _load_and_validate(run_root: Path):
    missing = [
        run_root / optimizer / filename
        for optimizer in OPTIMIZER_ORDER
        for filename in REQUIRED_AGGREGATE_FILES
        if not (run_root / optimizer / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Run all three baseline notebooks with the same RG_BASELINE_RUN_ROOT. "
            "Missing paths:\n" + "\n".join(f"  - {path}" for path in missing)
        )

    manifests: dict[str, dict[str, Any]] = {}
    performance_frames: list[pd.DataFrame] = []
    spectral_frames: list[pd.DataFrame] = []
    for optimizer in OPTIMIZER_ORDER:
        directory = run_root / optimizer
        manifests[optimizer] = json.loads(
            (directory / "replicate_manifest.json").read_text(encoding="utf-8")
        )
        performance = pd.read_csv(
            directory / "performance_by_epoch_and_seed.csv"
        )
        performance["optimizer"] = optimizer
        performance["optimizer_label"] = OPTIMIZER_LABELS[optimizer]
        performance_frames.append(performance)
        spectral = pd.read_csv(
            directory / "spectral_metrics_by_epoch_layer_and_seed.csv"
        )
        spectral["optimizer"] = optimizer
        spectral["optimizer_label"] = OPTIMIZER_LABELS[optimizer]
        spectral_frames.append(spectral)

    seed_sets = {
        optimizer: tuple(int(seed) for seed in manifests[optimizer]["seeds"])
        for optimizer in OPTIMIZER_ORDER
    }
    if len(set(seed_sets.values())) != 1:
        raise RuntimeError(f"optimizer seed tuples differ: {seed_sets}")
    seeds = seed_sets[OPTIMIZER_ORDER[0]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise RuntimeError(f"expected exactly three unique seeds; observed {seeds}")

    for key in SHARED_CONFIG_KEYS:
        values = {
            optimizer: manifests[optimizer]["config_template"].get(key)
            for optimizer in OPTIMIZER_ORDER
        }
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            raise RuntimeError(f"shared configuration key {key!r} differs: {values}")

    epochs = int(manifests[OPTIMIZER_ORDER[0]]["config_template"]["epochs"])
    inventory_rows = []
    for optimizer in OPTIMIZER_ORDER:
        for seed in seeds:
            seed_dir = run_root / optimizer / "seeds" / f"seed_{seed}"
            missing_seed = [
                path for path in _required_seed_paths(seed_dir, epochs)
                if not path.is_file()
            ]
            if missing_seed:
                raise RuntimeError(
                    f"incomplete artifacts for {optimizer}, seed={seed}:\n"
                    + "\n".join(f"  - {path}" for path in missing_seed)
                )
            completion = json.loads(
                (seed_dir / "run_complete.json").read_text(encoding="utf-8")
            )
            if completion.get("completed") is not True:
                raise RuntimeError(f"run is not complete: {seed_dir}")
            inventory_rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS[optimizer],
                    "seed": seed,
                    "epoch_checkpoints": epochs,
                    "best_validation_epoch": completion["best_validation_epoch"],
                    "final_state": str(seed_dir / "final_state.pt"),
                }
            )

    performance = pd.concat(performance_frames, ignore_index=True, sort=False)
    spectral = pd.concat(spectral_frames, ignore_index=True, sort=False)
    expected_perf = {
        (seed, epoch) for seed in seeds for epoch in range(epochs + 1)
    }
    for optimizer in OPTIMIZER_ORDER:
        rows = performance.loc[performance["optimizer"].eq(optimizer)]
        observed = set(
            zip(rows["seed"].astype(int), rows["epoch"].astype(int), strict=False)
        )
        if observed != expected_perf or len(rows) != len(expected_perf):
            raise RuntimeError(f"incomplete or duplicate performance grid for {optimizer}")
        if not rows["test_monitoring_only"].astype(int).eq(1).all():
            raise RuntimeError(f"test policy violation for {optimizer}")

    valid = spectral.loc[spectral["status"].astype(str).eq("ok")].copy()
    expected_spectral = {
        (seed, epoch, layer)
        for seed in seeds
        for epoch in range(epochs + 1)
        for layer in LAYER_ORDER
    }
    for optimizer in OPTIMIZER_ORDER:
        rows = valid.loc[valid["optimizer"].eq(optimizer)]
        observed = set(
            zip(
                rows["seed"].astype(int),
                rows["epoch"].astype(int),
                rows["layer"].astype(str),
                strict=False,
            )
        )
        if observed != expected_spectral or len(rows) != len(expected_spectral):
            raise RuntimeError(f"incomplete or duplicate spectral grid for {optimizer}")

    performance["train_perplexity"] = np.exp(performance["train_loss"].astype(float))
    performance["validation_perplexity"] = np.exp(
        performance["validation_loss"].astype(float)
    )
    performance["test_perplexity"] = np.exp(performance["test_loss"].astype(float))
    return (
        manifests,
        seeds,
        epochs,
        pd.DataFrame(inventory_rows),
        performance,
        spectral,
        valid,
    )


def _terminal_rows(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (optimizer, seed), run in performance.groupby(
        ["optimizer", "seed"], sort=True
    ):
        ordered = run.sort_values("epoch")
        final = ordered.iloc[-1].copy()
        final["checkpoint"] = "final"
        rows.append(final)
        selected = ordered.sort_values(
            ["validation_loss", "epoch"], ascending=[True, True]
        ).iloc[0].copy()
        selected["checkpoint"] = "validation_selected"
        rows.append(selected)
    return pd.DataFrame(rows).reset_index(drop=True)


def _convergence(performance: pd.DataFrame, seeds: tuple[int, ...], epochs: int):
    rows = []
    for optimizer in OPTIMIZER_ORDER:
        for seed in seeds:
            run = performance.loc[
                performance["optimizer"].eq(optimizer)
                & performance["seed"].astype(int).eq(seed)
            ].sort_values("epoch")
            selected = run.sort_values(
                ["validation_loss", "epoch"], ascending=[True, True]
            ).iloc[0]
            final = run.loc[run["epoch"].astype(int).eq(epochs)].iloc[0]
            row = {
                "optimizer": optimizer,
                "optimizer_label": OPTIMIZER_LABELS[optimizer],
                "seed": seed,
                "best_validation_loss": float(selected["validation_loss"]),
                "best_validation_epoch": int(selected["epoch"]),
                "validation_selected_test_accuracy": float(selected["test_accuracy"]),
                "validation_selected_test_loss": float(selected["test_loss"]),
                "final_test_accuracy": float(final["test_accuracy"]),
                "final_test_loss": float(final["test_loss"]),
            }
            for threshold in VALIDATION_ACCURACY_THRESHOLDS:
                reached = run.loc[
                    run["validation_accuracy"].astype(float).ge(threshold), "epoch"
                ]
                row[f"epoch_to_validation_accuracy_{threshold:.2f}"] = (
                    float(reached.min()) if not reached.empty else float("nan")
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    metrics = [
        column for column in frame.columns
        if column not in {"optimizer", "optimizer_label", "seed"}
    ]
    summary = summarize_numeric_metrics(
        frame,
        group_columns=("optimizer", "optimizer_label"),
        metrics=metrics,
        confidence=0.95,
    )
    return frame, summary


def _paired_terminal(terminal: pd.DataFrame, seeds: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for checkpoint in ("final", "validation_selected"):
        selected = terminal.loc[terminal["checkpoint"].eq(checkpoint)]
        for optimizer_a, optimizer_b in PAIR_ORDER:
            for metric in PAIR_METRICS:
                a = selected.loc[
                    selected["optimizer"].eq(optimizer_a), ["seed", metric]
                ].rename(columns={metric: "value_a"})
                b = selected.loc[
                    selected["optimizer"].eq(optimizer_b), ["seed", metric]
                ].rename(columns={metric: "value_b"})
                paired = a.merge(b, on="seed", validate="one_to_one").sort_values("seed")
                if tuple(paired["seed"].astype(int)) != tuple(sorted(seeds)):
                    raise RuntimeError(
                        (optimizer_a, optimizer_b, checkpoint, metric, "seed mismatch")
                    )
                deltas = (paired["value_a"] - paired["value_b"]).to_numpy(dtype=float)
                n = int(deltas.size)
                mean = float(np.mean(deltas))
                std = float(np.std(deltas, ddof=1))
                sem = std / math.sqrt(n)
                half = student_t_critical_95(n) * sem
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "optimizer_a": optimizer_a,
                        "optimizer_b": optimizer_b,
                        "contrast": (
                            f"{OPTIMIZER_LABELS[optimizer_a]} - "
                            f"{OPTIMIZER_LABELS[optimizer_b]}"
                        ),
                        "metric": metric,
                        "n": n,
                        "mean_difference": mean,
                        "std_difference": std,
                        "sem_difference": sem,
                        "ci_half_width": half,
                        "ci_low": mean - half,
                        "ci_high": mean + half,
                    }
                )
    return pd.DataFrame(rows)


def run_baseline_comparison(
    run_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    show_plots: bool = True,
) -> BaselineComparisonResult:
    root = Path(run_root).expanduser().resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir else root / "comparison"
    )
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    (
        manifests,
        seeds,
        epochs,
        inventory,
        performance,
        spectral,
        valid,
    ) = _load_and_validate(root)
    inventory.to_csv(output / "checkpoint_inventory.csv", index=False)
    performance.to_csv(
        output / "all_optimizers_performance_by_epoch_and_seed.csv", index=False
    )
    spectral.to_csv(
        output / "all_optimizers_spectral_metrics_by_epoch_layer_and_seed.csv",
        index=False,
    )

    performance_summary = summarize_numeric_metrics(
        performance,
        group_columns=("optimizer", "optimizer_label", "epoch"),
        metrics=[metric for metric in PERFORMANCE_METRICS if metric in performance],
        confidence=0.95,
    )
    spectral_summary = summarize_numeric_metrics(
        valid,
        group_columns=("optimizer", "optimizer_label", "layer", "epoch"),
        metrics=[metric for metric in SPECTRAL_METRICS if metric in valid],
        confidence=0.95,
    )
    terminal = _terminal_rows(performance)
    terminal_summary = summarize_numeric_metrics(
        terminal,
        group_columns=("optimizer", "optimizer_label", "checkpoint"),
        metrics=[
            metric for metric in PERFORMANCE_METRICS
            if metric in terminal and metric not in {"primary_lr", "auxiliary_lr"}
        ],
        confidence=0.95,
    )
    final_summary = terminal_summary.loc[
        terminal_summary["checkpoint"].eq("final")
    ].copy()
    convergence, convergence_summary = _convergence(performance, seeds, epochs)
    paired = _paired_terminal(terminal, seeds)

    table_map = {
        "performance_summary_95ci.csv": performance_summary,
        "spectral_summary_95ci.csv": spectral_summary,
        "terminal_by_seed.csv": terminal,
        "terminal_summary_95ci.csv": terminal_summary,
        "final_epoch_summary_95ci.csv": final_summary,
        "convergence_by_seed.csv": convergence,
        "convergence_summary_95ci.csv": convergence_summary,
        "paired_terminal_differences_95ci.csv": paired,
    }
    for filename, frame in table_map.items():
        frame.to_csv(output / filename, index=False)

    plot_paths = plot_all_comparisons(
        performance=performance,
        performance_summary=performance_summary,
        spectral=valid,
        spectral_summary=spectral_summary,
        output_dir=plots,
        seeds=seeds,
        optimizer_order=OPTIMIZER_ORDER,
        layer_order=LAYER_ORDER,
        labels=OPTIMIZER_LABELS,
        colors=OPTIMIZER_COLORS,
        show=show_plots,
    )
    manifest_path = output / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_run_root": str(root),
                "comparison_dir": str(output),
                "optimizers": list(OPTIMIZER_ORDER),
                "optimizer_labels": OPTIMIZER_LABELS,
                "seeds": list(seeds),
                "epochs": epochs,
                "confidence": 0.95,
                "selection_policy": (
                    "validation loss selects the reported checkpoint; the "
                    "official test set is monitoring-only"
                ),
                "error_bar_definition": (
                    "two-sided 95% Student-t confidence interval across "
                    "independent complete runs"
                ),
                "paired_contrast_definition": (
                    "optimizer_a minus optimizer_b within matched seed"
                ),
                "plot_files": [
                    str(path.relative_to(output)) for path in plot_paths
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    expected = (
        output / "checkpoint_inventory.csv",
        output / "all_optimizers_performance_by_epoch_and_seed.csv",
        output / "all_optimizers_spectral_metrics_by_epoch_layer_and_seed.csv",
        *(output / filename for filename in table_map),
        manifest_path,
        *plot_paths,
    )
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            "comparison did not persist every expected output:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    return BaselineComparisonResult(
        run_root=root,
        output_dir=output,
        seeds=seeds,
        epochs=epochs,
        manifests=manifests,
        checkpoint_inventory=inventory,
        performance=performance,
        spectral_metrics=spectral,
        performance_summary=performance_summary,
        spectral_summary=spectral_summary,
        terminal_by_seed=terminal,
        terminal_summary=terminal_summary,
        final_epoch_summary=final_summary,
        convergence_by_seed=convergence,
        convergence_summary=convergence_summary,
        paired_terminal_differences=paired,
        plot_paths=plot_paths,
        expected_outputs=tuple(expected),
    )

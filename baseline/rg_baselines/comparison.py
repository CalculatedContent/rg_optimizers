"""Load, validate, compare, and persist the three clean MNIST baselines."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .comparison_plotting import plot_all_comparisons
from .statistics import student_t_critical_95, summarize_numeric_metrics

OPTIMIZER_ORDER = ("sgd_momentum", "adamw", "sgd_momentum_muon")
OPTIMIZER_LABELS = {
    "sgd_momentum": "SGD + momentum",
    "adamw": "AdamW",
    "sgd_momentum_muon": "SGD + momentum + Muon",
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
    "num_workers",
    "grad_clip_norm",
    "train_eval_max_batches",
    "ww_min_evals",
    "ww_max_evals",
    "ww_svd_method",
    "ww_randomize",
    "strict_metrics",
    "save_epoch_checkpoints",
)
PERFORMANCE_METRICS = (
    "train_loss",
    "test_loss",
    "train_accuracy",
    "test_accuracy",
    "train_perplexity",
    "test_perplexity",
    "accuracy_generalization_gap",
    "loss_generalization_gap",
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
ACCURACY_THRESHOLDS = (0.90, 0.95, 0.97, 0.98)
PAIR_ORDER = (
    ("adamw", "sgd_momentum"),
    ("sgd_momentum_muon", "sgd_momentum"),
    ("sgd_momentum_muon", "adamw"),
)
PAIR_METRICS = (
    "test_accuracy",
    "test_loss",
    "test_perplexity",
    "train_accuracy",
    "train_loss",
    "accuracy_generalization_gap",
    "loss_generalization_gap",
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
    final_epoch_summary: pd.DataFrame
    convergence_by_seed: pd.DataFrame
    convergence_summary: pd.DataFrame
    paired_final_differences: pd.DataFrame
    plot_paths: tuple[Path, ...]
    expected_outputs: tuple[Path, ...]


def _required_seed_paths(seed_dir: Path, epochs: int) -> list[Path]:
    paths = [
        seed_dir / "final_state.pt",
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
        performance = pd.read_csv(directory / "performance_by_epoch_and_seed.csv")
        performance["optimizer"] = optimizer
        performance["optimizer_label"] = OPTIMIZER_LABELS[optimizer]
        performance_frames.append(performance)
        spectral = pd.read_csv(directory / "spectral_metrics_by_epoch_layer_and_seed.csv")
        spectral["optimizer"] = optimizer
        spectral["optimizer_label"] = OPTIMIZER_LABELS[optimizer]
        spectral_frames.append(spectral)

    seed_sets = {
        optimizer: tuple(int(seed) for seed in manifests[optimizer]["seeds"])
        for optimizer in OPTIMIZER_ORDER
    }
    if len(set(seed_sets.values())) != 1:
        raise RuntimeError(f"Optimizer seed tuples differ: {seed_sets}")
    seeds = seed_sets[OPTIMIZER_ORDER[0]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise RuntimeError(f"Expected exactly three unique seeds; observed {seeds}")

    for key in SHARED_CONFIG_KEYS:
        values = {
            optimizer: manifests[optimizer]["config_template"].get(key)
            for optimizer in OPTIMIZER_ORDER
        }
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            raise RuntimeError(f"Shared configuration key {key!r} differs: {values}")

    epochs = int(manifests[OPTIMIZER_ORDER[0]]["config_template"]["epochs"])
    if not all(
        bool(manifests[optimizer]["config_template"]["save_epoch_checkpoints"])
        for optimizer in OPTIMIZER_ORDER
    ):
        raise RuntimeError("Every baseline must enable save_epoch_checkpoints=True")

    inventory_rows = []
    for optimizer in OPTIMIZER_ORDER:
        for seed in seeds:
            seed_dir = run_root / optimizer / "seeds" / f"seed_{seed}"
            missing = [path for path in _required_seed_paths(seed_dir, epochs) if not path.is_file()]
            if missing:
                raise RuntimeError(
                    f"Incomplete persisted artifacts for {optimizer}, seed={seed}:\n"
                    + "\n".join(f"  - {path}" for path in missing)
                )
            inventory_rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS[optimizer],
                    "seed": seed,
                    "epoch_checkpoints": epochs,
                    "final_state": str(seed_dir / "final_state.pt"),
                }
            )

    performance = pd.concat(performance_frames, ignore_index=True)
    spectral = pd.concat(spectral_frames, ignore_index=True)
    expected_perf = {(seed, epoch) for seed in seeds for epoch in range(epochs + 1)}
    for optimizer in OPTIMIZER_ORDER:
        rows = performance.loc[performance["optimizer"].eq(optimizer)]
        observed = set(zip(rows["seed"].astype(int), rows["epoch"].astype(int)))
        if observed != expected_perf or len(rows) != len(expected_perf):
            raise RuntimeError(f"Incomplete or duplicate performance grid for {optimizer}")

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
            )
        )
        if observed != expected_spectral or len(rows) != len(expected_spectral):
            raise RuntimeError(f"Incomplete or duplicate spectral grid for {optimizer}")

    performance["train_perplexity"] = np.exp(performance["train_loss"].astype(float))
    performance["test_perplexity"] = np.exp(performance["test_loss"].astype(float))
    performance["accuracy_generalization_gap"] = (
        performance["train_accuracy"].astype(float)
        - performance["test_accuracy"].astype(float)
    )
    performance["loss_generalization_gap"] = (
        performance["test_loss"].astype(float)
        - performance["train_loss"].astype(float)
    )
    return manifests, seeds, epochs, pd.DataFrame(inventory_rows), performance, spectral, valid


def _convergence(performance: pd.DataFrame, seeds: tuple[int, ...], epochs: int):
    rows = []
    for optimizer in OPTIMIZER_ORDER:
        for seed in seeds:
            run = performance.loc[
                performance["optimizer"].eq(optimizer)
                & performance["seed"].astype(int).eq(seed)
            ].sort_values("epoch")
            best = float(run["test_accuracy"].max())
            final = run.loc[run["epoch"].astype(int).eq(epochs)].iloc[0]
            row = {
                "optimizer": optimizer,
                "optimizer_label": OPTIMIZER_LABELS[optimizer],
                "seed": seed,
                "best_test_accuracy": best,
                "best_test_accuracy_epoch": int(
                    run.loc[run["test_accuracy"].eq(best), "epoch"].min()
                ),
                "final_test_accuracy": float(final["test_accuracy"]),
                "final_test_loss": float(final["test_loss"]),
                "final_test_perplexity": float(final["test_perplexity"]),
                "mean_test_accuracy_over_epochs": float(run["test_accuracy"].mean()),
                "mean_test_loss_over_epochs": float(run["test_loss"].mean()),
            }
            for threshold in ACCURACY_THRESHOLDS:
                reached = run.loc[run["test_accuracy"].astype(float).ge(threshold), "epoch"]
                row[f"epoch_to_test_accuracy_{threshold:.2f}"] = (
                    float(reached.min()) if not reached.empty else float("nan")
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    metrics = [column for column in frame.columns if column not in {"optimizer", "optimizer_label", "seed"}]
    summary = summarize_numeric_metrics(
        frame,
        group_columns=("optimizer", "optimizer_label"),
        metrics=metrics,
        confidence=0.95,
    )
    return frame, summary


def _paired_final(performance: pd.DataFrame, seeds: tuple[int, ...], epochs: int) -> pd.DataFrame:
    final = performance.loc[performance["epoch"].astype(int).eq(epochs)].copy()
    rows = []
    for optimizer_a, optimizer_b in PAIR_ORDER:
        for metric in PAIR_METRICS:
            a = final.loc[final["optimizer"].eq(optimizer_a), ["seed", metric]].rename(
                columns={metric: "value_a"}
            )
            b = final.loc[final["optimizer"].eq(optimizer_b), ["seed", metric]].rename(
                columns={metric: "value_b"}
            )
            paired = a.merge(b, on="seed", validate="one_to_one").sort_values("seed")
            if tuple(paired["seed"].astype(int)) != tuple(sorted(seeds)):
                raise RuntimeError((optimizer_a, optimizer_b, metric, "seed mismatch"))
            deltas = (paired["value_a"] - paired["value_b"]).to_numpy(dtype=float)
            n = int(deltas.size)
            mean = float(np.mean(deltas))
            std = float(np.std(deltas, ddof=1))
            sem = float(std / math.sqrt(n))
            critical = student_t_critical_95(n)
            half = float(critical * sem)
            rows.append(
                {
                    "optimizer_a": optimizer_a,
                    "optimizer_b": optimizer_b,
                    "contrast": f"{OPTIMIZER_LABELS[optimizer_a]} - {OPTIMIZER_LABELS[optimizer_b]}",
                    "metric": metric,
                    "n": n,
                    "mean_difference": mean,
                    "std_difference": std,
                    "sem_difference": sem,
                    "critical_value": critical,
                    "ci_half_width": half,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                    "minimum_difference": float(np.min(deltas)),
                    "maximum_difference": float(np.max(deltas)),
                }
            )
    return pd.DataFrame(rows)


def run_baseline_comparison(
    run_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    show_plots: bool = True,
) -> BaselineComparisonResult:
    """Validate all persisted baselines and write the complete comparison package."""
    root = Path(run_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else root / "comparison"
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    manifests, seeds, epochs, inventory, performance, spectral, valid = _load_and_validate(root)
    inventory.to_csv(output / "checkpoint_inventory.csv", index=False)
    performance.to_csv(output / "all_optimizers_performance_by_epoch_and_seed.csv", index=False)
    spectral.to_csv(output / "all_optimizers_spectral_metrics_by_epoch_layer_and_seed.csv", index=False)

    perf_metrics = [metric for metric in PERFORMANCE_METRICS if metric in performance.columns]
    spectral_metrics = [metric for metric in SPECTRAL_METRICS if metric in valid.columns]
    performance_summary = summarize_numeric_metrics(
        performance,
        group_columns=("optimizer", "optimizer_label", "epoch"),
        metrics=perf_metrics,
        confidence=0.95,
    )
    spectral_summary = summarize_numeric_metrics(
        valid,
        group_columns=("optimizer", "optimizer_label", "layer", "epoch"),
        metrics=spectral_metrics,
        confidence=0.95,
    )
    final_summary = performance_summary.loc[
        performance_summary["epoch"].astype(int).eq(epochs)
    ].copy()
    convergence, convergence_summary = _convergence(performance, seeds, epochs)
    paired = _paired_final(performance, seeds, epochs)

    table_map = {
        "performance_summary_95ci.csv": performance_summary,
        "spectral_summary_95ci.csv": spectral_summary,
        "final_epoch_summary_95ci.csv": final_summary,
        "convergence_by_seed.csv": convergence,
        "convergence_summary_95ci.csv": convergence_summary,
        "paired_final_differences_95ci.csv": paired,
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
    manifest = {
        "source_run_root": str(root),
        "comparison_dir": str(output),
        "optimizers": [
            {"optimizer": optimizer, "optimizer_label": OPTIMIZER_LABELS[optimizer]}
            for optimizer in OPTIMIZER_ORDER
        ],
        "seeds": list(seeds),
        "replicate_count": len(seeds),
        "epochs": epochs,
        "final_epoch": epochs,
        "confidence": 0.95,
        "error_bar_definition": (
            "two-sided 95% Student-t confidence interval across independent complete runs"
        ),
        "paired_contrast_definition": "optimizer_a minus optimizer_b within matched seed",
        "derived_metrics": {
            "train_perplexity": "exp(train_loss)",
            "test_perplexity": "exp(test_loss)",
            "accuracy_generalization_gap": "train_accuracy - test_accuracy",
            "loss_generalization_gap": "test_loss - train_loss",
        },
        "plot_files": [str(path.relative_to(output)) for path in plot_paths],
    }
    manifest_path = output / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
            "Comparison did not persist every expected output:\n"
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
        final_epoch_summary=final_summary,
        convergence_by_seed=convergence,
        convergence_summary=convergence_summary,
        paired_final_differences=paired,
        plot_paths=plot_paths,
        expected_outputs=tuple(expected),
    )

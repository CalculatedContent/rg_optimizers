"""Independent-seed baseline runs and aggregate error-bar tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from .config import BaselineConfig
from .results import BaselineResult
from .runner import run_baseline
from .statistics import require_complete_summary, summarize_numeric_metrics

DEFAULT_BASELINE_SEEDS: tuple[int, ...] = (1337, 2027, 31415)

REQUIRED_PERFORMANCE_METRICS: tuple[str, ...] = (
    "train_loss", "test_loss", "train_accuracy", "test_accuracy"
)
PERFORMANCE_METRICS: tuple[str, ...] = (
    *REQUIRED_PERFORMANCE_METRICS,
    "online_train_loss", "online_train_accuracy",
    "mean_gradient_norm_before_clip", "median_gradient_norm_before_clip",
    "max_gradient_norm_before_clip", "parameter_l2_norm",
    "train_time_sec", "evaluation_time_sec", "weightwatcher_time_sec",
    "epoch_total_time_sec",
)
REQUIRED_SPECTRAL_METRICS: tuple[str, ...] = (
    "alpha", "num_traps", "detX_num", "num_pl_spikes", "ERG_gap",
    "m_midpoint", "trace_log_midpoint_per_eval", "trace_log_midpoint_total",
)
SPECTRAL_METRICS: tuple[str, ...] = (
    *REQUIRED_SPECTRAL_METRICS,
    "alpha_minus_2", "abs_alpha_minus_2", "geometric_mean_midpoint",
    "boundary_overlap_ratio", "frobenius_norm", "spectral_norm", "stable_rank",
    "participation_ratio", "entropy_effective_rank", "largest_eigenvalue",
    "smallest_positive_eigenvalue", "eigenvalue_condition_number",
    "top1_energy_fraction", "pl_energy_fraction", "detx_energy_fraction",
    "midpoint_energy_fraction", "midpoint_span_decades",
    "rescaled_eigenvalue_sum", "rescale_sum_minus_num_eigenvalues",
    "normalized_lambda_max", "normalized_lambda_midpoint_cut",
)


@dataclass
class BaselineReplicateResult:
    config_template: BaselineConfig
    seeds: tuple[int, ...]
    confidence: float
    results: tuple[BaselineResult, ...]
    performance: pd.DataFrame
    spectral_metrics: pd.DataFrame
    weightwatcher_details: pd.DataFrame
    optimizer_groups: pd.DataFrame
    combined_metrics: pd.DataFrame
    performance_summary: pd.DataFrame
    spectral_summary: pd.DataFrame

    @property
    def optimizer_label(self) -> str:
        return self.config_template.optimizer_label

    @property
    def replicate_count(self) -> int:
        return len(self.seeds)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.performance.to_csv(output / "performance_by_epoch_and_seed.csv", index=False)
        self.spectral_metrics.to_csv(
            output / "spectral_metrics_by_epoch_layer_and_seed.csv", index=False
        )
        self.weightwatcher_details.to_csv(
            output / "weightwatcher_details_by_epoch_and_seed.csv", index=False
        )
        self.optimizer_groups.to_csv(
            output / "optimizer_groups_by_epoch_and_seed.csv", index=False
        )
        self.combined_metrics.to_csv(
            output / "combined_metrics_by_epoch_layer_and_seed.csv", index=False
        )
        self.performance_summary.to_csv(output / "performance_summary_95ci.csv", index=False)
        self.spectral_summary.to_csv(output / "spectral_summary_95ci.csv", index=False)
        manifest = {
            "optimizer": self.config_template.optimizer,
            "optimizer_label": self.optimizer_label,
            "seeds": list(self.seeds),
            "replicate_count": self.replicate_count,
            "confidence": self.confidence,
            "error_bar_definition": (
                "two-sided 95% Student-t confidence interval across independent "
                "complete training runs"
            ),
            "config_template": asdict(self.config_template),
        }
        (output / "replicate_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def _tag_frame(frame: pd.DataFrame, *, config: BaselineConfig, replicate_index: int) -> pd.DataFrame:
    tagged = frame.copy()
    tagged["seed"] = int(config.seed)
    tagged["replicate"] = int(replicate_index)
    tagged["optimizer"] = config.optimizer
    tagged["optimizer_label"] = config.optimizer_label
    return tagged


def validate_replicate_result(result: BaselineReplicateResult) -> None:
    expected_seeds = set(result.seeds)
    if len(expected_seeds) != result.replicate_count:
        raise RuntimeError("replicate seeds are not unique")
    expected_epochs = set(range(result.config_template.epochs + 1))
    performance = result.performance.copy()
    if set(performance["epoch"].astype(int)) != expected_epochs:
        raise RuntimeError "replicate performance epochs are incomplete"
    for epoch in expected_epochs:
        present = set(performance.loc[performance["epoch"] == epoch, "seed"].astype(int))
        if present != expected_seeds:
            raise RuntimeError(f"epoch {epoch} is missing performance seeds")

    spectral = result.spectral_metrics[
        result.spectral_metrics["status"].astype(str).eq("ok")
    ].copy()
    for epoch in expected_epochs:
        for layer in ("fc1", "fc2", "fc3"):
            present = set(
                spectral.loc[
                    (spectral["epoch"].astype(int) == epoch)
                    & spectral["layer"].astype(str).eq(layer),
                    "seed",
                ].astype(int)
            )
            if present != expected_seeds:
                raise RuntimeError(
                    f"epoch {epoch}, layer {layer} is missing WeightWatcher seeds"
                )
    traps = spectral["num_traps"].to_numpy(dtype=float)
    if (traps < 0.0).any() or not np.allclose(traps, np.rint(traps)):
        raise RuntimeError("num_traps must contain non-negative integer counts")

    require_complete_summary(
        result.performance_summary,
        expected_replicates=result.replicate_count,
        required_metrics=REQUIRED_PERFORMANCE_METRICS,
    )
    require_complete_summary(
        result.spectral_summary,
        expected_replicates=result.replicate_count,
        required_metrics=REQUIRED_SPECTRAL_METRICS
   )


def run_baseline_replicates(
    config: BaselineConfig,
    *,
    seeds: Iterable[int] = DEFAULT_BASELINE_SEEDS,
    data_dir: str | Path = "./data",
    device: Optional[torch.device] = None,
    output_dir: Optional[str | Path] = None,
    progress: bool = True,
    confidence: float = 0.95,
) -> BaselineReplicateResult:
    ordered_seeds = tuple(int(seed) for seed in seeds)
    if len(ordered_seeds) < 2:
        raise ValueError("At least two independent seeds are required for error bars.")
    if len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("Replicate seeds must be unique.")

    results: list[BaselineResult] = []
    performance_frames: list[pd.DataFrame] = []
    spectral_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    combined_frames: list[pd.DataFrame] = []
    root = Path(output_dir) if output_dir is not None else None

    for replicate_index, seed in enumerate(ordered_seeds):
        run_config = replace(config, seed=seed)
        seed_output = root / "seeds" / f"seed_{seed}" if root is not None else None
        if progress:
            print(
                f"\n=== {run_config.optimizer_label}: replicate "
                f"{replicate_index + 1}/{len(ordered_seeds)}, seed={seed} ==="
            )
        run = run_baseline(
            run_config, data_dir=data_dir, device=device,
            output_dir=seed_output, progress=progress
        )
        results.append(run)
        performance_frames.append(_tag_frame(run.performance, config=run_config, replicate_index=replicate_index))
        spectral_frames.append(_tag_frame(run.spectral_metrics, config=run_config, replicate_index=replicate_index))
        detail_frames.append(_tag_frame(run.weightwatcher_details, config=run_config, replicate_index=replicate_index))
        group_frames.append(_tag_frame(run.optimizer_groups, config=run_config, replicate_index=replicate_index))
        combined_frames.append(_tag_frame(run.combined_metrics, config=run_config, replicate_index=replicate_index))

    performance = pd.concat(performance_frames, ignore_index=True)
    spectral_metrics = pd.concat(spectral_frames, ignore_index=True)
    weightwatcher_details = pd.concat(detail_frames, ignore_index=True)
    optimizer_groups = pd.concat(group_frames, ignore_index=True)
    combined_metrics = pd.concat(combined_frames, ignore_index=True)
    performance_metrics = [m for m in PERFORMANCE_METRICS if m in performance.columns]
    spectral_valid = spectral_metrics[spectral_metrics["status"].astype(str).eq("ok")].copy()
    spectral_metrics_to_summarize = [m for m in SPECTRAL_METRICS if m in spectral_valid.columns]

    performance_summary = summarize_numeric_metrics(
        performance,
        group_columns=("run", "optimizer", "optimizer_label", "epoch"),
        metrics=performance_metrics,
        confidence=confidence,
    )
    spectral_summary = summarize_numeric_metrics(
        spectral_valid,
        group_columns=("run", "optimizer", "optimizer_label", "layer", "epoch"),
        metrics=spectral_metrics_to_summarize,
        confidence=confidence,
    )
    aggregate = BaselineReplicateResult(
        config_template=config,
        seeds=ordered_seeds,
        confidence=float(confidence),
        results=tuple(results),
        performance=performance,
        spectral_metrics=spectral_metrics,
        weightwatcher_details=weightwatcher_details,
        optimizer_groups=optimizer_groups,
        combined_metrics=combined_metrics,
        performance_summary=performance_summary,
        spectral_summary=spectral_summary,
    )
    validate_replicate_result(aggregate)
    if root is not None:
        aggregate.save(root)
    return aggregate

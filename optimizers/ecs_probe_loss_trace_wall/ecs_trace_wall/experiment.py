"""Paired MLP3/MNIST baseline versus ECS probe-loss TraceWall experiments."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .optimizer import ECSProbeLossTraceWall
from .reporting import (
    correction_summary,
    performance_row,
    student_t_summary,
    validate_pairing,
)
from .runtime import (
    MLP3,
    WarmupCosineSchedule,
    build_base_optimizer,
    choose_device,
    evaluate,
    evaluation_loader,
    load_mnist,
    loader_for_indices,
    ordered_epoch_indices,
    parameter_l2_norm,
    set_seed,
    state_dict_checksum,
)
from .sampler import RotatingSubsetSampler
from .spectral import measure_spectral
from .training import train_baseline_epoch, train_trace_wall_epoch


@dataclass
class PairedExperimentResult:
    config: ExperimentConfig
    performance: pd.DataFrame
    spectral: pd.DataFrame
    corrections: pd.DataFrame
    performance_summary: pd.DataFrame
    spectral_summary: pd.DataFrame
    correction_summary: pd.DataFrame
    manifest: dict[str, Any]
    output_dir: Optional[Path] = None

    def save(self, output_dir: str | Path) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.performance.to_csv(
            directory / "performance_by_epoch_and_seed.csv", index=False
        )
        self.spectral.to_csv(
            directory / "spectral_metrics_by_epoch_layer_and_seed.csv", index=False
        )
        self.corrections.to_csv(
            directory / "trace_wall_corrections_by_step_layer_and_seed.csv",
            index=False,
        )
        self.performance_summary.to_csv(
            directory / "performance_summary_95ci.csv", index=False
        )
        self.spectral_summary.to_csv(
            directory / "spectral_summary_95ci.csv", index=False
        )
        self.correction_summary.to_csv(
            directory / "trace_wall_correction_summary.csv", index=False
        )
        with (directory / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.config.to_dict(), handle, indent=2, sort_keys=True)
        with (directory / "paired_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(self.manifest, handle, indent=2, sort_keys=True)
        self.output_dir = directory


def run_paired_experiment(
    config: ExperimentConfig,
    *,
    data_dir: str | Path = "./data",
    output_dir: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
    train_dataset: Optional[Dataset[Any]] = None,
    test_dataset: Optional[Dataset[Any]] = None,
    progress: bool = True,
    model_factory: Callable[[], nn.Module] = MLP3,
) -> PairedExperimentResult:
    """Run strictly paired baseline and TraceWall trajectories."""

    config.validate()
    device = device or choose_device()
    if train_dataset is None or test_dataset is None:
        train_dataset, test_dataset = load_mnist(data_dir)
    if len(train_dataset) < 1 or len(test_dataset) < 1:
        raise ValueError("datasets must be non-empty")

    steps_per_epoch = int(math.ceil(len(train_dataset) / config.batch_size))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(round(config.optimizer.warmup_epochs * steps_per_epoch))
    correction_interval = max(1, steps_per_epoch // config.corrections_per_epoch)
    correction_start = max(1, warmup_steps)
    runtime_trace_config = config.trace_wall.with_runtime_cadence(
        interval_steps=correction_interval,
        start_step=correction_start,
    )
    probe_examples = (
        runtime_trace_config.probe_batch_size
        * runtime_trace_config.probe_batches_per_correction
    )
    if probe_examples > len(train_dataset):
        raise ValueError("probe subset exceeds the training dataset")

    train_eval_loader = evaluation_loader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    test_loader = evaluation_loader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    performance_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    seed_manifest: list[dict[str, Any]] = []
    run_root = Path(output_dir) if output_dir is not None else None
    if run_root is not None:
        run_root.mkdir(parents=True, exist_ok=True)

    for seed in config.seeds:
        set_seed(seed)
        initial_model = model_factory()
        initial_state = copy.deepcopy(initial_model.state_dict())
        initial_checksum = state_dict_checksum(initial_state)

        baseline_model = model_factory().to(device)
        trace_model = model_factory().to(device)
        baseline_model.load_state_dict(initial_state)
        trace_model.load_state_dict(initial_state)
        baseline_checksum = state_dict_checksum(baseline_model.state_dict())
        trace_checksum = state_dict_checksum(trace_model.state_dict())
        if not baseline_checksum == trace_checksum == initial_checksum:
            raise RuntimeError("paired initial states do not match")

        baseline_optimizer = build_base_optimizer(baseline_model, config.optimizer)
        trace_base_optimizer = build_base_optimizer(trace_model, config.optimizer)
        baseline_schedule = WarmupCosineSchedule(
            baseline_optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            minimum_ratio=config.optimizer.minimum_learning_rate_ratio,
        )
        trace_schedule = WarmupCosineSchedule(
            trace_base_optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            minimum_ratio=config.optimizer.minimum_learning_rate_ratio,
        )
        trace_wrapper = ECSProbeLossTraceWall(
            trace_model,
            trace_base_optimizer,
            runtime_trace_config,
        )
        probe_sampler = RotatingSubsetSampler(
            len(train_dataset),
            seed=int(seed) + runtime_trace_config.probe_seed_offset,
        )

        baseline_step = 0
        previous_ranks = {"baseline": {}, "trace_wall": {}}
        seed_dir = run_root / "seeds" / f"seed_{seed}" if run_root else None
        checkpoint_dir = seed_dir / "checkpoints" if seed_dir else None
        if checkpoint_dir and config.save_epoch_checkpoints:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def measure_epoch(
            epoch: int,
            *,
            baseline_lr: float,
            trace_lr: float,
            baseline_time: float,
            trace_time: float,
            attempts: int,
            accepts: int,
        ) -> None:
            baseline_train = evaluate(
                baseline_model,
                train_eval_loader,
                device=device,
                max_batches=config.train_eval_max_batches,
            )
            baseline_test = evaluate(
                baseline_model,
                test_loader,
                device=device,
            )
            trace_train = evaluate(
                trace_model,
                train_eval_loader,
                device=device,
                max_batches=config.train_eval_max_batches,
            )
            trace_test = evaluate(trace_model, test_loader, device=device)
            performance_rows.extend(
                [
                    performance_row(
                        arm="baseline",
                        seed=seed,
                        epoch=epoch,
                        global_step=baseline_step,
                        train_metrics=baseline_train,
                        test_metrics=baseline_test,
                        learning_rate=baseline_lr,
                        parameter_norm=parameter_l2_norm(baseline_model),
                        epoch_train_time_sec=baseline_time,
                        correction_attempts=0,
                        correction_accepts=0,
                    ),
                    performance_row(
                        arm="trace_wall",
                        seed=seed,
                        epoch=epoch,
                        global_step=trace_wrapper.global_step,
                        train_metrics=trace_train,
                        test_metrics=trace_test,
                        learning_rate=trace_lr,
                        parameter_norm=parameter_l2_norm(trace_model),
                        epoch_train_time_sec=trace_time,
                        correction_attempts=attempts,
                        correction_accepts=accepts,
                    ),
                ]
            )
            spectral_rows.extend(
                measure_spectral(
                    baseline_model,
                    arm="baseline",
                    seed=seed,
                    epoch=epoch,
                    global_step=baseline_step,
                    config=config,
                    previous_ranks=previous_ranks["baseline"],
                )
            )
            spectral_rows.extend(
                measure_spectral(
                    trace_model,
                    arm="trace_wall",
                    seed=seed,
                    epoch=epoch,
                    global_step=trace_wrapper.global_step,
                    config=config,
                    previous_ranks=previous_ranks["trace_wall"],
                )
            )
            if progress:
                baseline_row = performance_rows[-2]
                trace_row = performance_rows[-1]
                print(
                    f"seed={seed} epoch={epoch:03d} | "
                    f"baseline test loss={baseline_row['test_loss']:.4f} "
                    f"acc={baseline_row['test_accuracy']:.4f} | "
                    f"TraceWall test loss={trace_row['test_loss']:.4f} "
                    f"acc={trace_row['test_accuracy']:.4f} | "
                    f"corrections={accepts}/{attempts}",
                    flush=True,
                )

        initial_lr = config.optimizer.peak_learning_rate / max(warmup_steps, 1)
        measure_epoch(
            0,
            baseline_lr=initial_lr,
            trace_lr=initial_lr,
            baseline_time=0.0,
            trace_time=0.0,
            attempts=0,
            accepts=0,
        )

        for epoch in range(1, config.epochs + 1):
            order = ordered_epoch_indices(
                len(train_dataset), seed=seed, epoch=epoch
            )
            baseline_loader = loader_for_indices(
                train_dataset,
                order,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
            trace_loader = loader_for_indices(
                train_dataset,
                order,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            start = time.perf_counter()
            baseline_step, _, _, baseline_lr = train_baseline_epoch(
                baseline_model,
                baseline_optimizer,
                baseline_schedule,
                baseline_loader,
                device=device,
                global_step=baseline_step,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            baseline_time = time.perf_counter() - start

            start = time.perf_counter()
            _, _, trace_lr, epoch_corrections, attempts, accepts = (
                train_trace_wall_epoch(
                    trace_model,
                    trace_wrapper,
                    trace_schedule,
                    trace_loader,
                    train_dataset=train_dataset,
                    probe_sampler=probe_sampler,
                    device=device,
                    seed=seed,
                    epoch=epoch,
                    gradient_clip_norm=config.gradient_clip_norm,
                )
            )
            trace_time = time.perf_counter() - start
            correction_rows.extend(epoch_corrections)

            measure_epoch(
                epoch,
                baseline_lr=baseline_lr,
                trace_lr=trace_lr,
                baseline_time=baseline_time,
                trace_time=trace_time,
                attempts=attempts,
                accepts=accepts,
            )

            if checkpoint_dir and config.save_epoch_checkpoints:
                torch.save(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "model": baseline_model.state_dict(),
                        "optimizer": baseline_optimizer.state_dict(),
                        "schedule": baseline_schedule.state_dict(),
                    },
                    checkpoint_dir / f"baseline_epoch_{epoch:03d}.pt",
                )
                torch.save(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "model": trace_model.state_dict(),
                        "trace_wall_optimizer": trace_wrapper.state_dict(),
                        "schedule": trace_schedule.state_dict(),
                        "probe_sampler": probe_sampler.state_dict(),
                    },
                    checkpoint_dir / f"trace_wall_epoch_{epoch:03d}.pt",
                )

        seed_manifest.append(
            {
                "seed": int(seed),
                "initial_checksum": initial_checksum,
                "baseline_initial_checksum": baseline_checksum,
                "trace_wall_initial_checksum": trace_checksum,
                "baseline_final_checksum": state_dict_checksum(
                    baseline_model.state_dict()
                ),
                "trace_wall_final_checksum": state_dict_checksum(
                    trace_model.state_dict()
                ),
                "baseline_global_step": int(baseline_step),
                "trace_wall_global_step": int(trace_wrapper.global_step),
                "probe_sampler_cycle": int(probe_sampler.cycle),
                "probe_sampler_position": int(probe_sampler.position),
            }
        )
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "seed": seed,
                    "model": baseline_model.state_dict(),
                    "optimizer": baseline_optimizer.state_dict(),
                    "schedule": baseline_schedule.state_dict(),
                },
                seed_dir / "baseline_final_state.pt",
            )
            torch.save(
                {
                    "seed": seed,
                    "model": trace_model.state_dict(),
                    "trace_wall_optimizer": trace_wrapper.state_dict(),
                    "schedule": trace_schedule.state_dict(),
                    "probe_sampler": probe_sampler.state_dict(),
                },
                seed_dir / "trace_wall_final_state.pt",
            )

    performance = pd.DataFrame(performance_rows)
    spectral = pd.DataFrame(spectral_rows)
    corrections = pd.DataFrame(correction_rows)
    validate_pairing(performance, spectral, config)

    performance_summary = student_t_summary(
        performance,
        id_columns=["arm", "epoch"],
        metrics=[
            "train_loss",
            "test_loss",
            "train_accuracy",
            "test_accuracy",
            "train_perplexity",
            "test_perplexity",
            "accuracy_generalization_gap",
            "loss_generalization_gap",
            "learning_rate",
            "parameter_l2_norm",
            "epoch_train_time_sec",
        ],
    )
    spectral_summary = student_t_summary(
        spectral,
        id_columns=["arm", "layer", "epoch"],
        metrics=[
            "ecs_rank",
            "ecs_fractional_rank",
            "ecs_rank_fraction",
            "ecs_normalization_dimension",
            "ecs_bulk_effective_count",
            "ecs_trace_log",
            "ecs_trace_log_per_eval",
            "retained_energy_fraction",
            "stable_rank",
            "participation_ratio",
            "alpha",
            "ERG_gap",
            "detX_num",
            "num_pl_spikes",
        ],
    )
    manifest = {
        "optimizer": config.optimizer.name,
        "seeds": list(config.seeds),
        "epochs": config.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "correction_interval_steps": correction_interval,
        "correction_start_step": correction_start,
        "corrections_per_epoch": config.corrections_per_epoch,
        "probe_examples_per_correction": probe_examples,
        "probe_source": "rotating subset of the MNIST training set",
        "official_test_set_used_for_optimization": False,
        "device": str(device),
        "seed_runs": seed_manifest,
    }
    result = PairedExperimentResult(
        config=config,
        performance=performance,
        spectral=spectral,
        corrections=corrections,
        performance_summary=performance_summary,
        spectral_summary=spectral_summary,
        correction_summary=correction_summary(corrections),
        manifest=manifest,
        output_dir=run_root,
    )
    if run_root is not None:
        result.save(run_root)
    return result

"""Restartable long-horizon MNIST/MLP3 training for tangent-RG experiments.

This runner deliberately separates the optimization horizon from the learning
rate horizon.  Warm-up/cosine scheduling ends after ``lr_schedule_epochs`` and
then remains at the declared nonzero floor for the rest of a 1,000--10,000
epoch run.  Online metrics are cheap and per-epoch; full evaluation,
WeightWatcher, model snapshots, and ESD persistence occur only on the resolved
sparse analysis schedule.  The official test split is monitoring-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import shutil
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..config import BaselineConfig
from ..engine import choose_device, evaluate, parameter_l2_norm, set_seed
from ..io_utils import atomic_csv, atomic_npz
from ..model import MLP3
from ..optimizers import (
    build_optimizer as build_canonical_optimizer,
    optimizer_group_rows,
    warmup_cosine_learning_rate,
)
from .capture import (
    finalize_step_capture,
    list_capture_files,
    parse_capture_name,
    prepare_step_capture,
)
from .checkpoints import (
    BEST_CHECKPOINT_NAME,
    FINAL_CHECKPOINT_NAME,
    LATEST_CHECKPOINT_NAME,
    capture_rng_state,
    ensure_tail_checkpoint_cache,
    finalize_tail_checkpoint_cache,
    inspect_full_checkpoint,
    list_analysis_checkpoints,
    load_verified_tail_checkpoint_refs,
    load_full_checkpoint,
    quarantine_tail_checkpoint_cache_after_boundary,
    restore_rng_state,
    save_analysis_checkpoint,
    save_full_checkpoint,
    save_tail_checkpoint,
    verify_tail_checkpoint_cache_prefix,
)
from .config import TangentRGConfig
from .muonclip import MuonClipRMSWithAuxAdamW
from .protocol import (
    AnalysisPlan,
    RunLayout,
    atomic_json,
    build_analysis_plan,
    indices_sha256,
    make_run_layout,
    make_tail_checkpoint_layout,
    protocol_fingerprint,
    tail_checkpoint_epochs,
    validate_disjoint_checkpoint_layouts,
)
from .weightwatcher_fit import (
    WEIGHTWATCHER_PRIMARY_TAIL_SUPPORT_SOURCES,
    analyze_weightwatcher_dual,
    extract_weight_esds,
    validate_weightwatcher_measurement,
    weightwatcher_trace_log_rows,
)


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MODEL_CONTRACT = "MLP3:784-512-512-10:relu"
_ESD_NAME_PATTERN = re.compile(r"^esd_epoch_(\d+)_step_(\d+)\.npz$")


@dataclass(frozen=True)
class DataBundle:
    """Deterministic data split, loaders, and training sampler RNG."""

    train: DataLoader
    train_evaluation: DataLoader
    validation: DataLoader
    test: DataLoader
    train_generator: torch.Generator
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass
class TrainingResult:
    """Completed or loaded run metadata plus optional live training objects."""

    config: TangentRGConfig
    run_dir: Path
    completion: dict[str, Any]
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return choose_device()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _make_data_bundle(
    config: TangentRGConfig,
    *,
    device: torch.device,
) -> DataBundle:
    """Create the fixed 55k/5k split; importing torchvision stays runtime-only."""

    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    full_train = datasets.MNIST(
        config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test = datasets.MNIST(
        config.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    if len(full_train) != 60_000 or len(test) != 10_000:
        raise RuntimeError("unexpected torchvision MNIST split sizes")

    split_generator = torch.Generator(device="cpu").manual_seed(config.split_seed)
    permutation = torch.randperm(len(full_train), generator=split_generator).tolist()
    validation_indices = tuple(permutation[: config.validation_size])
    train_indices = tuple(permutation[config.validation_size :])
    train_subset = Subset(full_train, train_indices)
    validation_subset = Subset(full_train, validation_indices)

    train_generator = torch.Generator(device="cpu").manual_seed(config.seed + 101)
    workers = 0 if device.type == "mps" else config.num_workers
    common = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    return DataBundle(
        train=train_loader,
        train_evaluation=DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=False,
            **common,
        ),
        validation=DataLoader(
            validation_subset,
            batch_size=config.batch_size,
            shuffle=False,
            **common,
        ),
        test=DataLoader(
            test,
            batch_size=config.batch_size,
            shuffle=False,
            **common,
        ),
        train_generator=train_generator,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def _canonical_optimizer_config(config: TangentRGConfig) -> BaselineConfig:
    """Translate an AdamW/Muon arm to the repository's audited constructor."""

    optimizer = "adamw" if config.optimizer == "adamw" else "sgd_momentum_muon"
    kwargs: dict[str, Any] = {
        "optimizer": optimizer,
        "seed": config.seed,
        "epochs": max(2, int(math.ceil(config.lr_schedule_epochs))),
        "batch_size": config.batch_size,
        "validation_size": config.validation_size,
        "split_seed": config.split_seed,
        "num_workers": config.num_workers,
        "grad_clip_norm": config.grad_clip_norm,
        "train_eval_max_batches": config.train_eval_max_batches,
        "test_monitoring_only": True,
        "save_epoch_checkpoints": False,
        "strict_metrics": False,
    }
    if config.optimizer == "adamw":
        profile = config.adamw
        kwargs.update(
            {
                "adamw_learning_rate": profile.learning_rate,
                "adamw_min_learning_rate": profile.min_learning_rate,
                "adamw_warmup_epochs": int(round(profile.warmup_epochs)),
                "adamw_beta1": profile.beta1,
                "adamw_beta2": profile.beta2,
                "adamw_eps": profile.epsilon,
                "adamw_weight_decay": profile.weight_decay,
            }
        )
    else:
        profile = config.muon
        kwargs.update(
            {
                "muon_parameter_names": profile.parameter_names,
                "muon_learning_rate": profile.matrix_learning_rate,
                "muon_min_learning_rate": profile.matrix_min_learning_rate,
                "muon_warmup_epochs": int(round(profile.warmup_epochs)),
                "muon_momentum": profile.momentum,
                "muon_nesterov": profile.nesterov,
                "muon_weight_decay": profile.weight_decay,
                "muon_newton_schulz_steps": profile.newton_schulz_steps,
                "muon_eps": profile.epsilon,
                "muon_aux_learning_rate": profile.auxiliary_learning_rate,
                "muon_aux_min_learning_rate": profile.auxiliary_min_learning_rate,
                "muon_aux_beta1": profile.auxiliary_beta1,
                "muon_aux_beta2": profile.auxiliary_beta2,
                "muon_aux_eps": profile.auxiliary_epsilon,
                "muon_aux_weight_decay": profile.auxiliary_weight_decay,
            }
        )
    return BaselineConfig(**kwargs)


def build_training_optimizer(
    model: torch.nn.Module,
    config: TangentRGConfig,
) -> torch.optim.Optimizer:
    """Build AdamW, canonical Muon, or the explicit MuonClip-RMS arm."""

    config.validate()
    if config.optimizer in {"adamw", "muon"}:
        optimizer = build_canonical_optimizer(model, _canonical_optimizer_config(config))
        if config.optimizer == "adamw":
            names = {id(parameter): name for name, parameter in model.named_parameters()}
            for group in optimizer.param_groups:
                group["names"] = [names[id(parameter)] for parameter in group["params"]]
                group["kind"] = (
                    "adamw_decay"
                    if float(group.get("weight_decay", 0.0)) > 0.0
                    else "adamw_no_decay"
                )
        return optimizer
    profile = config.muonclip_rms
    return MuonClipRMSWithAuxAdamW(
        model.named_parameters(),
        muon_parameter_names=profile.parameter_names,
        muon_lr=profile.matrix_learning_rate,
        muon_momentum=profile.momentum,
        muon_nesterov=profile.nesterov,
        muon_weight_decay=profile.weight_decay,
        newton_schulz_steps=profile.newton_schulz_steps,
        muon_eps=profile.epsilon,
        rms_scale=profile.rms_scale,
        auxiliary_lr=profile.auxiliary_learning_rate,
        auxiliary_betas=(profile.auxiliary_beta1, profile.auxiliary_beta2),
        auxiliary_eps=profile.auxiliary_epsilon,
        auxiliary_weight_decay=profile.auxiliary_weight_decay,
    )


def scheduled_learning_rates(
    config: TangentRGConfig,
    plan: AnalysisPlan,
    *,
    update_index: int,
) -> dict[str, float]:
    """Return update-level rates, clamped at their floors after schedule end."""

    if config.optimizer == "adamw":
        profile = config.adamw
        primary = (profile.learning_rate, profile.min_learning_rate)
        auxiliary = None
    elif config.optimizer == "muon":
        profile = config.muon
        primary = (profile.matrix_learning_rate, profile.matrix_min_learning_rate)
        auxiliary = (
            profile.auxiliary_learning_rate,
            profile.auxiliary_min_learning_rate,
        )
    else:
        profile = config.muonclip_rms
        primary = (profile.matrix_learning_rate, profile.matrix_min_learning_rate)
        auxiliary = (
            profile.auxiliary_learning_rate,
            profile.auxiliary_min_learning_rate,
        )
    warmup_steps = int(round(profile.warmup_epochs * plan.steps_per_epoch))
    values = {
        "primary": warmup_cosine_learning_rate(
            update_index,
            total_steps=plan.lr_schedule_steps,
            warmup_steps=warmup_steps,
            peak_lr=primary[0],
            min_lr=primary[1],
        )
    }
    if auxiliary is not None:
        values["auxiliary"] = warmup_cosine_learning_rate(
            update_index,
            total_steps=plan.lr_schedule_steps,
            warmup_steps=warmup_steps,
            peak_lr=auxiliary[0],
            min_lr=auxiliary[1],
        )
    return values


def set_scheduled_learning_rates(
    optimizer: torch.optim.Optimizer,
    config: TangentRGConfig,
    plan: AnalysisPlan,
    *,
    update_index: int,
) -> dict[str, float]:
    """Apply the long-horizon schedule to matrix and auxiliary groups."""

    values = scheduled_learning_rates(config, plan, update_index=update_index)
    if config.optimizer == "adamw":
        for group in optimizer.param_groups:
            group["lr"] = values["primary"]
        return values
    for group in optimizer.param_groups:
        group["lr"] = (
            values["primary"]
            if group.get("kind") in {"muon", "muonclip_rms"}
            else values["auxiliary"]
        )
    return values


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    for distribution in ("torchvision", "numpy", "pandas", "weightwatcher", "powerlaw"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _determinism_settings(device: torch.device) -> dict[str, Any]:
    """Record numerical determinism knobs that can change replay semantics."""

    settings: dict[str, Any] = {
        "seeded_python_numpy_torch": True,
        "fixed_split_seed": True,
        "fixed_train_sampler_generator": True,
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "device": str(device),
    }
    if hasattr(torch.backends, "cudnn"):
        settings["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        settings["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        settings["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        settings["cuda_matmul_allow_tf32"] = bool(
            torch.backends.cuda.matmul.allow_tf32
        )
    return settings


def _optimizer_contract(config: TangentRGConfig) -> dict[str, Any]:
    if config.optimizer == "adamw":
        return {
            "implementation": "rg_baselines.optimizers.build_optimizer/torch.optim.AdamW",
            "profile": asdict(config.adamw),
        }
    if config.optimizer == "muon":
        return {
            "implementation": "rg_baselines.muon.MuonWithAuxAdamW",
            "polar_map": "rg_baselines.muon.zeropower_via_newton_schulz_5",
            "profile": asdict(config.muon),
        }
    return {
        "implementation": "rg_baselines.tangent_rg.muonclip.MuonClipRMSWithAuxAdamW",
        "polar_map": "rg_baselines.muon.zeropower_via_newton_schulz_5",
        "matrix_rule": "direction = polar(source) * rms_scale / RMS(polar(source))",
        "zero_source_policy": "zero direction; no artificial orientation",
        "qk_clipping_applicable": False,
        "qk_clipping_reason": "MLP3 has no attention query/key pair",
        "profile": asdict(config.muonclip_rms),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _upsert(
    frame: pd.DataFrame,
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    keys: list[str],
) -> pd.DataFrame:
    incoming = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if incoming.empty:
        return frame
    combined = incoming.copy() if frame.empty else pd.concat([frame, incoming], ignore_index=True, sort=False)
    return combined.drop_duplicates(keys, keep="last").reset_index(drop=True)


def _truncate(frame: pd.DataFrame, epoch: int) -> pd.DataFrame:
    if frame.empty or "epoch" not in frame:
        return frame
    return frame[pd.to_numeric(frame["epoch"], errors="coerce") <= int(epoch)].copy()


def _table_paths(layout: RunLayout) -> dict[str, Path]:
    return {
        "training": layout.metrics / "training_by_epoch.csv",
        "validation": layout.metrics / "validation_by_epoch.csv",
        "performance": layout.metrics / "performance_by_analysis_epoch.csv",
        "optimizer_groups": layout.metrics / "optimizer_groups_by_analysis_epoch.csv",
        "weightwatcher": layout.metrics / "weightwatcher_fits.csv",
        "trace_log": layout.metrics / "trace_log.csv",
        "spectral": layout.metrics / "spectral_metrics_by_analysis_epoch.csv",
        "analysis_status": layout.metrics / "analysis_status_by_epoch.csv",
    }


def _persist_tables(tables: dict[str, pd.DataFrame], paths: dict[str, Path]) -> None:
    for name, frame in tables.items():
        # A resume truncation may legitimately remove every row.  If a table
        # already exists, replace it with its header-only truncated form rather
        # than leaving future rows from the discarded trajectory on disk.
        if not frame.empty or (paths[name].is_file() and len(frame.columns) > 0):
            atomic_csv(frame, paths[name])


def _quarantine_artifacts_after_resume_boundary(
    layout: RunLayout,
    *,
    epoch: int,
    global_step: int,
    best_validation_epoch: int,
    protocol_fingerprint: str,
) -> tuple[str, ...]:
    """Move stale future artifacts aside after restoring an older latest state."""

    stale: list[Path] = []
    stale.extend(
        ref.path
        for ref in list_analysis_checkpoints(layout.checkpoints)
        if ref.epoch > epoch or ref.global_step > global_step
    )
    for path in (layout.metrics / "esd").glob("esd_epoch_*_step_*.npz"):
        match = _ESD_NAME_PATTERN.fullmatch(path.name)
        if match and (int(match.group(1)) > epoch or int(match.group(2)) > global_step):
            stale.append(path)
    stale.extend(
        path
        for path in list_capture_files(layout.captures)
        if parse_capture_name(path) > global_step
    )

    def full_checkpoint_metadata(path: Path) -> dict[str, Any] | None:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None
        if not isinstance(payload, dict) or payload.get("checkpoint_kind") != "full_restart":
            return None
        return payload

    def payload_epoch(payload: dict[str, Any] | None) -> int | None:
        try:
            return int(payload["epoch"]) if payload is not None else None
        except (KeyError, TypeError, ValueError):
            return None

    best_path = layout.checkpoints / BEST_CHECKPOINT_NAME
    if best_path.is_file():
        best_payload = full_checkpoint_metadata(best_path)
        best_epoch = payload_epoch(best_payload)
        best_matches_latest = bool(
            best_payload is not None
            and best_epoch == int(best_validation_epoch)
            and best_epoch is not None
            and best_epoch <= int(epoch)
            and str(best_payload.get("checkpoint_role")) == "best_validation"
            and str(best_payload.get("protocol_fingerprint"))
            == str(protocol_fingerprint)
        )
        if not best_matches_latest:
            stale.append(best_path)

    # ``run_complete.json`` is checked before this function is called.  Without
    # that commit marker, final is a mutable by-product of a partial/discarded
    # completion sequence.  Quarantine it and require the resumed path to
    # recreate it after all final guards pass, even if its epoch equals latest.
    final_path = layout.checkpoints / FINAL_CHECKPOINT_NAME
    if final_path.is_file():
        stale.append(final_path)
    if not stale:
        return ()

    quarantine = (
        layout.root
        / "resume_quarantine"
        / f"boundary_epoch_{epoch:05d}_step_{global_step:09d}_{time.time_ns()}"
    )
    moved: list[str] = []
    for source in sorted(set(stale)):
        relative = source.relative_to(layout.root)
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        moved.append(str(relative))
    atomic_json(
        {
            "reason": (
                "artifact is after the restored boundary or mutable checkpoint "
                "metadata disagrees with checkpoint_latest"
            ),
            "resume_epoch": int(epoch),
            "resume_global_step": int(global_step),
            "best_validation_epoch_from_latest": int(best_validation_epoch),
            "moved": moved,
        },
        quarantine / "quarantine_manifest.json",
    )
    return tuple(moved)


def _joined_spectral_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join independently supported PL trace-log rows onto WW fit variants."""

    fits = tables["weightwatcher"]
    traces = tables["trace_log"]
    if fits.empty:
        return pd.DataFrame()
    trace_columns = [
        "optimizer",
        "seed",
        "epoch",
        "global_step",
        "layer",
        "fit_variant",
        "trace_log_total",
        "trace_log_per_eval",
        "support_rank",
        "support_rank_source",
        "support_selected_from_same_trace_log",
        "normalization_dimension",
        "lambda_cut_scaled",
        "trace_log_status",
        "support_window_start_descending_zero_based",
        "support_window_end_descending_exclusive",
        "support_window_source",
        "pl_support_rank_before_finger_clip",
        "effective_fit_tail_rank",
        "primary_effective_fit_tail_rank",
        "sensitivity_only",
        "certification_eligible",
        "qualification_role",
    ]
    if traces.empty:
        joined = fits.copy()
        for column in trace_columns[6:]:
            joined[column] = np.nan
        return joined
    primary = traces[
        traces["support_rank_source"].isin(
            WEIGHTWATCHER_PRIMARY_TAIL_SUPPORT_SOURCES
        )
    ][trace_columns].copy()
    return fits.merge(
        primary,
        on=[
            "optimizer",
            "seed",
            "epoch",
            "global_step",
            "layer",
            "fit_variant",
        ],
        how="left",
        validate="one_to_one",
    )


def _checkpoint_arguments(
    *,
    config: TangentRGConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    best_validation_epoch: int,
    train_generator: torch.Generator,
    fingerprint: str,
    role: str,
) -> dict[str, Any]:
    return {
        "config": config.to_dict(),
        "model": model,
        "optimizer": optimizer,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "best_validation_epoch": best_validation_epoch,
        "train_generator": train_generator,
        "protocol_fingerprint": fingerprint,
        "checkpoint_role": role,
    }


def _train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    *,
    config: TangentRGConfig,
    plan: AnalysisPlan,
    device: torch.device,
    epoch: int,
    global_step: int,
    layout: RunLayout,
    fingerprint: str,
) -> dict[str, Any]:
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    gradient_norms: list[float] = []
    last_lrs: dict[str, float] = {"primary": float("nan")}
    completed = int(global_step)

    for batch_index, (inputs, targets) in enumerate(loader):
        next_completed = completed + 1
        burst = plan.burst_for_completed_step(next_completed)
        is_calibration_step = (
            burst is not None and next_completed == burst.first_completed_step
        )
        calibration_inputs = inputs.detach().cpu().clone() if is_calibration_step else None
        calibration_targets = targets.detach().cpu().clone() if is_calibration_step else None
        pre_forward_rng = capture_rng_state() if is_calibration_step else None
        last_lrs = set_scheduled_learning_rates(
            optimizer,
            config,
            plan,
            update_index=completed,
        )
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip_norm,
        )
        pending = None
        if burst is not None:
            pending = prepare_step_capture(
                model,
                optimizer,
                optimizer_name=config.optimizer,
                parameter_names=config.capture_parameter_names,
                epoch=epoch - 1,
                batch_index=batch_index,
                completed_step=next_completed,
                burst=burst,
                learning_rates=last_lrs,
                protocol_fingerprint=fingerprint,
                calibration_inputs=calibration_inputs,
                calibration_targets=calibration_targets,
                rng_state_before_forward=pre_forward_rng,
                grad_clip_norm=config.grad_clip_norm,
            )
        optimizer.step()
        if pending is not None:
            finalize_step_capture(pending, capture_root=layout.captures)
        completed = next_completed

        examples = int(targets.numel())
        loss_sum += float(loss.detach().cpu()) * examples
        correct += int((logits.argmax(1) == targets).sum().detach().cpu())
        seen += examples
        gradient_norms.append(float(torch.as_tensor(norm).detach().cpu()))

    norms = np.asarray(gradient_norms, dtype=float)
    return {
        "optimizer": config.optimizer,
        "seed": config.seed,
        "epoch": int(epoch),
        "global_step": completed,
        "online_train_loss": loss_sum / max(seen, 1),
        "online_train_accuracy": correct / max(seen, 1),
        "examples": seen,
        "batches": len(gradient_norms),
        "mean_gradient_norm_before_clip": float(norms.mean()),
        "median_gradient_norm_before_clip": float(np.median(norms)),
        "max_gradient_norm_before_clip": float(norms.max()),
        "primary_lr": float(last_lrs["primary"]),
        "auxiliary_lr": float(last_lrs.get("auxiliary", np.nan)),
        "parameter_l2_norm": parameter_l2_norm(model),
    }


def _validation_row(
    result: dict[str, Any],
    *,
    config: TangentRGConfig,
    epoch: int,
    global_step: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "optimizer": config.optimizer,
        "seed": config.seed,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "step": int(global_step),
        "validation_loss": float(result["loss"]),
        "validation_accuracy": float(result["accuracy"]),
        "validation_examples": int(result["examples"]),
        "evaluation_reason": str(reason),
        "checkpoint_selection_eligible": 1,
        "test_used_for_selection": 0,
    }


def _measure_sparse_analysis(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    config: TangentRGConfig,
    plan: AnalysisPlan,
    data: DataBundle,
    device: torch.device,
    epoch: int,
    global_step: int,
    online: Optional[dict[str, Any]],
    layout: RunLayout,
    fingerprint: str,
    tables: dict[str, pd.DataFrame],
    table_paths: dict[str, Path],
) -> dict[str, Any]:
    """Evaluate and run dual WW fits at one scheduled model state only."""

    evaluation_started = time.perf_counter()
    train_result = evaluate(
        model,
        data.train_evaluation,
        device=device,
        max_batches=config.train_eval_max_batches,
    )
    validation_result = evaluate(model, data.validation, device=device)
    test_result = evaluate(model, data.test, device=device)
    evaluation_time = time.perf_counter() - evaluation_started
    rates = scheduled_learning_rates(
        config,
        plan,
        update_index=min(global_step, plan.total_steps - 1),
    )

    tables["validation"] = _upsert(
        tables["validation"],
        [
            _validation_row(
                validation_result,
                config=config,
                epoch=epoch,
                global_step=global_step,
                reason="sparse_analysis",
            )
        ],
        keys=["optimizer", "seed", "epoch"],
    )

    metadata = {
        "optimizer": config.optimizer,
        "seed": config.seed,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "step": int(global_step),
        "protocol_fingerprint": fingerprint,
    }
    esds = extract_weight_esds(model)
    esd_path = (
        layout.metrics
        / "esd"
        / f"esd_epoch_{epoch:05d}_step_{global_step:09d}.npz"
    )
    atomic_npz(esds, esd_path)

    ww_status = "disabled"
    ww_error = ""
    ww_structural_errors: tuple[str, ...] = ()
    ww_primary_fit_failures: tuple[str, ...] = ()
    ww_raw_audit_warnings: tuple[str, ...] = ()
    ww_time = 0.0
    if config.weightwatcher_enabled:
        rng_state = capture_rng_state()
        ww_started = time.perf_counter()
        try:
            measurement = analyze_weightwatcher_dual(
                model,
                min_evals=config.weightwatcher_min_evals,
                max_evals=config.weightwatcher_max_evals,
                max_fingers=config.weightwatcher_max_fingers,
                svd_method=config.weightwatcher_svd_method,
                randomize=config.weightwatcher_randomize,
                analysis_seed=(
                    config.seed
                    + config.weightwatcher_analysis_seed_offset
                    + int(epoch)
                ),
                primary_variant=config.weightwatcher_primary_variant,
                metadata=metadata,
            )
            tables["weightwatcher"] = _upsert(
                tables["weightwatcher"],
                measurement.details,
                keys=["optimizer", "seed", "epoch", "layer", "fit_variant"],
            )
            # Persist standardized rows even when the strict primary gate below
            # fails; this is the forensic artifact needed to diagnose the run.
            tables["spectral"] = _joined_spectral_table(tables)
            validation = validate_weightwatcher_measurement(
                measurement,
                primary_variant=config.weightwatcher_primary_variant,
            )
            ww_structural_errors = validation.structural_errors
            ww_primary_fit_failures = validation.primary_fit_failures
            ww_raw_audit_warnings = validation.raw_audit_warnings
            if ww_structural_errors:
                raise RuntimeError(
                    "structurally unusable WeightWatcher acquisition: "
                    + " | ".join(ww_structural_errors)
                )
            trace_rows = pd.concat(
                [
                    weightwatcher_trace_log_rows(
                        measurement,
                        fit_variant=variant,
                        metadata=metadata,
                    )
                    for variant in ("raw", "clip_xmax")
                ],
                ignore_index=True,
                sort=False,
            )
            tables["trace_log"] = _upsert(
                tables["trace_log"],
                trace_rows,
                keys=[
                    "optimizer",
                    "seed",
                    "epoch",
                    "layer",
                    "fit_variant",
                    "support_rank_source",
                ],
            )
            tables["spectral"] = _joined_spectral_table(tables)
            nonfatal = [
                *(f"primary fit nonfatal: {value}" for value in ww_primary_fit_failures),
                *(f"raw audit nonfatal: {value}" for value in ww_raw_audit_warnings),
            ]
            if nonfatal:
                ww_status = "ok_with_nonfatal_fit_failures"
                ww_error = " | ".join(nonfatal)
            else:
                ww_status = "ok"
        except Exception as error:
            ww_status = "failed"
            ww_error = f"{type(error).__name__}: {error}"
        finally:
            restore_rng_state(rng_state)
            ww_time = time.perf_counter() - ww_started

    status_row = {
        **metadata,
        "weightwatcher_status": ww_status,
        "weightwatcher_error": ww_error,
        "structural_error_count": len(ww_structural_errors),
        "structural_errors": " | ".join(ww_structural_errors),
        "primary_fit_failure_count": len(ww_primary_fit_failures),
        "primary_fit_failures": " | ".join(ww_primary_fit_failures),
        "raw_audit_warning_count": len(ww_raw_audit_warnings),
        "raw_audit_warnings": " | ".join(ww_raw_audit_warnings),
        "fit_failure_policy": "persist_reject_qualification_do_not_abort_training",
        "weightwatcher_required": int(config.weightwatcher_required),
        "esd_path": str(esd_path.relative_to(layout.root)),
        "evaluation_time_sec": evaluation_time,
        "weightwatcher_time_sec": ww_time,
    }
    tables["analysis_status"] = _upsert(
        tables["analysis_status"],
        [status_row],
        keys=["optimizer", "seed", "epoch"],
    )

    online_values = online or {}
    performance_row = {
        **metadata,
        "primary_lr": float(rates["primary"]),
        "auxiliary_lr": float(rates.get("auxiliary", np.nan)),
        "train_loss": float(train_result["loss"]),
        "train_accuracy": float(train_result["accuracy"]),
        "train_examples_evaluated": int(train_result["examples"]),
        "validation_loss": float(validation_result["loss"]),
        "validation_accuracy": float(validation_result["accuracy"]),
        "validation_examples_evaluated": int(validation_result["examples"]),
        "test_loss": float(test_result["loss"]),
        "test_accuracy": float(test_result["accuracy"]),
        "test_examples_evaluated": int(test_result["examples"]),
        "test_monitoring_only": 1,
        "test_used_for_selection": 0,
        "online_train_loss": float(online_values.get("online_train_loss", np.nan)),
        "online_train_accuracy": float(
            online_values.get("online_train_accuracy", np.nan)
        ),
        "parameter_l2_norm": parameter_l2_norm(model),
        "evaluation_time_sec": evaluation_time,
        "weightwatcher_time_sec": ww_time,
        "weightwatcher_status": ww_status,
        "esd_path": str(esd_path.relative_to(layout.root)),
        "device": str(device),
    }
    tables["performance"] = _upsert(
        tables["performance"],
        [performance_row],
        keys=["optimizer", "seed", "epoch"],
    )
    tables["optimizer_groups"] = _upsert(
        tables["optimizer_groups"],
        pd.DataFrame(
            optimizer_group_rows(
                optimizer,
                epoch=epoch,
                optimizer_label=config.optimizer_label,
            )
        ).assign(optimizer=config.optimizer, seed=config.seed),
        keys=["optimizer", "seed", "epoch", "group_index"],
    )
    _persist_tables(tables, table_paths)
    if ww_status == "failed" and config.weightwatcher_required:
        raise RuntimeError(
            f"required WeightWatcher analysis failed at epoch {epoch}: {ww_error}"
        )
    return {
        "validation": validation_result,
        "test": test_result,
        "weightwatcher_status": ww_status,
    }


def _completion_from_disk(path: Path, fingerprint: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("protocol_fingerprint")) != str(fingerprint):
        raise RuntimeError("completed run uses a different protocol fingerprint")
    return payload


def run_training(
    config: TangentRGConfig,
    *,
    output_root: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
    resume: bool = False,
    overwrite: bool = False,
    progress: bool = True,
) -> TrainingResult:
    """Train, resume, or load one preregistered optimizer/seed run.

    ``output_root`` replaces only the root component; the suite, optimizer, and
    seed directories remain protocol-defined.  ``resume`` and ``overwrite``
    are mutually exclusive.  No test-set value affects best-checkpoint choice.
    """

    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if output_root is not None:
        config = config.with_overrides(run_root=str(output_root))
    config.validate()
    layout = make_run_layout(config)
    tail_layout = make_tail_checkpoint_layout(config)
    validate_disjoint_checkpoint_layouts(layout, tail_layout)
    expected_tail_epochs = tail_checkpoint_epochs(config.epochs)
    requested_device = str(device) if device is not None else config.device
    selected_device = _resolve_device(requested_device)
    set_seed(config.seed)
    data = _make_data_bundle(config, device=selected_device)
    model = MLP3().to(selected_device)
    optimizer = build_training_optimizer(model, config)
    plan = build_analysis_plan(config, steps_per_epoch=len(data.train))
    software_versions = _software_versions()
    determinism_settings = _determinism_settings(selected_device)

    fingerprint_payload = {
        "config": config.to_dict(),
        "analysis_plan": plan.to_dict(),
        "dataset": "torchvision.datasets.MNIST",
        "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD},
        "model": MODEL_CONTRACT,
        "initialization": MLP3.initialization_name,
        "resolved_device": str(selected_device),
        "software_versions": software_versions,
        "determinism_settings": determinism_settings,
        "train_indices_sha256": indices_sha256(data.train_indices),
        "validation_indices_sha256": indices_sha256(data.validation_indices),
        "test_monitoring_only": True,
    }
    fingerprint = protocol_fingerprint(fingerprint_payload)
    tail_cache_preexisting = tail_layout.root.exists() and any(
        tail_layout.root.iterdir()
    )
    if overwrite and layout.root.exists():
        shutil.rmtree(layout.root)
    if overwrite and tail_layout.root.exists():
        shutil.rmtree(tail_layout.root)
    elif layout.root.exists() and any(layout.root.iterdir()) and not resume:
        raise FileExistsError(
            f"run directory is non-empty; pass resume=True or overwrite=True: {layout.root}"
        )
    elif tail_cache_preexisting and not resume:
        raise FileExistsError(
            "tail checkpoint cache is non-empty; pass resume=True or overwrite=True: "
            f"{tail_layout.root}"
        )
    layout.create()

    if layout.manifest.is_file():
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
        if str(manifest.get("protocol_fingerprint")) != fingerprint:
            raise RuntimeError("existing run directory has a different protocol")
        manifest_identity = (
            str(manifest.get("suite_name")),
            str(manifest.get("optimizer")),
            str(manifest.get("seed")),
        )
        expected_identity = (config.suite_name, config.optimizer, str(config.seed))
        if manifest_identity != expected_identity:
            raise RuntimeError(
                "existing manifest suite/optimizer/seed identity is inconsistent"
            )
        expected_tail_contract = {
            "cache_dir": str(tail_layout.root),
            "checkpoint_count": len(expected_tail_epochs),
            "first_epoch": expected_tail_epochs[0],
            "last_epoch": expected_tail_epochs[-1],
            "initialization_epoch_included": False,
        }
        if manifest.get("tail_checkpoint_cache") != expected_tail_contract:
            raise RuntimeError(
                "existing run manifest tail-checkpoint cache contract is missing "
                "or inconsistent"
            )
    else:
        manifest = {
            "schema_version": config.schema_version,
            "suite_name": config.suite_name,
            "protocol_fingerprint": fingerprint,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "optimizer": config.optimizer,
            "optimizer_label": config.optimizer_label,
            "optimizer_contract": _optimizer_contract(config),
            "seed": config.seed,
            "dataset": "torchvision.datasets.MNIST",
            "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD},
            "optimization_examples": len(data.train_indices),
            "validation_examples": len(data.validation_indices),
            "test_examples": 10_000,
            "train_indices_sha256": indices_sha256(data.train_indices),
            "validation_indices_sha256": indices_sha256(data.validation_indices),
            "test_monitoring_only": True,
            "test_selection_policy": "monitoring only; validation loss selects best",
            "model": MODEL_CONTRACT,
            "initialization": MLP3.initialization_name,
            "analysis_plan": plan.to_dict(),
            "device": str(selected_device),
            "software_versions": software_versions,
            "determinism_settings": determinism_settings,
            "tail_checkpoint_cache": {
                "cache_dir": str(tail_layout.root),
                "checkpoint_count": len(expected_tail_epochs),
                "first_epoch": expected_tail_epochs[0],
                "last_epoch": expected_tail_epochs[-1],
                "initialization_epoch_included": False,
            },
        }
        atomic_json(manifest, layout.manifest)
    if not layout.resolved_config.is_file():
        atomic_json(
            {
                "config": config.to_dict(),
                "analysis_plan": plan.to_dict(),
                "protocol_fingerprint": fingerprint,
            },
            layout.resolved_config,
        )
    else:
        resolved = json.loads(layout.resolved_config.read_text(encoding="utf-8"))
        if (
            str(resolved.get("protocol_fingerprint")) != fingerprint
            or protocol_fingerprint({"config": resolved.get("config")})
            != protocol_fingerprint({"config": config.to_dict()})
            or protocol_fingerprint(
                {"analysis_plan": resolved.get("analysis_plan")}
            )
            != protocol_fingerprint({"analysis_plan": plan.to_dict()})
        ):
            raise RuntimeError(
                "existing resolved_config.json is inconsistent with the frozen protocol"
            )

    # Validate the permanent run contract before creating or changing anything
    # under ephemeral /tmp.  A mistyped resume config must not strand a cache
    # manifest with the wrong fingerprint and block a subsequent correct resume.
    ensure_tail_checkpoint_cache(
        tail_layout,
        suite_name=config.suite_name,
        optimizer_name=config.optimizer,
        seed=config.seed,
        total_epochs=config.epochs,
        steps_per_epoch=plan.steps_per_epoch,
        protocol_fingerprint=fingerprint,
    )

    if layout.completion.is_file():
        completion = _completion_from_disk(layout.completion, fingerprint)
        try:
            completion_epoch = int(completion["epochs"])
            completion_step = int(completion["global_step"])
            completion_best_epoch = int(completion["best_validation_epoch"])
            completion_seed = int(completion["seed"])
            completion_tail_count = int(completion["tail_checkpoint_count"])
            completion_tail_first = int(completion["tail_checkpoint_first_epoch"])
            completion_tail_last = int(completion["tail_checkpoint_last_epoch"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("completion marker has invalid horizon metadata") from error
        if completion.get("completed") is not True:
            raise RuntimeError("completion marker is not committed as completed=true")
        if (
            str(completion.get("optimizer")) != config.optimizer
            or completion_seed != int(config.seed)
        ):
            raise RuntimeError("completion marker optimizer/seed identity is invalid")
        if (
            completion_epoch != int(config.epochs)
            or completion_step != int(plan.total_steps)
            or not 0 <= completion_best_epoch <= int(config.epochs)
        ):
            raise RuntimeError(
                "completion marker horizon or best-validation epoch is invalid"
            )
        if (
            str(completion.get("tail_checkpoint_cache_dir"))
            != str(tail_layout.root)
            or completion_tail_count != len(expected_tail_epochs)
            or completion_tail_first != expected_tail_epochs[0]
            or completion_tail_last != expected_tail_epochs[-1]
        ):
            raise RuntimeError(
                "completion marker tail-checkpoint cache contract is invalid"
            )
        inspect_full_checkpoint(
            layout.checkpoints / FINAL_CHECKPOINT_NAME,
            expected_fingerprint=fingerprint,
            expected_role="final",
            expected_epoch=config.epochs,
            expected_global_step=plan.total_steps,
        )
        inspect_full_checkpoint(
            layout.checkpoints / LATEST_CHECKPOINT_NAME,
            expected_fingerprint=fingerprint,
            expected_role="latest",
            expected_epoch=config.epochs,
            expected_global_step=plan.total_steps,
        )
        inspect_full_checkpoint(
            layout.checkpoints / BEST_CHECKPOINT_NAME,
            expected_fingerprint=fingerprint,
            expected_role="best_validation",
            expected_epoch=completion_best_epoch,
            expected_global_step=completion_best_epoch * plan.steps_per_epoch,
        )
        try:
            load_verified_tail_checkpoint_refs(
                tail_layout.root,
                expected_suite_name=config.suite_name,
                expected_optimizer_name=config.optimizer,
                expected_seed=config.seed,
                expected_fingerprint=fingerprint,
                expected_epochs=expected_tail_epochs,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "the run is complete but its /tmp tail-checkpoint cache is missing "
                "or incomplete; it cannot be reconstructed from sparse checkpoints "
                "without retraining, so no duplicate training was started"
            ) from error
        return TrainingResult(config, layout.root, completion)

    paths = _table_paths(layout)
    tables = {name: _read_csv(path) for name, path in paths.items()}
    latest_path = layout.checkpoints / LATEST_CHECKPOINT_NAME
    best_path = layout.checkpoints / BEST_CHECKPOINT_NAME
    final_path = layout.checkpoints / FINAL_CHECKPOINT_NAME
    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    best_validation_epoch = -1

    if resume and latest_path.is_file():
        state = load_full_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            train_generator=data.train_generator,
            expected_fingerprint=fingerprint,
            expected_role="latest",
        )
        model.to(selected_device)
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        if not 0 <= start_epoch <= config.epochs:
            raise RuntimeError("latest checkpoint epoch lies outside the run horizon")
        expected_step = start_epoch * plan.steps_per_epoch
        if global_step != expected_step:
            raise RuntimeError(
                f"epoch-boundary checkpoint has step {global_step}, expected {expected_step}"
            )
        best_validation_loss = float(state["best_validation_loss"])
        best_validation_epoch = int(state["best_validation_epoch"])
        if not -1 <= best_validation_epoch <= start_epoch:
            raise RuntimeError(
                "latest checkpoint best_validation_epoch lies outside its history"
            )
        if best_validation_epoch >= 0 and not np.isfinite(best_validation_loss):
            raise RuntimeError("latest checkpoint has a nonfinite best validation loss")
        quarantined = _quarantine_artifacts_after_resume_boundary(
            layout,
            epoch=start_epoch,
            global_step=global_step,
            best_validation_epoch=best_validation_epoch,
            protocol_fingerprint=fingerprint,
        )
        quarantined_tail = quarantine_tail_checkpoint_cache_after_boundary(
            tail_layout,
            epoch=start_epoch,
            global_step=global_step,
            expected_fingerprint=fingerprint,
        )
        try:
            verify_tail_checkpoint_cache_prefix(
                tail_layout,
                through_epoch=start_epoch,
                expected_fingerprint=fingerprint,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "the /tmp tail-checkpoint history at or before the resume epoch "
                "is incomplete; those immutable states cannot be reconstructed "
                "from checkpoint_latest, so training continuation was not started"
            ) from error
        tables = {name: _truncate(frame, start_epoch) for name, frame in tables.items()}
        _persist_tables(tables, paths)
        if progress:
            print(
                f"[tangent-rg] resume {config.optimizer} seed={config.seed} "
                f"epoch={start_epoch} step={global_step} "
                "quarantined_future_artifacts="
                f"{len(quarantined) + len(quarantined_tail)}"
            )
    elif resume:
        has_progress = any(
            path.is_file()
            for directory in (layout.metrics, layout.checkpoints, layout.captures)
            for path in directory.rglob("*")
        ) or any(tail_layout.checkpoints.glob("*.pt"))
        if has_progress:
            raise FileNotFoundError(f"cannot resume without {latest_path}")

    def save_restart(path: Path, role: str, epoch: int) -> None:
        save_full_checkpoint(
            path,
            **_checkpoint_arguments(
                config=config,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                best_validation_epoch=best_validation_epoch,
                train_generator=data.train_generator,
                fingerprint=fingerprint,
                role=role,
            ),
        )

    analysis_epochs = set(plan.analysis_epochs)
    tail_epochs = set(expected_tail_epochs)
    has_initial_performance = (
        not tables["performance"].empty
        and (pd.to_numeric(tables["performance"]["epoch"], errors="coerce") == 0).any()
    )
    initial_status = tables["analysis_status"]
    if initial_status.empty:
        has_successful_initial_status = False
    else:
        initial_rows = initial_status[
            pd.to_numeric(initial_status["epoch"], errors="coerce").eq(0)
        ]
        accepted_statuses = {
            "ok",
            "ok_with_nonfatal_fit_failures",
            "disabled",
        }
        has_successful_initial_status = bool(
            not initial_rows.empty
            and initial_rows["weightwatcher_status"].astype(str).isin(accepted_statuses).any()
        )
    has_initial_analysis = has_initial_performance and has_successful_initial_status
    if start_epoch == 0 and not has_initial_analysis:
        set_scheduled_learning_rates(optimizer, config, plan, update_index=0)
        save_analysis_checkpoint(
            layout.checkpoints,
            model=model,
            epoch=0,
            global_step=0,
            protocol_fingerprint=fingerprint,
            optimizer_name=config.optimizer,
            seed=config.seed,
        )
        save_restart(latest_path, "latest", 0)
        measured = _measure_sparse_analysis(
            model,
            optimizer,
            config=config,
            plan=plan,
            data=data,
            device=selected_device,
            epoch=0,
            global_step=0,
            online=None,
            layout=layout,
            fingerprint=fingerprint,
            tables=tables,
            table_paths=paths,
        )
        best_validation_loss = float(measured["validation"]["loss"])
        best_validation_epoch = 0
        save_restart(best_path, "best_validation", 0)
        save_restart(latest_path, "latest", 0)

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_started = time.perf_counter()
        online = _train_one_epoch(
            model,
            optimizer,
            data.train,
            config=config,
            plan=plan,
            device=selected_device,
            epoch=epoch,
            global_step=global_step,
            layout=layout,
            fingerprint=fingerprint,
        )
        online["train_time_sec"] = time.perf_counter() - train_started
        global_step = int(online["global_step"])
        tables["training"] = _upsert(
            tables["training"],
            [online],
            keys=["optimizer", "seed", "epoch"],
        )
        atomic_csv(tables["training"], paths["training"])

        if epoch in tail_epochs:
            save_tail_checkpoint(
                tail_layout,
                model=model,
                epoch=epoch,
                global_step=global_step,
                total_epochs=config.epochs,
                protocol_fingerprint=fingerprint,
                optimizer_name=config.optimizer,
                seed=config.seed,
            )

        validation_result: Optional[dict[str, Any]] = None
        if epoch in analysis_epochs:
            save_analysis_checkpoint(
                layout.checkpoints,
                model=model,
                epoch=epoch,
                global_step=global_step,
                protocol_fingerprint=fingerprint,
                optimizer_name=config.optimizer,
                seed=config.seed,
            )
            measured = _measure_sparse_analysis(
                model,
                optimizer,
                config=config,
                plan=plan,
                data=data,
                device=selected_device,
                epoch=epoch,
                global_step=global_step,
                online=online,
                layout=layout,
                fingerprint=fingerprint,
                tables=tables,
                table_paths=paths,
            )
            validation_result = measured["validation"]
        elif epoch % config.validation_every_epochs == 0:
            validation_result = evaluate(model, data.validation, device=selected_device)
            tables["validation"] = _upsert(
                tables["validation"],
                [
                    _validation_row(
                        validation_result,
                        config=config,
                        epoch=epoch,
                        global_step=global_step,
                        reason="selection_cadence",
                    )
                ],
                keys=["optimizer", "seed", "epoch"],
            )
            atomic_csv(tables["validation"], paths["validation"])

        if (
            validation_result is not None
            and float(validation_result["loss"]) < best_validation_loss
        ):
            best_validation_loss = float(validation_result["loss"])
            best_validation_epoch = epoch
            save_restart(best_path, "best_validation", epoch)

        if epoch % config.latest_every_epochs == 0 or epoch == config.epochs:
            save_restart(latest_path, "latest", epoch)
        if progress and (
            validation_result is not None
            or epoch == 1
            or epoch % max(1, config.epochs // 100) == 0
        ):
            validation_text = (
                ""
                if validation_result is None
                else f" val={float(validation_result['accuracy']):.4f}"
            )
            print(
                f"epoch={epoch:05d} optimizer={config.optimizer} "
                f"lr={online['primary_lr']:.3e} "
                f"online_train={online['online_train_accuracy']:.4f}{validation_text}"
            )

    if int(global_step) != int(plan.total_steps):
        raise RuntimeError(
            f"refusing final checkpoint at step {global_step}; "
            f"expected {plan.total_steps}"
        )
    final_rows = tables["performance"]
    if final_rows.empty or not {"epoch", "global_step"}.issubset(final_rows.columns):
        raise RuntimeError(
            "final epoch/step is missing from the sparse analysis schedule; "
            "refusing to commit final checkpoint"
        )
    final_match = final_rows[
        pd.to_numeric(final_rows["epoch"], errors="coerce").eq(config.epochs)
        & pd.to_numeric(final_rows["global_step"], errors="coerce").eq(
            plan.total_steps
        )
    ]
    if final_match.empty:
        raise RuntimeError(
            "final epoch/step is missing from the sparse analysis schedule; "
            "refusing to commit final checkpoint"
        )
    final_row = final_match.iloc[-1]
    tail_refs = finalize_tail_checkpoint_cache(
        tail_layout,
        expected_fingerprint=fingerprint,
    )
    save_restart(final_path, "final", config.epochs)
    save_restart(latest_path, "latest", config.epochs)
    completion = {
        "completed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": config.optimizer,
        "optimizer_label": config.optimizer_label,
        "seed": config.seed,
        "epochs": config.epochs,
        "global_step": global_step,
        "lr_schedule_epochs": config.lr_schedule_epochs,
        "lr_schedule_steps": plan.lr_schedule_steps,
        "best_validation_epoch": best_validation_epoch,
        "best_validation_loss": best_validation_loss,
        "final_test_loss_monitoring_only": float(final_row["test_loss"]),
        "final_test_accuracy_monitoring_only": float(final_row["test_accuracy"]),
        "test_used_for_selection": False,
        "protocol_fingerprint": fingerprint,
        "tail_checkpoint_cache_dir": str(tail_layout.root),
        "tail_checkpoint_count": len(tail_refs),
        "tail_checkpoint_first_epoch": tail_refs[0].epoch,
        "tail_checkpoint_last_epoch": tail_refs[-1].epoch,
    }
    atomic_json(completion, layout.completion)
    return TrainingResult(config, layout.root, completion, model, optimizer)


__all__ = [
    "DataBundle",
    "TrainingResult",
    "build_training_optimizer",
    "run_training",
    "scheduled_learning_rates",
    "set_scheduled_learning_rates",
]

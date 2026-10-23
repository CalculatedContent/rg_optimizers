from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from .checkpoints import (
    save_epoch_model_checkpoint,
    save_training_checkpoint,
)
from .config import (
    SUPPORTED_OPTIMIZERS,
    epoch_step_map,
    load_config,
    lr_schedule_steps,
    max_steps,
    optimizer_profile,
    protocol_fingerprint,
    tokens_per_step,
    warmup_steps,
)
from .data import load_memmaps
from .evaluation import (
    evaluate_probe,
    fixed_bleu_probe,
    fixed_probe,
    random_batch,
)
from .model import GPT, GPTConfig
from .optimizers import (
    make_optimizer_handles,
    optimizer_step,
    set_learning_rates,
    zero_grad,
)
from .random_canaries import RandomCanaryExperiment
from .run_utils import (
    EPOCH_FIELDS,
    METRIC_FIELDS,
    checkpoint_eval,
    prepare_csv,
    run_directory,
    write_manifest,
)
from .runtime import (
    gradient_norm,
    model_weight_norm,
    parameter_snapshot,
    runtime_metadata,
    seed_everything,
    synchronize,
    tree_to_cpu,
    update_norm,
)
from .spectral import run_weightwatcher


@dataclass(frozen=True)
class DistributedBatchPlan:
    world_size: int
    batch_size_per_rank: int
    block_size: int
    global_grad_accum_steps: int
    local_grad_accum_steps: int
    global_sequences_per_step: int
    global_tokens_per_step: int
    local_tokens_per_step: int


def derive_distributed_batch_plan(
    cfg: dict[str, Any],
    *,
    world_size: int,
) -> DistributedBatchPlan:
    """Preserve the configured global optimizer batch across TPU ranks.

    The historical single-device protocol interprets ``grad_accum_steps`` as
    the total number of micro-batches contributing to one optimizer update.
    In data parallel execution those micro-batches are partitioned evenly
    across ranks; they are not multiplied by the number of devices.
    """

    world_size = int(world_size)
    if world_size < 2:
        raise ValueError("distributed TPU execution requires world_size >= 2")

    batch_size = int(cfg["training"]["batch_size"])
    block_size = int(cfg["model"]["block_size"])
    global_accum = int(cfg["training"]["grad_accum_steps"])

    if global_accum % world_size != 0:
        raise ValueError(
            "training.grad_accum_steps must be divisible by the TPU world size "
            f"to preserve the configured global batch: "
            f"grad_accum_steps={global_accum}, world_size={world_size}"
        )

    local_accum = global_accum // world_size
    if local_accum < 1:
        raise ValueError(
            "training.grad_accum_steps is smaller than the TPU world size"
        )

    global_sequences = batch_size * global_accum
    global_tokens = global_sequences * block_size
    local_tokens = batch_size * local_accum * block_size

    if local_tokens * world_size != global_tokens:
        raise AssertionError("distributed batch derivation is inconsistent")
    if global_tokens != tokens_per_step(cfg):
        raise AssertionError("configured tokens_per_step changed unexpectedly")

    return DistributedBatchPlan(
        world_size=world_size,
        batch_size_per_rank=batch_size,
        block_size=block_size,
        global_grad_accum_steps=global_accum,
        local_grad_accum_steps=local_accum,
        global_sequences_per_step=global_sequences,
        global_tokens_per_step=global_tokens,
        local_tokens_per_step=local_tokens,
    )


def distributed_protocol_config(
    cfg: dict[str, Any],
    *,
    plan: DistributedBatchPlan,
) -> dict[str, Any]:
    """Return an additive distributed protocol variant.

    The original configuration remains unchanged. Extra runtime metadata makes
    the distributed run fingerprint distinct from every existing single-device
    run while preserving the configured global batch, schedule, and token
    budget.
    """

    distributed = deepcopy(cfg)
    runtime = distributed.setdefault("runtime", {})
    runtime["distributed_tpu"] = {
        "enabled": True,
        "strategy": "multiprocess_data_parallel",
        "world_size": int(plan.world_size),
        "gradient_reduction": "mean_before_optimizer_step",
        "rank_zero_owns_evaluation": True,
        "rank_zero_owns_weightwatcher": True,
        "rank_zero_owns_checkpoints": True,
        "batch_size_per_rank": int(plan.batch_size_per_rank),
        "global_grad_accum_steps": int(plan.global_grad_accum_steps),
        "local_grad_accum_steps": int(plan.local_grad_accum_steps),
        "global_sequences_per_step": int(plan.global_sequences_per_step),
        "global_tokens_per_step": int(plan.global_tokens_per_step),
        "local_tokens_per_step": int(plan.local_tokens_per_step),
    }
    return distributed


def _evaluation_due(
    step: int,
    *,
    cfg: dict[str, Any],
    epoch_steps: dict[int, float],
    total_steps: int,
) -> bool:
    return (
        step % int(cfg["training"]["eval_interval_steps"]) == 0
        or step in epoch_steps
        or step == total_steps
    )


def _checkpoint_due(
    step: int,
    *,
    cfg: dict[str, Any],
    epoch_steps: dict[int, float],
    total_steps: int,
) -> bool:
    return (
        step % int(cfg["training"]["checkpoint_interval_steps"]) == 0
        or step in epoch_steps
        or step == total_steps
    )


def _require_finite_model(model: torch.nn.Module, *, step: int) -> None:
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if not (parameter.is_floating_point() or parameter.is_complex()):
            continue
        if not bool(torch.isfinite(parameter).all().detach().cpu()):
            bad.append(name)
    if bad:
        raise FloatingPointError(
            f"non-finite model parameters at step={step}: " + ", ".join(bad)
        )


def _require_finite_metrics(
    *,
    step: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    values = {
        "train_loss": float(train_metrics["loss"]),
        "train_perplexity": float(train_metrics["perplexity"]),
        "train_bits_per_token": float(train_metrics["bits_per_token"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_top5_accuracy": float(train_metrics["top5_accuracy"]),
        "val_loss": float(val_metrics["loss"]),
        "val_perplexity": float(val_metrics["perplexity"]),
        "val_bits_per_token": float(val_metrics["bits_per_token"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_top5_accuracy": float(val_metrics["top5_accuracy"]),
    }
    bad = [name for name, value in values.items() if not math.isfinite(value)]
    if bad:
        raise FloatingPointError(
            f"non-finite metrics at step={step}: " + ", ".join(bad)
        )


def _configure_distributed_runtime(
    cfg: dict[str, Any],
    *,
    expected_world_size: int,
):
    # Imports are intentionally inside the spawned worker. PyTorch/XLA requires
    # device access to occur below torch_xla.launch().
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    world_size = int(xr.world_size())
    rank = int(xr.global_ordinal())
    if world_size != int(expected_world_size):
        raise RuntimeError(
            "TPU process count does not match the requested distributed run: "
            f"observed={world_size}, expected={expected_world_size}"
        )
    if str(xr.device_type()).upper() != "TPU":
        raise RuntimeError(f"PJRT device is not TPU: {xr.device_type()!r}")

    reduced_precision = [
        name
        for name in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16")
        if str(os.environ.get(name, "")).lower() in {"1", "true", "yes", "on"}
    ]
    if reduced_precision:
        raise RuntimeError(
            "the float32 protocol refuses TPU reduced precision from "
            + ", ".join(reduced_precision)
        )

    torch.set_float32_matmul_precision(
        str(cfg["runtime"].get("matmul_precision", "high"))
    )
    torch.use_deterministic_algorithms(
        bool(cfg["runtime"].get("deterministic_algorithms", False)),
        warn_only=bool(cfg["runtime"].get("deterministic_warn_only", True)),
    )

    device = torch_xla.device()
    return device, rank, world_size, xm, xr


def _rendezvous(xm, tag: str) -> None:
    xm.rendezvous(str(tag))


def _reduce_gradients(handles, xm) -> None:
    # Each optimizer owns a disjoint parameter subset. Reducing both handles
    # exactly once gives the mean global gradient before Muon/AdamW updates.
    for handle in handles:
        xm.reduce_gradients(handle.optimizer)


def _rank_record(
    *,
    run_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    plan: DistributedBatchPlan,
) -> None:
    distributed_dir = run_dir / "distributed"
    distributed_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": int(rank),
        "world_size": int(world_size),
        "pid": int(os.getpid()),
        "device": str(device),
        "runtime": runtime_metadata(device),
        "batch_plan": asdict(plan),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = distributed_dir / f"rank_{rank:02d}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _master_metric_row(
    *,
    step: int,
    cfg: dict[str, Any],
    train_tokens: int,
    elapsed: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    last_update_lrs: dict[str, float],
    last_grad_pre: float,
    last_grad_post: float,
    last_clipped: bool,
    weight_norm: float,
    update_norm_since_eval: float,
) -> dict[str, Any]:
    tokens_seen = int(step * tokens_per_step(cfg))
    actual_epoch = tokens_seen / max(1, int(train_tokens))
    return {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "epoch": float(actual_epoch),
        "elapsed_sec": float(elapsed),
        "tokens_per_sec": float(tokens_seen / max(elapsed, 1e-9)),
        "primary_lr": float(last_update_lrs.get("primary", float("nan"))),
        "auxiliary_lr": float(last_update_lrs.get("auxiliary", float("nan"))),
        "train_loss": float(train_metrics["loss"]),
        "train_perplexity": float(train_metrics["perplexity"]),
        "train_bits_per_token": float(train_metrics["bits_per_token"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_top5_accuracy": float(train_metrics["top5_accuracy"]),
        "val_loss": float(val_metrics["loss"]),
        "val_perplexity": float(val_metrics["perplexity"]),
        "val_bits_per_token": float(val_metrics["bits_per_token"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_top5_accuracy": float(val_metrics["top5_accuracy"]),
        "test_loss": float("nan"),
        "test_perplexity": float("nan"),
        "test_bits_per_token": float("nan"),
        "test_accuracy": float("nan"),
        "test_top5_accuracy": float("nan"),
        "test_bleu": float("nan"),
        "test_continuation_token_accuracy": float("nan"),
        "test_continuation_exact_match": float("nan"),
        "val_generalization_gap": float(
            val_metrics["loss"] - train_metrics["loss"]
        ),
        "test_generalization_gap": float("nan"),
        "grad_norm_pre_clip": float(last_grad_pre),
        "grad_norm_post_clip": float(last_grad_post),
        "gradient_clipped": int(last_clipped),
        "weight_norm": float(weight_norm),
        "update_norm_since_eval": float(update_norm_since_eval),
        "update_to_weight_ratio": float(
            update_norm_since_eval / max(weight_norm, 1e-30)
        ),
        "mps_current_allocated_mb": float("nan"),
        "mps_driver_allocated_mb": float("nan"),
    }


def _worker(index: int, payload: dict[str, Any]) -> None:
    cfg = payload["config"]
    optimizer_name = str(payload["optimizer"])
    seed = int(payload["seed"])
    data_root = Path(payload["data_root"])
    results_root = Path(payload["results_root"])
    expected_world_size = int(payload["world_size"])
    progress = bool(payload["progress"])

    device, rank, world_size, xm, _ = _configure_distributed_runtime(
        cfg,
        expected_world_size=expected_world_size,
    )
    plan = derive_distributed_batch_plan(cfg, world_size=world_size)
    is_master = rank == 0

    run_dir = run_directory(results_root, optimizer_name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    _rank_record(
        run_dir=run_dir,
        rank=rank,
        world_size=world_size,
        device=device,
        plan=plan,
    )
    _rendezvous(xm, "fourway-rank-records")

    if is_master:
        rank_files = sorted((run_dir / "distributed").glob("rank_*.json"))
        if len(rank_files) != world_size:
            raise RuntimeError(
                "not every TPU rank registered before training: "
                f"found={len(rank_files)}, expected={world_size}"
            )

    data_metadata, arrays = load_memmaps(data_root, cfg)
    train_tokens = int(data_metadata["splits"]["train"])
    total_steps = max_steps(cfg, train_tokens)
    profile = optimizer_profile(cfg, optimizer_name)
    schedule_steps = lr_schedule_steps(cfg, profile, train_tokens)
    warmup = warmup_steps(profile, schedule_steps)
    epoch_steps = epoch_step_map(cfg, train_tokens)
    fingerprint = protocol_fingerprint(
        cfg,
        optimizer=optimizer_name,
        seed=seed,
        data_metadata=data_metadata,
    )

    seed_everything(seed, device)
    model = GPT(GPTConfig(**cfg["model"])).to(device)
    # Guarantee identical starting weights even if host-process RNG behavior
    # changes across PyTorch/XLA versions.
    xm.broadcast_master_param(model)
    handles = make_optimizer_handles(model, profile)

    # Rank-specific training windows partition the original global sequence of
    # eight accumulated micro-batches into two micro-batches on each of four
    # ranks. Model and optimizer initialization remain identical on every rank.
    train_generator = torch.Generator(device="cpu").manual_seed(
        seed + 11 + rank * 1_000_003
    )

    batch_size = int(cfg["training"]["batch_size"])
    block_size = int(cfg["model"]["block_size"])
    eval_batches = int(cfg["training"]["eval_batches"])
    eval_cfg = cfg["evaluation"]

    train_probe = val_probe = test_probe = bleu_probe = None
    if is_master:
        train_probe = fixed_probe(
            arrays["train"],
            batch_size=batch_size,
            block_size=block_size,
            n_batches=eval_batches,
            seed=int(eval_cfg["train_probe_seed"]),
        )
        val_probe = fixed_probe(
            arrays["val"],
            batch_size=batch_size,
            block_size=block_size,
            n_batches=eval_batches,
            seed=int(eval_cfg["validation_probe_seed"]),
        )
        test_probe = fixed_probe(
            arrays["test"],
            batch_size=batch_size,
            block_size=block_size,
            n_batches=eval_batches,
            seed=int(eval_cfg["test_probe_seed"]),
        )
        bleu_probe = fixed_bleu_probe(
            arrays["test"],
            examples=int(eval_cfg["bleu_examples"]),
            prompt_tokens=int(eval_cfg["bleu_prompt_tokens"]),
            continuation_tokens=int(eval_cfg["bleu_continuation_tokens"]),
            seed=int(eval_cfg["bleu_probe_seed"]),
        )

    canary_dir = (
        run_dir
        if is_master
        else run_dir / ".distributed_aux" / f"rank_{rank:02d}"
    )
    canaries = RandomCanaryExperiment.from_config(
        cfg,
        seed=seed,
        total_steps=total_steps,
        run_dir=canary_dir,
    )

    initial_checkpoint = run_dir / "checkpoint_initial.pt"
    latest_checkpoint = run_dir / "checkpoint_latest.pt"
    best_checkpoint = run_dir / "checkpoint_best.pt"
    final_checkpoint = run_dir / "checkpoint_final.pt"
    metrics_path = run_dir / "metrics.csv"
    epoch_metrics_path = run_dir / "epoch_metrics.csv"

    metrics_handle = epoch_handle = None
    metrics_writer = epoch_writer = None
    previous_snapshot: list[torch.Tensor] | None = None

    best_validation_loss = float("inf")
    best_validation_step = 0
    last_grad_pre = float("nan")
    last_grad_post = float("nan")
    last_clipped = False
    last_update_lrs = {
        "primary": 0.0,
        "auxiliary": (
            0.0
            if any(handle.role == "auxiliary" for handle in handles)
            else float("nan")
        ),
    }

    if is_master:
        write_manifest(
            run_dir,
            cfg=cfg,
            data_metadata=data_metadata,
            optimizer_name=optimizer_name,
            profile=profile,
            seed=seed,
            device=device,
            data_root=data_root,
            results_root=results_root,
            total_steps=total_steps,
            schedule_steps=schedule_steps,
            warmup=warmup,
            fingerprint=fingerprint,
            model=model,
        )
        distributed_metadata = {
            "schema_version": 1,
            "strategy": "multiprocess_data_parallel",
            "world_size": world_size,
            "batch_plan": asdict(plan),
            "gradient_semantics": (
                "each rank averages its local micro-batches; gradients are "
                "then averaged across all ranks before clipping and optimizer step"
            ),
            "rank_zero_owns": [
                "evaluation",
                "WeightWatcher",
                "checkpoints",
                "metrics",
                "test audit",
            ],
        }
        (run_dir / "distributed_plan.json").write_text(
            json.dumps(distributed_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        prepare_csv(metrics_path, METRIC_FIELDS, None)
        prepare_csv(epoch_metrics_path, EPOCH_FIELDS, None)
        metrics_handle = metrics_path.open("a", newline="", encoding="utf-8")
        epoch_handle = epoch_metrics_path.open(
            "a", newline="", encoding="utf-8"
        )
        metrics_writer = csv.DictWriter(
            metrics_handle,
            fieldnames=METRIC_FIELDS,
        )
        epoch_writer = csv.DictWriter(
            epoch_handle,
            fieldnames=EPOCH_FIELDS,
        )

        save_training_checkpoint(
            initial_checkpoint,
            model=model,
            handles=handles,
            step=0,
            best_validation_loss=best_validation_loss,
            best_validation_step=best_validation_step,
            elapsed_seconds=0.0,
            fingerprint=fingerprint,
            cfg=cfg,
            optimizer_name=optimizer_name,
            seed=seed,
            train_generator=train_generator,
        )
        previous_snapshot = parameter_snapshot(model)

    _rendezvous(xm, "fourway-initialized")
    started = time.time()

    try:
        for completed_steps in range(0, total_steps + 1):
            schedule_index = min(completed_steps, schedule_steps - 1)
            next_update_lrs = set_learning_rates(
                handles,
                update_index=schedule_index,
                total_steps=schedule_steps,
                warmup_steps=warmup,
            )

            evaluation_due = _evaluation_due(
                completed_steps,
                cfg=cfg,
                epoch_steps=epoch_steps,
                total_steps=total_steps,
            )
            epoch_due = completed_steps in epoch_steps

            if evaluation_due:
                _rendezvous(xm, f"fourway-eval-start-{completed_steps}")

                if is_master:
                    assert train_probe is not None
                    assert val_probe is not None
                    assert metrics_writer is not None
                    assert metrics_handle is not None
                    assert epoch_writer is not None
                    assert epoch_handle is not None

                    synchronize(device)
                    train_metrics = evaluate_probe(model, train_probe, device)
                    val_metrics = evaluate_probe(model, val_probe, device)
                    _require_finite_metrics(
                        step=completed_steps,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                    )
                    _require_finite_model(model, step=completed_steps)

                    elapsed = time.time() - started
                    if val_metrics["loss"] < best_validation_loss:
                        best_validation_loss = float(val_metrics["loss"])
                        best_validation_step = int(completed_steps)
                        save_training_checkpoint(
                            best_checkpoint,
                            model=model,
                            handles=handles,
                            step=completed_steps,
                            best_validation_loss=best_validation_loss,
                            best_validation_step=best_validation_step,
                            elapsed_seconds=elapsed,
                            fingerprint=fingerprint,
                            cfg=cfg,
                            optimizer_name=optimizer_name,
                            seed=seed,
                            train_generator=train_generator,
                        )

                    current_snapshot = parameter_snapshot(model)
                    delta_norm = update_norm(
                        previous_snapshot,
                        current_snapshot,
                    )
                    previous_snapshot = current_snapshot
                    weight_norm = model_weight_norm(model)
                    row = _master_metric_row(
                        step=completed_steps,
                        cfg=cfg,
                        train_tokens=train_tokens,
                        elapsed=elapsed,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        last_update_lrs=last_update_lrs,
                        last_grad_pre=last_grad_pre,
                        last_grad_post=last_grad_post,
                        last_clipped=last_clipped,
                        weight_norm=weight_norm,
                        update_norm_since_eval=delta_norm,
                    )
                    metrics_writer.writerow(row)
                    metrics_handle.flush()

                    if canaries is not None:
                        canary_summary = canaries.evaluate(
                            model,
                            device=device,
                            step=completed_steps,
                            epoch=float(row["epoch"]),
                        )
                        if progress:
                            print(
                                "[one-head-fourway-canary] "
                                f"optimizer={optimizer_name} seed={seed} "
                                f"step={completed_steps} "
                                f"exact={100 * canary_summary['exact_match']:.2f}% "
                                f"token={100 * canary_summary['token_accuracy']:.2f}%",
                                flush=True,
                            )

                    if epoch_due:
                        nominal_epoch = float(epoch_steps[completed_steps])
                        checkpoint_path = save_epoch_model_checkpoint(
                            run_dir,
                            model=model,
                            step=completed_steps,
                            nominal_epoch=nominal_epoch,
                            actual_epoch=float(row["epoch"]),
                            fingerprint=fingerprint,
                            cfg=cfg,
                            optimizer_name=optimizer_name,
                            seed=seed,
                        )
                        epoch_writer.writerow(
                            {
                                **row,
                                "nominal_epoch": nominal_epoch,
                                "checkpoint_path": str(checkpoint_path),
                                "test_monitoring_only": 1,
                                "test_held_out": 1,
                            }
                        )
                        epoch_handle.flush()

                        ww_summary = run_weightwatcher(
                            model,
                            run_dir,
                            step=completed_steps,
                            tokens_seen=int(row["tokens_seen"]),
                            train_tokens=train_tokens,
                            config=cfg["weightwatcher"],
                            seed=seed,
                            fingerprint=fingerprint,
                        )
                        if progress:
                            print(
                                "[one-head-fourway-ww] "
                                f"optimizer={optimizer_name} seed={seed} "
                                f"epoch={nominal_epoch:.3f} "
                                f"alpha_clip="
                                f"{ww_summary.get('alpha_clip_xmax_median', ww_summary.get('alpha_median', float('nan'))):.3f} "
                                f"alpha_raw="
                                f"{ww_summary.get('alpha_raw_median', float('nan')):.3f}",
                                flush=True,
                            )

                    if progress:
                        remaining = total_steps - completed_steps
                        rate = completed_steps / max(elapsed, 1e-9)
                        eta = remaining / rate if rate > 0 else float("nan")
                        eta_text = (
                            "unknown"
                            if not math.isfinite(eta)
                            else f"{eta / 60:.1f}m"
                        )
                        print(
                            "[one-head-fourway] "
                            f"world_size={world_size} "
                            f"optimizer={optimizer_name} seed={seed} "
                            f"step={completed_steps}/{total_steps} "
                            f"epoch={float(row['epoch']):.3f} "
                            f"val_loss={val_metrics['loss']:.4f} "
                            f"val_acc={100 * val_metrics['accuracy']:.2f}% "
                            f"throughput={float(row['tokens_per_sec']):,.0f} tok/s "
                            f"eta={eta_text}",
                            flush=True,
                        )

                _rendezvous(xm, f"fourway-eval-end-{completed_steps}")

            if completed_steps == total_steps:
                break

            zero_grad(handles)
            for local_micro_index in range(plan.local_grad_accum_steps):
                x_cpu, y_cpu = random_batch(
                    arrays["train"],
                    batch_size=batch_size,
                    block_size=block_size,
                    generator=train_generator,
                )
                if canaries is not None:
                    global_micro_index = (
                        rank * plan.local_grad_accum_steps
                        + local_micro_index
                    )
                    x_cpu, y_cpu = canaries.inject(
                        x_cpu,
                        y_cpu,
                        completed_step=completed_steps,
                        micro_index=global_micro_index,
                    )
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError(
                        "training forward pass did not return loss"
                    )
                (loss / plan.local_grad_accum_steps).backward()

            _reduce_gradients(handles, xm)

            grad_pre_tensor = gradient_norm(model.parameters())
            clip = float(cfg["training"]["grad_clip"])
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip,
                    foreach=False,
                )
            grad_post_tensor = gradient_norm(model.parameters())

            optimizer_step(handles)
            xm.mark_step()
            last_update_lrs = dict(next_update_lrs)

            new_step = completed_steps + 1
            if is_master and (
                _evaluation_due(
                    new_step,
                    cfg=cfg,
                    epoch_steps=epoch_steps,
                    total_steps=total_steps,
                )
                or _checkpoint_due(
                    new_step,
                    cfg=cfg,
                    epoch_steps=epoch_steps,
                    total_steps=total_steps,
                )
            ):
                last_grad_pre = float(grad_pre_tensor.detach().cpu())
                last_grad_post = float(grad_post_tensor.detach().cpu())
                last_clipped = bool(last_grad_pre > clip) if clip > 0 else False

            if _checkpoint_due(
                new_step,
                cfg=cfg,
                epoch_steps=epoch_steps,
                total_steps=total_steps,
            ):
                _rendezvous(xm, f"fourway-checkpoint-start-{new_step}")
                if is_master:
                    _require_finite_model(model, step=new_step)
                    save_training_checkpoint(
                        latest_checkpoint,
                        model=model,
                        handles=handles,
                        step=new_step,
                        best_validation_loss=best_validation_loss,
                        best_validation_step=best_validation_step,
                        elapsed_seconds=time.time() - started,
                        fingerprint=fingerprint,
                        cfg=cfg,
                        optimizer_name=optimizer_name,
                        seed=seed,
                        train_generator=train_generator,
                    )
                _rendezvous(xm, f"fourway-checkpoint-end-{new_step}")

        _rendezvous(xm, "fourway-finalize-start")
        if is_master:
            elapsed_total = time.time() - started
            for checkpoint in (final_checkpoint, latest_checkpoint):
                save_training_checkpoint(
                    checkpoint,
                    model=model,
                    handles=handles,
                    step=total_steps,
                    best_validation_loss=best_validation_loss,
                    best_validation_step=best_validation_step,
                    elapsed_seconds=elapsed_total,
                    fingerprint=fingerprint,
                    cfg=cfg,
                    optimizer_name=optimizer_name,
                    seed=seed,
                    train_generator=train_generator,
                )

            assert test_probe is not None
            assert bleu_probe is not None
            final_state = tree_to_cpu(model.state_dict())
            final_test = checkpoint_eval(
                final_checkpoint,
                model=model,
                test_probe=test_probe,
                bleu_probe=bleu_probe,
                device=device,
                bleu_batch_size=int(eval_cfg["bleu_batch_size"]),
            )
            best_test = checkpoint_eval(
                best_checkpoint,
                model=model,
                test_probe=test_probe,
                bleu_probe=bleu_probe,
                device=device,
                bleu_batch_size=int(eval_cfg["bleu_batch_size"]),
            )
            model.load_state_dict(final_state)
            model.to(device)
            synchronize(device)

            test_results = {
                "policy": (
                    "test is held out until post-training audit; validation "
                    "selects checkpoint_best.pt and test never tunes the protocol"
                ),
                "distributed_tpu": {
                    "world_size": world_size,
                    "batch_plan": asdict(plan),
                },
                "final": final_test,
                "validation_selected": best_test,
            }
            (run_dir / "test_results.json").write_text(
                json.dumps(test_results, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            completion = {
                "completed": True,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "optimizer": optimizer_name,
                "seed": seed,
                "optimizer_steps": int(total_steps),
                "train_epochs": float(
                    total_steps * tokens_per_step(cfg) / train_tokens
                ),
                "elapsed_seconds": float(elapsed_total),
                "best_validation_step": int(best_validation_step),
                "best_validation_loss": float(best_validation_loss),
                "final_test_loss": float(final_test["loss"]),
                "final_test_perplexity": float(final_test["perplexity"]),
                "final_test_bits_per_token": float(
                    final_test["bits_per_token"]
                ),
                "final_test_accuracy": float(final_test["accuracy"]),
                "final_test_top5_accuracy": float(
                    final_test["top5_accuracy"]
                ),
                "final_test_bleu": float(final_test["bleu"]),
                "final_test_continuation_token_accuracy": float(
                    final_test["continuation_token_accuracy"]
                ),
                "final_test_continuation_exact_match": float(
                    final_test["continuation_exact_match"]
                ),
                "fingerprint": fingerprint,
                "distributed_tpu_world_size": int(world_size),
                "distributed_tpu_local_grad_accum_steps": int(
                    plan.local_grad_accum_steps
                ),
                "global_tokens_per_step": int(plan.global_tokens_per_step),
            }
            temporary = run_dir / "run_complete.json.tmp"
            temporary.write_text(
                json.dumps(completion, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(run_dir / "run_complete.json")

            if progress:
                print(
                    "[one-head-fourway] COMPLETE "
                    f"optimizer={optimizer_name} seed={seed} "
                    f"world_size={world_size} "
                    f"elapsed={elapsed_total:.1f}s "
                    f"run_dir={run_dir}",
                    flush=True,
                )

        _rendezvous(xm, "fourway-finalize-end")
    finally:
        if metrics_handle is not None:
            metrics_handle.close()
        if epoch_handle is not None:
            epoch_handle.close()
        if not is_master:
            shutil.rmtree(
                run_dir / ".distributed_aux" / f"rank_{rank:02d}",
                ignore_errors=True,
            )


def _resolve_path(
    explicit: str | None,
    *,
    env_name: str,
    fallback_root: Path,
    suffix: str,
) -> Path:
    if explicit:
        return Path(explicit)
    if os.environ.get(env_name):
        return Path(os.environ[env_name])
    return fallback_root / suffix


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    plan = derive_distributed_batch_plan(
        cfg,
        world_size=int(args.world_size),
    )
    cfg = distributed_protocol_config(cfg, plan=plan)

    root = Path(
        os.environ.get(
            "RG_NANOGPT_ONE_HEAD_ROOT",
            "/tmp/rg-nanogpt-one-head",
        )
    )
    data_root = _resolve_path(
        args.data_root,
        env_name="RG_NANOGPT_ONE_HEAD_DATA_ROOT",
        fallback_root=root,
        suffix="data",
    )
    results_root = _resolve_path(
        args.results_root,
        env_name="RG_NANOGPT_ONE_HEAD_RESULTS_ROOT",
        fallback_root=root,
        suffix="results",
    )

    # Validate corpus identity before spawning device processes.
    data_metadata, arrays = load_memmaps(data_root, cfg)
    del arrays

    run_dir = run_directory(
        results_root,
        str(args.optimizer),
        int(args.seed),
    )
    if run_dir.exists():
        if args.overwrite:
            shutil.rmtree(run_dir)
        elif any(run_dir.iterdir()):
            raise FileExistsError(
                f"distributed run already exists: {run_dir}; "
                "choose a new results root or pass --overwrite"
            )
    run_dir.mkdir(parents=True, exist_ok=True)

    preflight = {
        "config": cfg,
        "optimizer": str(args.optimizer),
        "seed": int(args.seed),
        "data_root": str(data_root),
        "results_root": str(results_root),
        "world_size": int(args.world_size),
        "progress": not bool(args.quiet),
        "data_metadata": data_metadata,
        "batch_plan": asdict(plan),
    }
    (run_dir / "distributed_preflight.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in preflight.items()
                if key != "config"
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fresh data-parallel nanoGPT replicate across every TPU "
            "device while preserving the configured global optimizer batch"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--optimizer",
        choices=SUPPORTED_OPTIMIZERS,
        default="muon",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="required TPU process count; v5litepod-4 uses 4",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete only this distributed optimizer/seed run before launch",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = _preflight(args)

    # Importing and launching happens only after CPU-side config/data preflight.
    import torch_xla

    torch_xla.launch(
        _worker,
        args=(payload,),
        debug_single_process=False,
    )


if __name__ == "__main__":
    main()

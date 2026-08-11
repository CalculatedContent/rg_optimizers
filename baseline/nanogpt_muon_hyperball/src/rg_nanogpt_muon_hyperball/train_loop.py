from __future__ import annotations

import csv
import math
from pathlib import Path
import time

import torch

from .checkpoints import save_epoch_model_checkpoint, save_training_checkpoint
from .config import lr_schedule_steps, optimizer_profile
from .evaluation import evaluate_bleu, evaluate_probe, random_batch
from .optimizers import optimizer_step, set_learning_rates, zero_grad
from .runtime import (
    empty_mps_cache,
    gradient_norm,
    model_weight_norm,
    mps_memory_megabytes,
    parameter_snapshot,
    synchronize,
    update_norm,
)
from .spectral import run_weightwatcher


def _accumulate_hyperball(
    accumulator: dict[str, torch.Tensor] | None,
    step_summary: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    if step_summary is None:
        return accumulator
    if accumulator is None:
        return {key: value.detach().clone() for key, value in step_summary.items()}

    for key in ("matrix_updates", "active_updates", "scale_sum", "radius_sum"):
        accumulator[key] = accumulator[key] + step_summary[key]
    accumulator["scale_min"] = torch.minimum(
        accumulator["scale_min"], step_summary["scale_min"]
    )
    for key in (
        "proposed_uwr_max",
        "applied_uwr_max",
        "proposed_update_norm_max",
        "applied_update_norm_max",
    ):
        accumulator[key] = torch.maximum(accumulator[key], step_summary[key])
    return accumulator


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _hyperball_metrics(
    accumulator: dict[str, torch.Tensor] | None,
    handles,
) -> dict[str, float]:
    radius = float("nan")
    for handle in handles:
        if handle.role == "primary" and handle.optimizer.param_groups:
            radius = float(
                handle.optimizer.param_groups[0].get(
                    "relative_radius", float("nan")
                )
            )
            break

    if accumulator is None:
        return {
            "hyperball_relative_radius": radius,
            "hyperball_matrix_updates_since_eval": 0.0,
            "hyperball_active_fraction": float("nan"),
            "hyperball_mean_scale": float("nan"),
            "hyperball_min_scale": float("nan"),
            "hyperball_mean_radius": float("nan"),
            "hyperball_max_proposed_update_to_weight_ratio": float("nan"),
            "hyperball_max_applied_update_to_weight_ratio": float("nan"),
            "hyperball_max_proposed_update_norm": float("nan"),
            "hyperball_max_applied_update_norm": float("nan"),
        }

    count = max(_scalar(accumulator["matrix_updates"]), 1.0)
    return {
        "hyperball_relative_radius": radius,
        "hyperball_matrix_updates_since_eval": count,
        "hyperball_active_fraction": _scalar(accumulator["active_updates"]) / count,
        "hyperball_mean_scale": _scalar(accumulator["scale_sum"]) / count,
        "hyperball_min_scale": _scalar(accumulator["scale_min"]),
        "hyperball_mean_radius": _scalar(accumulator["radius_sum"]) / count,
        "hyperball_max_proposed_update_to_weight_ratio": _scalar(
            accumulator["proposed_uwr_max"]
        ),
        "hyperball_max_applied_update_to_weight_ratio": _scalar(
            accumulator["applied_uwr_max"]
        ),
        "hyperball_max_proposed_update_norm": _scalar(
            accumulator["proposed_update_norm_max"]
        ),
        "hyperball_max_applied_update_norm": _scalar(
            accumulator["applied_update_norm_max"]
        ),
    }


def _require_finite_metrics(
    *,
    completed_steps: int,
    train_metrics: dict,
    val_metrics: dict,
) -> None:
    values = {
        "train_loss": float(train_metrics["loss"]),
        "train_perplexity": float(train_metrics["perplexity"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "val_loss": float(val_metrics["loss"]),
        "val_perplexity": float(val_metrics["perplexity"]),
        "val_accuracy": float(val_metrics["accuracy"]),
    }
    bad = [name for name, value in values.items() if not math.isfinite(value)]
    if bad:
        raise FloatingPointError(
            "non-finite training state at "
            f"step={completed_steps}: {', '.join(bad)}. "
            "Aborting before checkpoint selection or WeightWatcher."
        )


def execute_training_loop(
    *,
    cfg: dict,
    model,
    handles,
    arrays: dict,
    train_probe,
    val_probe,
    test_probe,
    bleu_probe,
    device: torch.device,
    optimizer_name: str,
    seed: int,
    train_tokens: int,
    total_steps: int,
    warmup: int,
    start_step: int,
    best_validation_loss: float,
    best_validation_step: int,
    elapsed_offset: float,
    fingerprint: str,
    train_generator: torch.Generator,
    epoch_steps: dict[int, float],
    metrics_writer: csv.DictWriter,
    metrics_handle,
    epoch_writer: csv.DictWriter,
    epoch_handle,
    run_dir: Path,
    latest_checkpoint: Path,
    best_checkpoint: Path,
    progress: bool,
) -> tuple[float, int, float]:
    batch_size = int(cfg["training"]["batch_size"])
    grad_accum = int(cfg["training"]["grad_accum_steps"])
    block_size = int(cfg["model"]["block_size"])
    step_tokens = batch_size * grad_accum * block_size
    eval_cfg = cfg["evaluation"]

    profile = optimizer_profile(cfg, optimizer_name)
    schedule_total_steps = lr_schedule_steps(cfg, profile, train_tokens)
    if not 0 <= warmup < schedule_total_steps:
        raise ValueError(
            f"warmup={warmup} must be smaller than "
            f"lr_schedule_steps={schedule_total_steps}"
        )

    previous_snapshot = parameter_snapshot(model)
    last_grad_pre = float("nan")
    last_grad_post = float("nan")
    last_clipped = False
    hyperball_interval: dict[str, torch.Tensor] | None = None
    started = time.time()

    last_update_lrs = {
        "primary": 0.0,
        "auxiliary": (
            0.0
            if any(handle.role == "auxiliary" for handle in handles)
            else float("nan")
        ),
    }
    if start_step > 0:
        for handle in handles:
            last_update_lrs[handle.role] = float(handle.lr)

    for completed_steps in range(start_step, total_steps + 1):
        schedule_index = min(completed_steps, schedule_total_steps - 1)
        next_update_lrs = set_learning_rates(
            handles,
            update_index=schedule_index,
            total_steps=schedule_total_steps,
            warmup_steps=warmup,
        )
        epoch_due = completed_steps in epoch_steps
        evaluation_due = (
            completed_steps
            % int(cfg["training"]["eval_interval_steps"])
            == 0
            or epoch_due
            or completed_steps == total_steps
        )

        if evaluation_due:
            synchronize(device)
            train_metrics = evaluate_probe(model, train_probe, device)
            val_metrics = evaluate_probe(model, val_probe, device)
            _require_finite_metrics(
                completed_steps=completed_steps,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
            elapsed = elapsed_offset + time.time() - started

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
                    seed=int(seed),
                    train_generator=train_generator,
                )

            test_metrics = {
                "loss": float("nan"),
                "perplexity": float("nan"),
                "accuracy": float("nan"),
            }
            bleu_metrics = {"bleu": float("nan")}
            if epoch_due or completed_steps == total_steps:
                test_metrics = evaluate_probe(model, test_probe, device)
                bleu_metrics = evaluate_bleu(
                    model,
                    bleu_probe,
                    device=device,
                    batch_size=int(eval_cfg["bleu_batch_size"]),
                )

            tokens_seen = int(completed_steps * step_tokens)
            actual_epoch = tokens_seen / max(1, train_tokens)
            current_snapshot = parameter_snapshot(model)
            delta_norm = update_norm(previous_snapshot, current_snapshot)
            previous_snapshot = current_snapshot
            weight_norm = model_weight_norm(model)
            current_mps, driver_mps = mps_memory_megabytes(device)
            row = {
                "step": int(completed_steps),
                "tokens_seen": tokens_seen,
                "epoch": float(actual_epoch),
                "elapsed_sec": float(elapsed),
                "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
                "primary_lr": float(
                    last_update_lrs.get("primary", float("nan"))
                ),
                "auxiliary_lr": float(
                    last_update_lrs.get("auxiliary", float("nan"))
                ),
                "train_loss": float(train_metrics["loss"]),
                "train_perplexity": float(train_metrics["perplexity"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_perplexity": float(val_metrics["perplexity"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
                "test_perplexity": float(test_metrics["perplexity"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_bleu": float(bleu_metrics["bleu"]),
                "val_generalization_gap": float(
                    val_metrics["loss"] - train_metrics["loss"]
                ),
                "test_generalization_gap": float(
                    test_metrics["loss"] - train_metrics["loss"]
                ),
                "grad_norm_pre_clip": float(last_grad_pre),
                "grad_norm_post_clip": float(last_grad_post),
                "gradient_clipped": int(last_clipped),
                "weight_norm": float(weight_norm),
                "update_norm_since_eval": float(delta_norm),
                "update_to_weight_ratio": float(
                    delta_norm / max(weight_norm, 1e-30)
                ),
                **_hyperball_metrics(hyperball_interval, handles),
                "mps_current_allocated_mb": float(current_mps),
                "mps_driver_allocated_mb": float(driver_mps),
            }
            metrics_writer.writerow(row)
            metrics_handle.flush()
            hyperball_interval = None

            if epoch_due:
                nominal_epoch = float(epoch_steps[completed_steps])
                checkpoint_path = save_epoch_model_checkpoint(
                    run_dir,
                    model=model,
                    step=completed_steps,
                    nominal_epoch=nominal_epoch,
                    actual_epoch=actual_epoch,
                    fingerprint=fingerprint,
                    cfg=cfg,
                    optimizer_name=optimizer_name,
                    seed=int(seed),
                )
                epoch_writer.writerow(
                    {
                        **row,
                        "nominal_epoch": nominal_epoch,
                        "checkpoint_path": str(checkpoint_path),
                        "test_monitoring_only": 1,
                    }
                )
                epoch_handle.flush()

                ww_summary = run_weightwatcher(
                    model,
                    run_dir,
                    step=completed_steps,
                    tokens_seen=tokens_seen,
                    train_tokens=train_tokens,
                    config=cfg["weightwatcher"],
                    seed=int(seed),
                )
                if progress:
                    print(
                        "[muon-hyperball-ww] "
                        f"optimizer={optimizer_name} seed={seed} "
                        f"epoch={nominal_epoch:.2f} "
                        f"alpha={ww_summary.get('alpha_median', float('nan')):.3f} "
                        f"ERG_gap={ww_summary.get('ERG_gap_median', float('nan')):.3f} "
                        f"num_traps={ww_summary.get('num_traps_mean', float('nan')):.2f}",
                        flush=True,
                    )
                if bool(
                    cfg["runtime"].get(
                        "empty_mps_cache_after_weightwatcher", True
                    )
                ):
                    empty_mps_cache(device)

            if progress:
                remaining = total_steps - completed_steps
                rate = completed_steps / max(elapsed, 1e-9)
                eta = remaining / rate if rate > 0 else float("nan")
                eta_text = (
                    "unknown"
                    if not math.isfinite(eta)
                    else f"{eta / 60:.1f}m"
                )
                active = row["hyperball_active_fraction"]
                active_text = (
                    "n/a"
                    if not math.isfinite(active)
                    else f"{100 * active:.1f}%"
                )
                print(
                    "[muon-hyperball-train] "
                    f"optimizer={optimizer_name} seed={seed} "
                    f"step={completed_steps}/{total_steps} "
                    f"epoch={actual_epoch:.3f} "
                    f"last_lr={last_update_lrs.get('primary', float('nan')):.3e} "
                    f"next_lr={next_update_lrs.get('primary', float('nan')):.3e} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_ppl={val_metrics['perplexity']:.2f} "
                    f"val_acc={100 * val_metrics['accuracy']:.2f}% "
                    f"ball_active={active_text} "
                    f"eta={eta_text}",
                    flush=True,
                )

        if completed_steps == total_steps:
            break

        zero_grad(handles)
        for _ in range(grad_accum):
            x_cpu, y_cpu = random_batch(
                arrays["train"],
                batch_size=batch_size,
                block_size=block_size,
                generator=train_generator,
            )
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("training forward pass did not return loss")
            (loss / grad_accum).backward()

        grad_pre_tensor = gradient_norm(model.parameters())
        last_grad_pre = float(grad_pre_tensor.detach().cpu())
        clip = float(cfg["training"]["grad_clip"])
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        grad_post_tensor = gradient_norm(model.parameters())
        last_grad_post = float(grad_post_tensor.detach().cpu())
        last_clipped = bool(last_grad_pre > clip) if clip > 0 else False

        step_hyperball = optimizer_step(handles)
        hyperball_interval = _accumulate_hyperball(
            hyperball_interval, step_hyperball
        )
        last_update_lrs = dict(next_update_lrs)

        new_step = completed_steps + 1
        checkpoint_due = (
            new_step
            % int(cfg["training"]["checkpoint_interval_steps"])
            == 0
            or new_step in epoch_steps
            or new_step == total_steps
        )
        if checkpoint_due:
            save_training_checkpoint(
                latest_checkpoint,
                model=model,
                handles=handles,
                step=new_step,
                best_validation_loss=best_validation_loss,
                best_validation_step=best_validation_step,
                elapsed_seconds=elapsed_offset + time.time() - started,
                fingerprint=fingerprint,
                cfg=cfg,
                optimizer_name=optimizer_name,
                seed=int(seed),
                train_generator=train_generator,
            )

    return (
        float(best_validation_loss),
        int(best_validation_step),
        float(elapsed_offset + time.time() - started),
    )

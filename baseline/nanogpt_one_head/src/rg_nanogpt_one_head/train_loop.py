from __future__ import annotations

import csv
import math
from pathlib import Path
import time

import torch

from .checkpoints import (
    save_epoch_model_checkpoint,
    save_training_checkpoint,
)
from .evaluation import evaluate_probe, random_batch
from .optimizers import optimizer_step, set_learning_rates, zero_grad
from .runtime import (
    empty_mps_cache,
    gradient_norm,
    mark_step,
    model_weight_norm,
    mps_memory_megabytes,
    parameter_snapshot,
    synchronize,
    update_norm,
)
from .spectral import run_weightwatcher


def _require_finite_metrics(
    *,
    completed_steps: int,
    train_metrics: dict,
    val_metrics: dict,
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
    bad = [
        name
        for name, value in values.items()
        if not math.isfinite(value)
    ]
    if bad:
        raise FloatingPointError(
            "non-finite training state at "
            f"step={completed_steps}: {', '.join(bad)}. "
            "Aborting before checkpoint selection or WeightWatcher."
        )


def _require_finite_model(
    model,
    *,
    completed_steps: int,
) -> None:
    checks = [
        (name, torch.isfinite(parameter).all())
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() or parameter.is_complex()
    ]
    if not checks:
        return
    all_finite = torch.stack([value for _, value in checks]).all()
    if bool(all_finite.detach().cpu()):
        return
    bad = [
        name
        for name, value in checks
        if not bool(value.detach().cpu())
    ]
    raise FloatingPointError(
        "non-finite model parameters at "
        f"step={completed_steps}: {', '.join(bad)}"
    )


def _evaluation_due(
    step: int,
    *,
    cfg: dict,
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
    cfg: dict,
    epoch_steps: dict[int, float],
    total_steps: int,
) -> bool:
    return (
        step % int(cfg["training"]["checkpoint_interval_steps"]) == 0
        or step in epoch_steps
        or step == total_steps
    )


def _resume_diagnostics_due(
    step: int,
    *,
    cfg: dict,
    epoch_steps: dict[int, float],
    total_steps: int,
) -> bool:
    # A rolling checkpoint must carry finite gradient diagnostics even when it
    # falls between evaluation rows. Otherwise a checkpoint interval shorter
    # than the evaluation interval persists the initialization NaN sentinels.
    return _evaluation_due(
        step,
        cfg=cfg,
        epoch_steps=epoch_steps,
        total_steps=total_steps,
    ) or _checkpoint_due(
        step,
        cfg=cfg,
        epoch_steps=epoch_steps,
        total_steps=total_steps,
    )


def _resume_diagnostics(
    previous_eval_snapshot: list[torch.Tensor],
    *,
    last_grad_pre: float,
    last_grad_post: float,
    last_clipped: bool,
) -> dict:
    return {
        "previous_eval_snapshot": previous_eval_snapshot,
        "last_grad_pre": float(last_grad_pre),
        "last_grad_post": float(last_grad_post),
        "last_clipped": bool(last_clipped),
    }


def execute_training_loop(
    *,
    cfg: dict,
    model,
    handles,
    arrays: dict,
    train_probe,
    val_probe,
    device: torch.device,
    optimizer_name: str,
    seed: int,
    train_tokens: int,
    total_steps: int,
    schedule_steps: int,
    warmup: int,
    start_step: int,
    best_validation_loss: float,
    best_validation_step: int,
    elapsed_offset: float,
    resume_diagnostics: dict | None,
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
) -> tuple[float, int, float, dict]:
    batch_size = int(cfg["training"]["batch_size"])
    grad_accum = int(cfg["training"]["grad_accum_steps"])
    block_size = int(cfg["model"]["block_size"])
    step_tokens = batch_size * grad_accum * block_size

    if not 1 <= schedule_steps <= total_steps:
        raise ValueError(
            f"schedule_steps={schedule_steps} must be in [1, {total_steps}]"
        )
    if not 0 <= warmup < schedule_steps:
        raise ValueError(
            f"warmup={warmup} must be smaller than "
            f"schedule_steps={schedule_steps}"
        )

    if resume_diagnostics is None:
        if start_step > 0:
            raise RuntimeError(
                "checkpoint has no deterministic resume diagnostics; refusing "
                "to rewrite monitoring rows with reset gradient/update values"
            )
        previous_snapshot = parameter_snapshot(model)
        last_grad_pre = float("nan")
        last_grad_post = float("nan")
        last_clipped = False
    else:
        previous_snapshot = resume_diagnostics["previous_eval_snapshot"]
        last_grad_pre = float(resume_diagnostics["last_grad_pre"])
        last_grad_post = float(resume_diagnostics["last_grad_post"])
        last_clipped = bool(resume_diagnostics["last_clipped"])
    started = time.time()
    final_resume_diagnostics: dict | None = None

    # CSV rows describe the model state at `completed_steps`, so the recorded
    # LR must be the LR used by the update that produced that state. At step
    # zero there has been no update. On resume, optimizer state carries the LR
    # used by the most recently completed update.
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
        schedule_index = min(completed_steps, schedule_steps - 1)
        next_update_lrs = set_learning_rates(
            handles,
            update_index=schedule_index,
            total_steps=schedule_steps,
            warmup_steps=warmup,
        )
        epoch_due = completed_steps in epoch_steps
        evaluation_due = _evaluation_due(
            completed_steps,
            cfg=cfg,
            epoch_steps=epoch_steps,
            total_steps=total_steps,
        )

        if evaluation_due:
            # These are the diagnostics that an uninterrupted run uses for the
            # row at this exact model state. Keep them even after advancing the
            # in-memory evaluation snapshot so a crash after evaluation can
            # reproduce the row byte-for-byte (apart from wall-clock fields).
            current_state_resume_diagnostics = _resume_diagnostics(
                previous_snapshot,
                last_grad_pre=last_grad_pre,
                last_grad_post=last_grad_post,
                last_clipped=last_clipped,
            )
            synchronize(device)
            train_metrics = evaluate_probe(model, train_probe, device)
            val_metrics = evaluate_probe(model, val_probe, device)
            _require_finite_metrics(
                completed_steps=completed_steps,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
            _require_finite_model(
                model,
                completed_steps=completed_steps,
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
                "bits_per_token": float("nan"),
                "accuracy": float("nan"),
                "top5_accuracy": float("nan"),
            }
            bleu_metrics = {
                "bleu": float("nan"),
                "continuation_token_accuracy": float("nan"),
                "continuation_exact_match": float("nan"),
            }
            # Keep the test split genuinely held out during optimization.
            # Final and validation-selected checkpoints are evaluated once,
            # after training, by engine.checkpoint_eval.

            tokens_seen = int(completed_steps * step_tokens)
            actual_epoch = tokens_seen / max(1, train_tokens)
            current_snapshot = parameter_snapshot(model)
            delta_norm = update_norm(
                previous_snapshot,
                current_snapshot,
            )
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
                    last_update_lrs.get(
                        "primary",
                        float("nan"),
                    )
                ),
                "auxiliary_lr": float(
                    last_update_lrs.get(
                        "auxiliary",
                        float("nan"),
                    )
                ),
                "train_loss": float(train_metrics["loss"]),
                "train_perplexity": float(
                    train_metrics["perplexity"]
                ),
                "train_bits_per_token": float(
                    train_metrics["bits_per_token"]
                ),
                "train_accuracy": float(
                    train_metrics["accuracy"]
                ),
                "train_top5_accuracy": float(
                    train_metrics["top5_accuracy"]
                ),
                "val_loss": float(val_metrics["loss"]),
                "val_perplexity": float(
                    val_metrics["perplexity"]
                ),
                "val_bits_per_token": float(
                    val_metrics["bits_per_token"]
                ),
                "val_accuracy": float(
                    val_metrics["accuracy"]
                ),
                "val_top5_accuracy": float(
                    val_metrics["top5_accuracy"]
                ),
                "test_loss": float(test_metrics["loss"]),
                "test_perplexity": float(
                    test_metrics["perplexity"]
                ),
                "test_bits_per_token": float(
                    test_metrics["bits_per_token"]
                ),
                "test_accuracy": float(
                    test_metrics["accuracy"]
                ),
                "test_top5_accuracy": float(
                    test_metrics["top5_accuracy"]
                ),
                "test_bleu": float(bleu_metrics["bleu"]),
                "test_continuation_token_accuracy": float(
                    bleu_metrics.get(
                        "continuation_token_accuracy", float("nan")
                    )
                ),
                "test_continuation_exact_match": float(
                    bleu_metrics.get(
                        "continuation_exact_match", float("nan")
                    )
                ),
                "val_generalization_gap": float(
                    val_metrics["loss"] - train_metrics["loss"]
                ),
                "test_generalization_gap": float(
                    test_metrics["loss"]
                    - train_metrics["loss"]
                ),
                "grad_norm_pre_clip": float(last_grad_pre),
                "grad_norm_post_clip": float(last_grad_post),
                "gradient_clipped": int(last_clipped),
                "weight_norm": float(weight_norm),
                "update_norm_since_eval": float(delta_norm),
                "update_to_weight_ratio": float(
                    delta_norm / max(weight_norm, 1e-30)
                ),
                "mps_current_allocated_mb": float(current_mps),
                "mps_driver_allocated_mb": float(driver_mps),
            }
            metrics_writer.writerow(row)
            metrics_handle.flush()

            if epoch_due:
                nominal_epoch = float(
                    epoch_steps[completed_steps]
                )
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
                        "test_held_out": 1,
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
                    fingerprint=fingerprint,
                )
                if progress:
                    print(
                        "[one-head-ww] "
                        f"optimizer={optimizer_name} seed={seed} "
                        f"epoch={nominal_epoch:.2f} "
                        f"alpha_clip={ww_summary.get('alpha_clip_xmax_median', ww_summary.get('alpha_median', float('nan'))):.3f} "
                        f"alpha_raw={ww_summary.get('alpha_raw_median', float('nan')):.3f} "
                        f"ERG_gap={ww_summary.get('ERG_gap_median', float('nan')):.3f} "
                        f"num_traps={ww_summary.get('num_traps_mean', float('nan')):.2f}",
                        flush=True,
                    )
                if bool(
                    cfg["runtime"].get(
                        "empty_mps_cache_after_weightwatcher",
                        True,
                    )
                ):
                    empty_mps_cache(device)

            if progress:
                remaining = total_steps - completed_steps
                rate = completed_steps / max(elapsed, 1e-9)
                eta = (
                    remaining / rate
                    if rate > 0
                    else float("nan")
                )
                eta_text = (
                    "unknown"
                    if not math.isfinite(eta)
                    else f"{eta / 60:.1f}m"
                )
                print(
                    "[one-head-train] "
                    f"optimizer={optimizer_name} seed={seed} "
                    f"step={completed_steps}/{total_steps} "
                    f"epoch={actual_epoch:.3f} "
                    f"last_lr={last_update_lrs.get('primary', float('nan')):.3e} "
                    f"next_lr={next_update_lrs.get('primary', float('nan')):.3e} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_ppl={val_metrics['perplexity']:.2f} "
                    f"val_acc={100 * val_metrics['accuracy']:.2f}% "
                    f"eta={eta_text}",
                    flush=True,
                )

            if completed_steps == total_steps:
                final_resume_diagnostics = current_state_resume_diagnostics

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
                raise RuntimeError(
                    "training forward pass did not return loss"
                )
            (loss / grad_accum).backward()

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
        mark_step(device)
        last_update_lrs = dict(next_update_lrs)

        new_step = completed_steps + 1
        if _resume_diagnostics_due(
            new_step,
            cfg=cfg,
            epoch_steps=epoch_steps,
            total_steps=total_steps,
        ):
            # Materialize gradient diagnostics only when the next state is
            # actually recorded. On XLA this avoids a host sync every step.
            last_grad_pre = float(
                grad_pre_tensor.detach().cpu()
            )
            last_grad_post = float(
                grad_post_tensor.detach().cpu()
            )
            last_clipped = (
                bool(last_grad_pre > clip)
                if clip > 0
                else False
            )

        checkpoint_due = _checkpoint_due(
            new_step,
            cfg=cfg,
            epoch_steps=epoch_steps,
            total_steps=total_steps,
        )
        if checkpoint_due:
            _require_finite_model(
                model,
                completed_steps=new_step,
            )
            save_training_checkpoint(
                latest_checkpoint,
                model=model,
                handles=handles,
                step=new_step,
                best_validation_loss=best_validation_loss,
                best_validation_step=best_validation_step,
                elapsed_seconds=(
                    elapsed_offset + time.time() - started
                ),
                fingerprint=fingerprint,
                cfg=cfg,
                optimizer_name=optimizer_name,
                seed=int(seed),
                train_generator=train_generator,
                resume_diagnostics=_resume_diagnostics(
                    previous_snapshot,
                    last_grad_pre=last_grad_pre,
                    last_grad_post=last_grad_post,
                    last_clipped=last_clipped,
                ),
            )

    synchronize(device)
    if final_resume_diagnostics is None:
        raise RuntimeError(
            "final model state was not evaluated; deterministic resume "
            "diagnostics are unavailable"
        )
    return (
        float(best_validation_loss),
        int(best_validation_step),
        float(elapsed_offset + time.time() - started),
        final_resume_diagnostics,
    )

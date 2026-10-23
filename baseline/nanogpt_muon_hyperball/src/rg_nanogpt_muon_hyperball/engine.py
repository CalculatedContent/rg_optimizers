from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import torch

from .completion import validate_completed_run
from .checkpoints import load_training_checkpoint, save_training_checkpoint
from .config import (
    SUPPORTED_OPTIMIZERS,
    epoch_step_map,
    max_steps,
    optimizer_profile,
    protocol_fingerprint,
    tokens_per_step,
    warmup_steps,
)
from .data import load_memmaps
from .evaluation import fixed_bleu_probe, fixed_probe
from .model import GPT, GPTConfig
from .optimizers import make_optimizer_handles
from .run_utils import (
    EPOCH_FIELDS,
    METRIC_FIELDS,
    checkpoint_eval,
    prepare_csv,
    run_directory,
    truncate_spectral_after,
    write_manifest,
)
from .runtime import choose_device, configure_runtime, seed_everything
from .train_loop import execute_training_loop


def run_one(
    *,
    cfg: dict,
    data_root: str | Path,
    results_root: str | Path,
    optimizer_name: str,
    seed: int,
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    optimizer_name = str(optimizer_name).lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")

    data_root = Path(data_root)
    results_root = Path(results_root)
    run_dir = run_directory(results_root, optimizer_name, int(seed))
    completion_path = run_dir / "run_complete.json"
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"incomplete run exists: {run_dir}; enable resume or overwrite"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    data_metadata, arrays = load_memmaps(data_root, cfg)
    train_tokens = int(data_metadata["splits"]["train"])
    total_steps = max_steps(cfg, train_tokens)
    profile = optimizer_profile(cfg, optimizer_name)
    warmup = warmup_steps(profile, total_steps)
    fingerprint = protocol_fingerprint(
        cfg,
        optimizer=optimizer_name,
        seed=int(seed),
        data_metadata=data_metadata,
    )
    if completion_path.is_file():
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=optimizer_name,
            expected_seed=int(seed),
            expected_total_steps=total_steps,
            verify_checkpoints=True,
        )
        if progress:
            print(
                "[one-head-train] reuse verified completed "
                f"{optimizer_name} seed={seed}"
            )
        return run_dir

    resolved_device = choose_device(device)
    configure_runtime(resolved_device, cfg)
    seed_everything(int(seed))
    model = GPT(GPTConfig(**cfg["model"])).to(resolved_device)
    handles = make_optimizer_handles(model, profile)
    train_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 11)

    batch_size = int(cfg["training"]["batch_size"])
    eval_batches = int(cfg["training"]["eval_batches"])
    block_size = int(cfg["model"]["block_size"])
    epoch_steps = epoch_step_map(cfg, train_tokens)
    eval_cfg = cfg["evaluation"]

    # Evaluation examples are deliberately independent of the training seed.
    # This makes paired optimizer comparisons and across-seed uncertainty use
    # the same train/validation/test probe windows in every complete run.
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

    start_step = 0
    best_validation_loss = float("inf")
    best_validation_step = 0
    elapsed_offset = 0.0
    latest_checkpoint = run_dir / "checkpoint_latest.pt"
    best_checkpoint = run_dir / "checkpoint_best.pt"
    final_checkpoint = run_dir / "checkpoint_final.pt"
    if resume and latest_checkpoint.is_file():
        (
            start_step,
            best_validation_loss,
            best_validation_step,
            elapsed_offset,
        ) = load_training_checkpoint(
            latest_checkpoint,
            model=model,
            handles=handles,
            expected_fingerprint=fingerprint,
            train_generator=train_generator,
        )
        model.to(resolved_device)
        truncate_spectral_after(run_dir, start_step)
        if progress:
            print(
                f"[one-head-train] resume {optimizer_name} "
                f"seed={seed} step={start_step}"
            )
    elif run_dir.exists() and any(run_dir.iterdir()) and resume:
        nontrivial = [
            path for path in run_dir.iterdir() if path.name != "manifest.json"
        ]
        if nontrivial and not latest_checkpoint.is_file():
            raise FileNotFoundError(
                f"cannot resume {run_dir}: checkpoint_latest.pt is missing"
            )

    write_manifest(
        run_dir,
        cfg=cfg,
        data_metadata=data_metadata,
        optimizer_name=optimizer_name,
        profile=profile,
        seed=int(seed),
        device=resolved_device,
        total_steps=total_steps,
        warmup=warmup,
        fingerprint=fingerprint,
        model=model,
    )

    metrics_path = run_dir / "metrics.csv"
    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    prepare_csv(
        metrics_path,
        METRIC_FIELDS,
        start_step if start_step else None,
    )
    prepare_csv(
        epoch_metrics_path,
        EPOCH_FIELDS,
        start_step if start_step else None,
    )
    with (
        metrics_path.open("a", newline="", encoding="utf-8") as metrics_handle,
        epoch_metrics_path.open(
            "a", newline="", encoding="utf-8"
        ) as epoch_handle,
    ):
        best_validation_loss, best_validation_step, elapsed_total = (
            execute_training_loop(
                cfg=cfg,
                model=model,
                handles=handles,
                arrays=arrays,
                train_probe=train_probe,
                val_probe=val_probe,
                test_probe=test_probe,
                bleu_probe=bleu_probe,
                device=resolved_device,
                optimizer_name=optimizer_name,
                seed=int(seed),
                train_tokens=train_tokens,
                total_steps=total_steps,
                warmup=warmup,
                start_step=start_step,
                best_validation_loss=best_validation_loss,
                best_validation_step=best_validation_step,
                elapsed_offset=elapsed_offset,
                fingerprint=fingerprint,
                train_generator=train_generator,
                epoch_steps=epoch_steps,
                metrics_writer=csv.DictWriter(
                    metrics_handle, fieldnames=METRIC_FIELDS
                ),
                metrics_handle=metrics_handle,
                epoch_writer=csv.DictWriter(
                    epoch_handle, fieldnames=EPOCH_FIELDS
                ),
                epoch_handle=epoch_handle,
                run_dir=run_dir,
                latest_checkpoint=latest_checkpoint,
                best_checkpoint=best_checkpoint,
                progress=progress,
            )
        )

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
            seed=int(seed),
            train_generator=train_generator,
        )

    final_state = torch.load(
        final_checkpoint,
        map_location="cpu",
        weights_only=False,
    )["model"]
    final_test = checkpoint_eval(
        final_checkpoint,
        model=model,
        test_probe=test_probe,
        bleu_probe=bleu_probe,
        device=resolved_device,
        bleu_batch_size=int(eval_cfg["bleu_batch_size"]),
    )
    best_test = checkpoint_eval(
        best_checkpoint,
        model=model,
        test_probe=test_probe,
        bleu_probe=bleu_probe,
        device=resolved_device,
        bleu_batch_size=int(eval_cfg["bleu_batch_size"]),
    )
    model.load_state_dict(final_state)
    model.to(resolved_device)

    test_results = {
        "policy": (
            "test is monitoring-only; validation loss selects "
            "checkpoint_best.pt"
        ),
        "final": final_test,
        "validation_selected": best_test,
    }
    (run_dir / "test_results.json").write_text(
        json.dumps(test_results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    completion = {
        "completed": True,
        "optimizer": optimizer_name,
        "seed": int(seed),
        "optimizer_steps": int(total_steps),
        "train_epochs": float(
            total_steps * tokens_per_step(cfg) / train_tokens
        ),
        "elapsed_seconds": float(elapsed_total),
        "best_validation_step": int(best_validation_step),
        "best_validation_loss": float(best_validation_loss),
        "final_test_loss": float(final_test["loss"]),
        "final_test_perplexity": float(final_test["perplexity"]),
        "final_test_accuracy": float(final_test["accuracy"]),
        "final_test_bleu": float(final_test["bleu"]),
        "fingerprint": fingerprint,
    }
    temporary = run_dir / "run_complete.json.tmp"
    temporary.write_text(
        json.dumps(completion, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(completion_path)
    if progress:
        print(
            f"[one-head-train] complete {optimizer_name} seed={seed}: {run_dir}"
        )
    return run_dir

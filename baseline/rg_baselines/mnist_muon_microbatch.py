"""Train the reference MNIST MLP3 with Muon and capture every microbatch.

This is an opt-in diagnostic runner. It reuses the baseline's exact MLP3,
MNIST split, Muon-with-auxiliary-AdamW optimizer, gradient clipping, and
update-level warmup/cosine learning-rate schedule. Only the three weight
matrices are persisted at microbatch cadence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .config import BaselineConfig
from .engine import choose_device, evaluate, set_seed
from .model import MLP3
from .muon_microbatch_capture import (
    DEFAULT_MATRIX_NAMES,
    MuonMicrobatchCheckpointRecorder,
    estimated_capture_bytes,
)
from .optimizers import build_optimizer, set_scheduled_learning_rates
from .runner import _make_datasets_and_loaders


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_metrics(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "epoch",
        "global_step",
        "examples_seen",
        "online_train_loss",
        "online_train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "test_loss",
        "test_accuracy",
        "primary_lr",
        "auxiliary_lr",
        "partial_epoch",
    ]
    rows: list[dict[str, Any]] = []
    if path.is_file() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    epoch = int(row["epoch"])
    rows = [item for item in rows if int(item["epoch"]) != epoch]
    rows.append({key: row.get(key, "") for key in fields})
    rows.sort(key=lambda item: int(item["epoch"]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolve_device(value: str) -> torch.device:
    name = str(value).lower()
    if name == "auto":
        return choose_device()
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _capture_count(total_steps: int, capture_every: int, max_capture_step: int) -> int:
    upper = min(total_steps, max_capture_step) if max_capture_step else total_steps
    return 1 + upper // capture_every


def run_muon_microbatch_capture(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 128,
    seed: int = 1337,
    device: str | torch.device = "auto",
    capture_every: int = 1,
    max_steps: int = 0,
    max_capture_step: int = 0,
    checkpoint_dtype: str = "float32",
    allow_large_capture: bool = False,
    large_capture_gib: float = 8.0,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Run the exact Muon baseline recipe with matrix-only step checkpoints."""

    config = BaselineConfig(
        optimizer="sgd_momentum_muon",
        epochs=int(epochs),
        batch_size=int(batch_size),
        seed=int(seed),
        strict_metrics=False,
    )
    config.validate()
    if int(max_steps) < 0 or int(max_capture_step) < 0:
        raise ValueError("max_steps and max_capture_step must be nonnegative")
    if int(capture_every) < 1:
        raise ValueError("capture_every must be positive")
    if not math.isfinite(float(large_capture_gib)) or large_capture_gib <= 0:
        raise ValueError("large_capture_gib must be positive and finite")

    resolved_device = (
        device if isinstance(device, torch.device) else _resolve_device(device)
    )
    set_seed(config.seed)
    (
        train_loader,
        _,
        validation_loader,
        test_loader,
        _,
        train_indices,
        validation_indices,
    ) = _make_datasets_and_loaders(
        config, data_dir=data_dir, device=resolved_device
    )

    model = MLP3().to(resolved_device)
    optimizer = build_optimizer(model, config)
    steps_per_epoch = len(train_loader)
    total_steps = int(config.epochs) * steps_per_epoch
    training_limit = min(total_steps, int(max_steps)) if max_steps else total_steps
    capture_count = _capture_count(
        training_limit, int(capture_every), int(max_capture_step)
    )
    raw_bytes = estimated_capture_bytes(
        model,
        matrix_names=DEFAULT_MATRIX_NAMES,
        dtype=checkpoint_dtype,
        checkpoint_count=capture_count,
    )
    estimated_gib = raw_bytes / float(1024**3)
    if estimated_gib > float(large_capture_gib) and not allow_large_capture:
        raise RuntimeError(
            "requested microbatch capture is estimated at "
            f"{estimated_gib:.2f} GiB of raw tensors. Re-run with "
            "allow_large_capture=True/--allow-large-capture, increase "
            "--capture-every, lower --max-steps, or use float16 checkpoints."
        )

    run_dir = Path(output_dir)
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {run_dir}; use --overwrite"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    _atomic_json(
        {
            "schema_version": 1,
            "purpose": "mnist_mlp3_muon_microbatch_training",
            "config": asdict(config),
            "device": str(resolved_device),
            "train_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "test_examples": 10_000,
            "steps_per_epoch": steps_per_epoch,
            "baseline_total_steps": total_steps,
            "training_step_limit": training_limit,
            "capture_every": int(capture_every),
            "max_capture_step": int(max_capture_step),
            "checkpoint_dtype": checkpoint_dtype,
            "estimated_checkpoint_count": capture_count,
            "estimated_raw_capture_gib": estimated_gib,
            "completed": False,
        },
        run_dir / "manifest.json",
    )

    recorder = MuonMicrobatchCheckpointRecorder(
        run_dir=run_dir,
        model=model,
        capture_every=int(capture_every),
        max_capture_step=int(max_capture_step),
        dtype=checkpoint_dtype,
    )
    initial_lrs = set_scheduled_learning_rates(
        optimizer,
        config,
        update_index=0,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
    )
    recorder.capture(
        global_step=0,
        epoch=0,
        batch_index=0,
        examples_seen=0,
        learning_rates=initial_lrs,
    )

    global_step = 0
    examples_seen = 0
    last_lrs = dict(initial_lrs)
    metrics_path = run_dir / "training_metrics.csv"
    stop = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        batches = 0
        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            last_lrs = set_scheduled_learning_rates(
                optimizer,
                config,
                update_index=global_step,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
            )
            inputs = inputs.to(resolved_device)
            targets = targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.grad_clip_norm)
            )
            optimizer.step()

            global_step += 1
            batch_examples = int(targets.numel())
            examples_seen += batch_examples
            seen += batch_examples
            batches += 1
            loss_value = float(loss.detach().cpu())
            loss_sum += loss_value * batch_examples
            correct += int((logits.argmax(1) == targets).sum().detach().cpu())
            recorder.capture(
                global_step=global_step,
                epoch=epoch,
                batch_index=batch_index,
                examples_seen=examples_seen,
                training_loss=loss_value,
                learning_rates=last_lrs,
            )

            if progress and (global_step == 1 or global_step % 100 == 0):
                print(
                    "[mnist-muon-microbatch] "
                    f"step={global_step}/{training_limit} epoch={epoch} "
                    f"loss={loss_value:.5f} capture_gib_est={estimated_gib:.2f}",
                    flush=True,
                )
            if global_step >= training_limit:
                stop = True
                break

        validation = evaluate(model, validation_loader, device=resolved_device)
        test = (
            evaluate(model, test_loader, device=resolved_device)
            if stop or epoch == config.epochs
            else {"loss": float("nan"), "accuracy": float("nan")}
        )
        _append_metrics(
            metrics_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "examples_seen": examples_seen,
                "online_train_loss": loss_sum / max(seen, 1),
                "online_train_accuracy": correct / max(seen, 1),
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
                "test_loss": float(test["loss"]),
                "test_accuracy": float(test["accuracy"]),
                "primary_lr": float(last_lrs.get("primary", float("nan"))),
                "auxiliary_lr": float(last_lrs.get("auxiliary", float("nan"))),
                "partial_epoch": int(batches < steps_per_epoch),
            },
        )
        if stop:
            break

    torch.save(
        {
            "schema_version": 1,
            "purpose": "mnist_mlp3_muon_microbatch_final_state",
            "global_step": global_step,
            "examples_seen": examples_seen,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
        },
        run_dir / "final_state.pt",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "completed": bool(global_step >= training_limit),
            "global_step": global_step,
            "examples_seen": examples_seen,
        }
    )
    _atomic_json(manifest, run_dir / "manifest.json")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the reference MNIST MLP3 Muon baseline and save the three "
            "weight matrices at microbatch cadence."
        )
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--output-dir", default="./results/mnist_mlp3_muon_microbatch"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--capture-every", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop training after this many optimizer steps; zero means full run.",
    )
    parser.add_argument(
        "--max-capture-step",
        type=int,
        default=0,
        help="Stop writing matrix checkpoints after this step; zero means no limit.",
    )
    parser.add_argument(
        "--checkpoint-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--allow-large-capture", action="store_true")
    parser.add_argument("--large-capture-gib", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_muon_microbatch_capture(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        capture_every=args.capture_every,
        max_steps=args.max_steps,
        max_capture_step=args.max_capture_step,
        checkpoint_dtype=args.checkpoint_dtype,
        allow_large_capture=args.allow_large_capture,
        large_capture_gib=args.large_capture_gib,
        overwrite=args.overwrite,
        progress=not args.quiet,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import torch

from .completion import (
    CompletedRunValidationError,
    validate_completed_run,
)
from .config import tokens_per_step
from .evaluation import evaluate_bleu, evaluate_probe
from .model import GPT

METRIC_FIELDS = [
    "step", "tokens_seen", "epoch", "elapsed_sec", "tokens_per_sec",
    "primary_lr", "auxiliary_lr", "train_loss", "train_perplexity",
    "train_accuracy", "val_loss", "val_perplexity", "val_accuracy",
    "test_loss", "test_perplexity", "test_accuracy", "test_bleu",
    "val_generalization_gap", "test_generalization_gap",
    "grad_norm_pre_clip", "grad_norm_post_clip", "gradient_clipped",
    "weight_norm", "update_norm_since_eval", "update_to_weight_ratio",
    "mps_current_allocated_mb", "mps_driver_allocated_mb",
]
EPOCH_FIELDS = [
    *METRIC_FIELDS, "nominal_epoch", "checkpoint_path", "test_monitoring_only"
]


def run_directory(results_root: str | Path, optimizer: str, seed: int) -> Path:
    return Path(results_root) / str(optimizer) / f"seed_{int(seed)}"


def run_is_complete(results_root: str | Path, optimizer: str, seed: int) -> bool:
    run_dir = run_directory(results_root, optimizer, seed)
    try:
        validate_completed_run(
            run_dir,
            expected_optimizer=str(optimizer),
            expected_seed=int(seed),
            verify_checkpoints=False,
        )
    except (CompletedRunValidationError, OSError):
        return False
    return True


def prepare_csv(path: Path, fields: list[str], resume_step: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and resume_step is not None:
        frame = pd.read_csv(path)
        if "step" in frame.columns:
            frame = frame[pd.to_numeric(frame["step"], errors="coerce") < int(resume_step)]
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    if not path.is_file() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()


def truncate_spectral_after(run_dir: Path, resume_step: int) -> None:
    spectral_root = run_dir / "spectral"
    for filename in ("layers.csv", "summary.csv"):
        path = spectral_root / filename
        if path.is_file():
            frame = pd.read_csv(path)
            if "step" in frame.columns:
                frame = frame[pd.to_numeric(frame["step"], errors="coerce") < int(resume_step)]
            temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(path)
    raw_root = spectral_root / "raw"
    if raw_root.is_dir():
        for path in raw_root.glob("weightwatcher_step_*.csv"):
            try:
                step = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if step >= int(resume_step):
                path.unlink(missing_ok=True)
    for path in spectral_root.glob("status_step_*.json"):
        try:
            step = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if step >= int(resume_step):
            path.unlink(missing_ok=True)


def write_manifest(
    run_dir: Path,
    *,
    cfg: dict,
    data_metadata: dict,
    optimizer_name: str,
    profile: dict,
    seed: int,
    device: torch.device,
    total_steps: int,
    warmup: int,
    fingerprint: str,
    model: GPT,
) -> None:
    payload = {
        "schema_version": 1,
        "protocol": cfg["protocol"],
        "optimizer": optimizer_name,
        "optimizer_profile": profile,
        "seed": int(seed),
        "device": str(device),
        "torch_version": torch.__version__,
        "model": cfg["model"],
        "parameter_count": model.parameter_count(),
        "data_metadata": data_metadata,
        "training": cfg["training"],
        "evaluation": cfg["evaluation"],
        "weightwatcher": cfg["weightwatcher"],
        "tokens_per_step": tokens_per_step(cfg),
        "max_steps": int(total_steps),
        "warmup_steps": int(warmup),
        "planned_training_tokens": int(total_steps * tokens_per_step(cfg)),
        "protocol_fingerprint": fingerprint,
        "test_policy": "fixed test probes are monitoring-only and never select checkpoints or tune schedules",
        "bleu_policy": "fixed greedy held-out continuation BLEU; secondary diagnostic, not translation BLEU",
    }
    temporary = run_dir / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    temporary.replace(run_dir / "manifest.json")


def checkpoint_eval(
    checkpoint: Path,
    *,
    model: GPT,
    test_probe,
    bleu_probe,
    device: torch.device,
    bleu_batch_size: int,
) -> dict[str, float]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    metrics = evaluate_probe(model, test_probe, device)
    bleu = evaluate_bleu(
        model, bleu_probe, device=device, batch_size=bleu_batch_size
    )
    return {
        "step": int(payload["step"]),
        "loss": float(metrics["loss"]),
        "perplexity": float(metrics["perplexity"]),
        "accuracy": float(metrics["accuracy"]),
        "bleu": float(bleu["bleu"]),
    }

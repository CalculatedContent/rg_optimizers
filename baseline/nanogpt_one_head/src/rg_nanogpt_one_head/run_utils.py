from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import sys

import pandas as pd
import torch

from .completion import (
    CompletedRunValidationError,
    validate_completed_run,
)
from .checkpoints import model_state_sha256
from .config import tokens_per_step
from .evaluation import evaluate_bleu, evaluate_probe
from .model import GPT
from .provenance import (
    repository_provenance,
    scientific_dependency_versions,
)
from .runtime import runtime_metadata

METRIC_FIELDS = [
    "step", "tokens_seen", "epoch", "elapsed_sec", "tokens_per_sec",
    "primary_lr", "auxiliary_lr", "train_loss", "train_perplexity",
    "train_bits_per_token", "train_accuracy", "train_top5_accuracy",
    "val_loss", "val_perplexity", "val_bits_per_token", "val_accuracy",
    "val_top5_accuracy", "test_loss", "test_perplexity",
    "test_bits_per_token", "test_accuracy", "test_top5_accuracy",
    "test_bleu", "test_continuation_token_accuracy",
    "test_continuation_exact_match",
    "val_generalization_gap", "test_generalization_gap",
    "grad_norm_pre_clip", "grad_norm_post_clip", "gradient_clipped",
    "weight_norm", "update_norm_since_eval", "update_to_weight_ratio",
    "mps_current_allocated_mb", "mps_driver_allocated_mb",
]
EPOCH_FIELDS = [
    *METRIC_FIELDS,
    "nominal_epoch",
    "checkpoint_path",
    "test_monitoring_only",
    "test_held_out",
]


def _package_versions() -> dict[str, str]:
    versions = scientific_dependency_versions()
    packages = (
        "rg-nanogpt-one-head",
        "papermill",
    )
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


_COMMON_RUNTIME_IDENTITY_FIELDS = (
    "platform",
    "machine",
    "python_version",
    "accelerator",
    "device",
    "torch_version",
    "float32_matmul_precision",
    "deterministic_algorithms",
    "deterministic_warn_only",
    "hardware_block_id",
    "hardware_block_id_source",
)
_ACCELERATOR_RUNTIME_IDENTITY_FIELDS = {
    "cuda": (
        "cuda_version",
        "cudnn_version",
        "cuda_device_name",
        "cuda_device_capability",
        "cuda_device_count",
        "cuda_device_uuid",
        "cuda_driver_version",
        "cuda_device_total_memory_bytes",
        "cuda_multi_processor_count",
        "cuda_nvidia_smi_memory_mib",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    ),
    "mps": (
        "mps_built",
        "mps_available",
        "mac_hardware_model",
        "mac_cpu_brand",
        "mac_memory_bytes",
    ),
    "tpu": (
        "torch_xla_version",
        "pjrt_device",
        "tpu_accelerator_type",
        "xla_process_count",
        "xla_process_index",
        "xla_addressable_device_count",
    ),
}


def _read_existing_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"existing manifest is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"existing manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"existing manifest is not a JSON object: {path}")
    return payload


def runtime_identity_payload(metadata: dict) -> dict:
    accelerator = str(metadata.get("accelerator", ""))
    fields = (
        *_COMMON_RUNTIME_IDENTITY_FIELDS,
        *_ACCELERATOR_RUNTIME_IDENTITY_FIELDS.get(accelerator, ()),
    )
    missing = [field for field in fields if field not in metadata]
    if missing:
        raise RuntimeError(
            "runtime metadata lacks identity fields: " + ", ".join(missing)
        )
    return {field: metadata[field] for field in fields}


def validate_existing_manifest_runtime(
    run_dir: str | Path,
    current_runtime: dict,
) -> dict | None:
    """Reject corrupt or cross-runtime reuse before any run artifacts mutate."""

    manifest_path = Path(run_dir) / "manifest.json"
    previous = _read_existing_manifest(manifest_path)
    if previous is None:
        ignored_pre_manifest_names = {"muonclip_walk_location.json"}
        substantive = [
            path
            for path in Path(run_dir).iterdir()
            if path.name not in ignored_pre_manifest_names
        ]
        if substantive:
            raise RuntimeError(
                "run artifacts exist without manifest.json; refusing to infer "
                "runtime/source provenance before reuse or resume: "
                + ", ".join(str(path) for path in substantive[:12])
            )
        return None
    previous_runtime = previous.get("runtime_environment")
    if not isinstance(previous_runtime, dict):
        raise RuntimeError(
            f"existing manifest has no runtime_environment mapping: {manifest_path}"
        )
    previous_identity = runtime_identity_payload(previous_runtime)
    current_identity = runtime_identity_payload(current_runtime)
    mismatches = {
        field: (
            previous_identity.get(field, "<missing>"),
            current_identity.get(field, "<missing>"),
        )
        for field in sorted(set(previous_identity) | set(current_identity))
        if previous_identity.get(field, "<missing>")
        != current_identity.get(field, "<missing>")
    }
    if mismatches:
        detail = "; ".join(
            f"{field}: existing={old!r}, current={new!r}"
            for field, (old, new) in mismatches.items()
        )
        raise RuntimeError(
            "Refusing cross-runtime reuse/resume before modifying artifacts: "
            + detail
        )
    return previous


def run_directory(
    results_root: str | Path,
    optimizer: str,
    seed: int,
) -> Path:
    return Path(results_root) / str(optimizer) / f"seed_{int(seed)}"


def run_is_complete(
    results_root: str | Path,
    optimizer: str,
    seed: int,
) -> bool:
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


def prepare_csv(
    path: Path,
    fields: list[str],
    resume_step: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and resume_step is not None:
        frame = pd.read_csv(path)
        if "step" in frame.columns:
            frame = frame[
                pd.to_numeric(
                    frame["step"],
                    errors="coerce",
                )
                < int(resume_step)
            ]
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    if not path.is_file() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()


def truncate_spectral_after(
    run_dir: Path,
    resume_step: int,
) -> None:
    spectral_root = run_dir / "spectral"
    for filename in ("layers.csv", "summary.csv"):
        path = spectral_root / filename
        if path.is_file():
            frame = pd.read_csv(path)
            if "step" in frame.columns:
                frame = frame[
                    pd.to_numeric(
                        frame["step"],
                        errors="coerce",
                    )
                    < int(resume_step)
                ]
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


def truncate_muonclip_qk_after(
    run_dir: Path,
    resume_step: int,
) -> None:
    """Discard QK diagnostics newer than the verified restart checkpoint.

    MuonClip flushes a diagnostic interval immediately after the optimizer
    update that closes it.  A process can therefore stop after the CSV row is
    durable but before ``checkpoint_latest.pt`` is replaced.  Retaining only
    rows at or before the checkpoint step makes a fresh-process resume
    idempotent; a row at the checkpoint itself is already represented by the
    checkpoint's reset diagnostic accumulator and remains valid.
    """

    path = Path(run_dir) / "muonclip_qk.csv"
    if not path.is_file():
        return
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(
            f"could not validate MuonClip QK diagnostics before resume: {path}"
        ) from exc
    if "step" not in frame.columns:
        raise RuntimeError(
            f"MuonClip QK diagnostics have no step column: {path}"
        )
    steps = pd.to_numeric(frame["step"], errors="coerce")
    if steps.isna().any() or not steps.mod(1).eq(0).all():
        raise RuntimeError(
            f"MuonClip QK diagnostics contain invalid steps: {path}"
        )
    frame = frame.loc[steps <= int(resume_step)].copy()
    retained_steps = pd.to_numeric(frame["step"], errors="raise")
    if retained_steps.duplicated().any():
        raise RuntimeError(
            f"MuonClip QK diagnostics contain duplicate verified steps: {path}"
        )
    frame = frame.sort_values("step")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_manifest(
    run_dir: Path,
    *,
    cfg: dict,
    data_metadata: dict,
    optimizer_name: str,
    profile: dict,
    seed: int,
    device: torch.device,
    data_root: str | Path,
    results_root: str | Path,
    total_steps: int,
    schedule_steps: int,
    warmup: int,
    fingerprint: str,
    model: GPT,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    manifest_path = run_dir / "manifest.json"
    started_at = now
    resume_count = 0
    current_runtime = runtime_metadata(device)
    previous = validate_existing_manifest_runtime(run_dir, current_runtime)
    current_model_hash = model_state_sha256(model.state_dict())
    if previous is not None:
        started_at = str(previous.get("run_started_at_utc", now))
        resume_count = int(previous.get("resume_count", 0)) + 1
        initial_model_hash = str(previous.get("initial_model_sha256", ""))
        if len(initial_model_hash) != 64:
            raise RuntimeError(
                "existing manifest has no valid initial-model tensor hash"
            )
    else:
        initial_model_hash = current_model_hash
    canonical_config = json.dumps(
        cfg,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    payload = {
        "schema_version": 3,
        "run_started_at_utc": started_at,
        "manifest_updated_at_utc": now,
        "resume_count": resume_count,
        "config_sha256": hashlib.sha256(
            canonical_config.encode("utf-8")
        ).hexdigest(),
        "invocation": {
            "argv": [str(value) for value in sys.argv],
            "campaign_command": str(
                os.environ.get("RG_NANOGPT_CAMPAIGN_COMMAND", "")
            ),
        },
        "protocol": cfg["protocol"],
        "optimizer": optimizer_name,
        "optimizer_profile": profile,
        "seed": int(seed),
        "device": str(device),
        "runtime_environment": current_runtime,
        "source_repository": repository_provenance(),
        "initial_model_sha256": initial_model_hash,
        "storage": {
            "data_root": str(Path(data_root)),
            "results_root": str(Path(results_root)),
            "run_dir": str(Path(run_dir)),
        },
        "torch_version": torch.__version__,
        "package_versions": _package_versions(),
        "model": cfg["model"],
        "parameter_count": model.parameter_count(),
        "data_metadata": data_metadata,
        "training": cfg["training"],
        "evaluation": cfg["evaluation"],
        "weightwatcher": cfg["weightwatcher"],
        "tokens_per_step": tokens_per_step(cfg),
        "max_steps": int(total_steps),
        "lr_schedule_steps": int(schedule_steps),
        "warmup_steps": int(warmup),
        "planned_training_tokens": int(
            total_steps * tokens_per_step(cfg)
        ),
        "protocol_fingerprint": fingerprint,
        "test_policy": (
            "test is held out until post-training; validation selects the "
            "best checkpoint and test never tunes the protocol"
        ),
        "bleu_policy": (
            "fixed greedy held-out continuation BLEU; secondary diagnostic, "
            "not translation BLEU"
        ),
    }
    temporary = run_dir / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def checkpoint_eval(
    checkpoint: Path,
    *,
    model: GPT,
    test_probe,
    bleu_probe,
    device: torch.device,
    bleu_batch_size: int,
) -> dict[str, float]:
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(payload["model"])
    metrics = evaluate_probe(model, test_probe, device)
    bleu = evaluate_bleu(
        model,
        bleu_probe,
        device=device,
        batch_size=bleu_batch_size,
    )
    return {
        "step": int(payload["step"]),
        "loss": float(metrics["loss"]),
        "perplexity": float(metrics["perplexity"]),
        "bits_per_token": float(metrics["bits_per_token"]),
        "accuracy": float(metrics["accuracy"]),
        "top5_accuracy": float(metrics["top5_accuracy"]),
        "bleu": float(bleu["bleu"]),
        "continuation_token_accuracy": float(
            bleu.get("continuation_token_accuracy", float("nan"))
        ),
        "continuation_exact_match": float(
            bleu.get("continuation_exact_match", float("nan"))
        ),
    }

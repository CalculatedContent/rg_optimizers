"""Explicit imports of full training state into separately identified extensions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch

from .checkpoints import load_training_checkpoint_for_resume
from .config import tokens_per_step, validate_config


class TrainingPaused(SystemExit):
    """Exit without making a deliberate pause consume the recovery budget."""

    def __init__(self, reason: str):
        print(f"[one-head-pause] {reason}; restart checkpoint retained", flush=True)
        super().__init__(75)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pause_reason(cfg: dict, run_dir: Path) -> str | None:
    training = cfg["training"]
    stop = training.get("stop_file")
    if stop and Path(stop).exists():
        return f"stop requested by {stop}"
    reserve = float(training.get("min_free_disk_gb", 0))
    if reserve > 0 and shutil.disk_usage(run_dir).free < reserve * 1024**3:
        return f"free disk space fell below {reserve:g} GiB"
    return None


def make_continuation_config(
    parent_path: Path, *, steps: int, learning_rate: float | None,
    test_interval: int, stop_file: Path, min_free_disk_gb: float,
) -> dict:
    if steps < 1 or test_interval < 1 or min_free_disk_gb < 0:
        raise ValueError("steps/test interval must be positive; disk reserve nonnegative")
    parent_path = Path(parent_path).resolve()
    before_hash = file_sha256(parent_path)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if before_hash != file_sha256(parent_path):
        raise RuntimeError("parent checkpoint changed during import; use an immutable final checkpoint")
    if parent.get("optimizer_name") != "muon_clip" or not parent.get("optimizers"):
        raise ValueError("continuation requires a full MuonClip training checkpoint")
    if int(parent.get("step", 0)) <= 0 or parent.get("resume_diagnostics") is None:
        raise ValueError("parent must contain trained state and deterministic resume diagnostics")
    cfg = deepcopy(parent["config"])
    if cfg.get("memorization", {}).get("enabled"):
        raise ValueError("continuation currently supports clean-corpus training only")
    profile = cfg["optimizer_profiles"]["muon_clip"]
    rate = float(profile["min_learning_rate"] if learning_rate is None else learning_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("continuation learning rate must be finite and positive")
    epochs = steps * tokens_per_step(cfg) / int(cfg["dataset"]["train_tokens"])
    cfg["protocol"] = {
        "name": "rg_nanogpt_muonclip_continuation", "version": 1,
        "description": "Full-state continuation with constant LR and fixed test monitoring",
    }
    cfg["training"].update(
        max_steps=int(steps), target_epochs=epochs,
        stop_file=str(stop_file.resolve()), min_free_disk_gb=float(min_free_disk_gb),
    )
    for item in cfg["optimizer_profiles"].values():
        item.pop("lr_schedule_epochs", None)
    profile.update(learning_rate=rate, min_learning_rate=rate, warmup_fraction=0.0)
    cfg["evaluation"]["test_interval_steps"] = int(test_interval)
    cfg["continuation"] = {
        "parent_checkpoint": str(parent_path), "parent_file_sha256": before_hash,
        "parent_fingerprint": parent["fingerprint"],
        "global_step_offset": int(parent.get("global_step", parent["step"])),
        "parent_local_step": int(parent["step"]),
        "seed": int(parent["seed"]),
        "policy": "preserve weights, momentum, Adam steps, RNG and sampler; reset only phase diagnostics and phase best-validation selection",
    }
    validate_config(cfg)
    return cfg


def import_parent_state(cfg, *, model, handles, train_generator, data_metadata, seed, current_runtime):
    """Changing run identity is allowed only through this recorded import path."""
    lineage = cfg["continuation"]
    path = Path(lineage["parent_checkpoint"])
    if file_sha256(path) != lineage["parent_file_sha256"]:
        raise RuntimeError("parent checkpoint file SHA-256 does not match")
    parent = torch.load(path, map_location="cpu", weights_only=False)
    if parent["optimizer_name"] != "muon_clip" or parent["seed"] != seed:
        raise RuntimeError("continuation parent optimizer/seed mismatch")
    if int(parent.get("global_step", parent["step"])) != lineage["global_step_offset"]:
        raise RuntimeError("continuation global step does not match parent")
    old = parent["config"]
    for key in ("model", "dataset", "runtime", "weightwatcher"):
        if old[key] != cfg[key]:
            raise RuntimeError(f"continuation cannot change {key}")
    for key in ("batch_size", "grad_accum_steps", "grad_clip", "eval_batches"):
        if old["training"][key] != cfg["training"][key]:
            raise RuntimeError(f"continuation cannot change training.{key}")
    ignored = {"learning_rate", "min_learning_rate", "warmup_fraction", "lr_schedule_epochs"}
    for key in set(old["optimizer_profiles"]["muon_clip"]) | set(cfg["optimizer_profiles"]["muon_clip"]):
        if key not in ignored and old["optimizer_profiles"]["muon_clip"].get(key) != cfg["optimizer_profiles"]["muon_clip"].get(key):
            raise RuntimeError(f"continuation cannot change optimizer {key}")
    for key, value in old["evaluation"].items():
        if key != "test_interval_steps" and cfg["evaluation"].get(key) != value:
            raise RuntimeError(f"continuation cannot change evaluation.{key}")
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest["protocol_fingerprint"] != lineage["parent_fingerprint"]:
        raise RuntimeError("parent manifest/checkpoint fingerprint mismatch")
    if manifest["data_metadata"] != data_metadata:
        raise RuntimeError("continuation data inventory differs from parent")
    numerical_fields = (
        "accelerator", "torch_version", "torch_xla_version", "float32_matmul_precision",
        "xla_matmul_precision", "xla_spmd", "xla_spmd_chips", "tpu_accelerator_type",
    )
    for key in numerical_fields:
        if manifest["runtime_environment"].get(key) != current_runtime.get(key):
            raise RuntimeError(f"continuation numerical runtime changed: {key}")
    loaded = load_training_checkpoint_for_resume(
        path, model=model, handles=handles,
        expected_fingerprint=lineage["parent_fingerprint"], train_generator=train_generator,
    )
    if loaded[0] != lineage["parent_local_step"] or loaded[4] is None:
        raise RuntimeError("parent restart state is incomplete")
    if file_sha256(path) != lineage["parent_file_sha256"]:
        raise RuntimeError("parent checkpoint changed while restoring state")
    # This counter controls ONLY QK CSV interval boundaries. Momentum buffers
    # and auxiliary Adam's per-parameter update counters remain untouched.
    for handle in handles:
        # The CLI may load MuonClip as __main__, so identity checks against a
        # second imported module would silently miss the active optimizer.
        reset = getattr(handle.optimizer, "reset_phase_diagnostics", None)
        if callable(reset):
            reset()
    return loaded[4]

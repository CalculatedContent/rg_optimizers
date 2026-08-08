from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .optimizers import (
    OptimizerHandle,
    load_optimizer_state_dict,
    optimizer_state_dict,
)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _capture_accelerator_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    if torch.cuda.is_available():
        state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and torch.backends.mps.is_available()
    ):
        state["mps_random_state"] = torch.mps.get_rng_state()
    return state


def _restore_accelerator_rng_state(payload: dict[str, Any]) -> None:
    if torch.cuda.is_available() and "cuda_random_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_random_state_all"])
    if (
        "mps_random_state" in payload
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(payload["mps_random_state"])


def save_training_checkpoint(
    path: str | Path,
    *,
    model,
    handles: list[OptimizerHandle],
    step: int,
    best_validation_loss: float,
    best_validation_step: int,
    elapsed_seconds: float,
    fingerprint: str,
    cfg: dict,
    optimizer_name: str,
    seed: int,
    train_generator: torch.Generator,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "model": model.state_dict(),
        "optimizers": optimizer_state_dict(handles),
        "step": int(step),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_step": int(best_validation_step),
        "elapsed_seconds": float(elapsed_seconds),
        "fingerprint": str(fingerprint),
        "config": cfg,
        "optimizer_name": str(optimizer_name),
        "seed": int(seed),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "train_generator_state": train_generator.get_state(),
        **_capture_accelerator_rng_state(),
    }
    return _atomic_torch_save(payload, Path(path))


def load_training_checkpoint(
    path: str | Path,
    *,
    model,
    handles: list[OptimizerHandle],
    expected_fingerprint: str,
    train_generator: torch.Generator,
) -> tuple[int, float, int, float]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("fingerprint")) != str(expected_fingerprint):
        raise RuntimeError(
            "checkpoint protocol fingerprint does not match the requested run"
        )
    model.load_state_dict(payload["model"])
    load_optimizer_state_dict(handles, payload["optimizers"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.random.set_rng_state(payload["torch_random_state"])
    train_generator.set_state(payload["train_generator_state"])
    _restore_accelerator_rng_state(payload)
    return (
        int(payload["step"]),
        float(payload["best_validation_loss"]),
        int(payload["best_validation_step"]),
        float(payload["elapsed_seconds"]),
    )


def save_epoch_model_checkpoint(
    run_dir: str | Path,
    *,
    model,
    step: int,
    nominal_epoch: float,
    actual_epoch: float,
    fingerprint: str,
    cfg: dict,
    optimizer_name: str,
    seed: int,
) -> Path:
    epoch_text = f"{float(nominal_epoch):06.3f}".replace(".", "p")
    path = (
        Path(run_dir)
        / "epoch_checkpoints"
        / f"model_epoch_{epoch_text}_step_{int(step):07d}.pt"
    )
    payload = {
        "schema_version": 1,
        "model": model.state_dict(),
        "step": int(step),
        "nominal_epoch": float(nominal_epoch),
        "actual_epoch": float(actual_epoch),
        "fingerprint": str(fingerprint),
        "config": cfg,
        "optimizer_name": str(optimizer_name),
        "seed": int(seed),
        "purpose": "per_epoch_model_only_analysis_checkpoint",
    }
    return _atomic_torch_save(payload, path)

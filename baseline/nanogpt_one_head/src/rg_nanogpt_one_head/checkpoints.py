from __future__ import annotations

import hashlib
import json
import math
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
from .runtime import (
    capture_accelerator_rng_state,
    model_device,
    restore_accelerator_rng_state,
    synchronize,
    tree_to_cpu,
)


def model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash model tensor names, shapes, dtypes, and exact bytes."""

    digest = hashlib.sha256()
    if not state:
        raise ValueError("model state is empty")
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"model state entry is not a tensor: {name}")
        tensor = value.detach().to("cpu").contiguous()
        metadata = json.dumps(
            {
                "name": str(name),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _update_state_digest(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().to("cpu").contiguous()
        metadata = json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(b"tensor\0")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        items = sorted(
            value.items(),
            key=lambda item: (type(item[0]).__name__, repr(item[0])),
        )
        digest.update(len(items).to_bytes(8, "big"))
        for key, item in items:
            _update_state_digest(digest, key)
            _update_state_digest(digest, item)
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_state_digest(digest, item)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
        return
    if isinstance(value, int):
        digest.update(b"int\0" + str(value).encode("ascii") + b"\0")
        return
    if isinstance(value, float):
        digest.update(b"float\0" + value.hex().encode("ascii") + b"\0")
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str\0" + len(encoded).to_bytes(8, "big") + encoded)
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0" + len(value).to_bytes(8, "big") + value)
        return
    raise TypeError(
        "unsupported optimizer-state value for integrity hashing: "
        f"{type(value).__name__}"
    )


def optimizer_state_sha256(states: list[dict[str, Any]]) -> str:
    """Hash the complete optimizer-state structure and exact tensor bytes."""

    if not states:
        raise ValueError("optimizer state inventory is empty")
    digest = hashlib.sha256()
    _update_state_digest(digest, states)
    return digest.hexdigest()


def _nonfinite_tensor_paths(value: Any, path: str) -> list[str]:
    bad: list[str] = []
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                bad.append(path)
        return bad
    if isinstance(value, float) and not math.isfinite(value):
        bad.append(path)
        return bad
    if isinstance(value, dict):
        for key, item in value.items():
            bad.extend(
                _nonfinite_tensor_paths(
                    item,
                    f"{path}.{key}",
                )
            )
        return bad
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            bad.extend(
                _nonfinite_tensor_paths(
                    item,
                    f"{path}[{index}]",
                )
            )
    return bad


def require_finite_checkpoint_state(
    *,
    model_state: dict[str, Any],
    optimizer_states: list[dict[str, Any]] | None,
    step: int,
) -> None:
    bad = _nonfinite_tensor_paths(model_state, "model")
    if optimizer_states is not None:
        bad.extend(
            _nonfinite_tensor_paths(
                optimizer_states,
                "optimizers",
            )
        )
    if bad:
        preview = ", ".join(bad[:12])
        suffix = "" if len(bad) <= 12 else f" (+{len(bad) - 12} more)"
        raise FloatingPointError(
            "refusing to write or load a contaminated checkpoint at "
            f"step={int(step)}; non-finite tensors: {preview}{suffix}"
        )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


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
    resume_diagnostics: dict[str, Any] | None = None,
) -> Path:
    device = model_device(model)
    synchronize(device)

    # Materialize model and optimizer state on CPU before touching the target
    # path. A Metal command-buffer recovery can otherwise leave finite-looking
    # Python control flow around corrupted accelerator tensors. The finite-state
    # gate ensures checkpoint_latest.pt always remains the last verified state.
    model_state = tree_to_cpu(model.state_dict())
    optimizer_states = tree_to_cpu(optimizer_state_dict(handles))
    require_finite_checkpoint_state(
        model_state=model_state,
        optimizer_states=optimizer_states,
        step=step,
    )

    persisted_diagnostics: dict[str, Any] | None = None
    if resume_diagnostics is not None:
        previous_snapshot = tree_to_cpu(
            resume_diagnostics.get("previous_eval_snapshot")
        )
        if not isinstance(previous_snapshot, list) or not previous_snapshot:
            raise ValueError(
                "resume diagnostics must contain a non-empty "
                "previous_eval_snapshot list"
            )
        expected_parameters = list(model.parameters())
        if len(previous_snapshot) != len(expected_parameters):
            raise ValueError(
                "resume diagnostic parameter inventory does not match the model"
            )
        for index, (snapshot, parameter) in enumerate(
            zip(previous_snapshot, expected_parameters, strict=True)
        ):
            if not torch.is_tensor(snapshot):
                raise TypeError(
                    "resume diagnostic snapshot entry is not a tensor: "
                    f"index={index}"
                )
            if tuple(snapshot.shape) != tuple(parameter.shape):
                raise ValueError(
                    "resume diagnostic snapshot shape does not match the model: "
                    f"index={index}, snapshot={tuple(snapshot.shape)}, "
                    f"parameter={tuple(parameter.shape)}"
                )
        bad_diagnostics = _nonfinite_tensor_paths(
            previous_snapshot,
            "resume_diagnostics.previous_eval_snapshot",
        )
        if bad_diagnostics:
            raise FloatingPointError(
                "refusing to write contaminated resume diagnostics: "
                + ", ".join(bad_diagnostics[:12])
            )
        last_grad_pre = float(resume_diagnostics["last_grad_pre"])
        last_grad_post = float(resume_diagnostics["last_grad_post"])
        if not math.isfinite(last_grad_pre) or not math.isfinite(last_grad_post):
            raise FloatingPointError(
                "refusing to write non-finite resume diagnostic gradients"
            )
        persisted_diagnostics = {
            "schema_version": 1,
            "previous_eval_snapshot": previous_snapshot,
            "last_grad_pre": last_grad_pre,
            "last_grad_post": last_grad_post,
            "last_clipped": bool(resume_diagnostics["last_clipped"]),
        }

    payload: dict[str, Any] = {
        "schema_version": 5,
        # CPU tensors keep checkpoints portable between MPS, CUDA, TPU/XLA,
        # and CPU environments.
        "model": model_state,
        "optimizers": optimizer_states,
        "model_state_sha256": model_state_sha256(model_state),
        "optimizer_state_sha256": optimizer_state_sha256(optimizer_states),
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
        "resume_diagnostics": persisted_diagnostics,
        **capture_accelerator_rng_state(device),
    }
    return _atomic_torch_save(payload, Path(path))


def _load_training_checkpoint(
    path: str | Path,
    *,
    model,
    handles: list[OptimizerHandle],
    expected_fingerprint: str,
    train_generator: torch.Generator,
) -> tuple[int, float, int, float, dict[str, Any] | None]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("fingerprint")) != str(expected_fingerprint):
        raise RuntimeError(
            "checkpoint protocol fingerprint does not match the requested run"
        )
    model_hash = model_state_sha256(payload["model"])
    if str(payload.get("model_state_sha256", "")) != model_hash:
        raise RuntimeError("checkpoint model-state SHA-256 does not match")
    optimizer_hash = optimizer_state_sha256(payload["optimizers"])
    if str(payload.get("optimizer_state_sha256", "")) != optimizer_hash:
        raise RuntimeError("checkpoint optimizer-state SHA-256 does not match")
    require_finite_checkpoint_state(
        model_state=payload["model"],
        optimizer_states=payload["optimizers"],
        step=int(payload.get("step", -1)),
    )
    model.load_state_dict(payload["model"])
    load_optimizer_state_dict(handles, payload["optimizers"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.random.set_rng_state(payload["torch_random_state"])
    train_generator.set_state(payload["train_generator_state"])
    restore_accelerator_rng_state(payload, model_device(model))
    diagnostics = payload.get("resume_diagnostics")
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise RuntimeError("checkpoint resume_diagnostics is not a mapping")
        if int(diagnostics.get("schema_version", -1)) != 1:
            raise RuntimeError(
                "checkpoint resume_diagnostics has an unsupported schema"
            )
        previous_snapshot = diagnostics.get("previous_eval_snapshot")
        expected_parameters = list(model.parameters())
        if (
            not isinstance(previous_snapshot, list)
            or len(previous_snapshot) != len(expected_parameters)
        ):
            raise RuntimeError(
                "checkpoint resume diagnostic parameter inventory is invalid"
            )
        for index, (snapshot, parameter) in enumerate(
            zip(previous_snapshot, expected_parameters, strict=True)
        ):
            if (
                not torch.is_tensor(snapshot)
                or tuple(snapshot.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(snapshot).all())
            ):
                raise RuntimeError(
                    "checkpoint resume diagnostic snapshot is invalid at "
                    f"parameter index {index}"
                )
        diagnostics = {
            "previous_eval_snapshot": tree_to_cpu(previous_snapshot),
            "last_grad_pre": float(diagnostics["last_grad_pre"]),
            "last_grad_post": float(diagnostics["last_grad_post"]),
            "last_clipped": bool(diagnostics["last_clipped"]),
        }
    return (
        int(payload["step"]),
        float(payload["best_validation_loss"]),
        int(payload["best_validation_step"]),
        float(payload["elapsed_seconds"]),
        diagnostics,
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model,
    handles: list[OptimizerHandle],
    expected_fingerprint: str,
    train_generator: torch.Generator,
) -> tuple[int, float, int, float]:
    """Load a checkpoint while preserving the historical four-value API."""

    loaded = _load_training_checkpoint(
        path,
        model=model,
        handles=handles,
        expected_fingerprint=expected_fingerprint,
        train_generator=train_generator,
    )
    return loaded[:4]


def load_training_checkpoint_for_resume(
    path: str | Path,
    *,
    model,
    handles: list[OptimizerHandle],
    expected_fingerprint: str,
    train_generator: torch.Generator,
) -> tuple[int, float, int, float, dict[str, Any] | None]:
    """Load training state plus deterministic monitoring diagnostics."""

    return _load_training_checkpoint(
        path,
        model=model,
        handles=handles,
        expected_fingerprint=expected_fingerprint,
        train_generator=train_generator,
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
    device = model_device(model)
    synchronize(device)
    model_state = tree_to_cpu(model.state_dict())
    require_finite_checkpoint_state(
        model_state=model_state,
        optimizer_states=None,
        step=step,
    )
    payload = {
        "schema_version": 3,
        "model": model_state,
        "model_state_sha256": model_state_sha256(model_state),
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

from __future__ import annotations

"""Opt-in early-step trajectory capture for the one-head MuonClip baseline.

The dedicated ``rg-onehead-muonclip-walk`` launcher installs the ordinary
MuonClip extension and then records the first N *optimizer steps* (effective
batches after gradient accumulation).  The standard MuonClip launcher and
historical configs are unchanged.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
from typing import Any
import uuid

import pandas as pd
import torch
import torch.nn as nn

from .model import GPT, GPTConfig, transformer_matrix_items
from .optimizers import zeropower_via_newton_schulz_5
from .runtime import model_device, synchronize, tree_to_cpu

_INSTALLED = False
_MAX_CAPTURE_STEPS = 20
_DEFAULT_CAPTURE_ROOT = Path("/tmp/rg-nanogpt-muonclip-walk")
_CAPTURE_ROOT_ENV = "RG_MUONCLIP_WALK_ROOT"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().clone()


def _tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).cpu())


def _tensor_rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.float().square().mean().sqrt().cpu())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 0.0:
        return float("nan")
    return float(torch.dot(left, right).div(denominator).cpu())


def _spectral_payload(value: torch.Tensor) -> dict[str, Any]:
    matrix = value.detach().float().cpu()
    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.numel() == 0:
        return {
            "singular_values": singular_values,
            "spectral_norm": 0.0,
            "frobenius_norm": 0.0,
            "nuclear_norm": 0.0,
            "stable_rank": 0.0,
            "effective_rank": 0.0,
            "sv_mean": 0.0,
            "sv_std": 0.0,
            "sv_min": 0.0,
            "sv_max": 0.0,
        }
    spectral_norm = float(singular_values.max())
    frobenius_norm = float(torch.linalg.vector_norm(matrix))
    nuclear_norm = float(singular_values.sum())
    stable_rank = (
        frobenius_norm * frobenius_norm
        / max(spectral_norm * spectral_norm, 1e-30)
    )
    total = singular_values.sum()
    if float(total) > 0.0:
        probabilities = singular_values / total
        positive = probabilities > 0
        entropy = -(
            probabilities[positive] * probabilities[positive].log()
        ).sum()
        effective_rank = float(entropy.exp())
    else:
        effective_rank = 0.0
    return {
        "singular_values": singular_values,
        "spectral_norm": spectral_norm,
        "frobenius_norm": frobenius_norm,
        "nuclear_norm": nuclear_norm,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "sv_mean": float(singular_values.mean()),
        "sv_std": (
            float(singular_values.std(unbiased=True))
            if singular_values.numel() > 1
            else 0.0
        ),
        "sv_min": float(singular_values.min()),
        "sv_max": float(singular_values.max()),
    }


def _append_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    keys: list[str],
) -> None:
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if path.is_file() and path.stat().st_size:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming
    combined = (
        combined.drop_duplicates(keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


class SavedWeightMatrixHolder(nn.Module):
    """Self-contained PyTorch holder accepted directly by WeightWatcher."""

    def __init__(
        self,
        matrices: dict[str, torch.Tensor],
        metadata: list[dict[str, Any]],
    ) -> None:
        super().__init__()
        self.matrix_metadata = metadata
        for item in metadata:
            name = str(item["matrix_name"])
            weight = matrices[name].detach().float().cpu()
            layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
            self.add_module(name, layer)


def load_weightwatcher_checkpoint(
    path: str | Path,
) -> tuple[SavedWeightMatrixHolder, dict[str, Any]]:
    """Load a captured six-matrix checkpoint for ``ww.WeightWatcher``."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("purpose") != "weightwatcher_matrix_checkpoint":
        raise ValueError(f"not a WeightWatcher matrix checkpoint: {path}")
    holder = SavedWeightMatrixHolder(
        payload["matrices"],
        list(payload["matrix_metadata"]),
    )
    return holder, payload


def load_full_model_checkpoint(
    path: str | Path,
) -> tuple[GPT, dict[str, Any]]:
    """Load a captured full-model checkpoint on CPU."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("purpose") != "muonclip_walk_full_model_checkpoint":
        raise ValueError(f"not a MuonClip walk model checkpoint: {path}")
    model = GPT(GPTConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


class MuonClipWalkRecorder:
    """Capture the first few effective MuonClip optimizer batches.

    A captured optimizer step is one update after all configured gradient-
    accumulation microbatches. Gradients in the trace are the actual gradients
    used by the optimizer *after* the global gradient clip.
    """

    def __init__(
        self,
        *,
        model: GPT,
        optimizer,
        profile: dict[str, Any],
        run_dir: Path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.profile = dict(profile)
        self.run_dir = Path(run_dir)
        self.max_steps = int(profile.get("walk_capture_steps", 0))
        self.save_full_model = bool(
            profile.get("walk_save_full_model", True)
        )
        self.save_weightwatcher = bool(
            profile.get("walk_save_weightwatcher", True)
        )
        self.save_optimizer_tensors = bool(
            profile.get("walk_save_optimizer_tensors", True)
        )
        self.pending: dict[str, Any] | None = None

        root_value = os.environ.get(
            _CAPTURE_ROOT_ENV,
            str(profile.get("walk_capture_root", _DEFAULT_CAPTURE_ROOT)),
        )
        self.capture_root = Path(root_value).expanduser()
        if not self.capture_root.is_absolute():
            raise ValueError("MuonClip walk capture root must be absolute")
        self.capture_root.mkdir(parents=True, exist_ok=True)

        self.pointer_path = self.run_dir / "muonclip_walk_location.json"
        self.capture_dir = self._resolve_capture_dir()
        self.checkpoint_dir = self.capture_dir / "model_checkpoints"
        self.ww_dir = self.capture_dir / "weightwatcher_checkpoints"
        self.step_dir = self.capture_dir / "step_traces"
        for path in (
            self.checkpoint_dir,
            self.ww_dir,
            self.step_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.matrix_items = transformer_matrix_items(model)
        self.matrix_metadata = [
            {
                "matrix_name": name,
                "matrix_type": matrix_type,
                "block": int(block),
                "shape": list(weight.shape),
            }
            for name, matrix_type, block, weight in self.matrix_items
        ]
        self.captured_steps = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in self.step_dir.glob("step_*.pt")
            if path.stem.rsplit("_", 1)[-1].isdigit()
        }
        self._write_capture_manifest()
        self._write_readme()

    def _resolve_capture_dir(self) -> Path:
        if self.pointer_path.is_file():
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
            path = Path(pointer["capture_dir"])
            if not path.is_dir():
                raise FileNotFoundError(
                    "MuonClip walk pointer exists but capture directory is "
                    f"missing: {path}"
                )
            return path

        run_id = (
            f"{self.run_dir.parent.name}_{self.run_dir.name}_"
            f"{_timestamp_id()}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        path = self.capture_root / run_id
        path.mkdir(parents=False, exist_ok=False)
        _atomic_json(
            self.pointer_path,
            {
                "schema_version": 1,
                "capture_dir": str(path),
                "created_at_utc": _utc_now(),
                "run_dir": str(self.run_dir),
                "capture_root_env": _CAPTURE_ROOT_ENV,
            },
        )
        return path

    def _run_manifest(self) -> dict[str, Any]:
        path = self.run_dir / "manifest.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _write_capture_manifest(self) -> None:
        manifest = self._run_manifest()
        payload = {
            "schema_version": 1,
            "purpose": "muonclip_early_optimizer_walk",
            "created_at_utc": _utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "run_dir": str(self.run_dir),
            "capture_dir": str(self.capture_dir),
            "capture_semantics": (
                "one captured batch equals one optimizer update after all "
                "gradient-accumulation microbatches"
            ),
            "gradient_semantics": "post_global_grad_clip",
            "max_optimizer_steps": self.max_steps,
            "hard_max_optimizer_steps": _MAX_CAPTURE_STEPS,
            "save_full_model": self.save_full_model,
            "save_weightwatcher": self.save_weightwatcher,
            "save_optimizer_tensors": self.save_optimizer_tensors,
            "matrix_metadata": self.matrix_metadata,
            "optimizer_profile": self.profile,
            "run_manifest": manifest,
        }
        _atomic_json(self.capture_dir / "walk_manifest.json", payload)

    def _write_readme(self) -> None:
        text = f"""# MuonClip early-step capture

This directory is append-only and belongs to:

```text
{self.run_dir}
```

It captures optimizer steps 1 through {self.max_steps}. Each step is one
**effective batch after gradient accumulation**, not one microbatch.

Files:

```text
model_checkpoints/model_step_0000000.pt
model_checkpoints/model_step_0000001.pt ...
weightwatcher_checkpoints/ww_step_0000000.pt
weightwatcher_checkpoints/ww_step_0000001.pt ...
step_traces/step_0000001.pt ...
checkpoint_index.csv
step_trajectory.csv
matrix_trajectory.csv
walk_manifest.json
```

`model_step_0000000.pt` is the initialized model. `model_step_N.pt` is the
full model after optimizer step N, including the auxiliary AdamW update.

Each step trace stores, for all six hidden matrices:

```text
weight_before
gradient_post_clip
momentum_before
momentum_after
update_source
rms_matched_orthogonal_update
predicted_weight_before_qk_clip
predicted_weight_after_qk_clip
weight_after
parameter_delta
prediction_residual
singular values before/after/delta
QK maximum logit and clipping scales
```

Load a WeightWatcher checkpoint:

```python
import weightwatcher as ww
from rg_nanogpt_one_head.muonclip_walk import load_weightwatcher_checkpoint

holder, metadata = load_weightwatcher_checkpoint(
    "weightwatcher_checkpoints/ww_step_0000010.pt"
)
details = ww.WeightWatcher(model=holder).analyze(
    ERG=True,
    randomize=True,
    plot=True,
    min_evals=20,
)
```

Load the full model:

```python
from rg_nanogpt_one_head.muonclip_walk import load_full_model_checkpoint

model, metadata = load_full_model_checkpoint(
    "model_checkpoints/model_step_0000010.pt"
)
```
"""
        path = self.capture_dir / "README.md"
        if not path.exists():
            path.write_text(text, encoding="utf-8")

    def _coordinates(self, step: int) -> dict[str, Any]:
        manifest = self._run_manifest()
        training = manifest.get("training", {})
        model = manifest.get("model", asdict(self.model.cfg))
        data_metadata = manifest.get("data_metadata", {})
        split_tokens = data_metadata.get("splits", {})
        train_tokens = int(split_tokens.get("train", 1))
        batch_size = int(training.get("batch_size", 1))
        grad_accum = int(training.get("grad_accum_steps", 1))
        block_size = int(model.get("block_size", self.model.cfg.block_size))
        tokens_per_step = batch_size * grad_accum * block_size
        tokens_seen = int(step * tokens_per_step)
        return {
            "step": int(step),
            "tokens_seen": tokens_seen,
            "epoch": tokens_seen / max(train_tokens, 1),
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum,
            "effective_batch_sequences": batch_size * grad_accum,
            "tokens_per_step": tokens_per_step,
            "train_tokens": train_tokens,
        }

    def _current_matrices(self) -> dict[str, torch.Tensor]:
        return {
            name: _cpu_clone(weight)
            for name, _, _, weight in transformer_matrix_items(self.model)
        }

    def _state_paths(self, step: int) -> tuple[Path, Path]:
        return (
            self.checkpoint_dir / f"model_step_{step:07d}.pt",
            self.ww_dir / f"ww_step_{step:07d}.pt",
        )

    def _save_state(self, step: int, *, stage: str) -> dict[str, str]:
        coordinates = self._coordinates(step)
        model_path, ww_path = self._state_paths(step)
        synchronize(model_device(self.model))
        matrices = self._current_matrices()
        run_manifest = self._run_manifest()

        if self.save_full_model and not model_path.exists():
            _atomic_torch_save(
                {
                    "schema_version": 1,
                    "purpose": "muonclip_walk_full_model_checkpoint",
                    "stage": stage,
                    **coordinates,
                    "model_config": asdict(self.model.cfg),
                    "model": tree_to_cpu(self.model.state_dict()),
                    "optimizer_name": "muon_clip",
                    "run_dir": str(self.run_dir),
                    "capture_dir": str(self.capture_dir),
                    "run_manifest": run_manifest,
                },
                model_path,
            )

        if self.save_weightwatcher and not ww_path.exists():
            _atomic_torch_save(
                {
                    "schema_version": 1,
                    "purpose": "weightwatcher_matrix_checkpoint",
                    "stage": stage,
                    **coordinates,
                    "matrices": matrices,
                    "matrix_metadata": self.matrix_metadata,
                    "optimizer_name": "muon_clip",
                    "run_dir": str(self.run_dir),
                    "capture_dir": str(self.capture_dir),
                },
                ww_path,
            )

        rows = [
            {
                **coordinates,
                "stage": stage,
                "model_checkpoint": (
                    str(model_path) if self.save_full_model else ""
                ),
                "model_sha256": (
                    _sha256(model_path)
                    if self.save_full_model and model_path.is_file()
                    else ""
                ),
                "weightwatcher_checkpoint": (
                    str(ww_path) if self.save_weightwatcher else ""
                ),
                "weightwatcher_sha256": (
                    _sha256(ww_path)
                    if self.save_weightwatcher and ww_path.is_file()
                    else ""
                ),
            }
        ]
        _append_rows(
            self.capture_dir / "checkpoint_index.csv",
            rows,
            keys=["step"],
        )
        return {
            "model_checkpoint": (
                str(model_path) if self.save_full_model else ""
            ),
            "weightwatcher_checkpoint": (
                str(ww_path) if self.save_weightwatcher else ""
            ),
        }

    def should_capture(self) -> bool:
        next_step = int(getattr(self.optimizer, "step_index", 0)) + 1
        return (
            1 <= next_step <= self.max_steps
            and next_step not in self.captured_steps
        )

    def _group_for_parameter(
        self,
        parameter: torch.nn.Parameter,
    ) -> dict[str, Any]:
        for group in self.optimizer.param_groups:
            if any(candidate is parameter for candidate in group["params"]):
                return group
        raise RuntimeError("captured matrix is missing from MuonClip parameter groups")

    def capture_before(self, handles) -> bool:
        if not self.should_capture():
            self.pending = None
            return False
        synchronize(model_device(self.model))
        step = int(self.optimizer.step_index) + 1
        if step == 1:
            self._save_state(0, stage="initialized_before_first_step")

        matrices: dict[str, dict[str, Any]] = {}
        for name, matrix_type, block, parameter in self.matrix_items:
            group = self._group_for_parameter(parameter)
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(
                    f"MuonClip walk capture found no gradient for {name}"
                )
            state = self.optimizer.state[parameter]
            momentum_before = state.get("momentum_buffer")
            if momentum_before is None:
                momentum_before = torch.zeros_like(parameter)
            matrices[name] = {
                "matrix_type": matrix_type,
                "block": int(block),
                "weight_before": _cpu_clone(parameter),
                "gradient_post_clip": _cpu_clone(gradient),
                "momentum_before": _cpu_clone(momentum_before),
                "lr": float(group["lr"]),
                "momentum": float(group["momentum"]),
                "nesterov": bool(group["nesterov"]),
                "weight_decay": float(group["weight_decay"]),
                "newton_schulz_steps": int(group["newton_schulz_steps"]),
                "eps": float(group["eps"]),
                "update_rms_scale": float(group["update_rms_scale"]),
            }

        qk_logits: dict[int, torch.Tensor] = {}
        for block_index, block in enumerate(self.model.blocks):
            value = getattr(block.attn, "_muonclip_max_logits", None)
            if value is not None:
                qk_logits[block_index] = _cpu_clone(value.reshape(-1))

        self.pending = {
            "step": step,
            "coordinates": self._coordinates(step),
            "matrices": matrices,
            "qk_logits": qk_logits,
            "primary_lr": float(handles[0].lr),
            "auxiliary_lr": (
                float(handles[1].lr) if len(handles) > 1 else float("nan")
            ),
        }
        return True

    def abort(self) -> None:
        self.pending = None

    def capture_after(self, handles) -> None:
        pending = self.pending
        if pending is None:
            return
        synchronize(model_device(self.model))
        step = int(pending["step"])
        if int(self.optimizer.step_index) != step:
            raise RuntimeError(
                "MuonClip step index changed unexpectedly during walk capture"
            )

        matrix_payload: dict[str, dict[str, Any]] = {}
        matrix_rows: list[dict[str, Any]] = []
        prediction_residuals: list[float] = []
        delta_norms: list[float] = []
        qk_max_values: list[float] = []
        qk_gamma_values: list[float] = []

        current_by_name = {
            name: parameter
            for name, _, _, parameter in transformer_matrix_items(self.model)
        }

        for name, before in pending["matrices"].items():
            parameter = current_by_name[name]
            weight_before = before["weight_before"]
            weight_after = _cpu_clone(parameter)
            gradient = before["gradient_post_clip"]
            state = self.optimizer.state[parameter]
            momentum_after_tensor = state.get("momentum_buffer")
            if momentum_after_tensor is None:
                raise RuntimeError(
                    f"MuonClip momentum state missing after step for {name}"
                )
            momentum_after = _cpu_clone(momentum_after_tensor)
            momentum_value = float(before["momentum"])
            update_source = (
                gradient + momentum_after * momentum_value
                if bool(before["nesterov"])
                else momentum_after
            )
            orthogonal_update = zeropower_via_newton_schulz_5(
                update_source,
                steps=int(before["newton_schulz_steps"]),
                eps=float(before["eps"]),
            )
            orthogonal_update = orthogonal_update * (
                float(before["update_rms_scale"])
                * math.sqrt(max(weight_before.shape))
            )
            decay_factor = max(
                0.0,
                1.0
                - float(before["lr"]) * float(before["weight_decay"]),
            )
            predicted_pre_qk = (
                weight_before * decay_factor
                - float(before["lr"]) * orthogonal_update
            )

            qk_logits = pending["qk_logits"].get(int(before["block"]))
            if qk_logits is None or qk_logits.numel() == 0:
                max_logit = float("nan")
                gamma = 1.0
            else:
                max_logit = float(qk_logits.max())
                threshold = float(self.optimizer.qk_clip_threshold)
                gamma = (
                    min(1.0, threshold / max(max_logit, 1e-30))
                    if max_logit > threshold
                    else 1.0
                )
            balance = float(self.optimizer.qk_clip_balance)
            q_scale = gamma**balance
            k_scale = gamma ** (1.0 - balance)
            applied_qk_scale = (
                q_scale
                if before["matrix_type"] == "W_Q"
                else k_scale
                if before["matrix_type"] == "W_K"
                else 1.0
            )
            predicted_final = predicted_pre_qk * applied_qk_scale
            parameter_delta = weight_after - weight_before
            prediction_residual = weight_after - predicted_final

            spectra_before = _spectral_payload(weight_before)
            spectra_after = _spectral_payload(weight_after)
            spectra_delta = _spectral_payload(parameter_delta)

            if math.isfinite(max_logit):
                qk_max_values.append(max_logit)
                qk_gamma_values.append(gamma)

            delta_norm = _tensor_norm(parameter_delta)
            residual_norm = _tensor_norm(prediction_residual)
            delta_norms.append(delta_norm)
            prediction_residuals.append(residual_norm)

            payload = {
                **before,
                "momentum_after": momentum_after,
                "update_source": update_source,
                "rms_matched_orthogonal_update": orthogonal_update,
                "predicted_weight_before_qk_clip": predicted_pre_qk,
                "qk_max_logit": max_logit,
                "qk_gamma": gamma,
                "q_scale": q_scale,
                "k_scale": k_scale,
                "applied_qk_scale": applied_qk_scale,
                "predicted_weight_after_qk_clip": predicted_final,
                "weight_after": weight_after,
                "parameter_delta": parameter_delta,
                "prediction_residual": prediction_residual,
                "spectra_before": spectra_before,
                "spectra_after": spectra_after,
                "spectra_delta": spectra_delta,
            }
            if not self.save_optimizer_tensors:
                for key in (
                    "gradient_post_clip",
                    "momentum_before",
                    "momentum_after",
                    "update_source",
                    "rms_matched_orthogonal_update",
                    "predicted_weight_before_qk_clip",
                    "predicted_weight_after_qk_clip",
                    "prediction_residual",
                ):
                    payload.pop(key, None)
            matrix_payload[name] = payload

            matrix_rows.append(
                {
                    **pending["coordinates"],
                    "matrix_name": name,
                    "matrix_type": before["matrix_type"],
                    "block": int(before["block"]),
                    "rows": int(weight_before.shape[0]),
                    "columns": int(weight_before.shape[1]),
                    "primary_lr": float(pending["primary_lr"]),
                    "auxiliary_lr": float(pending["auxiliary_lr"]),
                    "weight_decay": float(before["weight_decay"]),
                    "momentum": float(before["momentum"]),
                    "update_rms_scale": float(before["update_rms_scale"]),
                    "weight_norm_before": _tensor_norm(weight_before),
                    "weight_norm_after": _tensor_norm(weight_after),
                    "weight_rms_before": _tensor_rms(weight_before),
                    "weight_rms_after": _tensor_rms(weight_after),
                    "gradient_norm_post_clip": _tensor_norm(gradient),
                    "gradient_rms_post_clip": _tensor_rms(gradient),
                    "momentum_norm_before": _tensor_norm(
                        before["momentum_before"]
                    ),
                    "momentum_norm_after": _tensor_norm(momentum_after),
                    "orthogonal_update_norm": _tensor_norm(
                        orthogonal_update
                    ),
                    "orthogonal_update_rms": _tensor_rms(
                        orthogonal_update
                    ),
                    "delta_norm": delta_norm,
                    "delta_rms": _tensor_rms(parameter_delta),
                    "update_to_weight_ratio": delta_norm
                    / max(_tensor_norm(weight_before), 1e-30),
                    "cos_delta_descent_gradient": _cosine(
                        parameter_delta,
                        -gradient,
                    ),
                    "cos_delta_descent_momentum": _cosine(
                        parameter_delta,
                        -momentum_after,
                    ),
                    "cos_delta_descent_orthogonal_update": _cosine(
                        parameter_delta,
                        -orthogonal_update,
                    ),
                    "prediction_residual_norm": residual_norm,
                    "qk_max_logit": max_logit,
                    "qk_gamma": gamma,
                    "q_scale": q_scale,
                    "k_scale": k_scale,
                    "applied_qk_scale": applied_qk_scale,
                    "spectral_norm_before": spectra_before["spectral_norm"],
                    "spectral_norm_after": spectra_after["spectral_norm"],
                    "stable_rank_before": spectra_before["stable_rank"],
                    "stable_rank_after": spectra_after["stable_rank"],
                    "effective_rank_before": spectra_before["effective_rank"],
                    "effective_rank_after": spectra_after["effective_rank"],
                    "sv_mean_before": spectra_before["sv_mean"],
                    "sv_mean_after": spectra_after["sv_mean"],
                    "sv_std_before": spectra_before["sv_std"],
                    "sv_std_after": spectra_after["sv_std"],
                }
            )

        step_path = self.step_dir / f"step_{step:07d}.pt"
        if step_path.exists():
            raise FileExistsError(
                f"refusing to overwrite captured MuonClip step: {step_path}"
            )
        _atomic_torch_save(
            {
                "schema_version": 1,
                "purpose": "muonclip_optimizer_step_trace",
                **pending["coordinates"],
                "optimizer_name": "muon_clip",
                "run_dir": str(self.run_dir),
                "capture_dir": str(self.capture_dir),
                "primary_lr": float(pending["primary_lr"]),
                "auxiliary_lr": float(pending["auxiliary_lr"]),
                "gradient_semantics": "post_global_grad_clip",
                "qk_logits": pending["qk_logits"],
                "matrices": matrix_payload,
            },
            step_path,
        )
        state_paths = self._save_state(
            step,
            stage="after_optimizer_step",
        )

        step_rows = [
            {
                **pending["coordinates"],
                "primary_lr": float(pending["primary_lr"]),
                "auxiliary_lr": float(pending["auxiliary_lr"]),
                "mean_matrix_delta_norm": (
                    sum(delta_norms) / max(len(delta_norms), 1)
                ),
                "max_matrix_delta_norm": max(delta_norms, default=0.0),
                "max_prediction_residual_norm": max(
                    prediction_residuals,
                    default=0.0,
                ),
                "qk_max_logit": max(qk_max_values, default=float("nan")),
                "qk_min_gamma": min(qk_gamma_values, default=1.0),
                "step_trace": str(step_path),
                **state_paths,
            }
        ]
        _append_rows(
            self.capture_dir / "matrix_trajectory.csv",
            matrix_rows,
            keys=["step", "matrix_name"],
        )
        _append_rows(
            self.capture_dir / "step_trajectory.csv",
            step_rows,
            keys=["step"],
        )
        self.captured_steps.add(step)
        self.pending = None
        self._write_capture_manifest()


def _validate_walk_profile(profile: dict[str, Any]) -> None:
    steps = int(profile.get("walk_capture_steps", 0))
    if not 0 <= steps <= _MAX_CAPTURE_STEPS:
        raise ValueError(
            "walk_capture_steps must be between 0 and "
            f"{_MAX_CAPTURE_STEPS}"
        )
    if steps == 0:
        return
    root = Path(
        os.environ.get(
            _CAPTURE_ROOT_ENV,
            str(profile.get("walk_capture_root", _DEFAULT_CAPTURE_ROOT)),
        )
    ).expanduser()
    if not root.is_absolute():
        raise ValueError("walk_capture_root must be an absolute path")
    for key in (
        "walk_save_full_model",
        "walk_save_weightwatcher",
        "walk_save_optimizer_tensors",
    ):
        if key in profile and not isinstance(profile[key], bool):
            raise ValueError(f"{key} must be boolean")


def install_muonclip_walk_extension() -> None:
    """Install MuonClip plus the opt-in first-20-step capture wrapper."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import config as config_module
    from . import engine as engine_module
    from . import muonclip as muonclip_module
    from . import optimizers as optimizers_module
    from . import training as training_module
    from . import train_loop as train_loop_module

    muonclip_module.install_muonclip_extension()

    original_validate = config_module.validate_optimizer_profile
    original_make_handles = optimizers_module.make_optimizer_handles
    original_optimizer_step = optimizers_module.optimizer_step
    original_worker_module = training_module._mps_worker_module

    def validate_optimizer_profile(profile: dict[str, Any]) -> None:
        original_validate(profile)
        if str(profile.get("family", "")) == "muon_clip":
            _validate_walk_profile(profile)

    def make_optimizer_handles(model, profile: dict[str, Any]):
        handles = original_make_handles(model, profile)
        steps = int(profile.get("walk_capture_steps", 0))
        if str(profile.get("family", "")) != "muon_clip" or steps <= 0:
            return handles
        run_dir = muonclip_module._CURRENT_RUN_DIR
        if run_dir is None:
            raise RuntimeError(
                "MuonClip walk capture could not resolve the active run "
                "directory; use rg-onehead-muonclip-walk"
            )
        primary = handles[0].optimizer
        primary._muonclip_walk_recorder = MuonClipWalkRecorder(
            model=model,
            optimizer=primary,
            profile=profile,
            run_dir=Path(run_dir),
        )
        return handles

    def optimizer_step(handles) -> None:
        recorders = [
            getattr(handle.optimizer, "_muonclip_walk_recorder", None)
            for handle in handles
        ]
        recorders = [item for item in recorders if item is not None]
        active = [
            recorder
            for recorder in recorders
            if recorder.capture_before(handles)
        ]
        try:
            original_optimizer_step(handles)
        except Exception:
            for recorder in active:
                recorder.abort()
            raise
        for recorder in active:
            recorder.capture_after(handles)

    def worker_module(optimizer_name: str) -> str:
        if str(optimizer_name) == "muon_clip":
            return "rg_nanogpt_one_head.muonclip_walk"
        return original_worker_module(optimizer_name)

    config_module.validate_optimizer_profile = validate_optimizer_profile
    optimizers_module.make_optimizer_handles = make_optimizer_handles
    engine_module.make_optimizer_handles = make_optimizer_handles
    optimizers_module.optimizer_step = optimizer_step
    train_loop_module.optimizer_step = optimizer_step
    training_module._mps_worker_module = worker_module

    _INSTALLED = True


def main() -> None:
    install_muonclip_walk_extension()
    from .training import main as training_main

    training_main()


if __name__ == "__main__":
    main()

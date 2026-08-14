from __future__ import annotations

"""Opt-in MuonClip capture at initialization, microbatches, and updates.

The ordinary MuonClip launcher is unchanged.  This module extends the existing
optimizer-step walk recorder and is activated only by
``rg-onehead-muonclip-walk``.
"""

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from . import muonclip_walk as legacy

HARD_MAX_CHECKPOINTS = 500
_ACTIVE_RECORDER: "MuonClipCaptureRecorder | None" = None
_ORIGINAL_BACKWARD = None
_INSTALLED = False

SavedWeightMatrixHolder = legacy.SavedWeightMatrixHolder
load_full_model_checkpoint = legacy.load_full_model_checkpoint


def _effective_batches(profile: dict[str, Any]) -> int:
    modern = profile.get("walk_capture_effective_batches")
    old = profile.get("walk_capture_steps")
    if modern is not None and old is not None and int(modern) != int(old):
        raise ValueError(
            "walk_capture_effective_batches and walk_capture_steps must match"
        )
    return int(modern if modern is not None else old or 0)


def expected_weightwatcher_checkpoint_count(
    profile: dict[str, Any], *, grad_accum_steps: int
) -> int:
    batches = _effective_batches(profile)
    if batches <= 0:
        return 0
    count = 1 + batches
    if bool(profile.get("walk_capture_microbatches", False)):
        count += batches * int(grad_accum_steps)
    return count


def _validate_walk_profile(
    profile: dict[str, Any], *, grad_accum_steps: int | None = None
) -> None:
    batches = _effective_batches(profile)
    if batches < 0:
        raise ValueError("walk capture batch count must be nonnegative")
    cap = int(profile.get("walk_max_checkpoints", HARD_MAX_CHECKPOINTS))
    if not 1 <= cap <= HARD_MAX_CHECKPOINTS:
        raise ValueError("walk_max_checkpoints must be between 1 and 500")
    for key in (
        "walk_capture_microbatches",
        "walk_save_full_model",
        "walk_save_weightwatcher",
        "walk_save_optimizer_tensors",
        "walk_save_microbatch_gradients",
    ):
        if key in profile and not isinstance(profile[key], bool):
            raise ValueError(f"{key} must be boolean")
    root = Path(
        os.environ.get(
            legacy._CAPTURE_ROOT_ENV,
            str(profile.get("walk_capture_root", legacy._DEFAULT_CAPTURE_ROOT)),
        )
    ).expanduser()
    if batches and not root.is_absolute():
        raise ValueError("walk_capture_root must be absolute")
    if batches and grad_accum_steps is not None:
        count = expected_weightwatcher_checkpoint_count(
            profile, grad_accum_steps=grad_accum_steps
        )
        if count > cap:
            raise ValueError(
                f"requested capture writes {count} WeightWatcher checkpoints, "
                f"exceeding walk_max_checkpoints={cap}"
            )


def load_weightwatcher_checkpoint(
    path: str | Path, *, source: str = "weights"
) -> tuple[SavedWeightMatrixHolder, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("purpose") != "weightwatcher_matrix_checkpoint":
        raise ValueError(f"not a WeightWatcher checkpoint: {path}")
    key = {
        "weights": "matrices",
        "accumulated_gradients": "accumulated_gradients",
        "gradients": "accumulated_gradients",
    }.get(str(source))
    if key is None:
        raise ValueError("source must be weights or accumulated_gradients")
    matrices = payload.get(key)
    if matrices is None:
        raise ValueError(f"{path} does not contain source={source}")
    return SavedWeightMatrixHolder(
        matrices, list(payload["matrix_metadata"])
    ), payload


def _append_snapshot(path: Path, row: dict[str, Any]) -> None:
    incoming = pd.DataFrame([row])
    if path.is_file() and path.stat().st_size:
        frame = pd.concat([pd.read_csv(path), incoming], ignore_index=True)
    else:
        frame = incoming
    frame = (
        frame.drop_duplicates("timeline_index", keep="last")
        .sort_values("timeline_index")
        .reset_index(drop=True)
    )
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


class MuonClipCaptureRecorder(legacy.MuonClipWalkRecorder):
    """Legacy optimizer-step recorder plus optional microbatch snapshots."""

    def __init__(self, *, model, optimizer, profile, run_dir: Path) -> None:
        global _ACTIVE_RECORDER
        profile = dict(profile)
        batches = _effective_batches(profile)
        profile["walk_capture_steps"] = batches
        self.capture_microbatches = bool(
            profile.get("walk_capture_microbatches", False)
        )
        self.max_checkpoints = int(
            profile.get("walk_max_checkpoints", HARD_MAX_CHECKPOINTS)
        )
        self.save_microbatch_gradients = bool(
            profile.get("walk_save_microbatch_gradients", True)
        )
        self.microbatch_in_batch = 0
        super().__init__(
            model=model, optimizer=optimizer, profile=profile, run_dir=run_dir
        )
        self.max_steps = batches
        self.microbatch_dir = self.capture_dir / "microbatch_traces"
        self.microbatch_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_index = self.capture_dir / "snapshot_index.csv"
        model._muonclip_walk_recorder = self
        _ACTIVE_RECORDER = self

    def _context(self) -> dict[str, int]:
        manifest = self._run_manifest()
        training = manifest.get("training", {})
        model = manifest.get("model", {})
        splits = manifest.get("data_metadata", {}).get("splits", {})
        return {
            "batch_size": int(training.get("batch_size", 1)),
            "grad_accum_steps": int(training.get("grad_accum_steps", 1)),
            "block_size": int(model.get("block_size", self.model.cfg.block_size)),
            "train_tokens": int(splits.get("train", 1)),
        }

    def _count_ww(self) -> int:
        return sum(1 for _ in self.ww_dir.glob("ww_*.pt"))

    def _write_capture_manifest(self) -> None:
        super()._write_capture_manifest()
        path = self.capture_dir / "walk_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        ctx = self._context()
        payload.update(
            {
                "schema_version": 3,
                "purpose": "muonclip_microbatch_spectral_capture",
                "capture_microbatches": self.capture_microbatches,
                "hard_max_weightwatcher_checkpoints": HARD_MAX_CHECKPOINTS,
                "max_weightwatcher_checkpoints": self.max_checkpoints,
                "expected_weightwatcher_checkpoints": (
                    expected_weightwatcher_checkpoint_count(
                        self.profile,
                        grad_accum_steps=ctx["grad_accum_steps"],
                    )
                ),
                "written_weightwatcher_checkpoints": self._count_ww(),
            }
        )
        legacy._atomic_json(path, payload)

    def _timeline_row(
        self,
        *,
        kind: str,
        timeline: int,
        effective_batch: int,
        microbatch_index: int,
        global_microbatch: int,
        ww_path: Path,
        trace_path: Path | None = None,
    ) -> dict[str, Any]:
        ctx = self._context()
        tokens_seen = global_microbatch * ctx["batch_size"] * ctx["block_size"]
        step = max(effective_batch - 1, 0) if kind == "microbatch" else effective_batch
        return {
            "timeline_index": int(timeline),
            "snapshot_kind": kind,
            "step": int(step),
            "optimizer_step": int(step),
            "effective_batch": int(effective_batch),
            "microbatch_index": int(microbatch_index),
            "global_microbatch": int(global_microbatch),
            "tokens_seen": int(tokens_seen),
            "epoch": tokens_seen / max(ctx["train_tokens"], 1),
            "weightwatcher_checkpoint": str(ww_path),
            "trace_path": str(trace_path) if trace_path else "",
        }

    def _save_state(self, step: int, *, stage: str):
        _, ww_path = self._state_paths(step)
        if not ww_path.exists() and self._count_ww() >= self.max_checkpoints:
            raise RuntimeError("walk WeightWatcher checkpoint cap reached")
        result = super()._save_state(step, stage=stage)
        ctx = self._context()
        timeline = (
            step * (ctx["grad_accum_steps"] + 1)
            if self.capture_microbatches
            else step
        )
        trace = self.step_dir / f"step_{step:07d}.pt"
        _append_snapshot(
            self.snapshot_index,
            self._timeline_row(
                kind="initial" if step == 0 else "optimizer_step",
                timeline=timeline,
                effective_batch=step,
                microbatch_index=0,
                global_microbatch=step * ctx["grad_accum_steps"],
                ww_path=ww_path,
                trace_path=trace if trace.is_file() else None,
            ),
        )
        return result

    def capture_initial_state(self) -> None:
        if self.max_steps <= 0 or int(self.optimizer.step_index) != 0:
            return
        ww_path = self.ww_dir / "ww_step_0000000.pt"
        if ww_path.exists():
            return
        trace = self.step_dir / "step_0000000.pt"
        legacy._atomic_torch_save(
            {
                "schema_version": 3,
                "purpose": "muonclip_initialization_trace",
                "step": 0,
                "matrices": self._current_matrices(),
                "matrix_metadata": self.matrix_metadata,
            },
            trace,
        )
        self._save_state(0, stage="initialized_before_training")
        self._write_capture_manifest()
        print(
            f"[one-head-walk] saved initialization ww={ww_path} trace={trace}",
            flush=True,
        )

    def begin_effective_batch(self) -> None:
        self.microbatch_in_batch = 0

    def capture_after_backward(self, *, scaled_loss: torch.Tensor) -> None:
        if not self.capture_microbatches:
            return
        effective_batch = int(self.optimizer.step_index) + 1
        if not 1 <= effective_batch <= self.max_steps:
            return
        self.microbatch_in_batch += 1
        ctx = self._context()
        if self.microbatch_in_batch > ctx["grad_accum_steps"]:
            raise RuntimeError("too many backward calls in one effective batch")
        global_mb = (
            (effective_batch - 1) * ctx["grad_accum_steps"]
            + self.microbatch_in_batch
        )
        ww_path = self.ww_dir / f"ww_microbatch_{global_mb:07d}.pt"
        if ww_path.exists():
            return
        if self._count_ww() >= self.max_checkpoints:
            raise RuntimeError("walk WeightWatcher checkpoint cap reached")
        legacy.synchronize(legacy.model_device(self.model))
        matrices = self._current_matrices()
        gradients = {
            name: (
                torch.zeros(tuple(parameter.shape), dtype=torch.float32)
                if parameter.grad is None
                else legacy._cpu_clone(parameter.grad)
            )
            for name, _, _, parameter in legacy.transformer_matrix_items(self.model)
        }
        timeline = (
            (effective_batch - 1) * (ctx["grad_accum_steps"] + 1)
            + self.microbatch_in_batch
        )
        tokens_seen = global_mb * ctx["batch_size"] * ctx["block_size"]
        metadata = {
            "timeline_index": timeline,
            "snapshot_kind": "microbatch",
            "step": effective_batch - 1,
            "optimizer_step": effective_batch - 1,
            "effective_batch": effective_batch,
            "microbatch_index": self.microbatch_in_batch,
            "global_microbatch": global_mb,
            "tokens_seen": tokens_seen,
            "epoch": tokens_seen / max(ctx["train_tokens"], 1),
        }
        payload = {
            "schema_version": 3,
            "purpose": "weightwatcher_matrix_checkpoint",
            "stage": "after_backward_before_optimizer_step",
            **metadata,
            "matrices": matrices,
            "matrix_metadata": self.matrix_metadata,
            "optimizer_name": "muon_clip",
            "run_dir": str(self.run_dir),
            "capture_dir": str(self.capture_dir),
        }
        if self.save_microbatch_gradients:
            payload["accumulated_gradients"] = gradients
        legacy._atomic_torch_save(payload, ww_path)
        trace = self.microbatch_dir / f"microbatch_{global_mb:07d}.pt"
        legacy._atomic_torch_save(
            {
                "schema_version": 3,
                "purpose": "muonclip_microbatch_trace",
                **metadata,
                "scaled_loss": float(scaled_loss.detach().float().cpu()),
                "weightwatcher_checkpoint": str(ww_path),
            },
            trace,
        )
        rows = []
        for name, matrix_type, block, _ in self.matrix_items:
            rows.append(
                {
                    **metadata,
                    "matrix_name": name,
                    "matrix_type": matrix_type,
                    "block": int(block),
                    "weight_norm": legacy._tensor_norm(matrices[name]),
                    "accumulated_gradient_norm": legacy._tensor_norm(gradients[name]),
                    "accumulated_gradient_rms": legacy._tensor_rms(gradients[name]),
                }
            )
        legacy._append_rows(
            self.capture_dir / "microbatch_matrix_trajectory.csv",
            rows,
            keys=["global_microbatch", "matrix_name"],
        )
        legacy._append_rows(
            self.capture_dir / "microbatch_trajectory.csv",
            [
                {
                    **metadata,
                    "weightwatcher_checkpoint": str(ww_path),
                    "trace_path": str(trace),
                }
            ],
            keys=["global_microbatch"],
        )
        _append_snapshot(
            self.snapshot_index,
            self._timeline_row(
                kind="microbatch",
                timeline=timeline,
                effective_batch=effective_batch,
                microbatch_index=self.microbatch_in_batch,
                global_microbatch=global_mb,
                ww_path=ww_path,
                trace_path=trace,
            ),
        )
        self._write_capture_manifest()
        print(
            "[one-head-walk] saved microbatch "
            f"global={global_mb} batch={effective_batch} "
            f"micro={self.microbatch_in_batch}/{ctx['grad_accum_steps']} "
            f"ww={ww_path}",
            flush=True,
        )


def _validate_config(cfg: dict[str, Any]) -> None:
    profile = cfg.get("optimizer_profiles", {}).get("muon_clip")
    if not isinstance(profile, dict):
        return
    grad_accum = int(cfg["training"]["grad_accum_steps"])
    _validate_walk_profile(profile, grad_accum_steps=grad_accum)
    batches = _effective_batches(profile)
    tokens_per_step = (
        int(cfg["training"]["batch_size"])
        * grad_accum
        * int(cfg["model"]["block_size"])
    )
    total_steps = max(
        1,
        math.ceil(
            float(cfg["training"]["target_epochs"])
            * int(cfg["dataset"]["train_tokens"])
            / tokens_per_step
        ),
    )
    if batches > total_steps:
        raise ValueError(
            f"walk capture batches exceed training horizon: {batches} > {total_steps}"
        )


def install_muonclip_capture_extension() -> None:
    global _INSTALLED, _ORIGINAL_BACKWARD
    if _INSTALLED:
        return

    legacy.MuonClipWalkRecorder = MuonClipCaptureRecorder
    legacy._validate_walk_profile = _validate_walk_profile
    legacy.install_muonclip_walk_extension()

    from . import config as config_module
    from . import engine as engine_module
    from . import optimizers as optimizers_module
    from . import training as training_module
    from . import train_loop as train_loop_module

    original_validate_config = config_module.validate_config
    original_zero_grad = optimizers_module.zero_grad
    original_write_manifest = engine_module.write_manifest
    original_worker_module = training_module._mps_worker_module
    original_run_one = training_module.run_one
    _ORIGINAL_BACKWARD = torch.Tensor.backward

    def validate_config(cfg: dict[str, Any]) -> None:
        original_validate_config(cfg)
        _validate_config(cfg)

    def zero_grad(handles) -> None:
        original_zero_grad(handles)
        recorder = next(
            (
                getattr(h.optimizer, "_muonclip_walk_recorder", None)
                for h in handles
                if getattr(h.optimizer, "_muonclip_walk_recorder", None)
                is not None
            ),
            None,
        )
        if recorder is not None:
            recorder.begin_effective_batch()

    def backward(tensor, *args, **kwargs):
        result = _ORIGINAL_BACKWARD(tensor, *args, **kwargs)
        if _ACTIVE_RECORDER is not None:
            _ACTIVE_RECORDER.capture_after_backward(scaled_loss=tensor)
        return result

    def write_manifest(*args, **kwargs):
        result = original_write_manifest(*args, **kwargs)
        recorder = getattr(
            kwargs.get("model"), "_muonclip_walk_recorder", None
        )
        if recorder is not None:
            recorder.capture_initial_state()
        return result

    def worker_module(optimizer_name: str) -> str:
        if str(optimizer_name) == "muon_clip":
            return "rg_nanogpt_one_head.muonclip_capture"
        return original_worker_module(optimizer_name)

    def run_one(*args, **kwargs):
        global _ACTIVE_RECORDER
        try:
            return original_run_one(*args, **kwargs)
        finally:
            _ACTIVE_RECORDER = None

    config_module.validate_config = validate_config
    optimizers_module.zero_grad = zero_grad
    train_loop_module.zero_grad = zero_grad
    engine_module.write_manifest = write_manifest
    training_module._mps_worker_module = worker_module
    training_module.run_one = run_one
    torch.Tensor.backward = backward
    _INSTALLED = True


def main() -> None:
    install_muonclip_capture_extension()
    from .training import main as training_main

    training_main()


if __name__ == "__main__":
    main()

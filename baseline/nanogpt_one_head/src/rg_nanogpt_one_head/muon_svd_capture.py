from __future__ import annotations

"""Opt-in update-SVD capture for Muon and MuonClip.

The recorder is deliberately external to the optimizer kernels.  It reconstructs
exactly the matrix that each optimizer passes to Newton--Schulz, computes a thin
SVD of that matrix, and saves the corresponding physical and co-moving weight
matrices before and after the optimizer step.

For an update source

    M = U diag(S) Vh

we save the quotient/co-moving representative

    W_q = U.T @ W @ Vh.T

for every captured transformer matrix.  The optimizer equations themselves are
not changed by this module.
"""

import csv
import json
from pathlib import Path
from typing import Any

import torch

from .model import transformer_matrix_items
from .runtime import synchronize


_CAPTURE_DIRNAME = "muon_update_svd"


def _cpu_float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().clone()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_index(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "optimizer",
        "frame_path",
        "matrix_count",
        "svd_interval",
    ]
    rows: list[dict[str, Any]] = []
    if path.is_file() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    rows = [item for item in rows if int(item["step"]) != int(row["step"])]
    rows.append(row)
    rows.sort(key=lambda item: int(item["step"]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _profile_settings(profile: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(profile.get("update_svd_capture", False))
    interval = int(profile.get("update_svd_interval", 1))
    max_steps = int(profile.get("update_svd_max_steps", 0))
    if interval < 1:
        raise ValueError("update_svd_interval must be positive")
    if max_steps < 0:
        raise ValueError("update_svd_max_steps must be nonnegative")
    return {
        "enabled": enabled,
        "interval": interval,
        "max_steps": max_steps,
        "save_weights": bool(profile.get("update_svd_save_weights", True)),
        "save_quotient_weights": bool(
            profile.get("update_svd_save_quotient_weights", True)
        ),
        "save_update_source": bool(
            profile.get("update_svd_save_update_source", False)
        ),
    }


class MuonUpdateSVDCapture:
    """Capture the thin-SVD frame of every selected Muon/MuonClip update."""

    def __init__(
        self,
        *,
        model,
        handles,
        profile: dict[str, Any],
        optimizer_name: str,
        run_dir: Path,
        device: torch.device,
    ) -> None:
        self.model = model
        self.handles = handles
        self.profile = dict(profile)
        self.optimizer_name = str(optimizer_name)
        self.run_dir = Path(run_dir)
        self.device = device
        self.settings = _profile_settings(self.profile)
        self.enabled = bool(self.settings["enabled"])
        self.pending: dict[str, Any] | None = None

        self.primary = next(
            (handle.optimizer for handle in handles if handle.role == "primary"),
            None,
        )
        if self.enabled and self.primary is None:
            raise RuntimeError("update-SVD capture could not find primary optimizer")

        self.capture_dir = self.run_dir / _CAPTURE_DIRNAME
        self.frame_dir = self.capture_dir / "frames"
        self.index_path = self.capture_dir / "frame_index.csv"
        if self.enabled:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            self._write_manifest()

    @classmethod
    def from_training_state(
        cls,
        *,
        cfg: dict[str, Any],
        model,
        handles,
        optimizer_name: str,
        run_dir: Path,
        device: torch.device,
    ) -> "MuonUpdateSVDCapture | None":
        profile = cfg.get("optimizer_profiles", {}).get(str(optimizer_name))
        if not isinstance(profile, dict):
            return None
        family = str(profile.get("family", ""))
        if family not in {"muon", "muon_clip"}:
            return None
        recorder = cls(
            model=model,
            handles=handles,
            profile=profile,
            optimizer_name=optimizer_name,
            run_dir=run_dir,
            device=device,
        )
        return recorder if recorder.enabled else None

    def _write_manifest(self) -> None:
        _atomic_json(
            {
                "schema_version": 1,
                "purpose": "muon_update_svd_frame_capture",
                "optimizer": self.optimizer_name,
                "frame_source": "update_source_before_newton_schulz",
                "svd": "thin torch.linalg.svd(full_matrices=False) in float32 on CPU",
                "quotient_definition": "W_q = U.T @ W @ Vh.T",
                "capture_interval": int(self.settings["interval"]),
                "max_steps": int(self.settings["max_steps"]),
                "max_steps_semantics": "0 means no step limit",
                "save_weights": bool(self.settings["save_weights"]),
                "save_quotient_weights": bool(
                    self.settings["save_quotient_weights"]
                ),
                "save_update_source": bool(self.settings["save_update_source"]),
                "note": (
                    "This is diagnostic instrumentation only. Muon and MuonClip "
                    "continue to use their existing Newton--Schulz updates."
                ),
            },
            self.capture_dir / "manifest.json",
        )

    def _should_capture(self, step: int) -> bool:
        if not self.enabled:
            return False
        max_steps = int(self.settings["max_steps"])
        if max_steps and step > max_steps:
            return False
        if step % int(self.settings["interval"]) != 0:
            return False
        return not (self.frame_dir / f"frame_step_{step:07d}.pt").exists()

    def _group_for_parameter(self, parameter: torch.nn.Parameter) -> dict[str, Any]:
        assert self.primary is not None
        for group in self.primary.param_groups:
            if any(candidate is parameter for candidate in group["params"]):
                return group
        raise RuntimeError("captured matrix is missing from primary optimizer")

    def _reconstruct_update_source(
        self,
        *,
        gradient: torch.Tensor,
        momentum_before: torch.Tensor,
        group: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        momentum = float(group["momentum"])
        nesterov = bool(group["nesterov"])
        family = str(self.profile.get("family", ""))

        if family == "muon":
            # Muon.step: buffer.lerp_(gradient, 1 - momentum)
            momentum_after = (
                momentum_before * momentum + gradient * (1.0 - momentum)
            )
            # Muon.step: gradient.lerp(buffer, momentum)
            update_source = (
                gradient * (1.0 - momentum) + momentum_after * momentum
                if nesterov
                else momentum_after
            )
            return momentum_after, update_source

        if family == "muon_clip":
            # MuonClip.step: buffer.mul_(momentum).add_(gradient)
            momentum_after = momentum_before * momentum + gradient
            # MuonClip.step: gradient.add(buffer, alpha=momentum)
            update_source = (
                gradient + momentum_after * momentum
                if nesterov
                else momentum_after
            )
            return momentum_after, update_source

        raise RuntimeError(f"unsupported update-SVD family: {family}")

    def capture_before(self, *, step: int) -> bool:
        if not self._should_capture(int(step)):
            self.pending = None
            return False

        synchronize(self.device)
        matrices: dict[str, dict[str, Any]] = {}
        for name, matrix_type, block, parameter in transformer_matrix_items(self.model):
            gradient_tensor = parameter.grad
            if gradient_tensor is None:
                raise RuntimeError(
                    f"update-SVD capture found no gradient for {name} at step={step}"
                )
            group = self._group_for_parameter(parameter)
            state = self.primary.state[parameter]
            momentum_tensor = state.get("momentum_buffer")
            if momentum_tensor is None:
                momentum_tensor = torch.zeros_like(parameter)

            weight_before = _cpu_float(parameter)
            gradient = _cpu_float(gradient_tensor)
            momentum_before = _cpu_float(momentum_tensor)
            momentum_after, update_source = self._reconstruct_update_source(
                gradient=gradient,
                momentum_before=momentum_before,
                group=group,
            )

            try:
                u, s, vh = torch.linalg.svd(update_source, full_matrices=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"thin SVD failed for {name} at step={step}"
                ) from exc
            v = vh.transpose(-2, -1)
            quotient_before = u.transpose(-2, -1) @ weight_before @ v

            item: dict[str, Any] = {
                "matrix_type": matrix_type,
                "block": int(block),
                "shape": tuple(weight_before.shape),
                "lr": float(group["lr"]),
                "momentum": float(group["momentum"]),
                "nesterov": bool(group["nesterov"]),
                "U": u,
                "S": s,
                "Vh": vh,
                "momentum_before": momentum_before,
                "momentum_after_reconstructed": momentum_after,
            }
            if bool(self.settings["save_weights"]):
                item["weight_before"] = weight_before
            if bool(self.settings["save_quotient_weights"]):
                item["quotient_weight_before"] = quotient_before
            if bool(self.settings["save_update_source"]):
                item["update_source"] = update_source
            matrices[name] = item

        self.pending = {
            "step": int(step),
            "matrices": matrices,
        }
        return True

    def abort(self) -> None:
        self.pending = None

    def capture_after(self) -> Path | None:
        if self.pending is None:
            return None
        synchronize(self.device)
        step = int(self.pending["step"])
        current = {
            name: parameter
            for name, _, _, parameter in transformer_matrix_items(self.model)
        }

        for name, item in self.pending["matrices"].items():
            weight_after = _cpu_float(current[name])
            v = item["Vh"].transpose(-2, -1)
            quotient_after = item["U"].transpose(-2, -1) @ weight_after @ v
            if bool(self.settings["save_weights"]):
                item["weight_after"] = weight_after
            if bool(self.settings["save_quotient_weights"]):
                item["quotient_weight_after"] = quotient_after

        path = self.frame_dir / f"frame_step_{step:07d}.pt"
        _atomic_torch_save(
            {
                "schema_version": 1,
                "purpose": "muon_update_svd_frame",
                "step": step,
                "optimizer": self.optimizer_name,
                "frame_source": "update_source_before_newton_schulz",
                "quotient_definition": "W_q = U.T @ W @ Vh.T",
                "matrices": self.pending["matrices"],
            },
            path,
        )
        _append_index(
            self.index_path,
            {
                "step": step,
                "optimizer": self.optimizer_name,
                "frame_path": str(path),
                "matrix_count": len(self.pending["matrices"]),
                "svd_interval": int(self.settings["interval"]),
            },
        )
        self.pending = None
        return path


def load_update_svd_frame(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("purpose") != "muon_update_svd_frame":
        raise ValueError(f"not a Muon update-SVD frame: {path}")
    return payload

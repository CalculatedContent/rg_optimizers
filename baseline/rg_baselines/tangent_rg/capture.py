"""Dense one-step tangent captures around preregistered epoch anchors.

Captures are prepared after gradient clipping and immediately before
``optimizer.step``.  They preserve the matrix source presented to Muon's polar
map, the polar result, the optimizer's applied direction, and the exact
parameter delta observed after the step.  This makes later tangent/Jacobian
experiments independent of optimizer reimplementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional

import torch

from ..muon import zeropower_via_newton_schulz_5
from .checkpoints import atomic_torch_save
from .muonclip import matrix_update_components
from .protocol import BurstWindow


CAPTURE_SCHEMA_VERSION = 1
_CAPTURE_PATTERN = re.compile(r"^capture_step_(?P<step>\d+)\.pt$")


@dataclass
class PendingStepCapture:
    """In-memory pre-step tensors awaiting the actual post-step delta."""

    payload: dict[str, Any]
    parameter_refs: dict[str, torch.nn.Parameter]


def format_capture_name(completed_step: int) -> str:
    if completed_step < 1:
        raise ValueError("completed_step must be positive")
    return f"capture_step_{int(completed_step):09d}.pt"


def parse_capture_name(name: str | Path) -> int:
    match = _CAPTURE_PATTERN.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"not a dense-capture name: {Path(name).name!r}")
    return int(match.group("step"))


def _parameter_groups(
    optimizer: torch.optim.Optimizer,
) -> dict[int, Mapping[str, Any]]:
    return {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _cpu_clone(value: Any) -> Any:
    """Recursively detach tensors while preserving optimizer-state structure."""

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


@torch.no_grad()
def prepare_step_capture(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    optimizer_name: str,
    parameter_names: Iterable[str],
    epoch: int,
    batch_index: int,
    completed_step: int,
    burst: BurstWindow,
    learning_rates: Mapping[str, float],
    protocol_fingerprint: str,
    calibration_inputs: Optional[torch.Tensor] = None,
    calibration_targets: Optional[torch.Tensor] = None,
    rng_state_before_forward: Optional[Mapping[str, Any]] = None,
    loss_definition: str = "torch.nn.functional.cross_entropy:reduction=mean",
    grad_clip_norm: Optional[float] = None,
) -> PendingStepCapture:
    """Preview a captured update without mutating model or optimizer state."""

    if not burst.contains(completed_step):
        raise ValueError("completed_step does not belong to the supplied burst")
    named = dict(model.named_parameters())
    selected = tuple(str(name) for name in parameter_names)
    missing = [name for name in selected if name not in named]
    if missing:
        raise ValueError(f"capture parameters not found: {missing}")
    group_by_id = _parameter_groups(optimizer)
    tensors: dict[str, dict[str, Any]] = {}
    references: dict[str, torch.nn.Parameter] = {}

    for name in selected:
        parameter = named[name]
        if parameter.grad is None:
            raise RuntimeError(f"captured parameter has no gradient: {name}")
        gradient = parameter.grad.detach()
        group = group_by_id[id(parameter)]
        kind = str(group.get("kind", optimizer.__class__.__name__.lower()))
        state = optimizer.state[parameter]
        source: Optional[torch.Tensor] = gradient
        polar: Optional[torch.Tensor] = None
        direction: Optional[torch.Tensor] = None
        component_metadata: dict[str, Any] = {}

        if kind == "muon":
            momentum = float(group["momentum"])
            previous = state.get("momentum_buffer")
            if previous is None:
                previous = torch.zeros_like(gradient)
            next_buffer = previous.lerp(gradient, 1.0 - momentum)
            source = (
                gradient.lerp(next_buffer, momentum)
                if bool(group["nesterov"])
                else next_buffer
            )
            polar = zeropower_via_newton_schulz_5(
                source,
                steps=int(group["newton_schulz_steps"]),
                eps=float(group["eps"]),
            )
            shape_scale = math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
            direction = polar * shape_scale
            component_metadata = {"shape_scale": float(shape_scale)}
        elif kind == "muonclip_rms":
            parts = matrix_update_components(
                gradient,
                state.get("momentum_buffer"),
                momentum=float(group["momentum"]),
                nesterov=bool(group["nesterov"]),
                newton_schulz_steps=int(group["newton_schulz_steps"]),
                epsilon=float(group["eps"]),
                rms_scale=float(group["rms_scale"]),
            )
            source = parts["source"]
            polar = parts["polar"]
            direction = parts["direction"]
            component_metadata = {
                "declared_rms_scale": float(group["rms_scale"]),
                "polar_rms": float(parts["polar_rms"]),
                "effective_direction_rms": float(parts["effective_rms"]),
                "qk_clipping_applicable": False,
            }

        tensors[name] = {
            "optimizer_group_kind": kind,
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "weight_before": parameter.detach().cpu().clone(),
            "gradient_after_clipping": gradient.cpu().clone(),
            "update_source": source.detach().cpu().clone(),
            "polar_update": None if polar is None else polar.detach().cpu().clone(),
            "applied_update_direction": (
                None if direction is None else direction.detach().cpu().clone()
            ),
            **component_metadata,
        }
        references[name] = parameter

    payload: dict[str, Any] = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "capture_kind": "one_step_tangent_update",
        "optimizer": str(optimizer_name),
        "epoch_before_step": int(epoch),
        "batch_index": int(batch_index),
        "completed_step": int(completed_step),
        "global_step_before": int(completed_step) - 1,
        "anchor_epoch": int(burst.anchor_epoch),
        "anchor_step": int(burst.anchor_step),
        "learning_rates": {
            str(key): float(value) for key, value in learning_rates.items()
        },
        "protocol_fingerprint": str(protocol_fingerprint),
        "parameters": tensors,
    }
    is_first_in_burst = completed_step == burst.first_completed_step
    if is_first_in_burst:
        if (
            calibration_inputs is None
            or calibration_targets is None
            or rng_state_before_forward is None
        ):
            raise ValueError(
                "the first capture in every burst requires batch and pre-forward RNG state"
            )
        payload["calibration_state"] = {
            "state_phase": (
                "model and optimizer immediately before optimizer.step; batch and "
                "RNG-before-forward permit exact loss/gradient recomputation"
            ),
            "inputs": calibration_inputs.detach().cpu().clone(),
            "targets": calibration_targets.detach().cpu().clone(),
            "model_state_before_step": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "optimizer_state_before_step": _cpu_clone(optimizer.state_dict()),
            "scheduler_learning_rates": {
                str(key): float(value) for key, value in learning_rates.items()
            },
            "loss_definition": str(loss_definition),
            "gradient_clipping": {
                "kind": "torch.nn.utils.clip_grad_norm_",
                "max_norm": None if grad_clip_norm is None else float(grad_clip_norm),
            },
            "rng_state_before_forward": _cpu_clone(dict(rng_state_before_forward)),
        }
    return PendingStepCapture(
        payload=payload,
        parameter_refs=references,
    )


@torch.no_grad()
def finalize_step_capture(
    pending: PendingStepCapture,
    *,
    capture_root: str | Path,
) -> Path:
    """Attach post-step tensors and atomically persist a prepared capture."""

    for name, parameter in pending.parameter_refs.items():
        values = pending.payload["parameters"][name]
        after = parameter.detach().cpu().clone()
        before = values["weight_before"]
        values["weight_after"] = after
        values["applied_update"] = after - before
    anchor = int(pending.payload["anchor_epoch"])
    completed_step = int(pending.payload["completed_step"])
    destination = (
        Path(capture_root)
        / f"burst_epoch_{anchor:05d}"
        / format_capture_name(completed_step)
    )
    return atomic_torch_save(pending.payload, destination)


def list_capture_files(capture_root: str | Path) -> tuple[Path, ...]:
    """List valid dense captures in completed-step order."""

    candidates: list[tuple[int, Path]] = []
    for path in Path(capture_root).glob("burst_epoch_*/capture_step_*.pt"):
        try:
            step = parse_capture_name(path)
        except ValueError:
            continue
        candidates.append((step, path))
    return tuple(path for _, path in sorted(candidates))


def load_step_capture(
    path: str | Path,
    *,
    expected_fingerprint: Optional[str] = None,
) -> dict[str, Any]:
    """Load a capture and validate schema, filename identity, and provenance."""

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"capture payload is not a mapping in {path}")
    if payload.get("capture_kind") != "one_step_tangent_update":
        raise RuntimeError(f"{path} is not a tangent-RG one-step capture")
    if int(payload.get("schema_version", -1)) != CAPTURE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported capture schema in {path}")
    if int(payload.get("completed_step", -1)) != parse_capture_name(source):
        raise RuntimeError("capture payload completed_step disagrees with its filename")
    if expected_fingerprint is not None and str(
        payload.get("protocol_fingerprint")
    ) != str(expected_fingerprint):
        raise RuntimeError("capture protocol fingerprint does not match")
    return payload


def replay_calibrated_step(
    capture: str | Path | Mapping[str, Any],
    config: Any,
    *,
    parameter_perturbations: Optional[Mapping[str, torch.Tensor]] = None,
    device: str | torch.device = "cpu",
    expected_fingerprint: Optional[str] = None,
) -> dict[str, Any]:
    """Replay a first-in-burst local training map in an isolated RNG scope.

    ``parameter_perturbations`` are added at the captured base point before the
    stored batch loss is recomputed.  The function builds private model and
    optimizer instances, so it does not mutate a live training run.  Replaying
    on the original training device is the appropriate exactness check; CPU is
    the portable default for controlled finite-difference experiments.
    """

    from ..model import MLP3
    from .checkpoints import capture_rng_state, restore_rng_state
    from .training import build_training_optimizer

    payload = (
        load_step_capture(capture, expected_fingerprint=expected_fingerprint)
        if isinstance(capture, (str, Path))
        else dict(capture)
    )
    if expected_fingerprint is not None and str(
        payload.get("protocol_fingerprint")
    ) != str(expected_fingerprint):
        raise ValueError("capture protocol fingerprint does not match")
    calibration = payload.get("calibration_state")
    if not isinstance(calibration, Mapping):
        raise ValueError("only the first capture in a burst contains calibration_state")
    if str(payload.get("optimizer")) != str(config.optimizer):
        raise ValueError("capture optimizer does not match the supplied configuration")

    caller_rng = capture_rng_state()
    selected_device = torch.device(device)
    model = MLP3()
    model.load_state_dict(calibration["model_state_before_step"])
    model.to(selected_device)
    model.train()
    optimizer = build_training_optimizer(model, config)
    optimizer.load_state_dict(calibration["optimizer_state_before_step"])

    perturbations = dict(parameter_perturbations or {})
    named = dict(model.named_parameters())
    unknown = set(perturbations) - set(named)
    if unknown:
        raise ValueError(f"perturbation parameters not found: {sorted(unknown)}")
    with torch.no_grad():
        for name, perturbation in perturbations.items():
            value = perturbation.detach().to(device=selected_device, dtype=named[name].dtype)
            if value.shape != named[name].shape:
                raise ValueError(
                    f"perturbation shape for {name} is {tuple(value.shape)}, "
                    f"expected {tuple(named[name].shape)}"
                )
            named[name].add_(value)
    state_before = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }

    try:
        restore_rng_state(calibration["rng_state_before_forward"])
        inputs = calibration["inputs"].to(selected_device)
        targets = calibration["targets"].to(selected_device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(calibration["gradient_clipping"]["max_norm"]),
        )
        optimizer.step()
    finally:
        restore_rng_state(caller_rng)

    state_after = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    deltas = {
        name: state_after[name] - state_before[name]
        for name in state_before
        if torch.is_floating_point(state_before[name])
    }
    reference_error: dict[str, float] = {}
    if not perturbations:
        for name, values in payload.get("parameters", {}).items():
            if name in deltas and values.get("applied_update") is not None:
                reference_error[name] = float(
                    (deltas[name] - values["applied_update"]).abs().max()
                )
    return {
        "loss": float(loss.detach().cpu()),
        "gradient_norm_before_clip": float(torch.as_tensor(gradient_norm).detach().cpu()),
        "model_state_before_step": state_before,
        "model_state_after_step": state_after,
        "parameter_deltas": deltas,
        "optimizer_state_after_step": _cpu_clone(optimizer.state_dict()),
        "reference_max_abs_error": reference_error,
        "device": str(selected_device),
        "completed_step": int(payload["completed_step"]),
    }


__all__ = [
    "CAPTURE_SCHEMA_VERSION",
    "PendingStepCapture",
    "finalize_step_capture",
    "format_capture_name",
    "list_capture_files",
    "load_step_capture",
    "parse_capture_name",
    "prepare_step_capture",
    "replay_calibrated_step",
]

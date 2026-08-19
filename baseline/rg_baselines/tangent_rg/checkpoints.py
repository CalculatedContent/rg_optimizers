"""Restartable and analysis-only checkpoint contracts.

Only ``checkpoint_latest.pt`` is rewritten every cadence.  ``checkpoint_best``
and ``checkpoint_final`` retain restart state, while the numbered analysis
checkpoints contain model weights and immutable provenance only.  The parser
accepts arbitrary digit counts, so epochs 1,000 and 10,000 are first-class.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping

import numpy as np
import torch

from .protocol import (
    TailCheckpointLayout,
    atomic_json,
    tail_checkpoint_epochs,
)


CHECKPOINT_SCHEMA_VERSION = 1
TAIL_CHECKPOINT_CACHE_SCHEMA_VERSION = 1
LATEST_CHECKPOINT_NAME = "checkpoint_latest.pt"
BEST_CHECKPOINT_NAME = "checkpoint_best.pt"
FINAL_CHECKPOINT_NAME = "checkpoint_final.pt"
INITIAL_CHECKPOINT_NAME = "analysis_epoch_00000_step_000000000.pt"
_ANALYSIS_PATTERN = re.compile(
    r"^analysis_epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.pt$"
)


@dataclass(frozen=True, order=True)
class AnalysisCheckpointRef:
    """Parsed identity of a sparse model-only checkpoint."""

    epoch: int
    global_step: int
    path: Path


def capture_rng_state() -> dict[str, Any]:
    """Capture every training RNG stream supported by the current machine."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if (
        "mps" in state
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(state["mps"])


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a Torch payload through a sibling temporary file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def format_analysis_checkpoint_name(epoch: int, global_step: int) -> str:
    """Format a sortable name with minimum, not maximum, field widths."""

    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step must be non-negative")
    return f"analysis_epoch_{int(epoch):05d}_step_{int(global_step):09d}.pt"


def parse_analysis_checkpoint_name(name: str | Path) -> tuple[int, int]:
    """Parse numbered analysis checkpoints with any positive digit width."""

    match = _ANALYSIS_PATTERN.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"not an analysis checkpoint name: {Path(name).name!r}")
    return int(match.group("epoch")), int(match.group("step"))


def list_analysis_checkpoints(directory: str | Path) -> tuple[AnalysisCheckpointRef, ...]:
    """Return all valid model-only checkpoints in epoch/step order."""

    refs: list[AnalysisCheckpointRef] = []
    for path in Path(directory).glob("analysis_epoch_*_step_*.pt"):
        try:
            epoch, step = parse_analysis_checkpoint_name(path)
        except ValueError:
            continue
        refs.append(AnalysisCheckpointRef(epoch, step, path))
    return tuple(sorted(refs))


def full_checkpoint_payload(
    *,
    config: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    best_validation_epoch: int,
    train_generator: torch.Generator,
    protocol_fingerprint: str,
    checkpoint_role: str,
) -> dict[str, Any]:
    """Build an exact-resume checkpoint at an epoch boundary."""

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": "full_restart",
        "checkpoint_role": str(checkpoint_role),
        "config": dict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_epoch": int(best_validation_epoch),
        "train_generator_state": train_generator.get_state(),
        "rng_state": capture_rng_state(),
        "protocol_fingerprint": str(protocol_fingerprint),
    }


def save_full_checkpoint(path: str | Path, **payload_arguments: Any) -> Path:
    """Create and atomically save a full restart checkpoint."""

    return atomic_torch_save(full_checkpoint_payload(**payload_arguments), path)


def inspect_full_checkpoint(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
    expected_role: str | None = None,
    expected_epoch: int | None = None,
    expected_global_step: int | None = None,
) -> dict[str, Any]:
    """Validate a full checkpoint without restoring live training state."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"full checkpoint payload is not a mapping in {path}")
    if int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported full-checkpoint schema in {path}")
    if payload.get("checkpoint_kind") != "full_restart":
        raise RuntimeError(f"{path} is not a full restart checkpoint")
    if expected_fingerprint is not None and str(
        payload.get("protocol_fingerprint")
    ) != str(expected_fingerprint):
        raise RuntimeError("checkpoint protocol fingerprint does not match this run")
    try:
        epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best_epoch = int(payload["best_validation_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"checkpoint identity metadata is invalid in {path}") from error
    role = str(payload.get("checkpoint_role", "unknown"))
    if expected_role is not None and role != str(expected_role):
        raise RuntimeError(f"checkpoint role {role!r} does not match {expected_role!r}")
    if expected_epoch is not None and epoch != int(expected_epoch):
        raise RuntimeError(
            f"checkpoint epoch {epoch} does not match expected {int(expected_epoch)}"
        )
    if expected_global_step is not None and global_step != int(expected_global_step):
        raise RuntimeError(
            "checkpoint global_step "
            f"{global_step} does not match expected {int(expected_global_step)}"
        )
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_epoch": best_epoch,
        "checkpoint_role": role,
        "protocol_fingerprint": str(payload.get("protocol_fingerprint", "")),
    }


def load_full_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    expected_fingerprint: str,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Load and restore an exact-resume checkpoint, rejecting protocol drift."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"full checkpoint payload is not a mapping in {path}")
    if int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported full-checkpoint schema in {path}")
    if payload.get("checkpoint_kind") != "full_restart":
        raise RuntimeError(f"{path} is not a full restart checkpoint")
    if str(payload.get("protocol_fingerprint")) != str(expected_fingerprint):
        raise RuntimeError("checkpoint protocol fingerprint does not match this run")
    if expected_role is not None and str(payload.get("checkpoint_role")) != str(
        expected_role
    ):
        raise RuntimeError(
            f"checkpoint role {payload.get('checkpoint_role')!r} "
            f"does not match {expected_role!r}"
        )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    train_generator.set_state(payload["train_generator_state"])
    restore_rng_state(payload["rng_state"])
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "best_validation_loss": float(payload["best_validation_loss"]),
        "best_validation_epoch": int(payload["best_validation_epoch"]),
        "checkpoint_role": str(payload.get("checkpoint_role", "unknown")),
    }


def load_analysis_checkpoint(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Load a model-only checkpoint and validate its filename and provenance."""

    source = Path(path)
    epoch_from_name, step_from_name = parse_analysis_checkpoint_name(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"analysis checkpoint payload is not a mapping in {path}")
    if int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported analysis-checkpoint schema in {path}")
    if payload.get("checkpoint_kind") != "analysis_model_only":
        raise RuntimeError(f"{path} is not an analysis model-only checkpoint")
    if (
        int(payload.get("epoch", -1)) != epoch_from_name
        or int(payload.get("global_step", -1)) != step_from_name
    ):
        raise RuntimeError("analysis checkpoint payload disagrees with its filename")
    if expected_fingerprint is not None and str(
        payload.get("protocol_fingerprint")
    ) != str(expected_fingerprint):
        raise RuntimeError("analysis checkpoint protocol fingerprint does not match")
    if not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("analysis checkpoint has no model state mapping")
    return payload


def save_analysis_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    global_step: int,
    protocol_fingerprint: str,
    optimizer_name: str,
    seed: int,
) -> Path:
    """Save a sparse, immutable, model-only checkpoint for offline analysis."""

    destination = Path(directory) / format_analysis_checkpoint_name(epoch, global_step)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    if destination.exists():
        existing = load_analysis_checkpoint(
            destination,
            expected_fingerprint=protocol_fingerprint,
        )
        if (
            str(existing.get("optimizer")) != str(optimizer_name)
            or int(existing.get("seed", -1)) != int(seed)
        ):
            raise RuntimeError(
                "existing analysis checkpoint optimizer/seed provenance disagrees"
            )
        existing_state = existing["model"]
        if set(existing_state) != set(state) or any(
            not torch.is_tensor(existing_state[name])
            or not torch.equal(existing_state[name], tensor)
            for name, tensor in state.items()
        ):
            raise RuntimeError(
                "existing immutable analysis checkpoint model state disagrees"
            )
        return destination
    return atomic_torch_save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": "analysis_model_only",
            "model": state,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "optimizer": str(optimizer_name),
            "seed": int(seed),
            "protocol_fingerprint": str(protocol_fingerprint),
        },
        destination,
    )


def _tail_cache_manifest_contract(
    *,
    suite_name: str,
    optimizer_name: str,
    seed: int,
    total_epochs: int,
    steps_per_epoch: int,
    protocol_fingerprint: str,
) -> dict[str, Any]:
    expected_epochs = tail_checkpoint_epochs(total_epochs)
    if int(steps_per_epoch) < 1:
        raise ValueError("steps_per_epoch must be positive")
    return {
        "schema_version": TAIL_CHECKPOINT_CACHE_SCHEMA_VERSION,
        "cache_kind": "final_trained_epoch_model_checkpoints",
        "checkpoint_payload_kind": "analysis_model_only",
        "epoch_policy": "trained_epochs_max_1_total_minus_99_through_total",
        "initialization_epoch_included": False,
        "suite_name": str(suite_name),
        "optimizer": str(optimizer_name),
        "seed": int(seed),
        "protocol_fingerprint": str(protocol_fingerprint),
        "total_epochs": int(total_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "expected_epochs": list(expected_epochs),
        "expected_checkpoint_count": len(expected_epochs),
        "checkpoint_directory": "checkpoints",
    }


def ensure_tail_checkpoint_cache(
    layout: TailCheckpointLayout,
    *,
    suite_name: str,
    optimizer_name: str,
    seed: int,
    total_epochs: int,
    steps_per_epoch: int,
    protocol_fingerprint: str,
) -> dict[str, Any]:
    """Create or strictly validate the immutable tail-cache manifest."""

    contract = _tail_cache_manifest_contract(
        suite_name=suite_name,
        optimizer_name=optimizer_name,
        seed=seed,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        protocol_fingerprint=protocol_fingerprint,
    )
    layout.create()
    if layout.manifest.is_file():
        try:
            observed = json.loads(layout.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("tail checkpoint cache manifest is unreadable") from error
        if observed != contract:
            raise RuntimeError(
                "tail checkpoint cache manifest disagrees with this run protocol"
            )
    else:
        atomic_json(contract, layout.manifest)
    return contract


def save_tail_checkpoint(
    layout: TailCheckpointLayout,
    *,
    model: torch.nn.Module,
    epoch: int,
    global_step: int,
    total_epochs: int,
    protocol_fingerprint: str,
    optimizer_name: str,
    seed: int,
) -> Path | None:
    """Idempotently save one model-only snapshot when it is in the tail window."""

    if int(epoch) not in set(tail_checkpoint_epochs(total_epochs)):
        return None
    manifest = _load_tail_cache_manifest(
        layout,
        expected_fingerprint=protocol_fingerprint,
    )
    if (
        int(manifest["total_epochs"]) != int(total_epochs)
        or str(manifest["optimizer"]) != str(optimizer_name)
        or int(manifest["seed"]) != int(seed)
    ):
        raise RuntimeError("tail checkpoint save identity disagrees with manifest")
    expected_step = int(epoch) * int(manifest["steps_per_epoch"])
    if int(global_step) != expected_step:
        raise RuntimeError(
            f"tail checkpoint step {int(global_step)} does not match {expected_step}"
        )
    return save_analysis_checkpoint(
        layout.checkpoints,
        model=model,
        epoch=int(epoch),
        global_step=int(global_step),
        protocol_fingerprint=protocol_fingerprint,
        optimizer_name=optimizer_name,
        seed=seed,
    )


def _load_tail_cache_manifest(
    layout: TailCheckpointLayout,
    *,
    expected_fingerprint: str | None,
) -> dict[str, Any]:
    if not layout.manifest.is_file():
        raise RuntimeError(f"tail checkpoint manifest is missing: {layout.manifest}")
    try:
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("tail checkpoint cache manifest is unreadable") from error
    if not isinstance(manifest, Mapping):
        raise RuntimeError("tail checkpoint cache manifest must be a mapping")
    try:
        canonical = _tail_cache_manifest_contract(
            suite_name=str(manifest["suite_name"]),
            optimizer_name=str(manifest["optimizer"]),
            seed=int(manifest["seed"]),
            total_epochs=int(manifest["total_epochs"]),
            steps_per_epoch=int(manifest["steps_per_epoch"]),
            protocol_fingerprint=str(manifest["protocol_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("tail checkpoint cache manifest is invalid") from error
    if dict(manifest) != canonical:
        raise RuntimeError("tail checkpoint cache manifest contract is invalid")
    if expected_fingerprint is not None and str(
        manifest["protocol_fingerprint"]
    ) != str(expected_fingerprint):
        raise RuntimeError("tail checkpoint cache protocol fingerprint does not match")
    return dict(manifest)


def _tail_checkpoint_file_record(
    ref: AnalysisCheckpointRef,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    with ref.path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(Path("checkpoints") / ref.path.name),
        "epoch": int(ref.epoch),
        "global_step": int(ref.global_step),
        "size_bytes": int(ref.path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def verify_tail_checkpoint_cache(
    layout: TailCheckpointLayout,
    *,
    expected_fingerprint: str | None = None,
    require_complete: bool = True,
    validate_payloads: bool = True,
) -> tuple[AnalysisCheckpointRef, ...]:
    """Return cache refs after strict manifest, identity, and completeness checks."""

    manifest = _load_tail_cache_manifest(
        layout,
        expected_fingerprint=expected_fingerprint,
    )
    expected_epochs = tuple(int(value) for value in manifest["expected_epochs"])
    steps_per_epoch = int(manifest["steps_per_epoch"])
    expected_pairs = {
        (epoch, epoch * steps_per_epoch) for epoch in expected_epochs
    }
    refs = list_analysis_checkpoints(layout.checkpoints)
    actual_pairs = {(ref.epoch, ref.global_step) for ref in refs}
    all_pt_files = set(layout.checkpoints.glob("*.pt"))
    if {ref.path for ref in refs} != all_pt_files:
        raise RuntimeError("tail checkpoint cache contains an invalid .pt filename")
    if not actual_pairs.issubset(expected_pairs):
        raise RuntimeError("tail checkpoint cache contains an unexpected epoch/step")
    if len(actual_pairs) != len(refs):
        raise RuntimeError("tail checkpoint cache contains duplicate epoch/step identities")
    if require_complete and actual_pairs != expected_pairs:
        missing = sorted(expected_pairs.difference(actual_pairs))
        raise RuntimeError(
            f"tail checkpoint cache is incomplete; missing {len(missing)} checkpoint(s)"
        )
    if validate_payloads:
        for ref in refs:
            payload = load_analysis_checkpoint(
                ref.path,
                expected_fingerprint=str(manifest["protocol_fingerprint"]),
            )
            if (
                str(payload.get("optimizer")) != str(manifest["optimizer"])
                or int(payload.get("seed", -1)) != int(manifest["seed"])
            ):
                raise RuntimeError(
                    "tail checkpoint optimizer/seed provenance disagrees with manifest"
                )
    if require_complete:
        if not layout.completion.is_file():
            raise RuntimeError(
                f"tail checkpoint completion marker is missing: {layout.completion}"
            )
        try:
            completion = json.loads(layout.completion.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "tail checkpoint completion marker is unreadable"
            ) from error
        expected_files = [
            str(Path("checkpoints") / ref.path.name) for ref in refs
        ]
        expected_file_records = [
            _tail_checkpoint_file_record(ref) for ref in refs
        ]
        required_completion = {
            "schema_version": TAIL_CHECKPOINT_CACHE_SCHEMA_VERSION,
            "cache_kind": manifest["cache_kind"],
            "completed": True,
            "suite_name": manifest["suite_name"],
            "optimizer": manifest["optimizer"],
            "seed": manifest["seed"],
            "protocol_fingerprint": manifest["protocol_fingerprint"],
            "checkpoint_count": manifest["expected_checkpoint_count"],
            "first_epoch": expected_epochs[0],
            "last_epoch": expected_epochs[-1],
            "files": expected_files,
            "checkpoint_files": expected_file_records,
        }
        if not isinstance(completion, Mapping) or any(
            completion.get(key) != value
            for key, value in required_completion.items()
        ):
            raise RuntimeError("tail checkpoint completion marker is inconsistent")
        if not isinstance(completion.get("completed_utc"), str):
            raise RuntimeError("tail checkpoint completion timestamp is invalid")
    return refs


def verify_tail_checkpoint_cache_prefix(
    layout: TailCheckpointLayout,
    *,
    through_epoch: int,
    expected_fingerprint: str,
) -> tuple[AnalysisCheckpointRef, ...]:
    """Require the complete historical tail prefix at a resume boundary."""

    manifest = _load_tail_cache_manifest(
        layout,
        expected_fingerprint=expected_fingerprint,
    )
    boundary = int(through_epoch)
    total_epochs = int(manifest["total_epochs"])
    if not 0 <= boundary <= total_epochs:
        raise ValueError("tail checkpoint prefix boundary is outside the horizon")
    refs = verify_tail_checkpoint_cache(
        layout,
        expected_fingerprint=expected_fingerprint,
        require_complete=False,
        validate_payloads=True,
    )
    steps_per_epoch = int(manifest["steps_per_epoch"])
    required_pairs = {
        (epoch, epoch * steps_per_epoch)
        for epoch in (int(value) for value in manifest["expected_epochs"])
        if epoch <= boundary
    }
    observed_pairs = {(ref.epoch, ref.global_step) for ref in refs}
    if observed_pairs != required_pairs:
        missing = sorted(required_pairs.difference(observed_pairs))
        unexpected = sorted(observed_pairs.difference(required_pairs))
        raise RuntimeError(
            "tail checkpoint historical prefix is incomplete or extends beyond "
            f"the resume boundary (missing={len(missing)}, "
            f"unexpected={len(unexpected)})"
        )
    return refs


def load_verified_tail_checkpoint_refs(
    cache_seed_dir: str | Path,
    *,
    expected_suite_name: str,
    expected_optimizer_name: str,
    expected_seed: int,
    expected_fingerprint: str,
    expected_epochs: tuple[int, ...],
    validate_payloads: bool = True,
) -> tuple[AnalysisCheckpointRef, ...]:
    """Strict analysis-side loader for a completed optimizer/seed cache.

    Callers supply the expected run identity rather than trusting identity read
    from the cache itself.  This is the single provenance gate used by the
    Jacobian notebooks before they consume the shared checkpoint trajectory.
    """

    root = Path(cache_seed_dir)
    layout = TailCheckpointLayout(
        root=root,
        checkpoints=root / "checkpoints",
        manifest=root / "manifest.json",
        completion=root / "cache_complete.json",
        quarantine=root / "resume_quarantine",
    )
    manifest = _load_tail_cache_manifest(
        layout,
        expected_fingerprint=expected_fingerprint,
    )
    expected_identity = (
        str(expected_suite_name),
        str(expected_optimizer_name),
        int(expected_seed),
        tuple(int(value) for value in expected_epochs),
    )
    observed_identity = (
        str(manifest["suite_name"]),
        str(manifest["optimizer"]),
        int(manifest["seed"]),
        tuple(int(value) for value in manifest["expected_epochs"]),
    )
    if observed_identity != expected_identity:
        raise RuntimeError(
            "tail checkpoint cache suite/optimizer/seed/epoch identity does not match"
        )
    return verify_tail_checkpoint_cache(
        layout,
        expected_fingerprint=expected_fingerprint,
        require_complete=True,
        validate_payloads=validate_payloads,
    )


def finalize_tail_checkpoint_cache(
    layout: TailCheckpointLayout,
    *,
    expected_fingerprint: str,
) -> tuple[AnalysisCheckpointRef, ...]:
    """Certify that the cache contains exactly its declared final checkpoints."""

    manifest = _load_tail_cache_manifest(
        layout,
        expected_fingerprint=expected_fingerprint,
    )
    refs = verify_tail_checkpoint_cache(
        layout,
        expected_fingerprint=expected_fingerprint,
        require_complete=False,
        validate_payloads=True,
    )
    if len(refs) != int(manifest["expected_checkpoint_count"]):
        raise RuntimeError("cannot finalize an incomplete tail checkpoint cache")
    expected_epochs = tuple(int(value) for value in manifest["expected_epochs"])
    atomic_json(
        {
            "schema_version": TAIL_CHECKPOINT_CACHE_SCHEMA_VERSION,
            "cache_kind": manifest["cache_kind"],
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "suite_name": manifest["suite_name"],
            "optimizer": manifest["optimizer"],
            "seed": manifest["seed"],
            "protocol_fingerprint": manifest["protocol_fingerprint"],
            "checkpoint_count": len(refs),
            "first_epoch": expected_epochs[0],
            "last_epoch": expected_epochs[-1],
            "files": [
                str(Path("checkpoints") / ref.path.name) for ref in refs
            ],
            "checkpoint_files": [
                _tail_checkpoint_file_record(ref) for ref in refs
            ],
        },
        layout.completion,
    )
    return verify_tail_checkpoint_cache(
        layout,
        expected_fingerprint=expected_fingerprint,
        require_complete=True,
        validate_payloads=False,
    )


def quarantine_tail_checkpoint_cache_after_boundary(
    layout: TailCheckpointLayout,
    *,
    epoch: int,
    global_step: int,
    expected_fingerprint: str,
) -> tuple[str, ...]:
    """Move cache artifacts newer than an exact-resume boundary out of view."""

    refs = verify_tail_checkpoint_cache(
        layout,
        expected_fingerprint=expected_fingerprint,
        require_complete=False,
        # Future artifacts are about to be discarded and may be the residue of
        # an interrupted write.  Validate retained historical payloads in the
        # exact-prefix gate after these names have been quarantined.
        validate_payloads=False,
    )
    stale = [
        ref.path
        for ref in refs
        if ref.epoch > int(epoch) or ref.global_step > int(global_step)
    ]
    if layout.completion.is_file():
        stale.append(layout.completion)
    if not stale:
        return ()
    quarantine = (
        layout.quarantine
        / f"boundary_epoch_{int(epoch):05d}_step_{int(global_step):09d}_{time.time_ns()}"
    )
    moved: list[str] = []
    for source in sorted(set(stale)):
        relative = source.relative_to(layout.root)
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        moved.append(str(relative))
    atomic_json(
        {
            "reason": "tail artifact is after the restored exact-resume boundary",
            "resume_epoch": int(epoch),
            "resume_global_step": int(global_step),
            "protocol_fingerprint": str(expected_fingerprint),
            "moved": moved,
        },
        quarantine / "quarantine_manifest.json",
    )
    return tuple(moved)


__all__ = [
    "AnalysisCheckpointRef",
    "BEST_CHECKPOINT_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "TAIL_CHECKPOINT_CACHE_SCHEMA_VERSION",
    "FINAL_CHECKPOINT_NAME",
    "INITIAL_CHECKPOINT_NAME",
    "LATEST_CHECKPOINT_NAME",
    "atomic_torch_save",
    "capture_rng_state",
    "format_analysis_checkpoint_name",
    "full_checkpoint_payload",
    "ensure_tail_checkpoint_cache",
    "finalize_tail_checkpoint_cache",
    "inspect_full_checkpoint",
    "list_analysis_checkpoints",
    "load_verified_tail_checkpoint_refs",
    "load_analysis_checkpoint",
    "load_full_checkpoint",
    "parse_analysis_checkpoint_name",
    "quarantine_tail_checkpoint_cache_after_boundary",
    "restore_rng_state",
    "save_analysis_checkpoint",
    "save_full_checkpoint",
    "save_tail_checkpoint",
    "verify_tail_checkpoint_cache",
    "verify_tail_checkpoint_cache_prefix",
]

"""Auditable, asynchronous CPU WeightWatcher observations of immutable weights.

Importing this module does not import Torch, NumPy, or WeightWatcher. The trainer
only calls ``queue_snapshot`` on an already materialized CPU checkpoint. Run the
CLI in a separate CPU process to analyze snapshots; see METRICS.md for semantics.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
WEIGHTWATCHER_VERSION = "0.7.7"
KS_EPSILON = 0.02
ROLES = ("Q", "K", "V", "O", "MLP_IN", "MLP_OUT")
EMBEDDINGS = {"EMBED_WTE": "transformer.wte.weight", "EMBED_WPE": "transformer.wpe.weight"}
WW_COMMON_OPTIONS = {
    "ERG": True, "randomize": True, "mp_fit": True, "plot": False,
    "savefig": False, "fit": "PL", "pl_package": "ww",
    "max_fingers": 10, "min_evals": 20, "max_evals": 15000,
    # GPT-2's 50,257-row token embedding exceeds WW's default max_N=50,000.
    "max_N": 100000,
    # SciPy CPU SVD; cannot select a visible accelerator inside WeightWatcher.
    "svd_method": "accurate",
}
WW_SOURCE = "https://github.com/CalculatedContent/WeightWatcher/blob/bef487d1ab1ed4b4ab27c1e66dde7e55c0e2c605/weightwatcher/weightwatcher.py"
ERG_DEFINITION = "ERG_gap = detX_num - num_pl_spikes; num_ERG_spikes = detX_num"
REQUIRED_NATIVE_FIELDS = {
    "alpha", "D", "xmin", "xmax", "num_pl_spikes", "status",
    "detX_num", "num_ERG_spikes", "ERG_gap", "rand_num_spikes", "num_traps",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Keep every native column; represent nonfinite numbers as JSON null."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool, type(None))):
        return value
    return str(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_file(path: Path, writer: Callable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    data = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    _atomic_file(Path(path), lambda handle: handle.write(data))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@contextmanager
def _file_lock(path: Path, *, blocking: bool = True):
    """Advisory process lock, automatically released after crashes on Linux."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(handle, mode)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def expected_matrix_names(n_layers: int, include_embeddings: bool = False) -> list[str]:
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 1:
        raise ValueError("n_layers must be a positive integer")
    names = [f"L{block:02d}_W_{role}" for block in range(n_layers) for role in ROLES]
    return names + (list(EMBEDDINGS) if include_embeddings else [])


def _cpu_tensor(value):
    import torch
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.device.type != "cpu":
        raise ValueError("Spectral snapshots must use checkpoint CPU tensors, never live device tensors")
    if value.ndim != 2:
        raise ValueError("Spectral matrix must have exactly two dimensions")
    # Clone removes aliases to the checkpoint; in particular Q/K/V each own only
    # their slice, instead of serializing the packed QKV storage three times.
    return value.detach().to(dtype=torch.float32).contiguous().clone()


def extract_matrices(state: Mapping, n_layers: int = 12, include_embeddings: bool = False) -> dict:
    """Extract six logical matrices per block from row-packed stock GPT-2.

    Input is a CPU state_dict. The output owns detached float32 CPU storage. WTE
    and WPE, when requested, have embedding scope; a tied lm_head is not counted
    twice. Biases and LayerNorm vectors are outside this matrix-only analysis.
    """
    expected_matrix_names(n_layers, include_embeddings)
    matrices = {}
    for block in range(n_layers):
        prefix = f"transformer.h.{block}."
        packed_key = prefix + "attn.c_attn.weight"
        if packed_key not in state:
            raise KeyError(f"Missing packed stock GPT-2 weight: {packed_key}")
        packed = state[packed_key]
        if len(packed.shape) != 2 or packed.shape[0] != 3 * packed.shape[1]:
            raise ValueError(f"Expected row-packed [3*d,d] QKV at {packed_key}; found {tuple(packed.shape)}")
        width = packed.shape[1]
        for index, role in enumerate(("Q", "K", "V")):
            matrices[f"L{block:02d}_W_{role}"] = _cpu_tensor(packed[index * width:(index + 1) * width])
        for suffix, role, shape in (
            ("attn.c_proj.weight", "O", (width, width)),
            ("mlp.c_fc.weight", "MLP_IN", (4 * width, width)),
            ("mlp.c_proj.weight", "MLP_OUT", (width, 4 * width)),
        ):
            key = prefix + suffix
            if key not in state:
                raise KeyError(f"Missing stock GPT-2 weight: {key}")
            if tuple(state[key].shape) != shape:
                raise ValueError(f"Expected {shape} at {key}; found {tuple(state[key].shape)}")
            matrices[f"L{block:02d}_W_{role}"] = _cpu_tensor(state[key])
    if include_embeddings:
        for name, key in EMBEDDINGS.items():
            if key not in state:
                raise KeyError(f"Requested embedding is absent: {key}")
            matrices[name] = _cpu_tensor(state[key])
    return matrices


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _matrix_bytes(value) -> tuple[list[int], bytes]:
    import numpy as np
    if hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise ValueError("Only CPU snapshots can be measured")
        value = value.detach().float().numpy()
    array = np.asarray(value, dtype="<f4", order="C")
    return list(array.shape), array.tobytes(order="C")


def snapshot_fingerprint(payload: Mapping) -> str:
    """Content identity independent of Torch serialization container metadata."""
    digest = hashlib.sha256()
    digest.update(_canonical({k: payload[k] for k in ("run_id", "step", "tokens_seen", "validation", "manifest")}))
    for name in sorted(payload["matrices"]):
        shape, data = _matrix_bytes(payload["matrices"][name])
        digest.update(_canonical({"name": name, "shape": shape, "dtype": "float32"}))
        digest.update(data)
    return digest.hexdigest()


def _step_path(run_dir: Path, category: str, step: int, suffix: str) -> Path:
    return run_dir / "tracking" / category / f"{step:09d}{suffix}"


def _checkpoint_payload(checkpoint: Mapping, run_dir: Path, n_layers: int | None, include_embeddings: bool) -> dict:
    step = checkpoint["step"]
    tokens = checkpoint["tokens_seen"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError("tokens_seen must be a nonnegative integer")
    validation = json_safe(checkpoint["validation"])
    if validation.get("step", step) != step or validation.get("tokens_seen", tokens) != tokens:
        raise ValueError("Validation and spectral weights must describe the same update and token count")
    config = checkpoint.get("config", {})
    if hasattr(config, "as_dict"):
        config = config.as_dict()
    n_layers = n_layers or config.get("n_layer", config.get("n_layers", 12))
    matrices = extract_matrices(checkpoint["model"], n_layers, include_embeddings)
    manifest = json_safe(checkpoint.get("manifest", {}))
    manifest["spectral_tracking"] = {
        "schema_version": SCHEMA_VERSION, "n_layers": n_layers,
        "include_embeddings": include_embeddings,
        "keep_snapshots": bool(config.get("keep_snapshots", False)),
        "expected_matrices": expected_matrix_names(n_layers, include_embeddings),
        "qkv_layout": "packed_rows_Q_then_K_then_V", "dtype": "float32",
        "canonical_scope": "transformer_block_matrices", "embedding_lm_head_tied": True,
    }
    payload = {
        "schema_version": SCHEMA_VERSION, "step": step, "tokens_seen": tokens,
        "run_id": checkpoint.get("run_id", run_dir.name), "validation": validation,
        "matrices": matrices, "manifest": manifest,
    }
    payload["content_sha256"] = snapshot_fingerprint(payload)
    return payload


def queue_snapshot(run_dir, checkpoint: Mapping, *, n_layers: int | None = None,
                   include_embeddings: bool = False, max_pending: int = 4) -> dict:
    """Atomically queue a CPU snapshot; never overwrite or silently drop one.

    A ``queue_full`` receipt asks the caller to pause/retry. Already queued or
    measured identities are idempotent, including after the weight file is pruned.
    Reusing the same run/step for different weights or validation raises an error.
    """
    import torch
    if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
        raise ValueError("max_pending must be positive")
    run_dir = Path(run_dir)
    payload = _checkpoint_payload(checkpoint, run_dir, n_layers, include_embeddings)
    step = payload["step"]
    path = _step_path(run_dir, "snapshots", step, ".pt")
    measurement = _step_path(run_dir, "measurements", step, ".json")
    receipt = _step_path(run_dir, "receipts", step, ".json")
    result = {"path": str(path), "step": step, "content_sha256": payload["content_sha256"]}
    with _file_lock(run_dir / "tracking" / ".queue.lock"):
        if measurement.exists():
            previous = _read_json(measurement)
            if previous.get("content_sha256") != payload["content_sha256"]:
                raise ValueError(f"Immutable measured snapshot collision at step {step}")
            return {**result, "status": "already_measured"}
        if path.exists():
            previous = _read_json(receipt) if receipt.exists() else torch.load(path, map_location="cpu", weights_only=False)
            if previous.get("content_sha256") != payload["content_sha256"]:
                raise ValueError(f"Immutable queued snapshot collision at step {step}")
            return {**result, "status": "already_queued"}
        snapshot_dir = run_dir / "tracking" / "snapshots"
        pending = [p for p in snapshot_dir.glob("*.pt")
                   if not _step_path(run_dir, "measurements", int(p.stem), ".json").exists()]
        if len(pending) >= max_pending:
            return {**result, "status": "queue_full", "pending": len(pending)}
        _atomic_file(path, lambda handle: torch.save(payload, handle))
        write_json(receipt, {**result, "run_id": payload["run_id"], "tokens_seen": payload["tokens_seen"],
                             "queued_at": _now(), "snapshot_sha256": _file_sha256(path), "pruned": False})
    return {**result, "status": "queued"}


@contextmanager
def _diagnostic_rng(seed: int):
    """Restore caller Python/NumPy/CPU-Torch state, even on failed fits.

    No torch.manual_seed(): that function also changes accelerator generators.
    The CLI is a separate process; this guard also makes direct CPU tests safe.
    """
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch = sys.modules.get("torch")
    torch_state = torch.random.get_rng_state() if torch is not None else None
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        if torch is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            torch.random.set_rng_state(generator.get_state())
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)


def _holder_from(matrices):
    import torch
    holder = torch.nn.Module()
    for name, value in matrices.items():
        # Bypass Linear.reset_parameters entirely; no initialization RNG draws,
        # no meta-device dependence, and no additional CPU model allocation.
        layer = torch.nn.Linear.__new__(torch.nn.Linear)
        torch.nn.Module.__init__(layer)
        tensor = value.detach().float().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        layer.in_features, layer.out_features = tensor.shape[1], tensor.shape[0]
        layer.register_parameter("weight", torch.nn.Parameter(tensor, requires_grad=False))
        layer.register_parameter("bias", None)
        holder.add_module(name, layer)
    return holder


def _weightwatcher_analyzer(matrices: Mapping, **options):
    """Use native fits, adding only the exact count inside each fitted interval."""
    import numpy as np
    import weightwatcher as ww

    class ObservedWeightWatcher(ww.WeightWatcher):
        def apply_powerlaw(self, layer, params=None):
            result = super().apply_powerlaw(layer, params)
            tail_count = None
            if getattr(layer, "status", None) == "success":
                xmin, xmax = _number(getattr(layer, "xmin", None)), _number(getattr(layer, "xmax", None))
                if xmin is not None and xmax is not None and 0 < xmin <= xmax:
                    evals = np.asarray(layer.evals)
                    tail_count = int(np.count_nonzero((evals >= xmin) & (evals <= xmax)))
            # Native num_pl_spikes deliberately remains untouched: native
            # ERG_gap uses it, even when clip_xmax excludes top eigenvalues.
            layer.add_column("observer_n_tail", tail_count)
            layer.add_column("observer_n_tail_source", "count_original_eigenvalues_within_fit_xmin_xmax")
            return result

    return ObservedWeightWatcher(model=_holder_from(matrices)).analyze(**options)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _matrix_metadata(name: str) -> dict:
    match = re.fullmatch(r"L(\d+)_W_(Q|K|V|O|MLP_IN|MLP_OUT)", name)
    if match:
        return {"matrix_name": name, "block": int(match.group(1)), "matrix_type": match.group(2), "scope": "transformer"}
    if name in EMBEDDINGS:
        return {"matrix_name": name, "block": None, "matrix_type": name.removeprefix("EMBED_"), "scope": "embedding"}
    return {"matrix_name": name, "block": None, "matrix_type": "UNKNOWN", "scope": "unexpected"}


def _frame_rows(frame, expected_names: list[str]) -> tuple[dict, list[dict]]:
    records = frame.to_dict("records") if hasattr(frame, "to_dict") else list(frame)
    found, unassigned = {}, []
    expected = set(expected_names)
    for source in records:
        native = json_safe(source)
        matches = set()
        for key in ("longname", "name", "matrix_name"):
            label = str(source.get(key, ""))
            if label in expected:
                matches.add(label)
            # Allow a module prefix or a .weight suffix, without substring
            # matching L01 against L010 or accepting two different names.
            parts = label.split(".")
            matches.update(part for part in parts if part in expected)
        if len(matches) != 1:
            unassigned.append({"reason": "unmatched_or_ambiguous_name", "native": native})
            continue
        name = matches.pop()
        if name in found:
            raise ValueError(f"WeightWatcher returned duplicate matrix {name}")
        found[name] = native
    return found, unassigned


def _normalize_fit(native: dict | None, fit: str, *, error: str | None = None) -> dict:
    if native is None:
        return {"status": "error" if error else "not_returned", "reason": error or "WeightWatcher returned no row",
                "alpha": None, "D": None, "n_tail": None, "xmin": None, "xmax": None,
                "n_tail_source": None, "ERG_gap": None, "ERG_status": "unavailable", "native": {}}
    alpha, D = _number(native.get("alpha")), _number(native.get("D"))
    xmin, xmax = _number(native.get("xmin")), _number(native.get("xmax"))
    # The observer count belongs to this exact fit. Native n_tail may be
    # provided by a compatible injected analyzer. Never borrow across fits.
    tail_source = native.get("observer_n_tail_source")
    n_tail = _number(native.get("observer_n_tail"))
    if n_tail is None and "n_tail" in native:
        n_tail, tail_source = _number(native.get("n_tail")), "native_n_tail"
    if n_tail is None and fit == "raw":
        n_tail, tail_source = _number(native.get("num_pl_spikes")), "native_unclipped_num_pl_spikes"
    reasons = []
    if native.get("status") != "success":
        reasons.append(f"native_status={native.get('status', 'missing')}")
    if alpha is None or alpha <= 0:
        reasons.append("invalid_alpha")
    if D is None or not 0 <= D <= 1:
        reasons.append("invalid_KS_D")
    if n_tail is None or n_tail <= 0 or not n_tail.is_integer():
        reasons.append("invalid_or_unavailable_matching_n_tail")
    if xmin is None or xmax is None or not 0 < xmin <= xmax:
        reasons.append("invalid_fit_interval")
    erg = _number(native.get("ERG_gap"))
    return {
        "status": "invalid" if reasons else "success", "reason": "; ".join(reasons) or None,
        "alpha": alpha, "D": D, "n_tail": int(n_tail) if n_tail is not None and n_tail.is_integer() else n_tail,
        "xmin": xmin, "xmax": xmax, "n_tail_source": tail_source,
        "ERG_gap": erg, "ERG_status": "available" if erg is not None else "unavailable",
        "native": native,
    }


def _weighted(values: list[tuple[float, float]]) -> dict:
    if not values:
        return {"mean": None, "valid_count": 0, "effective_n": 0.0, "weight_sum": 0.0}
    total = math.fsum(weight for _, weight in values)
    squared = math.fsum(weight * weight for _, weight in values)
    return {"mean": math.fsum(alpha * weight for alpha, weight in values) / total,
            "valid_count": len(values), "effective_n": total * total / squared, "weight_sum": total}


def summarize(rows: list[dict], identity: dict | None = None) -> dict:
    result = {**(identity or {}), "matrix_count": len(rows), "weight_KS_epsilon": KS_EPSILON}
    for fit, column, status_column in (("raw", "alpha_raw", "raw_fit_status"), ("clip_xmax", "alpha_clip_xmax", "clipped_fit_status")):
        valid = [row for row in rows if row[status_column] == "success"]
        values = [row[column] for row in valid]
        result.update({column + "_valid_count": len(valid), column + "_invalid_count": len(rows) - len(valid),
                       column + "_min": min(values) if values else None,
                       column + "_max": max(values) if values else None,
                       column + "_below_two": sum(value < 2 for value in values)})
        for label, weight in (
            ("", lambda r: 1.0),
            ("confidence_weighted_", lambda r: 1.0 / (r[f"D_{fit}"] + KS_EPSILON)),
            ("tail_weighted_", lambda r: math.sqrt(r[f"n_tail_{fit}"])),
            ("confidence_tail_weighted_", lambda r: math.sqrt(r[f"n_tail_{fit}"]) / (r[f"D_{fit}"] + KS_EPSILON)),
        ):
            aggregate = _weighted([(row[column], weight(row)) for row in valid])
            for key, value in aggregate.items():
                result[column + "_" + label + key] = value
        gaps = [_number(row.get(f"ERG_gap_{fit}")) for row in rows]
        gaps = [gap for gap in gaps if gap is not None]
        result[f"ERG_gap_{fit}_mean"] = math.fsum(gaps) / len(gaps) if gaps else None
        result[f"ERG_gap_{fit}_valid_count"] = len(gaps)
    return result


def _load_snapshot(snapshot) -> tuple[dict, str | None]:
    if isinstance(snapshot, (str, Path)):
        import torch
        path = Path(snapshot)
        # Only trusted snapshots written by this run's queue are accepted.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if path.stem.isdecimal() and payload.get("step") != int(path.stem):
            raise ValueError("Snapshot filename and payload update do not match")
        return payload, _file_sha256(path)
    return dict(snapshot), None


def _validate_payload(payload: Mapping) -> tuple[list[str], int]:
    for key in ("step", "tokens_seen", "run_id", "validation", "matrices", "manifest"):
        if key not in payload:
            raise ValueError(f"Snapshot is missing {key}")
    step, tokens = payload["step"], payload["tokens_seen"]
    if payload["validation"].get("step", step) != step or payload["validation"].get("tokens_seen", tokens) != tokens:
        raise ValueError("Validation/spectrum identity mismatch")
    tracking = payload["manifest"].get("spectral_tracking", {})
    n_layers = tracking.get("n_layers", 12)
    names = expected_matrix_names(n_layers, tracking.get("include_embeddings", False))
    if "expected_matrices" in tracking and tracking["expected_matrices"] != names:
        raise ValueError("Manifest matrix inventory does not match the declared model")
    if "content_sha256" in payload and payload["content_sha256"] != snapshot_fingerprint(payload):
        raise ValueError("Immutable snapshot content digest mismatch")
    return names, n_layers


def measure(snapshot, *, analyzer: Callable | None = None) -> dict:
    """Measure both independent fits and retain all native fields.

    Test analyzers accept ``analyzer(matrices, **WW_options)`` and return a
    DataFrame or list of row dicts. Missing/invalid rows are explicit. A failed
    fit pass yields invalid rows with its error; the other pass remains usable.
    """
    measurement_began = time.monotonic()
    payload, snapshot_digest = _load_snapshot(snapshot)
    names, n_layers = _validate_payload(payload)
    content_digest = payload.get("content_sha256") or snapshot_fingerprint(payload)
    validation = json_safe(payload["validation"])
    seed = int(hashlib.sha256(f"{payload['run_id']}:{payload['step']}".encode()).hexdigest()[:8], 16)
    version = importlib.metadata.version("weightwatcher") if analyzer is None else "injected_test_analyzer"
    if analyzer is None and version != WEIGHTWATCHER_VERSION:
        raise RuntimeError(f"Require weightwatcher=={WEIGHTWATCHER_VERSION}; found {version}")
    if analyzer is None:
        import torch  # ensure CPU RNG is covered before invoking the analyzer
        analyzer = _weightwatcher_analyzer
    identity = {"run_id": payload["run_id"], "step": payload["step"], "tokens_seen": payload["tokens_seen"],
                "snapshot_sha256": snapshot_digest, "content_sha256": content_digest,
                "diagnostic_seed": seed, "weightwatcher_version": version}
    for key in ("val_nll", "val_perplexity", "val_token_error", "val_accuracy", "val_error_count",
                "evaluation_tokens", "full_benchmark_evaluation", "validation_kind", "validation_offset"):
        identity[key] = validation.get(key)
    available = {name: payload["matrices"][name] for name in names if name in payload["matrices"]}
    frames, unassigned, errors, options = {}, {}, {}, {}
    for fit, fix in (("raw", False), ("clip_xmax", "clip_xmax")):
        options[fit] = {**WW_COMMON_OPTIONS, "fix_fingers": fix}
        try:
            # Same seed in each pass makes randomized controls comparable.
            with _diagnostic_rng(seed):
                frame = analyzer(available, **options[fit])
            frames[fit], unassigned[fit] = _frame_rows(frame, names)
        except Exception as exc:
            errors[fit] = f"{type(exc).__name__}: {exc}"
            frames[fit], unassigned[fit] = {}, []
    layers = []
    for name in names:
        row = {**identity, **_matrix_metadata(name),
               "shape": list(payload["matrices"][name].shape) if name in available else None}
        for fit, label in (("raw", "raw"), ("clip_xmax", "clipped")):
            normalized = _normalize_fit(frames[fit].get(name), fit, error=errors.get(fit))
            row[f"{label}_fit_status"] = normalized["status"]
            row[f"{label}_fit_reason"] = normalized["reason"]
            for field in ("alpha", "D", "n_tail", "xmin", "xmax", "n_tail_source", "ERG_gap", "ERG_status"):
                row[f"{field}_{fit}"] = normalized[field]
            row[f"native_{fit}"] = normalized["native"]
            for field in ("num_ERG_spikes", "detX_num", "num_pl_spikes", "num_fingers", "num_traps", "rand_num_spikes"):
                row[f"{field}_{fit}"] = normalized["native"].get(field)
        layers.append(row)
    canonical = [row for row in layers if row["scope"] == "transformer"]
    groups = []
    for role in ROLES:
        groups.append(summarize([r for r in canonical if r["matrix_type"] == role], {**identity, "group_type": "role", "group": role}))
    for block in range(n_layers):
        groups.append(summarize([r for r in canonical if r["block"] == block], {**identity, "group_type": "block", "group": block}))
    embeddings = [row for row in layers if row["scope"] == "embedding"]
    if embeddings:
        groups.append(summarize(embeddings, {**identity, "group_type": "scope", "group": "embeddings"}))
        for row in embeddings:
            groups.append(summarize([row], {**identity, "group_type": "embedding", "group": row["matrix_name"]}))
    expected_canonical = 6 * n_layers
    coverage = {"expected_transformer_matrices": expected_canonical, "expected_embedding_matrices": len(embeddings),
                "expected_total_matrices": len(names), "emitted_rows": len(layers),
                "missing_snapshot_matrices": [name for name in names if name not in available],
                "unexpected_snapshot_matrices": sorted(set(payload["matrices"]) - set(names)),
                "raw_returned_count": len(frames["raw"]), "clipped_returned_count": len(frames["clip_xmax"]),
                "raw_valid_count": sum(r["raw_fit_status"] == "success" for r in canonical),
                "clipped_valid_count": sum(r["clipped_fit_status"] == "success" for r in canonical),
                "raw_ERG_available_count": sum(r["ERG_status_raw"] == "available" for r in canonical),
                "clipped_ERG_available_count": sum(r["ERG_status_clip_xmax"] == "available" for r in canonical),
                "unassigned_rows": unassigned, "pass_errors": errors}
    coverage["all_expected_rows_present"] = len(layers) == len(names)
    coverage["all_transformer_fits_valid"] = coverage["raw_valid_count"] == coverage["clipped_valid_count"] == expected_canonical
    coverage["all_transformer_ERG_available"] = coverage["raw_ERG_available_count"] == coverage["clipped_ERG_available_count"] == expected_canonical
    coverage["status"] = "complete" if (coverage["all_transformer_fits_valid"] and coverage["all_transformer_ERG_available"] and not any((
        coverage["missing_snapshot_matrices"], coverage["unexpected_snapshot_matrices"],
        unassigned["raw"], unassigned["clip_xmax"], errors,
    )) and all(r["raw_fit_status"] == r["clipped_fit_status"] == "success" for r in embeddings)) else "warning"
    measured_seconds = time.monotonic() - measurement_began
    return json_safe({"schema_version": SCHEMA_VERSION, **identity, "validation": validation,
                      "measured_seconds": measured_seconds,
                      "manifest": payload["manifest"], "layers": layers,
                      "summary": summarize(canonical, {**identity, "scope": "transformer", "coverage_status": coverage["status"],
                                                       "measured_seconds": measured_seconds}),
                      "groups": groups, "coverage": coverage, "options": options,
                      "ERG_definition": ERG_DEFINITION, "weightwatcher_source": WW_SOURCE,
                      "measured_at": _now()})


def _csv_write(path: Path, rows: list[dict]) -> None:
    import io
    columns = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    if columns:
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    _atomic_file(path, lambda handle: handle.write(buffer.getvalue().encode()))


def collect(run_dir) -> dict:
    """Rebuild atomic CSV exports and status from committed measurement JSONs."""
    run_dir = Path(run_dir)
    tracking = run_dir / "tracking"
    with _file_lock(tracking / ".collect.lock"):
        results = [_read_json(path) for path in sorted((tracking / "measurements").glob("*.json"))]
        run_ids = {result["run_id"] for result in results}
        config_hashes = {result.get("manifest", {}).get("config_sha256") for result in results}
        config_hashes.discard(None)
        if len(run_ids) > 1 or len(config_hashes) > 1:
            raise ValueError("Refusing to aggregate measurements from different immutable runs/configurations")
        completed_steps = sorted(result["step"] for result in results)
        completed = set(completed_steps)
        failed_steps = sorted(int(path.stem) for path in (tracking / "failures").glob("*.json") if int(path.stem) not in completed)
        failed = set(failed_steps)
        pending_steps = sorted(int(path.stem) for path in (tracking / "snapshots").glob("*.pt") if int(path.stem) not in completed | failed)
        warnings = sorted(result["step"] for result in results if result["coverage"]["status"] != "complete")
        expected_file = tracking / "EXPECTED_STEPS.json"
        expected = _read_json(expected_file).get("steps") if expected_file.exists() else None
        if expected is not None and (not isinstance(expected, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in expected)):
            raise ValueError("EXPECTED_STEPS.json must contain nonnegative integer steps")
        missing = sorted(set(expected or []) - completed)
        training_done = (run_dir / "TRAINING_DONE").exists()
        status = {"schema_version": SCHEMA_VERSION, "updated_at": _now(), "completed": len(completed),
                  "run_id": next(iter(run_ids), None), "config_sha256": next(iter(config_hashes), None),
                  "failed": len(failed), "pending": len(pending_steps), "completed_steps": completed_steps,
                  "failed_steps": failed_steps, "pending_steps": pending_steps, "warning_steps": warnings,
                  "expected_steps": expected, "missing_expected_steps": missing,
                  "training_done": training_done,
                  "complete": training_done and not (pending_steps or failed_steps or missing),
                  "all_fits_valid": bool(results) and not warnings,
                  "last_completed_step": max(completed_steps) if completed_steps else None}
        _csv_write(tracking / "layers.csv", [row for result in results for row in result["layers"]])
        _csv_write(tracking / "summary.csv", [result["summary"] for result in results])
        _csv_write(tracking / "groups.csv", [row for result in results for row in result["groups"]])
        # Manifest commits the CSV generation only after all three replacements.
        write_json(tracking / "EXPORTS.json", {"schema_version": SCHEMA_VERSION, "completed_steps": completed_steps,
                   "files": {name: _file_sha256(tracking / name) for name in ("layers.csv", "summary.csv", "groups.csv")},
                   "committed_at": _now()})
        write_json(run_dir / "TRACKING_STATUS.json", status)
        return status


def _prune_committed_snapshot(run_dir: Path, path: Path) -> None:
    step = int(path.stem)
    measurement = _step_path(run_dir, "measurements", step, ".json")
    exports = run_dir / "tracking" / "EXPORTS.json"
    if not measurement.exists() or not exports.exists() or step not in _read_json(exports)["completed_steps"]:
        raise RuntimeError("Cannot prune a snapshot before measurement and CSV exports commit")
    receipt_path = _step_path(run_dir, "receipts", step, ".json")
    result = _read_json(measurement)
    receipt = _read_json(receipt_path) if receipt_path.exists() else {
        "step": step, "content_sha256": result["content_sha256"], "snapshot_sha256": result["snapshot_sha256"]}
    receipt.update(pruned=True, pruned_at=_now(), measurement_sha256=_file_sha256(measurement))
    write_json(receipt_path, receipt)
    path.unlink(missing_ok=True)
    _sync_directory(path.parent)


def process_pending(run_dir, *, analyzer: Callable | None = None, keep_snapshots: bool = False,
                    retry_failed: bool = False) -> dict:
    """Process current queue once; keep all failed snapshots for inspection."""
    run_dir = Path(run_dir)
    with _file_lock(run_dir / "tracking" / ".worker.lock", blocking=False):
        for path in sorted((run_dir / "tracking" / "snapshots").glob("*.pt")):
            step = int(path.stem)
            output = _step_path(run_dir, "measurements", step, ".json")
            failure = _step_path(run_dir, "failures", step, ".json")
            if failure.exists() and not retry_failed and not output.exists():
                continue
            if not output.exists():
                try:
                    result = measure(path, analyzer=analyzer)
                    # A fully failed analyzer is an operational failure, not an
                    # ordinary invalid power-law fit on an individual matrix.
                    if len(result["coverage"]["pass_errors"]) == 2:
                        raise RuntimeError(json.dumps(result["coverage"]["pass_errors"], sort_keys=True))
                    write_json(output, result)
                    failure.unlink(missing_ok=True)
                except Exception as exc:
                    write_json(failure, {"schema_version": SCHEMA_VERSION, "step": step, "error_type": type(exc).__name__,
                                        "error": str(exc), "snapshot_path": str(path), "failed_at": _now()})
                    collect(run_dir)
                    continue
            # If collection fails, propagate and retain weights. A restarted
            # worker sees the committed JSON and safely retries only exports.
            collect(run_dir)
            requested_keep = _read_json(output).get("manifest", {}).get("spectral_tracking", {}).get("keep_snapshots", False)
            if not (keep_snapshots or requested_keep):
                _prune_committed_snapshot(run_dir, path)
        return collect(run_dir)


def capability_check() -> dict:
    """Real tiny CPU probe; fail explicitly if pinned native fields are absent."""
    try:
        import numpy as np
        import torch
        version = importlib.metadata.version("weightwatcher")
        if version != WEIGHTWATCHER_VERSION:
            return {"ok": False, "version": version, "required_version": WEIGHTWATCHER_VERSION,
                    "error": "WeightWatcher version mismatch"}
        generator = np.random.default_rng(1701)
        matrix = generator.standard_t(df=2.4, size=(64, 64)).astype("float32")
        probes = {}
        for fit, fix in (("raw", False), ("clip_xmax", "clip_xmax")):
            with _diagnostic_rng(1701):
                frame = _weightwatcher_analyzer({"L00_W_Q": matrix}, **WW_COMMON_OPTIONS, fix_fingers=fix)
            rows, unassigned = _frame_rows(frame, ["L00_W_Q"])
            native = rows.get("L00_W_Q", {})
            missing = sorted(REQUIRED_NATIVE_FIELDS - set(native))
            normalized = _normalize_fit(native or None, fit)
            formula_matches = (normalized["ERG_gap"] is not None
                               and _number(native.get("detX_num")) is not None
                               and _number(native.get("num_pl_spikes")) is not None
                               and normalized["ERG_gap"] == float(native["detX_num"]) - float(native["num_pl_spikes"]))
            probes[fit] = {"missing_fields": missing, "fit_status": normalized["status"],
                           "fit_reason": normalized["reason"], "n_tail": normalized["n_tail"],
                           "ERG_status": normalized["ERG_status"], "ERG_gap": normalized["ERG_gap"],
                           "native_ERG_formula_matches": formula_matches,
                           "unassigned_rows": unassigned, "native_fields": sorted(native)}
        ok = all(not p["missing_fields"] and p["fit_status"] == "success" and p["ERG_status"] == "available"
                 and p["native_ERG_formula_matches"] for p in probes.values())
        return {"ok": ok, "version": version, "required_version": WEIGHTWATCHER_VERSION, "probes": probes,
                "options": WW_COMMON_OPTIONS, "ERG_definition": ERG_DEFINITION, "source": WW_SOURCE}
    except Exception as exc:
        return {"ok": False, "required_version": WEIGHTWATCHER_VERSION,
                "error_type": type(exc).__name__, "error": str(exc)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--once", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--keep-snapshots", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--check", action="store_true", help="Run the real native CPU capability probe")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.poll_seconds <= 0:
        parser.error("--threads and --poll-seconds must be positive")
    if not args.check and args.run_dir is None:
        parser.error("--run-dir is required")
    # These are set before importing numerical libraries in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PJRT_DEVICE"] = "CPU"
    os.environ["JAX_PLATFORMS"] = "cpu"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    check = capability_check()
    if args.check:
        print(json.dumps(check, indent=2, allow_nan=False))
        return 0 if check["ok"] else 1
    if not check["ok"]:
        write_json(args.run_dir / "tracking" / "CAPABILITY_ERROR.json", check)
        print(json.dumps(check, indent=2), file=sys.stderr)
        return 1
    write_json(args.run_dir / "tracking" / "CAPABILITIES.json", check)
    while True:
        status = process_pending(args.run_dir, keep_snapshots=args.keep_snapshots, retry_failed=args.retry_failed)
        print(json.dumps({key: status[key] for key in ("completed", "failed", "pending", "warning_steps", "complete")}), flush=True)
        if status["failed"]:
            # Let the supervisor pause the trainer promptly, rather than
            # filling the queue and waiting for a long backpressure timeout.
            return 1
        if args.once or not args.watch:
            return 0
        if status["training_done"] and not status["pending"]:
            return 0 if status["complete"] else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

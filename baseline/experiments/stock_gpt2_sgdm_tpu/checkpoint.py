"""Atomic, integrity-checked checkpoints for trusted local experiment files.

PyTorch checkpoints include Python/NumPy RNG state, so loading uses
weights_only=False. Never pass an untrusted checkpoint to this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_cpu(item) for item in value)
    return value


def rng_state(rt=None):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if rt is not None and rt.tpu:
        state["xla"] = rt.xm.get_rng_state(rt.device)
    return state


def restore_rng(state, rt=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if rt is not None and rt.tpu:
        if "xla" not in state:
            raise ValueError("A TPU checkpoint must contain XLA RNG state")
        rt.xm.set_rng_state(state["xla"], rt.device)


def _link_or_copy(source, destination):
    temporary = destination.with_name(destination.name + ".link.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def save_checkpoint(run_dir, payload, *, best=False, first_target=False):
    directory = Path(run_dir) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.pt"
    sidecar = latest.with_suffix(".json")
    # Retain one known-good prior state before replacing either latest file.
    if latest.exists() and sidecar.exists():
        prior = json.loads(sidecar.read_text())
        if sha256_file(latest) == prior["sha256"]:
            _link_or_copy(latest, directory / "previous.pt")
            write_json(directory / "previous.json", {**prior, "file": "previous.pt"})
    temporary = directory / "latest.pt.tmp"
    with temporary.open("wb") as handle:
        torch.save(to_cpu(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    receipt = {"schema_version": 1, "file": "latest.pt", "sha256": sha256_file(temporary),
               "config_sha256": payload["config_sha256"], "step": payload["step"],
               "tokens_seen": payload["tokens_seen"], "run_id": payload["run_id"],
               "validation": payload.get("validation")}
    os.replace(temporary, latest)
    write_json(sidecar, receipt)
    if best:
        _link_or_copy(latest, directory / "best.pt")
        write_json(directory / "best.json", {**receipt, "file": "best.pt"})
    if first_target and not ((directory / "first_target.pt").exists() and (directory / "first_target.json").exists()):
        _link_or_copy(latest, directory / "first_target.pt")
        write_json(directory / "first_target.json", {**receipt, "file": "first_target.pt"})
    return receipt


def recover_checkpoint_aliases(run_dir, payload):
    """Repair a crash after verified latest publication but before best/target aliases.

    Call only after load_checkpoint has verified latest. Older alias generations
    cannot be reconstructed from a later model; those are never silently replaced.
    """
    directory = Path(run_dir) / "checkpoints"
    latest = directory / "latest.pt"
    receipt = json.loads((directory / "latest.json").read_text())
    if any(receipt[key] != payload[key] for key in ("run_id", "step", "config_sha256")):
        raise ValueError("Latest changed while recovering checkpoint aliases")
    repaired = []
    for alias, event in (("best", payload.get("best_validation")),
                         ("first_target", payload.get("first_target_validation"))):
        if event is None or event["step"] != payload["step"]:
            continue
        target = directory / (alias + ".pt")
        metadata = target.with_suffix(".json")
        valid = False
        if target.exists() and metadata.exists():
            saved = json.loads(metadata.read_text())
            valid = (all(saved.get(key) == receipt[key] for key in ("run_id", "step", "config_sha256"))
                     and sha256_file(target) == saved.get("sha256"))
        if not valid:
            _link_or_copy(latest, target)
            write_json(metadata, {**receipt, "file": target.name})
            repaired.append(alias)
    return repaired


def load_checkpoint(path, config, *, expected_run_id=None):
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["config_sha256"] != config.config_sha256:
        raise ValueError("Resume configuration changed; use a fresh run for a new training horizon")
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("Checkpoint integrity check failed; inspect previous.pt explicitly")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("config_sha256") != config.config_sha256:
        raise ValueError("Checkpoint schema/configuration mismatch")
    for field in ("step", "tokens_seen", "run_id"):
        if payload[field] != receipt[field]:
            raise ValueError("Checkpoint receipt mismatch: " + field)
    if expected_run_id is not None and payload["run_id"] != expected_run_id:
        raise ValueError("Checkpoint belongs to another run")
    if payload["tokens_seen"] != payload["step"] * config.tokens_per_step:
        raise ValueError("Checkpoint token/step mismatch")
    if not 0 <= payload["step"] <= config.max_steps:
        raise ValueError("Checkpoint is outside the immutable training horizon")
    if payload["next_lr_factor"] != config.schedule_factor(payload["step"]):
        raise ValueError("Checkpoint learning-rate schedule mismatch")
    return payload

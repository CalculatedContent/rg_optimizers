"""Platform-safe execution wrapper for the pinned nanochat baselines.

The pinned upstream trainer compiles both the model and the fused AdamW/Muon
step functions unconditionally. That is the canonical CUDA d12 path, but it is
not a reliable contract for Apple MPS or generic CPU execution. The upstream
attention module already falls back to PyTorch SDPA on those devices, so this
wrapper makes compilation environment-controlled and disables it on non-CUDA
devices.

No architecture, initialization, optimizer mathematics, scaling law, data
loader, or learning-rate schedule is changed. A versioned runtime marker
prevents a run created under a different compile policy from being silently
reused.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import nanochat_reference as reference

NANOCHAT_RUNTIME_PATCH_VERSION = 3
DISABLE_COMPILE_ENV = "NANOCHAT_DISABLE_COMPILE"
MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"
_COMPILE_HELPER = "_nanochat_compile_if_enabled"


def _patch_model_compile(checkout_dir: Path) -> Path:
    path = checkout_dir / "scripts" / "base_train.py"
    text = path.read_text(encoding="utf-8")
    if DISABLE_COMPILE_ENV in text:
        return path

    old = """orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)\nmodel = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe\n"""
    new = """orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)\ndisable_compile = os.environ.get(\"NANOCHAT_DISABLE_COMPILE\", \"0\") == \"1\"\nif disable_compile:\n    print0(\"torch.compile disabled by NANOCHAT_DISABLE_COMPILE=1\")\n    model = orig_model\nelse:\n    model = torch.compile(model, dynamic=False) # canonical CUDA/server path\n"""
    if old not in text:
        raise RuntimeError(
            "pinned nanochat base_train.py no longer matches the audited "
            "compile-policy patch"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def _patch_optimizer_compile(checkout_dir: Path) -> Path:
    """Make the two fused optimizer decorators conditional at import time."""

    path = checkout_dir / "nanochat" / "optim.py"
    text = path.read_text(encoding="utf-8")
    if _COMPILE_HELPER in text:
        if text.count("@_nanochat_compile_if_enabled") != 2:
            raise RuntimeError("nanochat optimizer compile patch is incomplete")
        return path

    import_block = """import torch\nimport torch.distributed as dist\nfrom torch import Tensor\nfrom nanochat.common import COMPUTE_DTYPE\n"""
    replacement_import_block = """import os\nimport torch\nimport torch.distributed as dist\nfrom torch import Tensor\nfrom nanochat.common import COMPUTE_DTYPE\n\ndef _nanochat_compile_if_enabled(function):\n    if os.environ.get(\"NANOCHAT_DISABLE_COMPILE\", \"0\") == \"1\":\n        return function\n    return torch.compile(dynamic=False, fullgraph=True)(function)\n"""
    if import_block not in text:
        raise RuntimeError(
            "pinned nanochat optim.py import block no longer matches the "
            "audited compile-policy patch"
        )
    decorator = "@torch.compile(dynamic=False, fullgraph=True)"
    if text.count(decorator) != 2:
        raise RuntimeError(
            "pinned nanochat optim.py does not contain exactly two compiled "
            "optimizer kernels"
        )
    text = text.replace(import_block, replacement_import_block, 1)
    text = text.replace(decorator, "@_nanochat_compile_if_enabled")
    path.write_text(text, encoding="utf-8")
    return path


def _install_compile_policy_patch(checkout_dir: str | Path) -> Path:
    """Patch model and optimizer compilation using exact pinned-source edits."""

    checkout = Path(checkout_dir)
    trainer = _patch_model_compile(checkout)
    _patch_optimizer_compile(checkout)
    return trainer


def ensure_checkout(
    checkout_dir: str | Path,
    *,
    commit: str = reference.NANOCHAT_COMMIT,
) -> Path:
    """Pin upstream, install the seed patch, then install compile policy."""

    checkout = reference.ensure_checkout(Path(checkout_dir), commit=commit)
    _install_compile_policy_patch(checkout)
    return checkout


def compile_enabled_for_device(device_type: str) -> bool:
    normalized = str(device_type).strip().lower()
    if normalized not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device_type: {device_type!r}")
    return normalized == "cuda"


def _runtime_policy(
    *,
    config: reference.NanoChatConfig,
    seed: int,
    device_type: str,
    nproc_per_node: int,
) -> dict[str, Any]:
    compile_enabled = compile_enabled_for_device(device_type)
    return {
        "schema_version": 1,
        "runtime_patch_version": NANOCHAT_RUNTIME_PATCH_VERSION,
        "nanochat_commit": reference.NANOCHAT_COMMIT,
        "profile": config.profile_name,
        "config": asdict(config),
        "seed": int(seed),
        "device_type": str(device_type),
        "nproc_per_node": int(nproc_per_node),
        "model_compile_enabled": compile_enabled,
        "optimizer_kernel_compile_enabled": compile_enabled,
        "mps_cpu_fallback_enabled": str(device_type) == "mps",
        "architecture_modified": False,
        "optimizer_math_modified": False,
    }


def _write_or_validate_policy(path: Path, policy: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != policy:
            raise RuntimeError(
                "nanochat run directory belongs to a different runtime policy; "
                "use a new output directory or remove the incompatible run"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def run_seed(
    checkout_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    config: reference.NanoChatConfig,
    *,
    seed: int,
    device_type: str = "auto",
    nproc_per_node: int = 1,
    resume: bool = True,
) -> Path:
    """Run the pinned recipe under an explicit, versioned compile policy."""

    resolved = (
        reference.detect_device_type()
        if str(device_type).lower() == "auto"
        else str(device_type).lower()
    )
    if resolved not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device_type: {resolved!r}")
    effective_nproc = int(nproc_per_node) if resolved == "cuda" else 1
    policy = _runtime_policy(
        config=config,
        seed=int(seed),
        device_type=resolved,
        nproc_per_node=effective_nproc,
    )
    seed_dir = Path(output_dir).expanduser().resolve() / f"seed_{int(seed)}"
    policy_path = seed_dir / "runtime_policy.json"
    _write_or_validate_policy(policy_path, policy)

    previous_compile = os.environ.get(DISABLE_COMPILE_ENV)
    previous_fallback = os.environ.get(MPS_FALLBACK_ENV)
    os.environ[DISABLE_COMPILE_ENV] = (
        "0" if policy["model_compile_enabled"] else "1"
    )
    if resolved == "mps":
        # This is read by the fresh training subprocess before PyTorch import.
        # Native MPS kernels remain preferred; unsupported individual operations
        # may fall back to CPU instead of terminating a long reference run.
        os.environ[MPS_FALLBACK_ENV] = "1"
    try:
        return reference.run_seed(
            Path(checkout_dir),
            Path(cache_dir),
            Path(output_dir),
            config,
            seed=int(seed),
            device_type=resolved,
            nproc_per_node=effective_nproc,
            resume=resume,
        )
    finally:
        if previous_compile is None:
            os.environ.pop(DISABLE_COMPILE_ENV, None)
        else:
            os.environ[DISABLE_COMPILE_ENV] = previous_compile
        if previous_fallback is None:
            os.environ.pop(MPS_FALLBACK_ENV, None)
        else:
            os.environ[MPS_FALLBACK_ENV] = previous_fallback


def _capture_rng_state() -> dict[str, Any]:
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
        and torch.backends.mps.is_available()
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if (
        "mps" in state
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(state["mps"])


def analyze_weightwatcher_checkpoints(
    checkout_dir: str | Path,
    cache_dir: str | Path,
    *,
    config: reference.NanoChatConfig,
    seed: int,
    output_csv: str | Path,
    min_evals: int = 20,
) -> pd.DataFrame:
    """Run deterministic offline WW analysis without compiling optimizer code."""

    state = _capture_rng_state()
    previous = os.environ.get(DISABLE_COMPILE_ENV)
    diagnostic_seed = int(seed) + 3_000_017
    random.seed(diagnostic_seed)
    np.random.seed(diagnostic_seed % (2**32 - 1))
    torch.manual_seed(diagnostic_seed)
    os.environ[DISABLE_COMPILE_ENV] = "1"
    try:
        frame = reference.analyze_weightwatcher_checkpoints(
            Path(checkout_dir),
            Path(cache_dir),
            config=config,
            seed=int(seed),
            output_csv=Path(output_csv),
            min_evals=int(min_evals),
        ).copy()
        frame["diagnostic_seed"] = int(diagnostic_seed)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
        return frame
    finally:
        _restore_rng_state(state)
        if previous is None:
            os.environ.pop(DISABLE_COMPILE_ENV, None)
        else:
            os.environ[DISABLE_COMPILE_ENV] = previous


# Re-export the rest of the audited public interface for notebook convenience.
DEFAULT_NANOCHAT_SEEDS = reference.DEFAULT_NANOCHAT_SEEDS
NANOCHAT_COMMIT = reference.NANOCHAT_COMMIT
NanoChatD12Config = reference.NanoChatD12Config
NanoChatMacConfig = reference.NanoChatMacConfig
collect_metrics = reference.collect_metrics
detect_device_type = reference.detect_device_type
ensure_environment = reference.ensure_environment
prepare_data = reference.prepare_data
resolve_profile = reference.resolve_profile
summarize_training_metrics = reference.summarize_training_metrics
summarize_weightwatcher = reference.summarize_weightwatcher

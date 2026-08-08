"""Platform-safe execution wrapper for the pinned nanochat baselines.

The pinned upstream trainer calls ``torch.compile`` unconditionally. That is the
correct canonical path for the CUDA d12 reference, but it is not a reliable
contract for Apple MPS or generic CPU execution. The upstream attention module
already falls back to PyTorch SDPA on those devices, so this wrapper makes only
one additional runtime change: compilation is configurable and disabled on
non-CUDA devices.

No architecture, initialization, optimizer, scaling law, data loader, or
learning-rate schedule is changed. A versioned runtime marker prevents a run
created under a different compile policy from being silently reused.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from . import nanochat_reference as reference

NANOCHAT_RUNTIME_PATCH_VERSION = 2
DISABLE_COMPILE_ENV = "NANOCHAT_DISABLE_COMPILE"


def _install_compile_policy_patch(checkout_dir: str | Path) -> Path:
    """Patch the pinned trainer so compile can be disabled by environment.

    The replacement is exact and pinned-commit-specific. A source mismatch
    fails rather than guessing how to rewrite an unknown upstream version.
    """

    path = Path(checkout_dir) / "scripts" / "base_train.py"
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
    return {
        "schema_version": 1,
        "runtime_patch_version": NANOCHAT_RUNTIME_PATCH_VERSION,
        "nanochat_commit": reference.NANOCHAT_COMMIT,
        "profile": config.profile_name,
        "config": asdict(config),
        "seed": int(seed),
        "device_type": str(device_type),
        "nproc_per_node": int(nproc_per_node),
        "torch_compile_enabled": compile_enabled_for_device(device_type),
        "architecture_or_optimizer_modified": False,
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

    previous = os.environ.get(DISABLE_COMPILE_ENV)
    os.environ[DISABLE_COMPILE_ENV] = (
        "0" if policy["torch_compile_enabled"] else "1"
    )
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
        if previous is None:
            os.environ.pop(DISABLE_COMPILE_ENV, None)
        else:
            os.environ[DISABLE_COMPILE_ENV] = previous


# Re-export the rest of the audited public interface for notebook convenience.
DEFAULT_NANOCHAT_SEEDS = reference.DEFAULT_NANOCHAT_SEEDS
NANOCHAT_COMMIT = reference.NANOCHAT_COMMIT
NanoChatD12Config = reference.NanoChatD12Config
NanoChatMacConfig = reference.NanoChatMacConfig
analyze_weightwatcher_checkpoints = reference.analyze_weightwatcher_checkpoints
collect_metrics = reference.collect_metrics
detect_device_type = reference.detect_device_type
ensure_environment = reference.ensure_environment
prepare_data = reference.prepare_data
resolve_profile = reference.resolve_profile
summarize_training_metrics = reference.summarize_training_metrics
summarize_weightwatcher = reference.summarize_weightwatcher

#!/usr/bin/env python3
"""Safe, reproducible launcher for the 2026-08-21 one-head nanoGPT campaign.

The launcher intentionally imports only the Python standard library at module
load time.  This keeps path-policy tests and ``--help`` usable before the
scientific environment has been installed.  Scientific dependencies are
loaded only by the command that needs them.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


EXPERIMENT_ROOT_ENV = "RG_NANOGPT_EXPERIMENT_ROOT"
CANONICAL_OPTIMIZERS = ("adamw", "muon_clip")
CANONICAL_SEEDS = (1337, 2027, 4099, 31415, 271828)
EXPECTED_REPLICATES = 10
EXPECTED_MATRICES = 6
MINIMUM_PERMANENT_STATES = 10
PINNED_WEIGHTWATCHER = "0.7.7"
PINNED_PACKAGE_VERSION = "0.5.1"
FROZEN_CONFIG_SHA256 = (
    "ebbbdfa30efe96b0b0c1c68ae4fc81909361502d89ad336d1181d00fcb85876a"
)
MANIFEST_PACKAGES = (
    "python",
    "rg-nanogpt-one-head",
    "torch",
    "torch-xla",
    "numpy",
    "pandas",
    "scipy",
    "PyYAML",
    "datasets",
    "tiktoken",
    "sacrebleu",
    "weightwatcher",
    "powerlaw",
    "papermill",
    "packaging",
)

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]
NANOGPT_ROOT = REPOSITORY_ROOT / "baseline" / "nanogpt_one_head"
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "baseline.yaml"

REQUIRED_RUN_FILES = (
    "run_complete.json",
    "manifest.json",
    "metrics.csv",
    "epoch_metrics.csv",
    "checkpoint_initial.pt",
    "checkpoint_latest.pt",
    "checkpoint_best.pt",
    "checkpoint_final.pt",
    "test_results.json",
    "spectral/layers.csv",
    "spectral/summary.csv",
)

REQUIRED_ANALYSIS_FILES = (
    "SUMMARY.md",
    "report.html",
    "results_manifest.json",
    "campaign_runs.csv",
    "metrics_all.csv",
    "epoch_metrics_all.csv",
    "spectral_layers_all.csv",
    "spectral_summary_all.csv",
    "test_results_all.csv",
    "qk_diagnostics_all.csv",
    "qk_summary.csv",
    "performance_summary.csv",
    "paired_seed_differences.csv",
    "alpha_run_medians.csv",
    "alpha_across_seed_summary.csv",
    "saturation_diagnostics.csv",
    "saturation_integer_epoch_validation.csv",
    "saturation_across_seed_summary.csv",
    "checkpoint_sha256.csv",
    "plots/adamw_performance.png",
    "plots/adamw_alpha_raw_vs_clip_xmax.png",
    "plots/adamw_alpha_raw_vs_clip_xmax_zoomed.png",
    "plots/adamw_erg_gap_num_traps.png",
    "plots/muon_clip_performance.png",
    "plots/muon_clip_alpha_raw_vs_clip_xmax.png",
    "plots/muon_clip_alpha_raw_vs_clip_xmax_zoomed.png",
    "plots/muon_clip_erg_gap_num_traps.png",
    "notebooks/01_Performance_and_Spectra.executed.ipynb",
)

DEPENDENCIES = {
    "rg-nanogpt-one-head": "rg_nanogpt_one_head",
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "PyYAML": "yaml",
    "weightwatcher": "weightwatcher",
    "powerlaw": "powerlaw",
    "datasets": "datasets",
    "tiktoken": "tiktoken",
    "sacrebleu": "sacrebleu",
    "matplotlib": "matplotlib",
    "jupyter": "jupyter",
    "ipykernel": "ipykernel",
    "nbformat": "nbformat",
    "papermill": "papermill",
    "packaging": "packaging",
}


class CampaignError(ValueError):
    """An actionable campaign-policy or artifact-validation failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_descendant(path: Path, parent: Path) -> bool:
    try:
        relative = path.relative_to(parent)
    except ValueError:
        return False
    return relative != Path(".")


def resolve_experiment_root(
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve and validate the required campaign output root.

    The value must be an absolute *strict descendant* of the resolved ``/tmp``
    or ``/private/tmp`` root.  Resolving before validation prevents an existing
    symlink or ``..`` component from escaping.  A root that is HOME (or below
    HOME) is rejected independently, even if a caller has placed HOME in tmp.
    No directory is created by this function.
    """

    source = os.environ if env is None else env
    raw = source.get(EXPERIMENT_ROOT_ENV)
    if raw is None or not str(raw).strip():
        raise CampaignError(
            f"{EXPERIMENT_ROOT_ENV} is required and must name an absolute "
            "directory strictly below /tmp or /private/tmp"
        )
    value = str(raw).strip()
    if "~" in value:
        raise CampaignError(
            f"{EXPERIMENT_ROOT_ENV} must not contain '~': {value!r}"
        )
    candidate = Path(value)
    if not candidate.is_absolute():
        raise CampaignError(
            f"{EXPERIMENT_ROOT_ENV} must be absolute, observed {value!r}"
        )
    resolved = candidate.resolve(strict=False)
    allowed_roots = {
        Path("/tmp").resolve(strict=False),
        Path("/private/tmp").resolve(strict=False),
    }
    if not any(_strict_descendant(resolved, root) for root in allowed_roots):
        raise CampaignError(
            f"{EXPERIMENT_ROOT_ENV} resolved to {resolved}; it must be "
            "strictly below /tmp or /private/tmp"
        )

    home_raw = source.get("HOME")
    if home_raw:
        home = Path(str(home_raw)).expanduser().resolve(strict=False)
        if resolved == home or _strict_descendant(resolved, home):
            raise CampaignError(
                f"{EXPERIMENT_ROOT_ENV} must never be HOME or below HOME"
            )
    return resolved


def _within(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    return resolved == root or _strict_descendant(resolved, root)


def _require_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=False)
    if not _within(resolved, root):
        raise CampaignError(f"{label} escapes experiment root: {resolved}")
    return resolved


def _paths(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "data": root / "data",
        "results": root / "results",
        "logs": root / "logs",
        "analysis": root / "analysis",
        "tables": root / "analysis" / "tables",
        "plots": root / "analysis" / "plots",
        "provenance": root / "provenance",
        "cache": root / "cache",
        "tmp": root / "tmp",
    }
    for label, path in paths.items():
        _require_within(path, root, label)
    return paths


def _create_runtime_directories(paths: Mapping[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for relative in (
        "huggingface/datasets",
        "huggingface/hub",
        "huggingface/assets",
        "huggingface/modules",
        "huggingface/transformers",
        "tiktoken",
        "matplotlib",
        "xdg/cache",
        "xdg/config",
        "xdg/data",
        "xdg/state",
        "torch",
        "torch_extensions",
        "torchinductor",
        "cuda",
        "triton",
        "cupy",
        "xla",
        "pip",
        "uv",
        "numba",
        "joblib",
        "keras",
        "sacrebleu",
        "wandb/cache",
        "wandb/config",
        "wandb/data",
        "pycache",
        "jupyter/config",
        "jupyter/data",
        "jupyter/runtime",
        "ipython",
        "home",
    ):
        (paths["cache"] / relative).mkdir(parents=True, exist_ok=True)


def _child_environment(root: Path, paths: Mapping[str, Path]) -> dict[str, str]:
    cache = paths["cache"]
    additions = {
        EXPERIMENT_ROOT_ENV: str(root),
        "RG_NANOGPT_ONE_HEAD_ROOT": str(root),
        "RG_NANOGPT_ONE_HEAD_DATA_ROOT": str(paths["data"]),
        "RG_NANOGPT_ONE_HEAD_RESULTS_ROOT": str(paths["results"]),
        "RG_NANOGPT_ONE_HEAD_PLOTS_ROOT": str(paths["plots"]),
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "HF_ASSETS_CACHE": str(cache / "huggingface" / "assets"),
        "HF_MODULES_CACHE": str(cache / "huggingface" / "modules"),
        "TRANSFORMERS_CACHE": str(cache / "huggingface" / "transformers"),
        "TIKTOKEN_CACHE_DIR": str(cache / "tiktoken"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "XDG_CACHE_HOME": str(cache / "xdg" / "cache"),
        "XDG_CONFIG_HOME": str(cache / "xdg" / "config"),
        "XDG_DATA_HOME": str(cache / "xdg" / "data"),
        "XDG_STATE_HOME": str(cache / "xdg" / "state"),
        "TORCH_HOME": str(cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUPY_CACHE_DIR": str(cache / "cupy"),
        "XLA_PERSISTENT_CACHE_PATH": str(cache / "xla"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "JOBLIB_TEMP_FOLDER": str(cache / "joblib"),
        "KERAS_HOME": str(cache / "keras"),
        "SACREBLEU": str(cache / "sacrebleu"),
        "WANDB_CACHE_DIR": str(cache / "wandb" / "cache"),
        "WANDB_CONFIG_DIR": str(cache / "wandb" / "config"),
        "WANDB_DATA_DIR": str(cache / "wandb" / "data"),
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
        "JUPYTER_CONFIG_DIR": str(cache / "jupyter" / "config"),
        "JUPYTER_DATA_DIR": str(cache / "jupyter" / "data"),
        "JUPYTER_RUNTIME_DIR": str(cache / "jupyter" / "runtime"),
        "IPYTHONDIR": str(cache / "ipython"),
        # Some third-party libraries ignore XDG/cache-specific variables. Give
        # child processes an ephemeral HOME inside the required /tmp campaign
        # root so even those fallbacks can never touch the user's real home.
        "HOME": str(cache / "home"),
        "TMPDIR": str(paths["tmp"]),
        "TMP": str(paths["tmp"]),
        "TEMP": str(paths["tmp"]),
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MPLBACKEND": "Agg",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for name, value in additions.items():
        if name in {
            "PYTHONUNBUFFERED",
            "PYTORCH_ENABLE_MPS_FALLBACK",
            "CUBLAS_WORKSPACE_CONFIG",
            "MPLBACKEND",
            "TOKENIZERS_PARALLELISM",
        }:
            continue
        _require_within(Path(value), root, f"environment variable {name}")

    child = os.environ.copy()
    child.update(additions)
    source_path = str(NANOGPT_ROOT / "src")
    # Do not inherit arbitrary module-shadowing paths whose distribution
    # metadata can look pinned while different source code is imported.
    child["PYTHONPATH"] = source_path
    return child


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _runtime_block_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Return the comparison identity for one declared hardware block.

    Auto identities remain exact, including a CUDA UUID. A collaborator may
    deliberately assign the same user block ID to homogeneous machines; in
    that case host/install-path and accelerator-instance fields are recorded
    but do not prevent pooling distinct complete seeds.
    """

    identity = dict(runtime)
    if str(identity.get("hardware_block_id_source", "")) == "user":
        for field in (
            "python_executable",
            "processor",
            "cuda_device_uuid",
            "cuda_device_count",
            "xla_process_index",
        ):
            identity.pop(field, None)
    return identity


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            if value and not value.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(dict(payload), sort_keys=True, default=str) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, record.encode("utf-8"))
    finally:
        os.close(descriptor)


def _acquire_exclusive_lock(path: Path):
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - campaign targets Unix hosts
        raise CampaignError(
            "campaign subprocess locks require a Unix fcntl implementation"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise CampaignError(
            f"another campaign process holds {path}: {owner}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {"pid": os.getpid(), "started_at_utc": _utc_now()},
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _release_exclusive_lock(handle) -> None:
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _git(arguments: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=check,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CampaignError(f"git command failed: {exc}") from exc


def _git_provenance() -> dict[str, Any]:
    try:
        commit = _git(("rev-parse", "HEAD")).stdout.strip()
        branch = _git(("rev-parse", "--abbrev-ref", "HEAD")).stdout.strip()
        describe = _git(("describe", "--tags", "--always", "--dirty")).stdout.strip()
        tags = [
            line
            for line in _git(("tag", "--points-at", "HEAD")).stdout.splitlines()
            if line.strip()
        ]
        status = _git(("status", "--porcelain=v1", "--untracked-files=all")).stdout
        remote_result = _git(("remote", "get-url", "origin"), check=False)
        return {
            "available": True,
            "commit": commit,
            "branch": branch,
            "describe": describe,
            "tags_at_commit": tags,
            "tag_status": ",".join(tags) if tags else "untagged",
            "origin_url": remote_result.stdout.strip() if remote_result.returncode == 0 else None,
            "clean": not bool(status.strip()),
            "dirty": bool(status.strip()),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "status_lines": len(status.splitlines()),
        }
    except CampaignError:
        return {"available": False, "clean": False, "dirty": None}


def _require_clean_git() -> dict[str, Any]:
    provenance = _git_provenance()
    if not provenance.get("available"):
        raise CampaignError("a readable Git checkout is required")
    if not provenance.get("clean"):
        status = _git(("status", "--short", "--untracked-files=all")).stdout.strip()
        preview = "\n".join(status.splitlines()[:20])
        raise CampaignError(
            "production commands require a clean Git worktree. Commit or "
            f"remove all changes first. Current status:\n{preview}"
        )
    return provenance


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", str(value).strip()).lower()


def _installed_distribution_lock(
    scientific_packages: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    """Lock the installed dependency closure and reject opaque origins.

    Unrelated packages in a long-lived conda environment are retained by raw
    ``pip freeze`` but do not belong in the replay contract. Conversely, every
    installed transitive dependency reachable from a campaign requirement is
    locked. PEP 610 direct/VCS/file installs are rejected because ``Name==ver``
    would silently discard the only information capable of reproducing them.
    """

    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError as exc:
        raise CampaignError(
            "packaging is required to construct the dependency closure lock"
        ) from exc

    project_name = _normalized_distribution_name("rg-nanogpt-one-head")
    packages: dict[str, dict[str, str]] = {}
    pending = [
        name
        for name, version in scientific_packages.items()
        if name != "python"
        and str(version) != "not-installed"
        and _normalized_distribution_name(name) != project_name
    ]
    visited: set[str] = set()
    while pending:
        requested_name = pending.pop()
        requested_normalized = _normalized_distribution_name(requested_name)
        if requested_normalized in visited:
            continue
        visited.add(requested_normalized)
        try:
            distribution = importlib.metadata.distribution(requested_name)
        except importlib.metadata.PackageNotFoundError:
            # Optional requirements and platform markers may name packages not
            # installed on this hardware block.
            continue
        name = str(distribution.metadata.get("Name", "")).strip()
        version = str(distribution.version).strip()
        if not name or not version or "\n" in name or "\n" in version:
            raise CampaignError(
                "installed distribution has unsafe or missing name/version metadata"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise CampaignError(
                f"installed distribution name is not replay-safe: {name!r}"
            )
        normalized = _normalized_distribution_name(name)
        if normalized == project_name:
            continue
        raw_direct_url = distribution.read_text("direct_url.json")
        if raw_direct_url:
            try:
                direct_url = json.loads(raw_direct_url)
            except json.JSONDecodeError as exc:
                raise CampaignError(
                    f"installed dependency {name} has invalid direct_url.json"
                ) from exc
            origin = str(direct_url.get("url", "unknown"))
            raise CampaignError(
                f"installed dependency {name}=={version} uses a direct origin "
                f"({origin}); a portable exact replay cannot replace it with "
                "a name/version guess. Reinstall that dependency from a "
                "replayable package index or conda channel before production."
            )
        record = {"name": name, "version": version}
        previous = packages.get(normalized)
        if previous is not None and previous != record:
            raise CampaignError(
                "conflicting installed distribution metadata for "
                f"{normalized}: {previous!r} versus {record!r}"
            )
        packages[normalized] = record
        for requirement_text in distribution.requires or ():
            try:
                requirement = Requirement(requirement_text)
            except InvalidRequirement as exc:
                raise CampaignError(
                    f"installed dependency {name} has an invalid requirement: "
                    f"{requirement_text!r}"
                ) from exc
            dependency_normalized = _normalized_distribution_name(
                requirement.name
            )
            if dependency_normalized not in visited:
                pending.append(requirement.name)
    if not packages:
        raise CampaignError("could not enumerate the installed dependency closure")
    requirements = "\n".join(
        f"{record['name']}=={record['version']}"
        for _, record in sorted(packages.items())
    ) + "\n"
    payload = {
        "schema_version": 2,
        "python_version": platform.python_version(),
        "packages": packages,
        "scientific_packages": dict(scientific_packages),
        "excluded_project": "rg-nanogpt-one-head",
        "replay_policy": (
            "installed campaign dependency closure pinned by metadata "
            "name/version; opaque direct origins rejected; checked-out "
            "project installed separately with --no-deps"
        ),
    }
    return payload, requirements


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise CampaignError("PyYAML is required to validate the campaign config") from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignError(f"could not read valid YAML from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"config is not a YAML mapping: {path}")
    return payload


def _resolve_config(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    else:
        path = path.resolve(strict=False)
    if not path.is_file():
        raise CampaignError(f"config does not exist: {path}")
    if not _within(path, REPOSITORY_ROOT):
        raise CampaignError(f"config must be tracked inside the repository: {path}")
    try:
        relative = path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:  # pragma: no cover - guarded by _within above
        raise CampaignError(f"config escapes the repository: {path}") from exc
    tracked = _git(("ls-files", "--error-unmatch", "--", str(relative)), check=False)
    if tracked.returncode != 0:
        raise CampaignError(
            "config must be a Git-tracked file (ignored/untracked configs are "
            f"not reproducible): {path}"
        )
    return path


def _validate_protocol_config(config: Path) -> dict[str, Any]:
    cfg = _load_yaml(config)
    observed_config_sha = _canonical_sha256(cfg)
    if observed_config_sha != FROZEN_CONFIG_SHA256:
        raise CampaignError(
            "config does not exactly match the frozen dated campaign; "
            f"canonical_sha256={observed_config_sha}, "
            f"expected={FROZEN_CONFIG_SHA256}"
        )
    dataset = cfg.get("dataset", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    profiles = cfg.get("optimizer_profiles", {})
    ww = cfg.get("weightwatcher", {})
    runtime = cfg.get("runtime", {})
    expected_dataset = {
        "name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "revision": "593b3a867298afb8ce42625a270ef20ddcad28f9",
        "tokenizer": "gpt2",
        "train_tokens": 80_000_000,
        "val_tokens": 1_000_000,
        "test_tokens": 1_000_000,
    }
    if any(dataset.get(key) != value for key, value in expected_dataset.items()):
        raise CampaignError("config does not match the frozen FineWeb-Edu corpus protocol")
    expected_model = {
        "vocab_size": 50_257,
        "block_size": 256,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 128,
        "dropout": 0.0,
        "bias": False,
        "tie_weights": True,
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise CampaignError("config does not match the frozen one-head model")
    if tuple(training.get("seeds", ())) != CANONICAL_SEEDS:
        raise CampaignError(
            f"config training.seeds must be exactly {list(CANONICAL_SEEDS)}"
        )
    missing_profiles = set(CANONICAL_OPTIMIZERS).difference(profiles)
    if missing_profiles:
        raise CampaignError(
            "config lacks campaign optimizer profiles: "
            + ", ".join(sorted(missing_profiles))
        )
    if ww.get("fix_fingers") != "clip_xmax":
        raise CampaignError("weightwatcher.fix_fingers must equal clip_xmax")
    if int(ww.get("max_fingers", 0)) < 1:
        raise CampaignError("weightwatcher.max_fingers must be positive")
    if ww.get("require_raw_alpha") is not True:
        raise CampaignError("weightwatcher.require_raw_alpha must be true")
    if ww.get("enabled") is not True or ww.get("strict") is not True:
        raise CampaignError("WeightWatcher must be enabled and strict")
    target_epochs = float(training.get("target_epochs", 0.0))
    epoch_interval = float(training.get("epoch_interval", 0.0))
    if target_epochs != 4.0 or epoch_interval != 0.25:
        raise CampaignError("campaign horizon must be 4.0 epochs at 0.25-epoch states")
    expected_training = {
        "batch_size": 4,
        "grad_accum_steps": 8,
        "eval_interval_steps": 500,
        "eval_batches": 64,
        "checkpoint_interval_steps": 500,
        "grad_clip": 1.0,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise CampaignError("config does not match the frozen training/evaluation cadence")
    permanent_states = int(round(target_epochs / epoch_interval)) + 1
    if permanent_states < MINIMUM_PERMANENT_STATES:
        raise CampaignError(
            f"config yields only {permanent_states} permanent states; at least "
            f"{MINIMUM_PERMANENT_STATES} are required"
        )
    if (
        runtime.get("matmul_precision") != "highest"
        or runtime.get("allow_tf32") is not False
        or runtime.get("cudnn_benchmark") is not False
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("deterministic_warn_only") is not False
    ):
        raise CampaignError("runtime must use the frozen strict deterministic settings")
    return cfg


def _expected_total_steps(cfg: Mapping[str, Any]) -> int:
    training = cfg["training"]
    model = cfg["model"]
    dataset = cfg["dataset"]
    tokens_per_update = (
        int(training["batch_size"])
        * int(training["grad_accum_steps"])
        * int(model["block_size"])
    )
    return max(
        1,
        int(
            math.ceil(
                float(training["target_epochs"])
                * int(dataset["train_tokens"])
                / tokens_per_update
            )
        ),
    )


def _expected_permanent_steps(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    training = cfg["training"]
    train_tokens = int(cfg["dataset"]["train_tokens"])
    tokens_per_update = (
        int(training["batch_size"])
        * int(training["grad_accum_steps"])
        * int(cfg["model"]["block_size"])
    )
    target_epochs = float(training["target_epochs"])
    interval = float(training["epoch_interval"])
    total_steps = _expected_total_steps(cfg)
    epochs = [0.0]
    current = interval
    while current < target_epochs - 1e-12:
        epochs.append(round(current, 12))
        current += interval
    epochs.append(target_epochs)
    steps: dict[int, float] = {}
    for epoch in epochs:
        if epoch == 0.0:
            step = 0
        elif math.isclose(epoch, target_epochs, rel_tol=0.0, abs_tol=1e-12):
            step = total_steps
        else:
            step = int(round(epoch * train_tokens / tokens_per_update))
        steps[min(total_steps, max(0, step))] = epoch
    steps[total_steps] = target_epochs
    return tuple(sorted(steps))


def _require_dependency_contract() -> dict[str, str]:
    versions = _dependency_versions()
    missing = [name for name, version in versions.items() if version is None]
    if missing:
        raise CampaignError(
            "missing experiment dependencies: " + ", ".join(missing)
            + f". Install with: {sys.executable} -m pip install -e {NANOGPT_ROOT}"
        )
    if versions.get("weightwatcher") != PINNED_WEIGHTWATCHER:
        raise CampaignError(
            "WeightWatcher must be pinned exactly to "
            f"{PINNED_WEIGHTWATCHER}; observed {versions.get('weightwatcher')}"
        )
    if versions.get("rg-nanogpt-one-head") != PINNED_PACKAGE_VERSION:
        raise CampaignError(
            "the editable campaign package must be reinstalled at version "
            f"{PINNED_PACKAGE_VERSION}; observed "
            f"{versions.get('rg-nanogpt-one-head')}. Run: "
            f"{sys.executable} -m pip install -e {NANOGPT_ROOT}"
        )
    resolved = {
        name: str(version) for name, version in versions.items() if version is not None
    }
    resolved["python"] = platform.python_version()
    try:
        resolved["torch-xla"] = importlib.metadata.version("torch-xla")
    except importlib.metadata.PackageNotFoundError:
        resolved["torch-xla"] = "not-installed"
    # Enforce the same replayability policy before a multi-day run, rather
    # than discovering an opaque direct/VCS/file dependency only at archive.
    # Return the full closure in the spelling used by the run manifest so a
    # second host cannot begin a disjoint replicate with a silent transitive
    # dependency difference.
    dependency_lock, _ = _installed_distribution_lock(resolved)
    for record in dependency_lock["packages"].values():
        resolved.setdefault(str(record["name"]), str(record["version"]))
    return resolved


def _verified_data_metadata(
    paths: Mapping[str, Path], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    data_root = paths["data"]
    metadata_path = data_root / "meta.json"
    metadata = _read_json(metadata_path)
    expected_splits = {
        "train": int(cfg["dataset"]["train_tokens"]),
        "val": int(cfg["dataset"]["val_tokens"]),
        "test": int(cfg["dataset"]["test_tokens"]),
    }
    expected_metadata = {
        "schema_version": 2,
        "dataset_name": cfg["dataset"]["name"],
        "dataset_config": cfg["dataset"]["config"],
        "dataset_split": cfg["dataset"]["split"],
        "dataset_revision": cfg["dataset"]["revision"],
        "tokenizer": cfg["dataset"]["tokenizer"],
        "vocab_size": int(cfg["model"]["vocab_size"]),
        "eot_token": 50_256,
        "dtype": "uint16",
        "document_disjoint_splits": True,
        "splits": expected_splits,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise CampaignError(
            "prepared corpus metadata does not match the frozen campaign: "
            + _canonical_json(mismatches)
        )
    file_metadata = metadata.get("files")
    if not isinstance(file_metadata, dict):
        raise CampaignError("prepared corpus metadata has no file-hash inventory")
    for split, token_count in expected_splits.items():
        record = file_metadata.get(split)
        if not isinstance(record, dict):
            raise CampaignError(f"prepared corpus metadata lacks files.{split}")
        if record.get("path") != f"{split}.bin":
            raise CampaignError(f"prepared corpus files.{split}.path is not canonical")
        path = data_root / f"{split}.bin"
        expected_bytes = token_count * 2
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or int(record.get("bytes", -1)) != expected_bytes
        ):
            raise CampaignError(f"prepared {split} token file has an invalid byte count")
        observed_hash = _sha256(path)
        if not record.get("sha256") or observed_hash != str(record["sha256"]):
            raise CampaignError(f"prepared {split} token file SHA-256 mismatch")
    return metadata


def _inspect_runtime(
    config: Path,
    device: str,
    child_env: Mapping[str, str],
) -> dict[str, Any]:
    probe = """
import json
import sys
from rg_nanogpt_one_head.config import load_config
from rg_nanogpt_one_head.runtime import choose_device, configure_runtime, runtime_metadata
cfg = load_config(sys.argv[1])
resolved = choose_device(sys.argv[2])
configure_runtime(resolved, cfg)
print('RG_RUNTIME_JSON=' + json.dumps(runtime_metadata(resolved), sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe, str(config), str(device)],
            cwd=REPOSITORY_ROOT,
            env=dict(child_env),
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CampaignError(f"could not inspect requested runtime: {exc}") from exc
    prefix = "RG_RUNTIME_JSON="
    lines = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise CampaignError(
            "runtime inspection did not emit exactly one metadata record; "
            f"stderr={completed.stderr[-2000:]}"
        )
    try:
        payload = json.loads(lines[0][len(prefix) :])
    except json.JSONDecodeError as exc:
        raise CampaignError("runtime inspection emitted invalid JSON") from exc
    if not isinstance(payload, dict):
        raise CampaignError("runtime inspection metadata is not an object")
    return payload


def _run_context(
    *,
    cfg: Mapping[str, Any],
    git: Mapping[str, Any],
    data_metadata: Mapping[str, Any] | None = None,
    dependencies: Mapping[str, str] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "cfg": dict(cfg),
        "config_sha256": _canonical_sha256(cfg),
        "git_commit": str(git.get("commit", "")),
        "data_metadata": dict(data_metadata) if data_metadata is not None else None,
        "dependencies": dict(dependencies) if dependencies is not None else None,
        "runtime": dict(runtime) if runtime is not None else None,
        "total_steps": _expected_total_steps(cfg),
        "permanent_steps": _expected_permanent_steps(cfg),
    }


def _parse_optimizers(value: str) -> tuple[str, ...]:
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    if not requested:
        raise CampaignError("at least one optimizer is required")
    if len(requested) != len(set(requested)):
        raise CampaignError("optimizer list contains duplicates")
    invalid = set(requested).difference(CANONICAL_OPTIMIZERS)
    if invalid:
        raise CampaignError("unsupported campaign optimizers: " + ", ".join(sorted(invalid)))
    return tuple(name for name in CANONICAL_OPTIMIZERS if name in requested)


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        requested = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise CampaignError("seeds must be comma-separated integers") from exc
    if not requested:
        raise CampaignError("at least one seed is required")
    if len(requested) != len(set(requested)):
        raise CampaignError("seed list contains duplicates")
    invalid = set(requested).difference(CANONICAL_SEEDS)
    if invalid:
        raise CampaignError("unsupported campaign seeds: " + ", ".join(map(str, sorted(invalid))))
    return tuple(seed for seed in CANONICAL_SEEDS if seed in requested)


def _stream_command(
    command: Sequence[str],
    *,
    log_path: Path,
    environment: Mapping[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # A stable log path identifies one artifact-producing subprocess. Adjacent
    # locks allow different optimizer/seed jobs to run in parallel, while two
    # launchers cannot write the same replicate, corpus, report, or preflight.
    lock = _acquire_exclusive_lock(
        log_path.with_name(log_path.name + ".lock")
    )
    try:
        started = _utc_now()
        header = f"\n[{started}] $ {' '.join(command)}\n"
        print(header.rstrip(), flush=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(header)
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=REPOSITORY_ROOT,
                    env=dict(environment),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                log.write(f"launcher error: {exc}\n")
                raise CampaignError(f"could not start command: {exc}") from exc
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise
            return_code = process.wait()
            footer = f"[{_utc_now()}] exit_code={return_code}\n"
            print(footer.rstrip(), flush=True)
            log.write(footer)
        return return_code
    finally:
        _release_exclusive_lock(lock)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"JSON artifact is not an object: {path}")
    return payload


def _read_csv_header(path: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return set(next(reader))
    except (OSError, StopIteration, csv.Error) as exc:
        raise CampaignError(f"invalid or empty CSV artifact {path}: {exc}") from exc


def _run_dir(paths: Mapping[str, Path], optimizer: str, seed: int) -> Path:
    return paths["results"] / optimizer / f"seed_{int(seed)}"


def _validate_manifest_binding(
    manifest: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    optimizer: str,
    seed: int,
    expected_context: Mapping[str, Any] | None,
    run_dir: Path,
) -> None:
    try:
        manifest_seed = int(manifest.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"manifest seed is invalid in {run_dir}") from exc
    if manifest.get("optimizer") != optimizer or manifest_seed != seed:
        raise CampaignError(f"manifest identity mismatch in {run_dir}")
    initial_model_hash = str(manifest.get("initial_model_sha256", ""))
    if (
        len(initial_model_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in initial_model_hash.lower()
        )
    ):
        raise CampaignError(f"manifest has no initial-model tensor hash in {run_dir}")
    fingerprint = str(manifest.get("protocol_fingerprint", ""))
    if not fingerprint or str(completion.get("fingerprint", "")) != fingerprint:
        raise CampaignError(f"manifest/completion fingerprint mismatch in {run_dir}")

    source = manifest.get("source_repository")
    if not isinstance(source, dict):
        raise CampaignError(f"manifest has no source_repository object in {run_dir}")
    if source.get("available") is not True or source.get("dirty") is not False:
        raise CampaignError(f"run was not produced from a clean readable Git source: {run_dir}")
    if not str(source.get("commit", "")) or source.get("commit") == "unknown":
        raise CampaignError(f"manifest has no exact source commit in {run_dir}")

    packages = manifest.get("package_versions")
    if not isinstance(packages, dict):
        raise CampaignError(f"manifest has no package_versions object in {run_dir}")
    if str(packages.get("weightwatcher", "")) != PINNED_WEIGHTWATCHER:
        raise CampaignError(
            f"manifest did not use weightwatcher=={PINNED_WEIGHTWATCHER}: {run_dir}"
        )
    if str(packages.get("rg-nanogpt-one-head", "")) != PINNED_PACKAGE_VERSION:
        raise CampaignError(
            f"manifest did not use rg-nanogpt-one-head=={PINNED_PACKAGE_VERSION}: "
            f"{run_dir}"
        )

    runtime = manifest.get("runtime_environment")
    if not isinstance(runtime, dict):
        raise CampaignError(f"manifest has no runtime_environment object in {run_dir}")
    if (
        runtime.get("float32_matmul_precision") != "highest"
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("deterministic_warn_only") is not False
    ):
        raise CampaignError(f"manifest violates deterministic runtime policy in {run_dir}")
    if not str(runtime.get("hardware_block_id", "")).strip() or not str(
        runtime.get("hardware_block_id_source", "")
    ).strip():
        raise CampaignError(f"manifest has no hardware-block identity in {run_dir}")
    if runtime.get("accelerator") == "cuda" and (
        runtime.get("cuda_matmul_allow_tf32") is not False
        or runtime.get("cudnn_allow_tf32") is not False
    ):
        raise CampaignError(f"manifest enabled CUDA TF32 in {run_dir}")

    if expected_context is None:
        return
    cfg = expected_context["cfg"]
    expected_mappings = {
        "protocol": cfg["protocol"],
        "model": cfg["model"],
        "training": cfg["training"],
        "evaluation": cfg["evaluation"],
        "weightwatcher": cfg["weightwatcher"],
    }
    expected_profile = dict(cfg["optimizer_profiles"][optimizer])
    expected_profile["name"] = optimizer
    expected_mappings["optimizer_profile"] = expected_profile
    for key, expected in expected_mappings.items():
        if _canonical_json(manifest.get(key)) != _canonical_json(expected):
            raise CampaignError(f"manifest {key} differs from frozen config in {run_dir}")
    if str(manifest.get("config_sha256", "")) != str(
        expected_context["config_sha256"]
    ):
        raise CampaignError(f"manifest config SHA-256 mismatch in {run_dir}")
    if int(manifest.get("max_steps", -1)) != int(expected_context["total_steps"]):
        raise CampaignError(f"manifest optimizer-step horizon mismatch in {run_dir}")
    tokens_per_step = (
        int(cfg["training"]["batch_size"])
        * int(cfg["training"]["grad_accum_steps"])
        * int(cfg["model"]["block_size"])
    )
    if int(manifest.get("tokens_per_step", -1)) != tokens_per_step:
        raise CampaignError(f"manifest tokens_per_step mismatch in {run_dir}")
    if str(source.get("commit", "")) != str(expected_context["git_commit"]):
        raise CampaignError(
            f"run source commit differs from the checked-out commit in {run_dir}"
        )
    expected_data = expected_context.get("data_metadata")
    if expected_data is not None and _canonical_json(
        manifest.get("data_metadata")
    ) != _canonical_json(expected_data):
        raise CampaignError(f"manifest data metadata/hash inventory mismatch in {run_dir}")
    expected_dependencies = expected_context.get("dependencies")
    if isinstance(expected_dependencies, Mapping) and _canonical_json(
        packages
    ) != _canonical_json(expected_dependencies):
        expected_names = set(expected_dependencies)
        observed_names = set(packages)
        mismatched = sorted(
            name
            for name in expected_names & observed_names
            if str(expected_dependencies[name]) != str(packages[name])
        )
        raise CampaignError(
            "manifest dependency closure differs from the current production "
            f"environment in {run_dir}; missing={sorted(expected_names - observed_names)[:10]}, "
            f"extra={sorted(observed_names - expected_names)[:10]}, "
            f"version_mismatches={mismatched[:10]}"
        )
    expected_runtime = expected_context.get("runtime")
    if expected_runtime is not None and _canonical_json(
        _runtime_block_identity(runtime)
    ) != _canonical_json(_runtime_block_identity(expected_runtime)):
        raise CampaignError(
            f"manifest runtime/hardware differs from the requested runtime in {run_dir}"
        )


def _validate_completed_run(
    run_dir: Path,
    optimizer: str,
    seed: int,
    *,
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failure_marker = run_dir / "run_failed.json"
    if failure_marker.exists():
        raise CampaignError(f"failure marker exists: {failure_marker}")
    missing = [
        relative
        for relative in REQUIRED_RUN_FILES
        if not (run_dir / relative).is_file() or (run_dir / relative).stat().st_size == 0
    ]
    if missing:
        raise CampaignError(f"{run_dir} lacks required artifacts: {', '.join(missing)}")
    completion = _read_json(run_dir / "run_complete.json")
    manifest = _read_json(run_dir / "manifest.json")
    if completion.get("completed") is not True:
        raise CampaignError(f"{run_dir}/run_complete.json does not declare completed=true")
    try:
        recorded_seed = int(completion.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"completion seed is invalid in {run_dir}") from exc
    if completion.get("optimizer") != optimizer or recorded_seed != seed:
        raise CampaignError(f"completion identity mismatch for {optimizer}/seed_{seed}")
    _validate_manifest_binding(
        manifest,
        completion,
        optimizer=optimizer,
        seed=seed,
        expected_context=expected_context,
        run_dir=run_dir,
    )

    epoch_path = run_dir / "epoch_metrics.csv"
    metrics_path = run_dir / "metrics.csv"
    spectral_path = run_dir / "spectral" / "layers.csv"
    epoch_columns = _read_csv_header(epoch_path)
    metric_columns = _read_csv_header(metrics_path)
    spectral_columns = _read_csv_header(spectral_path)
    required_metrics = {
        f"{split}_{metric}"
        for split in ("train", "val", "test")
        for metric in (
            "loss",
            "perplexity",
            "bits_per_token",
            "accuracy",
            "top5_accuracy",
        )
    }
    required_metrics.update({
        "test_bleu",
        "test_continuation_token_accuracy",
        "test_continuation_exact_match",
    })
    for label, columns in (
        (str(metrics_path), metric_columns),
        (str(epoch_path), epoch_columns),
    ):
        missing_metrics = required_metrics.difference(columns)
        if missing_metrics:
            raise CampaignError(
                f"{label} lacks required campaign metrics: "
                + ", ".join(sorted(missing_metrics))
            )
    required_spectral = {
        "step", "matrix_name", "alpha", "raw_alpha", "alpha_raw",
        "alpha_clip_xmax", "alpha_delta", "num_fingers", "finger_policy",
        "primary_alpha_variant", "weightwatcher_analysis_calls",
    }
    missing_spectral = required_spectral.difference(spectral_columns)
    if missing_spectral:
        raise CampaignError(
            f"{spectral_path} lacks one-pass WeightWatcher columns: "
            + ", ".join(sorted(missing_spectral))
        )
    if "checkpoint_path" not in epoch_columns:
        raise CampaignError(f"{epoch_path} lacks checkpoint_path")
    if "test_held_out" not in epoch_columns:
        raise CampaignError(f"{epoch_path} lacks test_held_out policy markers")

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    if not metric_rows:
        raise CampaignError(f"{metrics_path} has no metric rows")
    with epoch_path.open("r", encoding="utf-8", newline="") as handle:
        epoch_rows = list(csv.DictReader(handle))
    held_out_curve_columns = (
        "test_loss",
        "test_perplexity",
        "test_bits_per_token",
        "test_accuracy",
        "test_top5_accuracy",
        "test_bleu",
        "test_continuation_token_accuracy",
        "test_continuation_exact_match",
        "test_generalization_gap",
    )
    for label, rows in ((metrics_path, metric_rows), (epoch_path, epoch_rows)):
        for row in rows:
            for column in held_out_curve_columns:
                raw = str(row.get(column, "")).strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise CampaignError(
                        f"invalid held-out placeholder {column} in {label}"
                    ) from exc
                if not math.isnan(value):
                    raise CampaignError(
                        f"{label} leaks held-out {column} into a training curve"
                    )
    if any(str(row.get("test_held_out", "")).strip() not in {"1", "1.0"} for row in epoch_rows):
        raise CampaignError(f"{epoch_path} does not mark every test curve held out")
    if len(epoch_rows) < MINIMUM_PERMANENT_STATES:
        raise CampaignError(
            f"{run_dir} has {len(epoch_rows)} permanent states; "
            f"at least {MINIMUM_PERMANENT_STATES} are required"
        )
    checkpoint_paths: set[Path] = set()
    for row in epoch_rows:
        recorded = Path(str(row.get("checkpoint_path", "")))
        candidate = recorded if recorded.is_file() else run_dir / "epoch_checkpoints" / recorded.name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise CampaignError(f"missing permanent checkpoint: {recorded}")
        checkpoint_paths.add(candidate.resolve())
    if len(checkpoint_paths) < MINIMUM_PERMANENT_STATES:
        raise CampaignError(f"{run_dir} has fewer than ten distinct permanent checkpoints")

    with spectral_path.open("r", encoding="utf-8", newline="") as handle:
        layer_rows = list(csv.DictReader(handle))
    by_step: dict[int, set[str]] = {}
    rows_by_step: dict[int, int] = {}
    for row in layer_rows:
        try:
            step_value = float(row.get("step", ""))
            step = int(step_value)
        except (TypeError, ValueError) as exc:
            raise CampaignError(f"invalid WeightWatcher step in {spectral_path}") from exc
        if not math.isfinite(step_value) or step_value != step:
            raise CampaignError(f"non-integer WeightWatcher step in {spectral_path}")
        by_step.setdefault(step, set()).add(str(row.get("matrix_name", "")))
        rows_by_step[step] = rows_by_step.get(step, 0) + 1
        if row.get("finger_policy") != "fix_fingers=clip_xmax":
            raise CampaignError(f"incorrect WeightWatcher finger policy in {spectral_path}")
        if row.get("primary_alpha_variant") != "clip_xmax":
            raise CampaignError(f"incorrect primary alpha variant in {spectral_path}")
        try:
            alpha_alias = float(row["alpha"])
            raw_alias = float(row["raw_alpha"])
            alpha = float(row["alpha_clip_xmax"])
            raw_alpha = float(row["alpha_raw"])
            alpha_delta = float(row["alpha_delta"])
            num_fingers = float(row["num_fingers"])
            calls = int(float(row["weightwatcher_analysis_calls"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignError(f"invalid WeightWatcher row in {spectral_path}") from exc
        values = (alpha_alias, raw_alias, alpha, raw_alpha, alpha_delta, num_fingers)
        if not all(math.isfinite(value) for value in values) or calls != 1:
            raise CampaignError(f"non-finite alpha or repeated WeightWatcher call in {spectral_path}")
        if alpha_alias != alpha or raw_alias != raw_alpha:
            raise CampaignError(f"WeightWatcher alpha aliases disagree in {spectral_path}")
        if not math.isclose(
            alpha_delta, raw_alpha - alpha, rel_tol=1e-12, abs_tol=1e-12
        ) or num_fingers < 0:
            raise CampaignError(f"invalid WeightWatcher finger correction in {spectral_path}")
    try:
        epoch_step_values = [float(row.get("step", "")) for row in epoch_rows]
        epoch_steps = {int(value) for value in epoch_step_values}
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"invalid permanent checkpoint step in {epoch_path}") from exc
    if any(
        not math.isfinite(value) or value != int(value)
        for value in epoch_step_values
    ) or len(epoch_steps) != len(epoch_rows):
        raise CampaignError(f"duplicate or non-integer permanent steps in {epoch_path}")
    if expected_context is not None:
        expected_steps = set(int(value) for value in expected_context["permanent_steps"])
        if epoch_steps != expected_steps:
            raise CampaignError(
                f"permanent checkpoint grid differs from the frozen 17-state grid in {run_dir}"
            )
    if set(by_step) != epoch_steps:
        raise CampaignError(f"WeightWatcher steps do not match permanent states in {run_dir}")
    if any(
        len(matrices) != EXPECTED_MATRICES or rows_by_step[step] != EXPECTED_MATRICES
        for step, matrices in by_step.items()
    ):
        raise CampaignError(f"not every WeightWatcher state has six matrices in {run_dir}")

    status_files = list((run_dir / "spectral").glob("status_step_*.json"))
    expected_status = {
        run_dir / "spectral" / f"status_step_{step:07d}.json"
        for step in epoch_steps
    }
    if set(status_files) != expected_status:
        raise CampaignError(f"WeightWatcher completion-record count mismatch in {run_dir}")
    for path in status_files:
        status = _read_json(path)
        try:
            calls = int(status.get("weightwatcher_analysis_calls", -1))
        except (TypeError, ValueError) as exc:
            raise CampaignError(
                f"invalid WeightWatcher call count in status: {path}"
            ) from exc
        if status.get("completed") is not True or calls != 1:
            raise CampaignError(f"incomplete WeightWatcher status: {path}")

    test_results = _read_json(run_dir / "test_results.json")
    policy = str(test_results.get("policy", "")).lower()
    if "held out" not in policy or "validation" not in policy or "never" not in policy:
        raise CampaignError(
            f"test_results.json does not declare the protected test policy in {run_dir}"
        )
    required_test_fields = {
        "step",
        "loss",
        "perplexity",
        "bits_per_token",
        "accuracy",
        "top5_accuracy",
        "bleu",
        "continuation_token_accuracy",
        "continuation_exact_match",
    }
    parsed_test_results: dict[str, dict[str, float]] = {}
    for checkpoint in ("final", "validation_selected"):
        values = test_results.get(checkpoint)
        if not isinstance(values, dict):
            raise CampaignError(
                f"test_results.json lacks object {checkpoint!r} in {run_dir}"
            )
        missing_test = required_test_fields.difference(values)
        if missing_test:
            raise CampaignError(
                f"test_results.json {checkpoint} lacks fields: "
                + ", ".join(sorted(missing_test))
            )
        try:
            numeric_test = [float(values[field]) for field in required_test_fields]
        except (TypeError, ValueError) as exc:
            raise CampaignError(
                f"test_results.json {checkpoint} contains nonnumeric metrics"
            ) from exc
        if not all(math.isfinite(value) for value in numeric_test):
            raise CampaignError(
                f"test_results.json {checkpoint} contains non-finite metrics"
            )
        parsed = {field: float(values[field]) for field in required_test_fields}
        if parsed["loss"] < 0.0 or parsed["bits_per_token"] < 0.0:
            raise CampaignError(f"test_results.json {checkpoint} has negative NLL/bits")
        if parsed["perplexity"] <= 0.0 or not math.isclose(
            math.log(parsed["perplexity"]),
            parsed["loss"],
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise CampaignError(
                f"test_results.json {checkpoint} perplexity is inconsistent with loss"
            )
        if not math.isclose(
            parsed["bits_per_token"] * math.log(2.0),
            parsed["loss"],
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise CampaignError(
                f"test_results.json {checkpoint} bits/token is inconsistent with loss"
            )
        bounded = (
            "accuracy",
            "top5_accuracy",
            "continuation_token_accuracy",
            "continuation_exact_match",
        )
        if any(not 0.0 <= parsed[field] <= 1.0 for field in bounded):
            raise CampaignError(
                f"test_results.json {checkpoint} has an accuracy outside [0, 1]"
            )
        if (
            parsed["top5_accuracy"] < parsed["accuracy"]
            or parsed["continuation_exact_match"]
            > parsed["continuation_token_accuracy"] + 1e-12
            or not 0.0 <= parsed["bleu"] <= 100.0
        ):
            raise CampaignError(
                f"test_results.json {checkpoint} violates metric bounds/order"
            )
        parsed_test_results[checkpoint] = parsed
    try:
        total_steps = int(completion["optimizer_steps"])
        selected_step = int(completion["best_validation_step"])
        final_test_step = int(test_results["final"]["step"])
        selected_test_step = int(test_results["validation_selected"]["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"completion/test checkpoint steps are invalid in {run_dir}") from exc
    if expected_context is not None and total_steps != int(
        expected_context["total_steps"]
    ):
        raise CampaignError(f"optimizer-step horizon differs from frozen config in {run_dir}")
    try:
        metric_step_values = [float(row.get("step", "")) for row in metric_rows]
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"invalid metric step in {metrics_path}") from exc
    if (
        any(not math.isfinite(value) or value != int(value) for value in metric_step_values)
        or 0 not in metric_step_values
        or float(total_steps) not in metric_step_values
        or max(metric_step_values) != float(total_steps)
    ):
        raise CampaignError(f"metrics do not span step zero through {total_steps} in {run_dir}")
    if final_test_step != total_steps or selected_test_step != selected_step:
        raise CampaignError(
            f"final or validation-selected test checkpoint is inconsistent in {run_dir}"
        )
    final_completion_fields = {
        "loss": "final_test_loss",
        "perplexity": "final_test_perplexity",
        "bits_per_token": "final_test_bits_per_token",
        "accuracy": "final_test_accuracy",
        "top5_accuracy": "final_test_top5_accuracy",
        "bleu": "final_test_bleu",
        "continuation_token_accuracy": "final_test_continuation_token_accuracy",
        "continuation_exact_match": "final_test_continuation_exact_match",
    }
    for metric, completion_key in final_completion_fields.items():
        try:
            recorded = float(completion[completion_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignError(f"completion lacks numeric {completion_key} in {run_dir}") from exc
        if not math.isfinite(recorded) or not math.isclose(
            recorded,
            parsed_test_results["final"][metric],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise CampaignError(
                f"completion {completion_key} disagrees with final test results in {run_dir}"
            )
    try:
        validation_rows = [
            (int(float(row["step"])), float(row["val_loss"])) for row in metric_rows
        ]
        recorded_best_loss = float(completion["best_validation_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"validation-selection metadata is invalid in {run_dir}") from exc
    if not validation_rows or not all(math.isfinite(loss) for _, loss in validation_rows):
        raise CampaignError(f"metrics.csv has non-finite validation loss in {run_dir}")
    observed_best_step, observed_best_loss = min(validation_rows, key=lambda item: item[1])
    if observed_best_step != selected_step or not math.isclose(
        observed_best_loss,
        recorded_best_loss,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise CampaignError(
            f"validation-selected checkpoint does not match metrics.csv in {run_dir}"
        )

    # The package validator torch-loads every standard and permanent
    # checkpoint, checks its embedded run identity and step/epoch metadata,
    # and verifies the exact MuonClip QK interval grid. Keep the launcher
    # import-free at startup, but never declare a production run complete from
    # filenames and CSVs alone.
    try:
        from rg_nanogpt_one_head.completion import (
            CompletedRunValidationError,
            validate_completed_run,
        )
    except ImportError as exc:
        raise CampaignError(
            "cannot validate checkpoint payloads; install the dated nanoGPT "
            "package and its PyTorch dependencies"
        ) from exc
    try:
        validate_completed_run(
            run_dir,
            expected_fingerprint=str(completion["fingerprint"]),
            expected_optimizer=optimizer,
            expected_seed=seed,
            expected_total_steps=total_steps,
            verify_checkpoints=True,
        )
    except CompletedRunValidationError as exc:
        raise CampaignError(str(exc)) from exc
    return completion


def _status_rows(
    paths: Mapping[str, Path],
    optimizers: Sequence[str],
    seeds: Sequence[int],
    *,
    expected_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for optimizer in optimizers:
        for seed in seeds:
            run_dir = _run_dir(paths, optimizer, seed)
            status = "missing"
            detail = "run directory does not exist"
            if (run_dir / "run_failed.json").is_file():
                status = "failed"
                detail = "run_failed.json exists"
            elif run_dir.exists():
                try:
                    completion = _validate_completed_run(
                        run_dir,
                        optimizer,
                        seed,
                        expected_context=expected_context,
                    )
                except CampaignError as exc:
                    status = "incomplete"
                    detail = str(exc)
                else:
                    status = "complete"
                    detail = f"steps={completion.get('optimizer_steps', 'unknown')}"
            rows.append({
                "optimizer": optimizer,
                "seed": seed,
                "status": status,
                "detail": detail,
                "run_dir": str(run_dir),
            })
    return rows


def _preflight_existing_campaign(
    paths: Mapping[str, Path],
    *,
    expected_context: Mapping[str, Any],
) -> None:
    """Reject a split-provenance campaign before launching another replicate."""

    ignored_pre_manifest_names = {"muonclip_walk_location.json"}
    for optimizer in CANONICAL_OPTIMIZERS:
        for seed in CANONICAL_SEEDS:
            run_dir = _run_dir(paths, optimizer, seed)
            if not run_dir.is_dir():
                continue
            substantive = [
                path for path in run_dir.iterdir()
                if path.name not in ignored_pre_manifest_names
            ]
            if not substantive:
                continue
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                raise CampaignError(
                    "existing campaign artifacts have no manifest and cannot be "
                    f"safely resumed: {run_dir}"
                )
            manifest = _read_json(manifest_path)
            completion_path = run_dir / "run_complete.json"
            if completion_path.is_file():
                _validate_completed_run(
                    run_dir,
                    optimizer,
                    seed,
                    expected_context=expected_context,
                )
            else:
                completion = {"fingerprint": manifest.get("protocol_fingerprint")}
                _validate_manifest_binding(
                    manifest,
                    completion,
                    optimizer=optimizer,
                    seed=seed,
                    expected_context=expected_context,
                    run_dir=run_dir,
                )


def _write_provenance(
    paths: Mapping[str, Path], config: Path, child_env: Mapping[str, str]
) -> dict[str, Any]:
    path_names = (
        EXPERIMENT_ROOT_ENV, "RG_NANOGPT_ONE_HEAD_DATA_ROOT",
        "RG_NANOGPT_ONE_HEAD_RESULTS_ROOT", "RG_NANOGPT_ONE_HEAD_PLOTS_ROOT",
        "HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE",
        "HF_ASSETS_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE",
        "TIKTOKEN_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR", "CUPY_CACHE_DIR", "XLA_PERSISTENT_CACHE_PATH",
        "PIP_CACHE_DIR", "UV_CACHE_DIR", "NUMBA_CACHE_DIR",
        "JOBLIB_TEMP_FOLDER", "KERAS_HOME", "SACREBLEU",
        "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR",
        "PYTHONPYCACHEPREFIX",
        "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "JUPYTER_RUNTIME_DIR",
        "IPYTHONDIR", "TMPDIR", "TMP", "TEMP",
        "CONDA_PREFIX", "PYTHONPATH", "HOME",
    )
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            cwd=REPOSITORY_ROOT,
            env=dict(child_env),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise CampaignError(f"could not capture pip freeze: {exc}") from exc
    freeze_path = paths["provenance"] / "pip_freeze.txt"
    _atomic_text(freeze_path, freeze)
    conda_record: dict[str, Any] | None = None
    conda_prefix = str(child_env.get("CONDA_PREFIX", "")).strip()
    if conda_prefix:
        conda_executable = shutil.which("conda", path=child_env.get("PATH"))
        if conda_executable is None:
            raise CampaignError(
                "CONDA_PREFIX is set but the conda executable is unavailable; "
                "cannot capture an exact conda replay lock"
            )
        try:
            explicit = subprocess.run(
                [conda_executable, "list", "--explicit", "--prefix", conda_prefix],
                cwd=REPOSITORY_ROOT,
                env=dict(child_env),
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            ).stdout
            conda_json_text = subprocess.run(
                [conda_executable, "list", "--json", "--prefix", conda_prefix],
                cwd=REPOSITORY_ROOT,
                env=dict(child_env),
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            ).stdout
            conda_packages = json.loads(conda_json_text)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise CampaignError(f"could not capture exact conda locks: {exc}") from exc
        if not explicit.lstrip().startswith("# This file may be used to create"):
            raise CampaignError("conda explicit lock has an unexpected format")
        if not isinstance(conda_packages, list):
            raise CampaignError("conda package inventory is not a JSON list")
        explicit_path = paths["provenance"] / "conda_explicit.txt"
        packages_path = paths["provenance"] / "conda_packages.json"
        _atomic_text(explicit_path, explicit)
        _atomic_text(
            packages_path,
            json.dumps(conda_packages, indent=2, sort_keys=True) + "\n",
        )
        conda_record = {
            "prefix": conda_prefix,
            "explicit_path": str(explicit_path),
            "explicit_sha256": _sha256(explicit_path),
            "packages_path": str(packages_path),
            "packages_sha256": _sha256(packages_path),
        }
    payload = {
        "schema_version": 1,
        "captured_at_utc": _utc_now(),
        "campaign_id": "nanogpt_one_head_2026_08_21_baseline_v3",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "git": _git_provenance(),
        "config": {"path": str(config), "sha256": _sha256(config)},
        "launcher": {"path": str(SCRIPT_PATH), "sha256": _sha256(SCRIPT_PATH)},
        "dependencies": _dependency_versions(),
        "pip_freeze": {
            "path": str(freeze_path),
            "sha256": _sha256(freeze_path),
            "line_count": len(freeze.splitlines()),
        },
        "conda": conda_record,
        "paths": {name: child_env.get(name) for name in path_names},
        "runtime_environment": {
            name: child_env.get(name)
            for name in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "CUBLAS_WORKSPACE_CONFIG",
                "MPLBACKEND",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }
    _atomic_json(paths["provenance"] / "provenance.json", payload)
    return payload


def _command_doctor(
    args: argparse.Namespace,
    root: Path,
    paths: Mapping[str, Path],
    child_env: Mapping[str, str],
) -> int:
    config = _resolve_config(args.config)
    _require_clean_git()
    cfg = _validate_protocol_config(config)
    versions = _require_dependency_contract()

    smoke_imports = ", ".join(DEPENDENCIES.values())
    import_command = [
        sys.executable,
        "-u",
        "-c",
        f"import {smoke_imports}; import rg_nanogpt_one_head; "
        "print('scientific imports: OK')",
    ]
    log_path = paths["logs"] / "doctor.log"
    if _stream_command(import_command, log_path=log_path, environment=child_env) != 0:
        raise CampaignError(f"dependency import smoke test failed; inspect {log_path}")
    backend_summary_path = paths["provenance"] / "doctor_backend_smoke.json"
    backend_summary_path.unlink(missing_ok=True)
    backend_command = [
        sys.executable,
        "-u",
        str(
            NANOGPT_ROOT
            / "src"
            / "rg_nanogpt_one_head"
            / "doctor_smoke.py"
        ),
        "--config",
        str(config),
        "--work-dir",
        str(paths["tmp"] / "doctor-smoke"),
        "--summary",
        str(backend_summary_path),
        "--device",
        args.device,
    ]
    if _stream_command(backend_command, log_path=log_path, environment=child_env) != 0:
        raise CampaignError(f"backend smoke test failed; inspect {log_path}")
    backend_summary = _read_json(backend_summary_path)
    if backend_summary.get("completed") is not True:
        raise CampaignError("backend smoke test did not declare completion")
    optimizer_smokes = backend_summary.get("optimizers")
    if (
        not isinstance(optimizer_smokes, list)
        or not all(isinstance(item, dict) for item in optimizer_smokes)
        or tuple(item.get("optimizer") for item in optimizer_smokes)
        != CANONICAL_OPTIMIZERS
        or not all(
            item.get("checkpoint_roundtrip") is True
            and item.get("resumed_optimizer_step") is True
            for item in optimizer_smokes
        )
    ):
        raise CampaignError("backend smoke optimizer/checkpoint inventory is incomplete")
    ww_smoke = backend_summary.get("weightwatcher")
    if (
        not isinstance(ww_smoke, dict)
        or int(ww_smoke.get("analysis_calls", -1)) != 1
        or int(ww_smoke.get("matrix_count", -1)) != EXPECTED_MATRICES
        or int(ww_smoke.get("raw_alpha_count", -1)) != EXPECTED_MATRICES
        or int(ww_smoke.get("clipped_alpha_count", -1)) != EXPECTED_MATRICES
    ):
        raise CampaignError("backend smoke WeightWatcher inventory is incomplete")

    dataset = cfg.get("dataset", {})
    training = cfg.get("training", {})
    effective_tokens = (
        int(training.get("batch_size", 0))
        * int(training.get("grad_accum_steps", 0))
        * int(cfg.get("model", {}).get("block_size", 0))
    )
    doctor = {
        "schema_version": 1,
        "completed": True,
        "checked_at_utc": _utc_now(),
        "experiment_root": str(root),
        "config": str(config),
        "config_sha256": _sha256(config),
        "device_request": args.device,
        "resolved_device": backend_summary.get("resolved_device"),
        "accelerator": backend_summary.get("accelerator"),
        "runtime": backend_summary.get("runtime"),
        "backend_smoke": {
            "summary_path": str(backend_summary_path),
            "summary_sha256": _sha256(backend_summary_path),
            "optimizer_count": len(optimizer_smokes),
            "weightwatcher_analysis_calls": int(ww_smoke["analysis_calls"]),
            "weightwatcher_matrix_count": int(ww_smoke["matrix_count"]),
        },
        "git": _git_provenance(),
        "dependencies": versions,
        "campaign": {
            "optimizers": list(CANONICAL_OPTIMIZERS),
            "seeds": list(CANONICAL_SEEDS),
            "replicates": EXPECTED_REPLICATES,
            "train_tokens": int(dataset.get("train_tokens", 0)),
            "target_epochs": float(training.get("target_epochs", 0.0)),
            "effective_tokens_per_update": effective_tokens,
        },
    }
    _atomic_json(paths["provenance"] / "doctor.json", doctor)
    _write_provenance(paths, config, child_env)
    print(json.dumps(doctor, indent=2, sort_keys=True))
    return 0


def _require_doctor_gate(
    paths: Mapping[str, Path],
    *,
    config: Path,
    git: Mapping[str, Any],
    dependencies: Mapping[str, str],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a successful smoke test bound to this production runtime."""

    doctor_path = paths["provenance"] / "doctor.json"
    backend_path = paths["provenance"] / "doctor_backend_smoke.json"
    if not doctor_path.is_file() or not backend_path.is_file():
        raise CampaignError(
            "the backend doctor gate has not completed for this campaign root; "
            "run `doctor --device <production-device>` before `run`"
        )
    doctor = _read_json(doctor_path)
    if int(doctor.get("schema_version", -1)) != 1 or doctor.get("completed") is not True:
        raise CampaignError(f"invalid or incomplete doctor gate: {doctor_path}")
    if Path(str(doctor.get("experiment_root", ""))).resolve(strict=False) != paths[
        "root"
    ].resolve(strict=False):
        raise CampaignError("doctor gate belongs to a different campaign root")
    if str(doctor.get("config_sha256", "")) != _sha256(config):
        raise CampaignError(
            "doctor gate used a different frozen config; rerun `doctor`"
        )

    doctor_git = doctor.get("git")
    if not isinstance(doctor_git, Mapping) or (
        doctor_git.get("available") is not True
        or doctor_git.get("clean") is not True
        or doctor_git.get("dirty") is not False
        or str(doctor_git.get("commit", "")) != str(git.get("commit", ""))
        or str(doctor_git.get("origin_url", "")) != str(git.get("origin_url", ""))
    ):
        raise CampaignError(
            "doctor gate is not bound to this clean source commit/origin; "
            "rerun `doctor`"
        )
    if _canonical_json(doctor.get("dependencies")) != _canonical_json(dependencies):
        raise CampaignError(
            "scientific dependencies changed after the doctor gate; rerun `doctor`"
        )

    doctor_runtime = doctor.get("runtime")
    if not isinstance(doctor_runtime, Mapping) or _canonical_json(
        _runtime_block_identity(doctor_runtime)
    ) != _canonical_json(_runtime_block_identity(runtime)):
        raise CampaignError(
            "runtime/hardware block differs from the successful doctor gate; "
            "rerun `doctor --device <production-device>`"
        )
    if (
        str(doctor.get("resolved_device", "")) != str(doctor_runtime.get("device", ""))
        or str(doctor.get("accelerator", ""))
        != str(doctor_runtime.get("accelerator", ""))
    ):
        raise CampaignError("doctor gate has inconsistent resolved-runtime metadata")

    backend_record = doctor.get("backend_smoke")
    if not isinstance(backend_record, Mapping):
        raise CampaignError("doctor gate has no backend-smoke binding")
    recorded_backend_path = Path(
        str(backend_record.get("summary_path", ""))
    ).resolve(strict=False)
    if recorded_backend_path != backend_path.resolve(strict=False):
        raise CampaignError("doctor gate points at an unexpected backend summary")
    if str(backend_record.get("summary_sha256", "")) != _sha256(backend_path):
        raise CampaignError("doctor backend summary changed after the gate completed")

    backend = _read_json(backend_path)
    optimizer_smokes = backend.get("optimizers")
    weightwatcher = backend.get("weightwatcher")
    if (
        int(backend.get("schema_version", -1)) != 1
        or backend.get("completed") is not True
        or _canonical_json(backend.get("runtime"))
        != _canonical_json(doctor_runtime)
        or not isinstance(optimizer_smokes, list)
        or tuple(
            item.get("optimizer") for item in optimizer_smokes if isinstance(item, dict)
        )
        != CANONICAL_OPTIMIZERS
        or not all(
            isinstance(item, dict)
            and item.get("checkpoint_roundtrip") is True
            and item.get("resumed_optimizer_step") is True
            for item in optimizer_smokes
        )
        or not isinstance(weightwatcher, Mapping)
        or int(weightwatcher.get("analysis_calls", -1)) != 1
        or int(weightwatcher.get("matrix_count", -1)) != EXPECTED_MATRICES
        or int(weightwatcher.get("raw_alpha_count", -1)) != EXPECTED_MATRICES
        or int(weightwatcher.get("clipped_alpha_count", -1)) != EXPECTED_MATRICES
    ):
        raise CampaignError("doctor backend summary no longer satisfies the smoke gate")
    return doctor


def _command_verify_lock(
    args: argparse.Namespace,
    root: Path,
    paths: Mapping[str, Path],
    child_env: Mapping[str, str],
) -> int:
    del root, paths, child_env
    lock_path = Path(args.lock).resolve(strict=False)
    if not lock_path.is_file():
        raise CampaignError(f"dependency lock does not exist: {lock_path}")
    lock = _read_json(lock_path)
    if int(lock.get("schema_version", -1)) != 2:
        raise CampaignError(f"unsupported dependency-lock schema: {lock_path}")
    expected_python = str(lock.get("python_version", ""))
    if expected_python != platform.python_version():
        raise CampaignError(
            "Python version differs from the archived lock: "
            f"expected={expected_python!r}, observed={platform.python_version()!r}"
        )
    expected_scientific = lock.get("scientific_packages")
    expected_packages = lock.get("packages")
    if not isinstance(expected_scientific, Mapping) or not isinstance(
        expected_packages, Mapping
    ):
        raise CampaignError(f"dependency lock is incomplete: {lock_path}")
    observed_scientific = _require_dependency_contract()
    if _canonical_json(observed_scientific) != _canonical_json(
        expected_scientific
    ):
        raise CampaignError(
            "scientific dependency map differs from the archived lock"
        )
    observed_lock, _ = _installed_distribution_lock(observed_scientific)
    observed_packages = observed_lock["packages"]
    if _canonical_json(observed_packages) != _canonical_json(expected_packages):
        expected_names = set(expected_packages)
        observed_names = set(observed_packages)
        mismatched = sorted(
            name
            for name in expected_names & observed_names
            if expected_packages[name] != observed_packages[name]
        )
        raise CampaignError(
            "installed distributions differ from the archived lock; "
            f"missing={sorted(expected_names - observed_names)[:10]}, "
            f"extra={sorted(observed_names - expected_names)[:10]}, "
            f"version_mismatches={mismatched[:10]}"
        )
    print(f"Dependency lock verified exactly: {lock_path}")
    return 0


def _command_prepare(
    args: argparse.Namespace,
    root: Path,
    paths: Mapping[str, Path],
    child_env: Mapping[str, str],
) -> int:
    del root
    config = _resolve_config(args.config)
    _require_clean_git()
    cfg = _validate_protocol_config(config)
    command = [
        sys.executable,
        "-u",
        "-m",
        "rg_nanogpt_one_head.data",
        "--config",
        str(config),
        "--output-dir",
        str(paths["data"]),
    ]
    if args.force:
        command.append("--force")
    log_path = paths["logs"] / "prepare.log"
    return_code = _stream_command(command, log_path=log_path, environment=child_env)
    if return_code != 0:
        raise CampaignError(
            f"corpus preparation exited with {return_code}; inspect {log_path}"
        )
    _verified_data_metadata(paths, cfg)
    _write_provenance(paths, config, child_env)
    print(f"Prepared and verified corpus: {paths['data']}")
    return 0


def _require_prepared_data(
    paths: Mapping[str, Path], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    required = [paths["data"] / "meta.json"] + [
        paths["data"] / f"{split}.bin" for split in ("train", "val", "test")
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise CampaignError(
            "prepared corpus is missing or empty; run `prepare` first: "
            + ", ".join(missing)
        )
    return _verified_data_metadata(paths, cfg)


def _command_run(
    args: argparse.Namespace,
    root: Path,
    paths: Mapping[str, Path],
    child_env: Mapping[str, str],
) -> int:
    del root
    config = _resolve_config(args.config)
    git = _require_clean_git()
    cfg = _validate_protocol_config(config)
    dependencies = _require_dependency_contract()
    runtime = _inspect_runtime(config, args.device, child_env)
    _require_doctor_gate(
        paths,
        config=config,
        git=git,
        dependencies=dependencies,
        runtime=runtime,
    )
    data_metadata = _require_prepared_data(paths, cfg)
    expected_context = _run_context(
        cfg=cfg,
        git=git,
        data_metadata=data_metadata,
        dependencies=dependencies,
        runtime=runtime,
    )
    _preflight_existing_campaign(paths, expected_context=expected_context)
    optimizers = _parse_optimizers(args.optimizers)
    seeds = _parse_seeds(args.seeds)
    failures: list[dict[str, Any]] = []
    history = paths["provenance"] / "command_history.jsonl"

    for optimizer in optimizers:
        for seed in seeds:
            run_dir = _run_dir(paths, optimizer, seed)
            try:
                _validate_completed_run(
                    run_dir,
                    optimizer,
                    seed,
                    expected_context=expected_context,
                )
            except CampaignError:
                pass
            else:
                print(f"Already complete and verified: {optimizer}/seed_{seed}")
                continue

            module = (
                "rg_nanogpt_one_head.muonclip"
                if optimizer == "muon_clip"
                else "rg_nanogpt_one_head.training"
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                module,
                "--config",
                str(config),
                "--optimizer",
                optimizer,
                "--seeds",
                str(seed),
                "--data-root",
                str(paths["data"]),
                "--results-root",
                str(paths["results"]),
                "--device",
                args.device,
                "--mps-retries",
                str(args.mps_retries),
                "--fail-fast",
            ]
            log_path = paths["logs"] / "runs" / optimizer / f"seed_{seed}.log"
            started = time.monotonic()
            _append_jsonl(history, {
                "event": "replicate_start",
                "at_utc": _utc_now(),
                "optimizer": optimizer,
                "seed": seed,
                "device_request": args.device,
                "command": command,
                "log": str(log_path),
            })
            replicate_env = dict(child_env)
            replicate_env["RG_NANOGPT_CAMPAIGN_COMMAND"] = shlex.join(command)
            try:
                return_code = _stream_command(
                    command, log_path=log_path, environment=replicate_env
                )
                if return_code != 0:
                    raise CampaignError(f"training process exited with {return_code}")
                _validate_completed_run(
                    run_dir,
                    optimizer,
                    seed,
                    expected_context=expected_context,
                )
            except (CampaignError, KeyboardInterrupt) as exc:
                failure = {
                    "optimizer": optimizer,
                    "seed": seed,
                    "error": str(exc),
                    "log": str(log_path),
                }
                failures.append(failure)
                _append_jsonl(history, {
                    "event": "replicate_end",
                    "at_utc": _utc_now(),
                    "optimizer": optimizer,
                    "seed": seed,
                    "completed": False,
                    "elapsed_seconds": time.monotonic() - started,
                    "error": str(exc),
                })
                print(
                    f"FAILED {optimizer}/seed_{seed}: {exc}; log={log_path}",
                    file=sys.stderr,
                    flush=True,
                )
                if isinstance(exc, KeyboardInterrupt):
                    raise
                if args.stop_on_error:
                    break
            else:
                _append_jsonl(history, {
                    "event": "replicate_end",
                    "at_utc": _utc_now(),
                    "optimizer": optimizer,
                    "seed": seed,
                    "completed": True,
                    "elapsed_seconds": time.monotonic() - started,
                })
                print(f"COMPLETE {optimizer}/seed_{seed}: {run_dir}")
        if failures and args.stop_on_error:
            break

    _write_provenance(paths, config, child_env)
    if failures:
        _atomic_json(paths["provenance"] / "last_run_failures.json", {
            "recorded_at_utc": _utc_now(), "failures": failures
        })
        raise CampaignError(
            f"{len(failures)} requested replicate(s) failed or remained incomplete"
        )
    return 0


def _command_monitor(
    args: argparse.Namespace,
    root: Path,
    paths: Mapping[str, Path],
    child_env: Mapping[str, str],
) -> int:
    del root
    command = [
        sys.executable,
        "-u",
        "-m",
        "rg_nanogpt_one_head.monitor",
        "--results-root",
        str(paths["results"]),
        "--optimizer",
        str(args.optimizer),
        "--seed",
        str(args.seed),
        "--interval",
        str(args.interval),
        "--recent",
        str(args.recent),
    ]
    if args.once:
        command.append("--once")
    if args.no_clear:
        command.append("--no-clear")
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=dict(child_env),
            check=False,
        )
    except OSError as exc:
        raise CampaignError(f"could not start live monitor: {exc}") from exc
    if completed.returncode != 0:
        raise CampaignError(f"live monitor exited with {completed.returncode}")
    return 0


def _print_status_table(rows: Sequence[Mapping[str, Any]]) -> None:
    widths = {
        "optimizer": max(len("optimizer"), *(len(str(row["optimizer"])) for row in rows)),
        "seed": max(len("seed"), *(len(str(row["seed"])) for row in rows)),
        "status": max(len("status"), *(len(str(row["status"])) for row in rows)),
    }
    print(
        f"{'optimizer':<{widths['optimizer']}}  "
        f"{'seed':>{widths['seed']}}  {'status':<{widths['status']}}  detail"
    )
    for row in rows:
        print(
            f"{str(row['optimizer']):<{widths['optimizer']}}  "
            f"{str(row['seed']):>{widths['seed']}}  "
            f"{str(row['status']):<{widths['status']}}  {row['detail']}"
        )


def _command_status(
    args: argparse.Namespace,
    root: Path,
    paths: Mapping[str, Path],
    child_env: Mapping[str, str],
) -> int:
    del root, child_env
    config = _resolve_config(args.config)
    cfg = _validate_protocol_config(config)
    expected_context = _run_context(cfg=cfg, git=_git_provenance())
    optimizers = _parse_optimizers(args.optimizers)
    seeds = _parse_seeds(args.seeds)
    rows = _status_rows(
        paths,
        optimizers,
        seeds,
        expected_context=expected_context,
    )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_status_table(rows)
    incomplete = [row for row in rows if row["status"] != "complete"]
    if incomplete:
        print(
            f"{len(incomplete)}/{len(rows)} requested replicates are not complete",
            file=sys.stderr,
        )
        return 1
    return 0


def _require_exact_campaign(
    paths: Mapping[str, Path],
    *,
    expected_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _status_rows(
        paths,
        CANONICAL_OPTIMIZERS,
        CANONICAL_SEEDS,
        expected_context=expected_context,
    )
    incomplete = [row for row in rows if row["status"] != "complete"]
    if incomplete:
        details = "; ".join(
            f"{row['optimizer']}/seed_{row['seed']}={row['status']}"
            for row in incomplete
        )
        raise CampaignError(
            f"the exact 2 x 5 campaign is not complete ({details})"
        )
    if len(rows) != EXPECTED_REPLICATES:
        raise CampaignError(
            f"campaign inventory contains {len(rows)} runs, expected {EXPECTED_REPLICATES}"
        )
    runtime_reference: str | None = None
    initial_hashes: dict[int, set[str]] = {
        seed: set() for seed in CANONICAL_SEEDS
    }
    for optimizer in CANONICAL_OPTIMIZERS:
        for seed in CANONICAL_SEEDS:
            manifest = _read_json(_run_dir(paths, optimizer, seed) / "manifest.json")
            runtime = manifest.get("runtime_environment")
            if not isinstance(runtime, Mapping):
                raise CampaignError(
                    f"run has no runtime identity: {optimizer}/seed_{seed}"
                )
            serialized = _canonical_json(_runtime_block_identity(runtime))
            initial_hashes[seed].add(str(manifest["initial_model_sha256"]))
            if runtime_reference is None:
                runtime_reference = serialized
            elif serialized != runtime_reference:
                raise CampaignError(
                    "the exact 2 x 5 campaign mixes hardware-block identities; "
                    "run each accelerator as a separate campaign root"
                )
    mismatched_initializations = {
        seed: sorted(values)
        for seed, values in initial_hashes.items()
        if len(values) != 1
    }
    if mismatched_initializations:
        raise CampaignError(
            "optimizer arms do not share identical step-zero tensors by seed: "
            + _canonical_json(mismatched_initializations)
        )
    return rows


def _command_analyze(
    args: argparse.Namespace,
    root: Path,
    paths: Mapping[str, Path],
    child_env: Mapping[str, str],
) -> int:
    del root
    config = _resolve_config(args.config)
    git = _require_clean_git()
    cfg = _validate_protocol_config(config)
    dependencies = _require_dependency_contract()
    data_metadata = _require_prepared_data(paths, cfg)
    expected_context = _run_context(
        cfg=cfg,
        git=git,
        data_metadata=data_metadata,
        dependencies=dependencies,
    )
    require_complete = not bool(args.allow_incomplete)
    if require_complete:
        _require_exact_campaign(paths, expected_context=expected_context)
    else:
        _preflight_existing_campaign(paths, expected_context=expected_context)
        rows = _status_rows(
            paths,
            CANONICAL_OPTIMIZERS,
            CANONICAL_SEEDS,
            expected_context=expected_context,
        )
        complete = [row for row in rows if row["status"] == "complete"]
        completed_optimizers = {str(row["optimizer"]) for row in complete}
        missing_optimizers = set(CANONICAL_OPTIMIZERS).difference(
            completed_optimizers
        )
        if missing_optimizers:
            raise CampaignError(
                "provisional analysis requires at least one complete replicate "
                "from every optimizer; missing: "
                + ", ".join(sorted(missing_optimizers))
            )
        complete_seed_sets = {
            optimizer: {
                int(row["seed"])
                for row in complete
                if row["optimizer"] == optimizer
            }
            for optimizer in CANONICAL_OPTIMIZERS
        }
        common_seeds = set.intersection(*complete_seed_sets.values())
        if not common_seeds:
            raise CampaignError(
                "provisional analysis requires at least one paired seed "
                "completed across both optimizers"
            )

    report_builder = EXPERIMENT_DIR / "scripts" / "build_report.py"
    source_notebook = (
        EXPERIMENT_DIR / "notebooks" / "01_Performance_and_Spectra.ipynb"
    )
    if not report_builder.is_file():
        raise CampaignError(f"report builder is missing: {report_builder}")
    if not source_notebook.is_file():
        raise CampaignError(f"source analysis notebook is missing: {source_notebook}")

    executed_dir = paths["analysis"] / "notebooks"
    executed_dir.mkdir(parents=True, exist_ok=True)
    executed_notebook = executed_dir / "01_Performance_and_Spectra.executed.ipynb"
    command = [
        sys.executable,
        "-u",
        "-m",
        "papermill",
        str(source_notebook),
        str(executed_notebook),
        "--cwd",
        str(REPOSITORY_ROOT),
        "-p",
        "RESULTS_ROOT",
        str(paths["results"]),
        "-p",
        "OUTPUT_ROOT",
        str(paths["analysis"]),
        "-p",
        "REQUIRE_COMPLETE",
        str(require_complete),
    ]
    log_path = paths["logs"] / "analyze.log"
    return_code = _stream_command(command, log_path=log_path, environment=child_env)
    if return_code != 0:
        raise CampaignError(
            f"analysis notebook exited with {return_code}; inspect {log_path}"
        )
    required_outputs = (
        paths["analysis"] / "SUMMARY.md",
        paths["analysis"] / "report.html",
        paths["analysis"] / "results_manifest.json",
        paths["analysis"] / "campaign_runs.csv",
        paths["analysis"] / "performance_summary.csv",
        paths["analysis"] / "paired_seed_differences.csv",
        paths["analysis"] / "spectral_layers_all.csv",
        paths["analysis"] / "alpha_across_seed_summary.csv",
        paths["analysis"] / "saturation_diagnostics.csv",
        executed_notebook,
    )
    missing = [
        str(path)
        for path in required_outputs
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise CampaignError(
            "analysis returned zero but required outputs are missing: "
            + ", ".join(missing)
        )
    for optimizer in CANONICAL_OPTIMIZERS:
        for suffix in (
            "performance",
            "alpha_raw_vs_clip_xmax",
            "alpha_raw_vs_clip_xmax_zoomed",
            "erg_gap_num_traps",
        ):
            figure = paths["plots"] / f"{optimizer}_{suffix}.png"
            if not figure.is_file() or figure.stat().st_size == 0:
                raise CampaignError(f"analysis did not produce required plot: {figure}")

    results_manifest_path = paths["analysis"] / "results_manifest.json"
    results_manifest = _read_json(results_manifest_path)
    results_manifest["executed_notebook"] = {
        "path": str(executed_notebook.relative_to(paths["analysis"])),
        "bytes": executed_notebook.stat().st_size,
        "sha256": _sha256(executed_notebook),
    }
    results_manifest["analysis_finalized_at_utc"] = _utc_now()
    _atomic_json(results_manifest_path, results_manifest)
    _validate_analysis_bundle(
        paths,
        expected_context=expected_context,
        require_complete=require_complete,
    )

    _write_provenance(paths, config, child_env)
    label = (
        "Final analysis report"
        if require_complete
        else "Provisional analysis report"
    )
    print(f"{label}: {paths['analysis'] / 'report.html'}")
    print(f"Executed notebook: {executed_notebook}")
    return 0


def _validated_artifact_inventory(
    records: Any,
    *,
    root: Path,
    label: str,
) -> dict[str, Path]:
    if not isinstance(records, list) or not records:
        raise CampaignError(f"analysis manifest has no {label} inventory")
    inventory: dict[str, Path] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CampaignError(f"analysis manifest {label}[{index}] is not an object")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise CampaignError(f"analysis manifest {label} has unsafe path: {relative}")
        unresolved = root / relative
        path = unresolved.resolve(strict=False)
        if not _within(path, root.resolve(strict=False)):
            raise CampaignError(f"analysis manifest {label} escapes its root: {relative}")
        key = relative.as_posix()
        if key in inventory:
            raise CampaignError(f"analysis manifest {label} duplicates {key}")
        if not path.is_file() or unresolved.is_symlink():
            raise CampaignError(f"analysis manifest {label} artifact is missing: {path}")
        try:
            expected_bytes = int(record.get("bytes", -1))
        except (TypeError, ValueError) as exc:
            raise CampaignError(f"analysis manifest {label} has invalid size: {key}") from exc
        if expected_bytes != path.stat().st_size or str(record.get("sha256", "")) != _sha256(path):
            raise CampaignError(f"analysis manifest {label} hash/size is stale: {key}")
        inventory[key] = path
    return inventory


def _expected_analysis_input_paths(paths: Mapping[str, Path]) -> set[str]:
    expected: set[str] = set()
    results_root = paths["results"].resolve()
    for optimizer in CANONICAL_OPTIMIZERS:
        for seed in CANONICAL_SEEDS:
            run_dir = _run_dir(paths, optimizer, seed)
            for relative in (
                "manifest.json",
                "run_complete.json",
                "metrics.csv",
                "epoch_metrics.csv",
                "spectral/layers.csv",
                "spectral/summary.csv",
                "test_results.json",
            ):
                expected.add(str((run_dir / relative).resolve().relative_to(results_root)))
            for step in _expected_permanent_steps(_load_yaml(DEFAULT_CONFIG)):
                status = run_dir / "spectral" / f"status_step_{step:07d}.json"
                expected.add(str(status.resolve().relative_to(results_root)))
            if optimizer == "muon_clip":
                expected.add(
                    str((run_dir / "muonclip_qk.csv").resolve().relative_to(results_root))
                )

    checkpoint_index = paths["analysis"] / "checkpoint_sha256.csv"
    try:
        with checkpoint_index.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise CampaignError(f"could not read checkpoint integrity index: {exc}") from exc
    if not rows:
        raise CampaignError("checkpoint integrity index is empty")
    for row in rows:
        relative = Path(str(row.get("checkpoint_relative_path", "")))
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise CampaignError(f"checkpoint index has unsafe path: {relative}")
        checkpoint = (results_root / relative).resolve(strict=False)
        if not _within(checkpoint, results_root):
            raise CampaignError(f"checkpoint index escapes results root: {relative}")
        if (
            not checkpoint.is_file()
            or int(row.get("bytes", -1)) != checkpoint.stat().st_size
            or str(row.get("sha256", "")) != _sha256(checkpoint)
        ):
            raise CampaignError(f"checkpoint index is stale for {relative}")
        expected.add(relative.as_posix())
    return expected


def _validate_analysis_bundle(
    paths: Mapping[str, Path],
    *,
    expected_context: Mapping[str, Any],
    require_complete: bool = True,
) -> dict[str, Any]:
    analysis_root = paths["analysis"].resolve()
    results_root = paths["results"].resolve()
    for relative in REQUIRED_ANALYSIS_FILES:
        path = analysis_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise CampaignError(f"analysis bundle lacks required artifact: {path}")
    manifest = _read_json(analysis_root / "results_manifest.json")
    if int(manifest.get("schema_version", -1)) != 2 or manifest.get("campaign") != (
        "nanogpt_one_head_2026_08_21_baseline"
    ):
        raise CampaignError("analysis manifest schema/campaign identity is invalid")
    exact = manifest.get("exact_campaign")
    valid_run_count = int(manifest.get("valid_run_count", -1))
    if not isinstance(exact, Mapping) or (
        exact.get("optimizers") != list(CANONICAL_OPTIMIZERS)
        or exact.get("seeds") != list(CANONICAL_SEEDS)
        or int(exact.get("expected_run_count", -1)) != EXPECTED_REPLICATES
        or bool(exact.get("require_complete")) is not bool(require_complete)
        or exact.get("allow_extra_runs") is not False
        or exact.get("allow_mixed_runtime") is not False
        or valid_run_count < 1
        or valid_run_count > EXPECTED_REPLICATES
        or (require_complete and valid_run_count != EXPECTED_REPLICATES)
    ):
        raise CampaignError(
            "analysis manifest does not match the requested 2 x 5 "
            "completion policy"
        )
    if Path(str(manifest.get("results_root", ""))).resolve(strict=False) != results_root:
        raise CampaignError("analysis manifest points at a different results root")
    if Path(str(manifest.get("output_root", ""))).resolve(strict=False) != analysis_root:
        raise CampaignError("analysis manifest points at a different output root")
    if str(manifest.get("source_git_commit", "")) != str(expected_context["git_commit"]):
        raise CampaignError("analysis manifest source commit is stale")
    frozen = manifest.get("frozen_config")
    if not isinstance(frozen, Mapping) or str(
        frozen.get("canonical_sha256", "")
    ) != str(expected_context["config_sha256"]):
        raise CampaignError("analysis manifest frozen config is stale")
    builder = manifest.get("report_builder")
    report_builder = EXPERIMENT_DIR / "scripts" / "build_report.py"
    if not isinstance(builder, Mapping) or str(builder.get("sha256", "")) != _sha256(
        report_builder
    ):
        raise CampaignError("analysis was generated by a different report builder")

    inputs = _validated_artifact_inventory(
        manifest.get("input_artifacts"), root=results_root, label="input_artifacts"
    )
    if require_complete:
        expected_inputs = _expected_analysis_input_paths(paths)
        if set(inputs) != expected_inputs:
            missing = sorted(expected_inputs.difference(inputs))[:10]
            extra = sorted(set(inputs).difference(expected_inputs))[:10]
            raise CampaignError(
                "analysis input inventory differs from the current campaign; "
                f"missing={missing}, extra={extra}"
            )
    outputs = _validated_artifact_inventory(
        manifest.get("artifacts"), root=analysis_root, label="artifacts"
    )
    required_hashed_outputs = set(REQUIRED_ANALYSIS_FILES).difference(
        {"results_manifest.json", "notebooks/01_Performance_and_Spectra.executed.ipynb"}
    )
    if not required_hashed_outputs.issubset(outputs):
        raise CampaignError("analysis output hash inventory is incomplete")
    notebook = manifest.get("executed_notebook")
    _validated_artifact_inventory(
        [notebook] if isinstance(notebook, Mapping) else None,
        root=analysis_root,
        label="executed_notebook",
    )
    return manifest


def _copy_small_artifact(
    source: Path,
    destination: Path,
    *,
    allowed_source_root: Path,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> None:
    if source.is_symlink():
        raise CampaignError(f"refusing to copy symlink into run record: {source}")
    resolved = source.resolve(strict=True)
    if not _within(resolved, allowed_source_root):
        raise CampaignError(f"run-record source escapes allowed root: {source}")
    if not resolved.is_file():
        raise CampaignError(f"run-record source is not a file: {source}")
    if resolved.suffix in {".pt", ".bin"} or ".partial" in resolved.name:
        raise CampaignError(f"large/raw payload is prohibited in run records: {source}")
    size = resolved.stat().st_size
    if size > maximum_bytes:
        raise CampaignError(
            f"run-record artifact is unexpectedly large ({size} bytes): {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved, destination)


def _command_archive(
    args: argparse.Namespace,
    root: Path,
    paths: Mapping[str, Path],
    child_env: Mapping[str, str],
) -> int:
    del root
    config = _resolve_config(args.config)
    git = _require_clean_git()
    cfg = _validate_protocol_config(config)
    dependencies = _require_dependency_contract()
    data_metadata = _require_prepared_data(paths, cfg)
    expected_context = _run_context(
        cfg=cfg,
        git=git,
        data_metadata=data_metadata,
        dependencies=dependencies,
    )
    _require_exact_campaign(paths, expected_context=expected_context)
    _validate_analysis_bundle(paths, expected_context=expected_context)
    _write_provenance(paths, config, child_env)
    _append_jsonl(paths["provenance"] / "command_history.jsonl", {
        "event": "run_record_snapshot",
        "at_utc": _utc_now(),
        "git_commit": git.get("commit"),
    })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_commit = str(git.get("commit", "unknown"))[:12]
    record_name = f"{stamp}_{short_commit}"
    runs_root = EXPERIMENT_DIR / "runs"
    if not _within(runs_root, EXPERIMENT_DIR):
        raise CampaignError(f"dated runs directory escapes experiment: {runs_root}")
    runs_root.mkdir(parents=True, exist_ok=True)
    target = runs_root / record_name
    if target.exists():
        raise CampaignError(f"refusing to overwrite existing run record: {target}")
    stage = Path(tempfile.mkdtemp(prefix=f".{record_name}.partial-", dir=runs_root))

    try:
        protocol_sources = {
            EXPERIMENT_DIR / "campaign.yaml": Path("protocol/campaign.yaml"),
            config: Path("protocol/baseline.yaml"),
        }
        for source, relative in protocol_sources.items():
            _copy_small_artifact(
                source,
                stage / relative,
                allowed_source_root=REPOSITORY_ROOT,
            )

        for source in sorted(paths["provenance"].rglob("*")):
            if not source.is_file():
                continue
            relative = source.resolve().relative_to(paths["provenance"].resolve())
            _copy_small_artifact(
                source,
                stage / "provenance" / relative,
                allowed_source_root=paths["provenance"],
            )

        dependency_lock, replay_requirements = _installed_distribution_lock(
            dependencies
        )
        _atomic_text(
            stage / "provenance" / "requirements_replay.txt",
            replay_requirements,
        )
        _atomic_json(
            stage / "provenance" / "dependency_lock.json",
            dependency_lock,
        )
        conda_packages_path = stage / "provenance" / "conda_packages.json"
        if conda_packages_path.is_file():
            try:
                conda_packages = json.loads(
                    conda_packages_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignError(
                    f"could not read captured conda package inventory: {exc}"
                ) from exc
            if not isinstance(conda_packages, list):
                raise CampaignError("captured conda package inventory is not a list")
            pip_overlay_names = {
                _normalized_distribution_name(str(record.get("name", "")))
                for record in conda_packages
                if isinstance(record, Mapping)
                and (
                    str(record.get("channel", "")).lower() == "pypi"
                    or str(record.get("build", "")).lower() == "pypi_0"
                    or str(record.get("build_string", "")).lower() == "pypi_0"
                )
            }
            pip_overlay_names.discard(
                _normalized_distribution_name("rg-nanogpt-one-head")
            )
            locked_packages = dependency_lock["packages"]
            missing_overlay = sorted(
                name for name in pip_overlay_names if name not in locked_packages
            )
            if missing_overlay:
                raise CampaignError(
                    "conda lists pip overlays absent from Python metadata: "
                    + ", ".join(missing_overlay[:20])
                )
            overlay_requirements = "\n".join(
                f"{locked_packages[name]['name']}=={locked_packages[name]['version']}"
                for name in sorted(pip_overlay_names)
            )
            if overlay_requirements:
                overlay_requirements += "\n"
            _atomic_text(
                stage / "provenance" / "pip_overlay_replay.txt",
                overlay_requirements,
            )
            environment_replay_lines = [
                "export REPLAY_CONDA_PREFIX=\"$RG_NANOGPT_EXPERIMENT_ROOT/replay-conda\"",
                "conda create --yes --prefix \"$REPLAY_CONDA_PREFIX\" --file \"$RUN_RECORD/provenance/conda_explicit.txt\"",
                "export REPLAY_PYTHON=\"$REPLAY_CONDA_PREFIX/bin/python\"",
                "\"$REPLAY_PYTHON\" -m pip install -r \"$RUN_RECORD/provenance/pip_overlay_replay.txt\"",
            ]
        else:
            environment_replay_lines = [
                "python -m venv \"$RG_NANOGPT_EXPERIMENT_ROOT/replay-venv\"",
                "export REPLAY_PYTHON=\"$RG_NANOGPT_EXPERIMENT_ROOT/replay-venv/bin/python\"",
                "\"$REPLAY_PYTHON\" -m pip install -r \"$RUN_RECORD/provenance/requirements_replay.txt\"",
            ]

        for source in sorted(paths["analysis"].rglob("*")):
            if not source.is_file():
                continue
            relative = source.resolve().relative_to(paths["analysis"].resolve())
            _copy_small_artifact(
                source,
                stage / "analysis" / relative,
                allowed_source_root=paths["analysis"],
            )

        run_manifest_rows: list[dict[str, Any]] = []
        for optimizer in CANONICAL_OPTIMIZERS:
            for seed in CANONICAL_SEEDS:
                run_dir = _run_dir(paths, optimizer, seed)
                manifest = _read_json(run_dir / "manifest.json")
                completion = _read_json(run_dir / "run_complete.json")
                run_manifest_rows.append({
                    "optimizer": optimizer,
                    "seed": seed,
                    "accelerator": manifest.get("runtime_environment", {}).get(
                        "accelerator"
                    ) if isinstance(manifest.get("runtime_environment"), dict) else None,
                    "hardware_block_id": manifest.get(
                        "runtime_environment", {}
                    ).get("hardware_block_id")
                    if isinstance(manifest.get("runtime_environment"), dict)
                    else None,
                    "hardware_block_id_source": manifest.get(
                        "runtime_environment", {}
                    ).get("hardware_block_id_source")
                    if isinstance(manifest.get("runtime_environment"), dict)
                    else None,
                    "torch_version": manifest.get(
                        "torch_version",
                        manifest.get("runtime_environment", {}).get("torch_version")
                        if isinstance(manifest.get("runtime_environment"), dict)
                        else None,
                    ),
                    "optimizer_steps": completion.get("optimizer_steps"),
                    "best_validation_step": completion.get("best_validation_step"),
                })
                for filename in (
                    "manifest.json",
                    "run_complete.json",
                    "test_results.json",
                ):
                    _copy_small_artifact(
                        run_dir / filename,
                        stage / "runs" / optimizer / f"seed_{seed}" / filename,
                        allowed_source_root=run_dir,
                    )

        accelerators = {str(row["accelerator"]) for row in run_manifest_rows}
        if len(accelerators) != 1:
            raise CampaignError(
                "archived campaign does not have one accelerator identity"
            )
        replay_device = next(iter(accelerators))
        if replay_device not in {"cpu", "mps", "cuda", "tpu"}:
            raise CampaignError(
                f"unsupported archived accelerator identity: {replay_device!r}"
            )
        hardware_blocks = {
            (
                str(row["hardware_block_id"]),
                str(row["hardware_block_id_source"]),
            )
            for row in run_manifest_rows
        }
        if len(hardware_blocks) != 1:
            raise CampaignError(
                "archived campaign does not have one hardware-block identity"
            )
        replay_hardware_block, replay_hardware_source = next(
            iter(hardware_blocks)
        )
        hardware_replay_lines = (
            [
                "export RG_NANOGPT_HARDWARE_BLOCK_ID="
                + shlex.quote(replay_hardware_block)
            ]
            if replay_hardware_source == "user"
            else []
        )
        tpu_replay_lines = (
            ["export RG_NANOGPT_ALLOW_EPHEMERAL_TPU_STORAGE=1"]
            if replay_device == "tpu"
            else []
        )
        tag = str(git.get("tag_status", "untagged"))
        record_lines = [
            "# One-head nanoGPT campaign run record",
            "",
            f"- Archived at (UTC): `{_utc_now()}`",
            f"- Source commit: `{git.get('commit', 'unknown')}`",
            f"- Git describe: `{git.get('describe', 'unknown')}`",
            f"- Tag status: `{tag}`",
            f"- Frozen config canonical SHA-256: `{FROZEN_CONFIG_SHA256}`",
            f"- Frozen config file SHA-256: `{_sha256(config)}`",
            "- Design: AdamW / MuonClip × seeds "
            "1337 / 2027 / 4099 / 31415 / 271828",
            "- Test policy: held out until post-training; validation NLL selects checkpoints",
            "- WeightWatcher: one clip_xmax call per state; clipped and raw alpha retained",
            "",
            "## Reproduce",
            "",
            "Start from the repository commit/tag that contains this archive. Copy the",
            "record outside Git before checking out the exact source commit used to run it.",
            "",
            "```bash",
            "export RG_NANOGPT_EXPERIMENT_ROOT=/tmp/rg-nanogpt-one-head-20260821",
            "mkdir -p \"$RG_NANOGPT_EXPERIMENT_ROOT/cache/home\"",
            "export HOME=\"$RG_NANOGPT_EXPERIMENT_ROOT/cache/home\"",
            f"export ARCHIVED_RUN_RECORD='baseline/experiments/nanogpt_one_head_2026_08_21_baseline/runs/{record_name}'",
            "export RUN_RECORD=\"$RG_NANOGPT_EXPERIMENT_ROOT/archived-run-record\"",
            "mkdir -p \"$RUN_RECORD\"",
            "cp -R \"$ARCHIVED_RUN_RECORD\"/. \"$RUN_RECORD\"/",
            f"git checkout {git.get('commit', 'COMMIT')}",
            "mkdir -p \"$RG_NANOGPT_EXPERIMENT_ROOT\"/{cache/{home,pip,xdg/{cache,config,data,state},matplotlib},tmp}",
            "export PIP_CACHE_DIR=\"$RG_NANOGPT_EXPERIMENT_ROOT/cache/pip\"",
            "export XDG_CACHE_HOME=\"$RG_NANOGPT_EXPERIMENT_ROOT/cache/xdg/cache\"",
            "export XDG_CONFIG_HOME=\"$RG_NANOGPT_EXPERIMENT_ROOT/cache/xdg/config\"",
            "export XDG_DATA_HOME=\"$RG_NANOGPT_EXPERIMENT_ROOT/cache/xdg/data\"",
            "export XDG_STATE_HOME=\"$RG_NANOGPT_EXPERIMENT_ROOT/cache/xdg/state\"",
            "export MPLCONFIGDIR=\"$RG_NANOGPT_EXPERIMENT_ROOT/cache/matplotlib\"",
            "export TMPDIR=\"$RG_NANOGPT_EXPERIMENT_ROOT/tmp\"",
            "export PYTORCH_ENABLE_MPS_FALLBACK=1",
            *tpu_replay_lines,
            *hardware_replay_lines,
            *environment_replay_lines,
            "\"$REPLAY_PYTHON\" -m pip install --no-deps -e './baseline/nanogpt_one_head'",
            "\"$REPLAY_PYTHON\" baseline/experiments/nanogpt_one_head_2026_08_21_baseline/scripts/run_experiment.py verify-lock --lock \"$RUN_RECORD/provenance/dependency_lock.json\"",
            f"\"$REPLAY_PYTHON\" baseline/experiments/nanogpt_one_head_2026_08_21_baseline/scripts/run_experiment.py doctor --device {replay_device}",
            "\"$REPLAY_PYTHON\" baseline/experiments/nanogpt_one_head_2026_08_21_baseline/scripts/run_experiment.py prepare",
            f"\"$REPLAY_PYTHON\" baseline/experiments/nanogpt_one_head_2026_08_21_baseline/scripts/run_experiment.py run --device {replay_device}",
            "\"$REPLAY_PYTHON\" baseline/experiments/nanogpt_one_head_2026_08_21_baseline/scripts/run_experiment.py analyze",
            "\"$REPLAY_PYTHON\" baseline/experiments/nanogpt_one_head_2026_08_21_baseline/scripts/run_experiment.py archive",
            "```",
            "",
            "The `analysis/` directory contains aggregate tables, separate optimizer plots,",
            "the HTML/Markdown report, and the executed notebook. `runs/` contains only",
            "lightweight per-replicate manifests and protected test outcomes. Corpus files,",
            "model checkpoints, caches, and raw large data are intentionally excluded.",
            "The raw `pip_freeze.txt` is retained for audit. `requirements_replay.txt`",
            "pins the complete installed campaign dependency closure by metadata version.",
            "Opaque direct/VCS/file origins are rejected instead of being rewritten as a",
            "misleading name/version pin; the project is installed separately with",
            "`--no-deps -e`. A conda run also includes an explicit platform lock and a",
            "pip-overlay lock. Large binary packages are not vendored into Git; preserve",
            "the public package-channel configuration or an external wheelhouse needed to",
            "resolve them. Replay fails closed unless `verify-lock` passes before training.",
            "",
            "## Replicate inventory",
            "",
            "| optimizer | seed | accelerator | torch | steps | best validation step |",
            "|---|---:|---|---|---:|---:|",
        ]
        for row in run_manifest_rows:
            record_lines.append(
                "| {optimizer} | {seed} | {accelerator} | {torch_version} | "
                "{optimizer_steps} | {best_validation_step} |".format(**row)
            )
        _atomic_text(stage / "RUN_RECORD.md", "\n".join(record_lines) + "\n")

        artifact_rows = []
        total_bytes = 0
        for artifact in sorted(stage.rglob("*")):
            if not artifact.is_file():
                continue
            size = artifact.stat().st_size
            total_bytes += size
            artifact_rows.append({
                "path": str(artifact.relative_to(stage)),
                "bytes": size,
                "sha256": _sha256(artifact),
            })
        maximum_total = 256 * 1024 * 1024
        if total_bytes > maximum_total:
            raise CampaignError(
                f"tracked run record would be {total_bytes} bytes; limit is {maximum_total}"
            )
        _atomic_json(stage / "archive_manifest.json", {
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "source_git": git,
            "config_sha256": _sha256(config),
            "config_canonical_sha256": FROZEN_CONFIG_SHA256,
            "file_count_excluding_manifest": len(artifact_rows),
            "total_bytes_excluding_manifest": total_bytes,
            "files": artifact_rows,
            "excluded": [
                "tokenized corpus", "model checkpoints", "cache directories", "training logs"
            ],
        })
        os.replace(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    print(f"Tracked-ready run record: {target}")
    print("This command intentionally made the Git worktree dirty; review and commit the run record.")
    return 0


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"frozen protocol config (default: {DEFAULT_CONFIG})",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run, audit, analyze, and archive the exact dated one-head "
            "nanoGPT AdamW/MuonClip campaign"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate source, dependencies, device, and paths")
    _add_config_argument(doctor)
    doctor.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps", "tpu", "xla"), default="auto"
    )
    doctor.set_defaults(handler=_command_doctor)

    verify_lock = subparsers.add_parser(
        "verify-lock",
        help="compare the active environment with an archived exact dependency lock",
    )
    verify_lock.add_argument(
        "--lock",
        required=True,
        help="path to an archived provenance/dependency_lock.json",
    )
    verify_lock.set_defaults(handler=_command_verify_lock)

    prepare = subparsers.add_parser("prepare", help="download and verify the pinned token corpus")
    _add_config_argument(prepare)
    prepare.add_argument("--force", action="store_true", help="rebuild even if verified data exist")
    prepare.set_defaults(handler=_command_prepare)

    run = subparsers.add_parser("run", help="run or resume requested campaign replicates")
    _add_config_argument(run)
    run.add_argument(
        "--optimizers",
        default=",".join(CANONICAL_OPTIMIZERS),
        help="canonical comma-separated subset; default is both arms",
    )
    run.add_argument(
        "--seeds",
        default=",".join(map(str, CANONICAL_SEEDS)),
        help="canonical comma-separated subset; default is all five",
    )
    run.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps", "tpu", "xla"), default="auto"
    )
    run.add_argument(
        "--mps-retries",
        type=int,
        default=2,
        help="fresh-process MPS resume attempts after failure (default: 2)",
    )
    run.add_argument("--stop-on-error", action="store_true")
    run.set_defaults(handler=_command_run)

    monitor = subparsers.add_parser(
        "monitor",
        help="live-display training plus raw and clip_xmax WeightWatcher alphas",
    )
    monitor.add_argument("--optimizer", choices=CANONICAL_OPTIMIZERS, default="muon_clip")
    monitor.add_argument("--seed", type=int, choices=CANONICAL_SEEDS, default=1337)
    monitor.add_argument("--interval", type=float, default=30.0)
    monitor.add_argument("--recent", type=int, default=8)
    monitor.add_argument("--once", action="store_true")
    monitor.add_argument("--no-clear", action="store_true")
    monitor.set_defaults(handler=_command_monitor)

    status = subparsers.add_parser("status", help="strictly inspect requested replicate artifacts")
    _add_config_argument(status)
    status.add_argument("--optimizers", default=",".join(CANONICAL_OPTIMIZERS))
    status.add_argument("--seeds", default=",".join(map(str, CANONICAL_SEEDS)))
    status.add_argument("--json", action="store_true", help="emit machine-readable status")
    status.set_defaults(handler=_command_status)

    analyze = subparsers.add_parser(
        "analyze",
        help="execute the report notebook (strict 2 x 5 by default)",
    )
    _add_config_argument(analyze)
    analyze.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "build a clearly marked provisional report from completed runs; "
            "archive still requires the exact 2 x 5 campaign"
        ),
    )
    analyze.set_defaults(handler=_command_analyze)

    archive = subparsers.add_parser(
        "archive",
        help="copy a small, check-in-ready run record into the dated experiment",
    )
    _add_config_argument(archive)
    archive.set_defaults(handler=_command_archive)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "mps_retries", 0) < 0:
        parser.error("--mps-retries must be nonnegative")
    if getattr(args, "interval", 1.0) <= 0:
        parser.error("--interval must be positive")
    if getattr(args, "recent", 1) < 1:
        parser.error("--recent must be positive")

    try:
        root = resolve_experiment_root()
    except CampaignError as exc:
        print(f"[campaign] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    paths = _paths(root)
    _create_runtime_directories(paths)
    child_env = _child_environment(root, paths)
    history = paths["provenance"] / "command_history.jsonl"
    invocation_id = hashlib.sha256(
        f"{os.getpid()}:{time.time_ns()}:{' '.join(sys.argv)}".encode("utf-8")
    ).hexdigest()[:16]
    _append_jsonl(history, {
        "event": "command_start",
        "invocation_id": invocation_id,
        "at_utc": _utc_now(),
        "argv": list(sys.argv if argv is None else [SCRIPT_PATH.name, *argv]),
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
        "experiment_root": str(root),
        "git": _git_provenance(),
    })
    exit_code = 1
    error: str | None = None
    try:
        exit_code = int(args.handler(args, root, paths, child_env))
    except CampaignError as exc:
        exit_code = 2
        error = str(exc)
        print(f"[campaign] ERROR: {exc}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        exit_code = 130
        error = "interrupted"
        print("[campaign] interrupted", file=sys.stderr, flush=True)
    finally:
        _append_jsonl(history, {
            "event": "command_end",
            "invocation_id": invocation_id,
            "at_utc": _utc_now(),
            "exit_code": exit_code,
            "error": error,
        })
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

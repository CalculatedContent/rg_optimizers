from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

from .provenance import (
    scientific_dependency_versions,
    source_fingerprint_payload,
)
from .runtime import is_tpu_environment

BASELINE_OPTIMIZERS = ("sgd_momentum", "adamw", "muon")
# ``adam`` is optional in the historical reference YAMLs, but is a first-class
# arm in the dated long-horizon campaign. Keeping it out of
# BASELINE_OPTIMIZERS preserves compatibility with old configs while exposing
# it whenever a profile is declared.
SUPPORTED_OPTIMIZERS = ("sgd_momentum", "adam", "adamw", "muon")
DEFAULT_ROOT = Path("/tmp/rg-nanogpt-one-head")
TPU_ROOT_ENV = "RG_NANOGPT_ONE_HEAD_TPU_ROOT"
TPU_PERSISTENT_ENV = "RG_TPU_PERSISTENT_ROOT"
TPU_EPHEMERAL_ENV = "RG_NANOGPT_ALLOW_EPHEMERAL_TPU_STORAGE"
_TPU_MOUNT_PREFIXES = (
    Path("/mnt/disks"),
    Path("/mnt/hyperdisk"),
    Path("/mnt/persistent"),
)


def _decode_mount_path(value: str) -> Path:
    return Path(
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _mounted_paths() -> tuple[Path, ...]:
    path = Path("/proc/self/mountinfo")
    if not path.is_file():
        return ()
    mounts: list[Path] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 5:
            mount = _decode_mount_path(fields[4])
            if mount.is_absolute():
                mounts.append(mount)
    return tuple(dict.fromkeys(mounts))


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _is_tpu_persistent_mount(path: Path) -> bool:
    return any(
        _within(path, prefix) and path != prefix
        for prefix in _TPU_MOUNT_PREFIXES
    )


def _rank_tpu_mount(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    preference = 0
    if "rg" in name:
        preference -= 4
    if "data" in name:
        preference -= 2
    if "persist" in name:
        preference -= 1
    return preference, len(path.parts), str(path)


def _all_mounts(
    mount_points: Iterable[Path] | None = None,
) -> tuple[Path, ...]:
    source = _mounted_paths() if mount_points is None else mount_points
    return tuple(dict.fromkeys(Path(value) for value in source))


def _tpu_mount_candidates(
    mount_points: Iterable[Path] | None = None,
) -> tuple[Path, ...]:
    mounts = _all_mounts(mount_points)
    candidates = [path for path in mounts if _is_tpu_persistent_mount(path)]
    return tuple(sorted(dict.fromkeys(candidates), key=_rank_tpu_mount))


def _allow_ephemeral_tpu_storage() -> bool:
    return str(os.environ.get(TPU_EPHEMERAL_ENV, "")).lower() in {
        "1", "true", "yes", "on"
    }


def _ensure_under_mount(
    root: Path,
    mounts: tuple[Path, ...],
    *,
    source: str,
) -> Path:
    excluded = (Path("/proc"), Path("/sys"), Path("/dev"), Path("/run"), Path("/boot"))
    eligible = [
        mount
        for mount in mounts
        if mount != Path("/")
        and not any(mount == item or _within(mount, item) for item in excluded)
    ]
    if any(root == mount or _within(root, mount) for mount in eligible):
        return root
    if _allow_ephemeral_tpu_storage():
        return root
    raise RuntimeError(
        f"{source}={root} is not located on a detected TPU persistent mount. "
        "Mount Hyperdisk/Persistent Disk under /mnt/disks or set "
        f"{TPU_EPHEMERAL_ENV}=1 only for a disposable smoke test."
    )


def _default_tpu_root(
    mount_points: Iterable[Path] | None = None,
) -> Path:
    all_mounts = _all_mounts(mount_points)
    candidates = _tpu_mount_candidates(all_mounts)

    exact = os.environ.get(TPU_ROOT_ENV)
    if exact:
        return _ensure_under_mount(Path(exact), all_mounts, source=TPU_ROOT_ENV)

    persistent = os.environ.get(TPU_PERSISTENT_ENV)
    if persistent:
        return _ensure_under_mount(
            Path(persistent) / "rg-nanogpt-one-head",
            all_mounts,
            source=TPU_PERSISTENT_ENV,
        )

    if candidates:
        return candidates[0] / "rg-nanogpt-one-head"
    if _allow_ephemeral_tpu_storage():
        return DEFAULT_ROOT
    raise RuntimeError(
        "A TPU environment was detected, but no persistent data volume was "
        "found. Attach and mount Hyperdisk/Persistent Disk under /mnt/disks, "
        f"or set {TPU_PERSISTENT_ENV} to the mounted volume. Use "
        f"{TPU_EPHEMERAL_ENV}=1 only for a disposable smoke test."
    )


def roots(
    device: str = "auto",
    *,
    mount_points: Iterable[Path] | None = None,
) -> dict[str, Path]:
    root_override = os.environ.get("RG_NANOGPT_ONE_HEAD_ROOT")
    data_override = os.environ.get("RG_NANOGPT_ONE_HEAD_DATA_ROOT")
    results_override = os.environ.get("RG_NANOGPT_ONE_HEAD_RESULTS_ROOT")
    plots_override = os.environ.get("RG_NANOGPT_ONE_HEAD_PLOTS_ROOT")
    tpu_environment = is_tpu_environment(device)
    all_mounts = _all_mounts(mount_points) if tpu_environment else ()

    if root_override:
        root = Path(root_override)
        if tpu_environment:
            root = _ensure_under_mount(
                root,
                all_mounts,
                source="RG_NANOGPT_ONE_HEAD_ROOT",
            )
    elif data_override and results_override:
        # Exact per-purpose paths are an intentional advanced override and may
        # use an organization-specific bind mount outside the conventional TPU
        # mount prefixes.
        try:
            root = Path(os.path.commonpath([data_override, results_override]))
        except ValueError:
            root = Path(results_override).parent
    elif tpu_environment:
        root = _default_tpu_root(all_mounts)
    else:
        root = DEFAULT_ROOT

    return {
        "root": root,
        "data": Path(data_override) if data_override else root / "data",
        "results": Path(results_override) if results_override else root / "results",
        "plots": Path(plots_override) if plots_override else root / "plots",
    }


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("configuration root must be a mapping")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required_sections = (
        "protocol", "dataset", "model", "training", "optimizer_profiles",
        "evaluation", "weightwatcher", "runtime",
    )
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"missing configuration section: {section}")

    if int(cfg["protocol"].get("version", 0)) < 1:
        raise ValueError("protocol.version must be positive")

    model = cfg["model"]
    for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
        if int(model[key]) < 1:
            raise ValueError(f"model.{key} must be positive")
    if int(model["n_head"]) != 1:
        raise ValueError("this experiment is fixed to exactly one attention head")
    if int(model["n_layer"]) != 1:
        raise ValueError("this experiment is fixed to one transformer block")
    if int(model["n_embd"]) % int(model["n_head"]) != 0:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if not 0.0 <= float(model.get("dropout", 0.0)) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    dataset = cfg["dataset"]
    for key in ("train_tokens", "val_tokens", "test_tokens"):
        if int(dataset[key]) <= int(model["block_size"]) + 1:
            raise ValueError(f"dataset.{key} is too small for the context length")

    training = cfg["training"]
    for key in (
        "batch_size", "grad_accum_steps", "target_epochs",
        "eval_interval_steps", "eval_batches", "checkpoint_interval_steps",
        "epoch_interval",
    ):
        if float(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    seeds = [int(seed) for seed in training["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds must contain unique values")
    if float(training["grad_clip"]) < 0:
        raise ValueError("training.grad_clip must be nonnegative")

    profiles = cfg["optimizer_profiles"]
    target_epochs = float(training["target_epochs"])
    # Optional launchers may extend SUPPORTED_OPTIMIZERS at runtime.  Those
    # extensions must not make their profile mandatory in the historical
    # three-optimizer reference config; validate them only when declared.
    for name in BASELINE_OPTIMIZERS:
        if name not in profiles:
            raise ValueError(f"missing optimizer profile: {name}")
    for name in SUPPORTED_OPTIMIZERS:
        if name not in profiles:
            continue
        profile = {**profiles[name], "name": name}
        validate_optimizer_profile(profile)
        schedule_epochs = float(profile.get("lr_schedule_epochs", target_epochs))
        if schedule_epochs > target_epochs:
            raise ValueError(
                f"optimizer_profiles.{name}.lr_schedule_epochs cannot exceed "
                "training.target_epochs"
            )

    evaluation = cfg["evaluation"]
    for key in (
        "bleu_examples", "bleu_prompt_tokens", "bleu_continuation_tokens",
        "bleu_batch_size",
    ):
        if int(evaluation[key]) < 1:
            raise ValueError(f"evaluation.{key} must be positive")
    if (
        int(evaluation["bleu_prompt_tokens"])
        + int(evaluation["bleu_continuation_tokens"])
        > int(model["block_size"])
    ):
        raise ValueError("BLEU prompt plus continuation exceeds context length")

    probe_keys = (
        "train_probe_seed", "validation_probe_seed", "test_probe_seed",
        "bleu_probe_seed",
    )
    probe_seeds = []
    for key in probe_keys:
        if key not in evaluation:
            raise ValueError(f"evaluation.{key} is required")
        value = int(evaluation[key])
        if value < 0:
            raise ValueError(f"evaluation.{key} must be nonnegative")
        probe_seeds.append(value)
    if len(set(probe_seeds)) != len(probe_seeds):
        raise ValueError("evaluation probe seeds must be distinct")

    ww = cfg["weightwatcher"]
    if not bool(ww.get("ERG", False)):
        raise ValueError("WeightWatcher ERG must be enabled")
    if not bool(ww.get("randomize", False)):
        raise ValueError("WeightWatcher randomize must be enabled")
    if int(ww["min_evals"]) < 5:
        raise ValueError("weightwatcher.min_evals must be at least 5")
    finger_policy = ww.get("fix_fingers", False)
    if finger_policy not in (False, "clip_xmax"):
        raise ValueError(
            "weightwatcher.fix_fingers must be false or 'clip_xmax'"
        )
    if finger_policy == "clip_xmax":
        if int(ww.get("max_fingers", 0)) < 1:
            raise ValueError(
                "weightwatcher.max_fingers must be positive when "
                "fix_fingers='clip_xmax'"
            )
        if not bool(ww.get("require_raw_alpha", True)):
            raise ValueError(
                "clip_xmax monitoring must retain WeightWatcher's raw_alpha"
            )

    runtime = cfg["runtime"]
    if str(runtime.get("matmul_precision", "high")) not in {
        "highest",
        "high",
        "medium",
    }:
        raise ValueError(
            "runtime.matmul_precision must be highest, high, or medium"
        )
    if bool(runtime.get("allow_tf32", False)) and str(
        runtime.get("matmul_precision", "high")
    ) == "highest":
        raise ValueError(
            "runtime.allow_tf32 cannot be true when matmul_precision=highest"
        )


def validate_optimizer_profile(profile: dict[str, Any]) -> None:
    family = str(profile.get("family", ""))
    if family not in {"sgd", "adam", "adamw", "muon"}:
        raise ValueError(f"unsupported optimizer family: {family}")
    warmup_fraction = float(profile.get("warmup_fraction", -1.0))
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if str(profile.get("schedule")) != "warmup_cosine":
        raise ValueError("the reference suite requires warmup_cosine schedules")
    if "lr_schedule_epochs" in profile and float(profile["lr_schedule_epochs"]) <= 0:
        raise ValueError("lr_schedule_epochs must be positive")

    if family in {"sgd", "adam", "adamw"}:
        peak = float(profile["learning_rate"])
        floor = float(profile["min_learning_rate"])
        if peak <= 0 or floor < 0 or floor > peak:
            raise ValueError("learning-rate peak/floor values are inconsistent")
    else:
        for peak_key, floor_key in (
            ("matrix_learning_rate", "matrix_min_learning_rate"),
            ("aux_learning_rate", "aux_min_learning_rate"),
        ):
            peak = float(profile[peak_key])
            floor = float(profile[floor_key])
            if peak <= 0 or floor < 0 or floor > peak:
                raise ValueError(f"Muon {peak_key}/{floor_key} values are inconsistent")
        if int(profile["newton_schulz_steps"]) < 1:
            raise ValueError("Muon newton_schulz_steps must be positive")


def optimizer_profile(cfg: dict[str, Any], optimizer: str) -> dict[str, Any]:
    optimizer = str(optimizer).lower()
    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"unsupported optimizer {optimizer!r}; choose from {SUPPORTED_OPTIMIZERS}"
        )
    profile = deepcopy(cfg["optimizer_profiles"][optimizer])
    profile["name"] = optimizer
    validate_optimizer_profile(profile)
    return profile


def canonical_seeds(cfg: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(seed) for seed in cfg["training"]["seeds"])


def tokens_per_step(cfg: dict[str, Any]) -> int:
    return (
        int(cfg["training"]["batch_size"])
        * int(cfg["training"]["grad_accum_steps"])
        * int(cfg["model"]["block_size"])
    )


def _steps_for_epochs(cfg: dict[str, Any], epochs: float, train_tokens: int) -> int:
    target_tokens = float(epochs) * int(train_tokens)
    return max(1, int(math.ceil(target_tokens / tokens_per_step(cfg))))


def max_steps(cfg: dict[str, Any], train_tokens: int | None = None) -> int:
    train_tokens = int(train_tokens or cfg["dataset"]["train_tokens"])
    return _steps_for_epochs(cfg, float(cfg["training"]["target_epochs"]), train_tokens)


def lr_schedule_steps(
    cfg: dict[str, Any],
    profile: dict[str, Any],
    train_tokens: int | None = None,
) -> int:
    """Return the LR horizon, which may be shorter than training."""
    train_tokens = int(train_tokens or cfg["dataset"]["train_tokens"])
    training_epochs = float(cfg["training"]["target_epochs"])
    schedule_epochs = float(profile.get("lr_schedule_epochs", training_epochs))
    if not 0 < schedule_epochs <= training_epochs:
        raise ValueError(
            "lr_schedule_epochs must be positive and cannot exceed training.target_epochs"
        )
    return min(
        max_steps(cfg, train_tokens),
        _steps_for_epochs(cfg, schedule_epochs, train_tokens),
    )


def warmup_steps(profile: dict[str, Any], schedule_steps: int) -> int:
    if schedule_steps < 2:
        return 0
    return min(
        schedule_steps - 1,
        max(1, int(round(schedule_steps * float(profile["warmup_fraction"])))),
    )


def epoch_step_map(
    cfg: dict[str, Any],
    train_tokens: int | None = None,
) -> dict[int, float]:
    """Map preregistered nominal epochs, including epoch zero, to steps."""
    train_tokens = int(train_tokens or cfg["dataset"]["train_tokens"])
    total_steps = max_steps(cfg, train_tokens)
    step_tokens = tokens_per_step(cfg)
    target_epochs = float(cfg["training"]["target_epochs"])
    interval = float(cfg["training"]["epoch_interval"])

    points = [0.0]
    current = interval
    while current < target_epochs - 1e-12:
        points.append(round(current, 12))
        current += interval
    points.append(target_epochs)

    result: dict[int, float] = {}
    for epoch in points:
        if epoch == 0:
            step = 0
        elif math.isclose(epoch, target_epochs, rel_tol=0.0, abs_tol=1e-12):
            step = total_steps
        else:
            step = int(round(epoch * train_tokens / step_tokens))
        result[min(total_steps, max(0, step))] = float(epoch)
    result[total_steps] = target_epochs
    return dict(sorted(result.items()))


def protocol_fingerprint(
    cfg: dict[str, Any],
    *,
    optimizer: str,
    seed: int,
    data_metadata: dict[str, Any],
) -> str:
    payload = {
        "protocol": cfg["protocol"],
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "training": cfg["training"],
        "optimizer_profile": optimizer_profile(cfg, optimizer),
        "evaluation": cfg["evaluation"],
        "weightwatcher": cfg["weightwatcher"],
        "runtime": cfg["runtime"],
        "source": source_fingerprint_payload(),
        "scientific_dependencies": scientific_dependency_versions(),
        "optimizer": str(optimizer),
        "seed": int(seed),
        "data_metadata": data_metadata,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

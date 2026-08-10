from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_OPTIMIZERS = ("sgd_momentum", "adamw", "muon")
DEFAULT_ROOT = Path("/tmp/rg-nanogpt-one-head")


def roots() -> dict[str, Path]:
    root = Path(os.environ.get("RG_NANOGPT_ONE_HEAD_ROOT", DEFAULT_ROOT))
    return {
        "root": root,
        "data": Path(
            os.environ.get("RG_NANOGPT_ONE_HEAD_DATA_ROOT", root / "data")
        ),
        "results": Path(
            os.environ.get("RG_NANOGPT_ONE_HEAD_RESULTS_ROOT", root / "results")
        ),
        "plots": Path(
            os.environ.get("RG_NANOGPT_ONE_HEAD_PLOTS_ROOT", root / "plots")
        ),
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
        "protocol",
        "dataset",
        "model",
        "training",
        "optimizer_profiles",
        "evaluation",
        "weightwatcher",
        "runtime",
    )
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"missing configuration section: {section}")

    protocol = cfg["protocol"]
    if int(protocol.get("version", 0)) < 1:
        raise ValueError("protocol.version must be positive")

    model = cfg["model"]
    for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
        if int(model[key]) < 1:
            raise ValueError(f"model.{key} must be positive")
    if int(model["n_head"]) != 1:
        raise ValueError(
            "this experiment is intentionally fixed to exactly one attention head"
        )
    if int(model["n_layer"]) != 1:
        raise ValueError(
            "this experiment is intentionally fixed to one transformer block"
        )
    if int(model["n_embd"]) % int(model["n_head"]) != 0:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if not 0.0 <= float(model.get("dropout", 0.0)) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    dataset = cfg["dataset"]
    for key in ("train_tokens", "val_tokens", "test_tokens"):
        if int(dataset[key]) <= int(model["block_size"]) + 1:
            raise ValueError(
                f"dataset.{key} is too small for the configured context length"
            )

    training = cfg["training"]
    for key in (
        "batch_size",
        "grad_accum_steps",
        "target_epochs",
        "eval_interval_steps",
        "eval_batches",
        "checkpoint_interval_steps",
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
    for name in SUPPORTED_OPTIMIZERS:
        if name not in profiles:
            raise ValueError(f"missing optimizer profile: {name}")
        validate_optimizer_profile({**profiles[name], "name": name})

    evaluation = cfg["evaluation"]
    for key in (
        "bleu_examples",
        "bleu_prompt_tokens",
        "bleu_continuation_tokens",
        "bleu_batch_size",
    ):
        if int(evaluation[key]) < 1:
            raise ValueError(f"evaluation.{key} must be positive")
    if (
        int(evaluation["bleu_prompt_tokens"])
        + int(evaluation["bleu_continuation_tokens"])
        > int(model["block_size"])
    ):
        raise ValueError(
            "BLEU prompt plus continuation must fit inside model.block_size"
        )

    probe_seed_keys = (
        "train_probe_seed",
        "validation_probe_seed",
        "test_probe_seed",
        "bleu_probe_seed",
    )
    probe_seeds = []
    for key in probe_seed_keys:
        if key not in evaluation:
            raise ValueError(
                f"evaluation.{key} is required so every optimizer and model seed "
                "uses the same fixed evaluation examples"
            )
        value = int(evaluation[key])
        if value < 0:
            raise ValueError(f"evaluation.{key} must be nonnegative")
        probe_seeds.append(value)
    if len(set(probe_seeds)) != len(probe_seeds):
        raise ValueError("evaluation probe seeds must be distinct")

    ww = cfg["weightwatcher"]
    if not bool(ww.get("ERG", False)):
        raise ValueError(
            "WeightWatcher ERG must be enabled for this reference experiment"
        )
    if not bool(ww.get("randomize", False)):
        raise ValueError(
            "WeightWatcher randomize must be enabled to obtain num_traps"
        )
    if int(ww["min_evals"]) < 5:
        raise ValueError("weightwatcher.min_evals must be at least 5")


def validate_optimizer_profile(profile: dict[str, Any]) -> None:
    family = str(profile.get("family", ""))
    if family not in {"sgd", "adamw", "muon"}:
        raise ValueError(f"unsupported optimizer family: {family}")
    warmup_fraction = float(profile.get("warmup_fraction", -1.0))
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if str(profile.get("schedule")) != "warmup_cosine":
        raise ValueError("the reference suite requires warmup_cosine schedules")

    if family in {"sgd", "adamw"}:
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
                raise ValueError(
                    f"Muon {peak_key}/{floor_key} values are inconsistent"
                )
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


def max_steps(cfg: dict[str, Any], train_tokens: int | None = None) -> int:
    import math

    train_tokens = int(train_tokens or cfg["dataset"]["train_tokens"])
    target_tokens = float(cfg["training"]["target_epochs"]) * train_tokens
    return max(1, int(math.ceil(target_tokens / tokens_per_step(cfg))))


def warmup_steps(profile: dict[str, Any], total_steps: int) -> int:
    if total_steps < 2:
        return 0
    return min(
        total_steps - 1,
        max(1, int(round(total_steps * float(profile["warmup_fraction"])))),
    )


def epoch_step_map(
    cfg: dict[str, Any], train_tokens: int | None = None
) -> dict[int, float]:
    """Map preregistered nominal epochs, including epoch zero, to optimizer steps."""

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
        step = (
            0
            if epoch == 0
            else int(round(epoch * train_tokens / step_tokens))
        )
        step = min(total_steps, max(0, step))
        result[step] = float(epoch)
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
        "optimizer": str(optimizer),
        "seed": int(seed),
        "data_metadata": data_metadata,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

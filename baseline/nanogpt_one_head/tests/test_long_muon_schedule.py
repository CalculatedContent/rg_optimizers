from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from rg_nanogpt_one_head.config import (
    epoch_step_map,
    load_config,
    lr_schedule_steps,
    max_steps,
    optimizer_profile,
    validate_config,
    warmup_steps,
)
from rg_nanogpt_one_head.optimizers import cosine_learning_rate

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def reference_config() -> dict:
    return load_config(EXPERIMENT_ROOT / "configs" / "reference.yaml")


def long_muon_config() -> dict:
    return load_config(EXPERIMENT_ROOT / "configs" / "muon_10epochs.yaml")


def test_reference_protocol_retains_matched_one_epoch_schedule() -> None:
    cfg = reference_config()
    profile = optimizer_profile(cfg, "muon")

    assert max_steps(cfg) == 9766
    assert lr_schedule_steps(cfg, profile) == 9766
    assert warmup_steps(profile, 9766) == 488


def test_long_muon_separates_training_and_lr_horizons() -> None:
    cfg = long_muon_config()
    profile = optimizer_profile(cfg, "muon")
    training_steps = max_steps(cfg)
    schedule_steps = lr_schedule_steps(cfg, profile)

    assert cfg["training"]["seeds"] == [1337]
    assert cfg["training"]["target_epochs"] == 10.0
    assert cfg["training"]["epoch_interval"] == 0.25
    assert training_steps == 97657
    assert schedule_steps == 9766
    assert warmup_steps(profile, schedule_steps) == 488

    mapping = epoch_step_map(cfg)
    assert len(mapping) == 41
    assert list(mapping.values())[0] == 0.0
    assert list(mapping.values())[-1] == 10.0
    assert list(mapping.keys())[-1] == training_steps
    assert len(set(mapping.values())) == len(mapping)


def test_long_muon_reaches_floor_at_one_epoch() -> None:
    cfg = long_muon_config()
    profile = optimizer_profile(cfg, "muon")
    schedule_steps = lr_schedule_steps(cfg, profile)
    warmup = warmup_steps(profile, schedule_steps)

    at_end = cosine_learning_rate(
        schedule_steps - 1,
        total_steps=schedule_steps,
        warmup_steps=warmup,
        peak_lr=float(profile["matrix_learning_rate"]),
        min_lr=float(profile["matrix_min_learning_rate"]),
    )

    assert at_end == pytest.approx(0.002)


def test_lr_schedule_cannot_exceed_training_horizon() -> None:
    cfg = deepcopy(long_muon_config())
    cfg["optimizer_profiles"]["muon"]["lr_schedule_epochs"] = 11.0

    with pytest.raises(ValueError, match="cannot exceed"):
        validate_config(cfg)

from copy import deepcopy
from pathlib import Path

import pytest

from rg_nanogpt_muon_hyperball.config import (
    load_config,
    lr_schedule_steps,
    max_steps,
    optimizer_profile,
    validate_config,
    warmup_steps,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "reference.yaml"


def test_reference_config_has_matched_long_muon_arms() -> None:
    cfg = load_config(CONFIG)
    assert cfg["training"]["seeds"] == [1337]
    assert cfg["training"]["target_epochs"] == 10.0
    assert set(cfg["optimizer_profiles"]) == {"muon", "muon_hyperball"}

    total = max_steps(cfg)
    assert total == 97657

    for name in ("muon", "muon_hyperball"):
        profile = optimizer_profile(cfg, name)
        assert lr_schedule_steps(cfg, profile) == 9766
        assert warmup_steps(profile, total) == 488
        assert profile["matrix_learning_rate"] == 0.02
        assert profile["matrix_min_learning_rate"] == 0.002

    hb = optimizer_profile(cfg, "muon_hyperball")
    assert hb["hyperball_relative_radius"] == 0.01


def test_nonpositive_hyperball_radius_is_rejected() -> None:
    cfg = load_config(CONFIG)
    broken = deepcopy(cfg)
    broken["optimizer_profiles"]["muon_hyperball"][
        "hyperball_relative_radius"
    ] = 0.0
    with pytest.raises(ValueError, match="hyperball_relative_radius"):
        validate_config(broken)

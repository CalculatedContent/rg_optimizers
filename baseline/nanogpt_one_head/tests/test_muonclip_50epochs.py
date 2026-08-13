from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.config import (
    epoch_step_map,
    lr_schedule_steps,
    max_steps,
    warmup_steps,
)


def _read(name: str) -> dict:
    return yaml.safe_load(
        (EXPERIMENT_ROOT / "configs" / name).read_text(encoding="utf-8")
    )


def test_50epoch_muon_and_muonclip_are_matched_except_optimizer_profile() -> None:
    muon = _read("muon_50epochs.yaml")
    clip = _read("muonclip_50epochs.yaml")

    for section in (
        "dataset",
        "model",
        "training",
        "evaluation",
        "weightwatcher",
        "runtime",
    ):
        assert clip[section] == muon[section]

    assert muon["training"]["seeds"] == [1337]
    assert muon["training"]["target_epochs"] == 50.0
    assert muon["training"]["epoch_interval"] == 0.25
    assert max_steps(muon) == 488_282
    assert max_steps(clip) == 488_282

    muon_profile = muon["optimizer_profiles"]["muon"]
    clip_profile = clip["optimizer_profiles"]["muon_clip"]

    assert lr_schedule_steps(muon, muon_profile) == 9_766
    assert lr_schedule_steps(clip, clip_profile) == 9_766
    assert warmup_steps(muon_profile, 9_766) == 488
    assert warmup_steps(clip_profile, 9_766) == 500

    muon_epochs = epoch_step_map(muon)
    clip_epochs = epoch_step_map(clip)
    assert muon_epochs == clip_epochs
    assert len(muon_epochs) == 201
    assert list(muon_epochs.values())[0] == 0.0
    assert list(muon_epochs.values())[-1] == 50.0
    assert list(muon_epochs.keys())[-1] == 488_282


def test_50epoch_muonclip_keeps_rms_and_qk_clip_settings() -> None:
    cfg = _read("muonclip_50epochs.yaml")
    profile = cfg["optimizer_profiles"]["muon_clip"]

    assert profile["family"] == "muon_clip"
    assert profile["learning_rate"] == pytest.approx(2e-4)
    assert profile["min_learning_rate"] == pytest.approx(2e-5)
    assert profile["lr_schedule_epochs"] == pytest.approx(1.0)
    assert profile["momentum"] == pytest.approx(0.95)
    assert profile["nesterov"] is False
    assert profile["weight_decay"] == pytest.approx(0.10)
    assert profile["update_rms_scale"] == pytest.approx(0.20)
    assert profile["qk_clip_threshold"] == pytest.approx(100.0)
    assert profile["qk_clip_balance"] == pytest.approx(0.50)
    assert profile["qk_diagnostics_interval"] == 250


def test_50epoch_pair_has_identical_weightwatcher_contract() -> None:
    muon = _read("muon_50epochs.yaml")
    clip = _read("muonclip_50epochs.yaml")

    assert muon["weightwatcher"] == clip["weightwatcher"] == {
        "enabled": True,
        "ERG": True,
        "randomize": True,
        "strict": True,
        "min_evals": 20,
    }

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

import rg_nanogpt_one_head as package
from rg_nanogpt_one_head.config import (
    epoch_step_map,
    lr_schedule_steps,
    max_steps,
    warmup_steps,
)


def test_analysis_fallback_keeps_training_registration_available() -> None:
    def fail_import(name: str):
        raise ImportError(f"forced failure for {name}")

    fallback = package._load_analysis_module(
        fail_import,
        register=False,
    )

    fallback.OPTIMIZER_LABELS["muon_clip"] = "MuonClip"
    fallback.OPTIMIZER_COLORS["muon_clip"] = "#CC79A7"

    assert fallback.OPTIMIZER_LABELS["muon_clip"] == "MuonClip"
    assert fallback.OPTIMIZER_COLORS["muon_clip"] == "#CC79A7"
    with pytest.raises(RuntimeError, match="analysis utilities are unavailable"):
        fallback.load_metrics("unused")


def test_muonclip_extension_installs_with_fallback_analysis() -> None:
    code = """
import sys
import rg_nanogpt_one_head as package

def fail_import(name):
    raise ImportError(f'forced failure for {name}')

fallback = package._load_analysis_module(fail_import, register=False)
package.analysis = fallback
sys.modules['rg_nanogpt_one_head.analysis'] = fallback

from rg_nanogpt_one_head.muonclip import install_muonclip_extension
install_muonclip_extension()
from rg_nanogpt_one_head import config
assert 'muon_clip' in config.SUPPORTED_OPTIMIZERS
print('fallback worker import passed')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=EXPERIMENT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().endswith("fallback worker import passed")


def test_24h_muonclip_config_is_125k_steps_and_100_states() -> None:
    cfg = yaml.safe_load(
        (
            EXPERIMENT_ROOT
            / "configs"
            / "muonclip_24h_125k.yaml"
        ).read_text(encoding="utf-8")
    )
    profile = cfg["optimizer_profiles"]["muon_clip"]
    points = epoch_step_map(cfg)

    assert max_steps(cfg) == 125_000
    assert len(points) == 100
    assert min(points) == 0
    assert max(points) == 125_000
    assert lr_schedule_steps(cfg, profile) == 9_766
    assert warmup_steps(profile, 9_766) == 500
    assert cfg["training"]["checkpoint_interval_steps"] == 2_500
    assert profile["qk_diagnostics_interval"] == 2_500

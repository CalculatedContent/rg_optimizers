from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT
    / "baseline"
    / "experiments"
    / "nanogpt_muonclip_large_2026_08_30"
)
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "muonclip_long_mps.yaml"
RUNNER_PATH = EXPERIMENT_ROOT / "scripts" / "run_experiment.py"
REPORT_PATH = EXPERIMENT_ROOT / "scripts" / "build_live_report.py"

sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from rg_nanogpt_one_head.config import epoch_step_map, max_steps, tokens_per_step
from rg_nanogpt_one_head.model import GPT, GPTConfig, transformer_matrix_items
from rg_nanogpt_one_head.muonclip import install_muonclip_extension
from rg_nanogpt_one_head.spectral import _validate_weightwatcher_frame
from rg_nanogpt_one_head.train_loop import (
    _evaluation_due,
    _resume_diagnostics_due,
)


def _load_config() -> dict:
    install_muonclip_extension()
    from rg_nanogpt_one_head.config import load_config

    return load_config(CONFIG_PATH)


def test_large_muonclip_protocol_has_the_declared_scale_and_schedule() -> None:
    cfg = _load_config()
    assert cfg["model"] == {
        "vocab_size": 50_257,
        "block_size": 512,
        "n_layer": 6,
        "n_head": 8,
        "n_embd": 384,
        "dropout": 0.0,
        "bias": False,
        "tie_weights": True,
    }
    assert cfg["dataset"]["train_tokens"] == 512_000_000
    assert cfg["training"]["seeds"] == [20260830]
    assert tokens_per_step(cfg) == 8_192
    assert max_steps(cfg) == 62_500
    assert len(epoch_step_map(cfg)) == 26
    profile = cfg["optimizer_profiles"]["muon_clip"]
    assert profile["learning_rate"] == pytest.approx(2e-4)
    assert profile["min_learning_rate"] == pytest.approx(1e-5)
    assert profile["warmup_fraction"] == pytest.approx(0.016)


def test_checkpoint_before_first_evaluation_materializes_gradients() -> None:
    cfg = _load_config()
    epoch_steps = epoch_step_map(cfg)
    total_steps = max_steps(cfg)

    assert cfg["training"]["checkpoint_interval_steps"] == 250
    assert cfg["training"]["eval_interval_steps"] == 500
    assert not _evaluation_due(
        250,
        cfg=cfg,
        epoch_steps=epoch_steps,
        total_steps=total_steps,
    )
    assert _resume_diagnostics_due(
        250,
        cfg=cfg,
        epoch_steps=epoch_steps,
        total_steps=total_steps,
    )


def test_gpt_and_matrix_inventory_support_multiple_blocks() -> None:
    model = GPT(
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layer=3,
            n_head=4,
            n_embd=32,
        )
    )
    matrices = transformer_matrix_items(model)
    assert len(matrices) == 18
    assert {block for _, _, block, _ in matrices} == {0, 1, 2}
    assert len({name for name, _, _, _ in matrices}) == 18


def test_weightwatcher_validator_accepts_full_multiblock_inventory() -> None:
    names = [f"L{block:02d}_{kind}" for block in range(2) for kind in (
        "W_Q", "W_K", "W_V", "W_O", "W_MLP_IN", "W_MLP_OUT"
    )]
    frame = pd.DataFrame(
        {
            "matrix_name": names,
            "alpha": 3.0,
            "alpha_raw": 3.0,
            "ERG_gap": 1.0,
            "num_traps": 0.0,
            "rand_distance": 0.2,
            "finger_policy": "none",
            "primary_alpha_variant": "raw",
            "weightwatcher_analysis_calls": 1,
            "run_seed": 1,
            "diagnostic_seed": 2,
            "protocol_fingerprint": "fingerprint",
            "model_state_sha256": "state",
        }
    )
    _validate_weightwatcher_frame(
        frame,
        finger_policy=False,
        expected_matrix_names=names,
    )
    with pytest.raises(RuntimeError, match="12 matrices"):
        _validate_weightwatcher_frame(
            frame.iloc[:-1],
            finger_policy=False,
            expected_matrix_names=names,
        )


def test_experiment_entrypoints_are_importable() -> None:
    assert (EXPERIMENT_ROOT / "README.md").is_file()
    assert CONFIG_PATH.is_file()
    for path in (RUNNER_PATH, REPORT_PATH):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

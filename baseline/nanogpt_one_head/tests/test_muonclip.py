from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.config import SUPPORTED_OPTIMIZERS
from rg_nanogpt_one_head.model import GPT, GPTConfig
from rg_nanogpt_one_head.muonclip import (
    MuonClip,
    _muonclip_attention_forward,
)
from rg_nanogpt_one_head.optimizers import zeropower_via_newton_schulz_5


def small_model() -> GPT:
    return GPT(
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
            dropout=0.0,
            bias=False,
        )
    )


def hidden_matrices(model: GPT) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("blocks.") and parameter.ndim == 2
    ]


def test_historical_launcher_remains_three_optimizer_reference() -> None:
    # Importing the class does not mutate the historical launcher. The extension
    # is installed only by rg-onehead-muonclip.
    assert SUPPORTED_OPTIMIZERS == (
        "sgd_momentum",
        "adamw",
        "muon",
    )


def test_extension_does_not_require_muonclip_in_historical_config() -> None:
    from rg_nanogpt_one_head.config import load_config

    config = load_config(EXPERIMENT_ROOT / "configs" / "reference.yaml")
    assert "muon_clip" not in config["optimizer_profiles"]


def test_muonclip_configs_use_reported_scaling_and_threshold() -> None:
    reference = yaml.safe_load(
        (EXPERIMENT_ROOT / "configs" / "muonclip_reference.yaml").read_text()
    )
    profile = reference["optimizer_profiles"]["muon_clip"]

    assert profile["learning_rate"] == pytest.approx(2e-4)
    assert profile["min_learning_rate"] == pytest.approx(2e-5)
    assert profile["weight_decay"] == pytest.approx(0.1)
    assert profile["update_rms_scale"] == pytest.approx(0.2)
    assert profile["qk_clip_threshold"] == pytest.approx(100.0)
    assert profile["qk_clip_balance"] == pytest.approx(0.5)
    assert round(9766 * profile["warmup_fraction"]) == 500

    long_cfg = yaml.safe_load(
        (EXPERIMENT_ROOT / "configs" / "muonclip_10epochs.yaml").read_text()
    )
    long_profile = long_cfg["optimizer_profiles"]["muon_clip"]
    assert long_cfg["training"]["target_epochs"] == 10.0
    assert long_profile["lr_schedule_epochs"] == 1.0


def test_rms_matched_update_uses_point_two_sqrt_max_dimension() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4, 2))
    gradient = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    parameter.grad = gradient.clone()

    class DummyModel:
        blocks = []

    optimizer = MuonClip(
        [parameter],
        model=DummyModel(),
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        update_rms_scale=0.2,
        qk_clip_threshold=100.0,
        qk_clip_balance=0.5,
    )
    optimizer._apply_qk_clip = lambda: None
    expected = -zeropower_via_newton_schulz_5(
        gradient,
        steps=5,
        eps=1e-7,
    ) * (0.2 * math.sqrt(4))

    optimizer.step()

    assert torch.allclose(parameter, expected)


def test_qk_clip_balances_query_and_key_scaling() -> None:
    model = small_model()
    optimizer = MuonClip(
        hidden_matrices(model),
        model=model,
        lr=0.0,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        update_rms_scale=0.2,
        qk_clip_threshold=100.0,
        qk_clip_balance=0.5,
        diagnostics_interval=1,
    )
    q_before = model.blocks[0].attn.q_proj.weight.detach().clone()
    k_before = model.blocks[0].attn.k_proj.weight.detach().clone()
    model.blocks[0].attn._muonclip_max_logits = torch.tensor([400.0])
    for parameter in hidden_matrices(model):
        parameter.grad = torch.zeros_like(parameter)

    optimizer.step()

    assert torch.allclose(
        model.blocks[0].attn.q_proj.weight,
        0.5 * q_before,
    )
    assert torch.allclose(
        model.blocks[0].attn.k_proj.weight,
        0.5 * k_before,
    )
    assert optimizer.last_diagnostics["active_fraction"] == pytest.approx(1.0)
    assert optimizer.last_diagnostics["min_gamma"] == pytest.approx(0.25)


def test_qk_clip_is_noop_below_threshold() -> None:
    model = small_model()
    optimizer = MuonClip(
        hidden_matrices(model),
        model=model,
        lr=0.0,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        update_rms_scale=0.2,
        qk_clip_threshold=100.0,
        qk_clip_balance=0.5,
        diagnostics_interval=1,
    )
    q_before = model.blocks[0].attn.q_proj.weight.detach().clone()
    k_before = model.blocks[0].attn.k_proj.weight.detach().clone()
    model.blocks[0].attn._muonclip_max_logits = torch.tensor([20.0])
    for parameter in hidden_matrices(model):
        parameter.grad = torch.zeros_like(parameter)

    optimizer.step()

    assert torch.equal(model.blocks[0].attn.q_proj.weight, q_before)
    assert torch.equal(model.blocks[0].attn.k_proj.weight, k_before)
    assert optimizer.last_diagnostics["active_fraction"] == pytest.approx(0.0)
    assert optimizer.last_diagnostics["min_gamma"] == pytest.approx(1.0)


def test_attention_observation_matches_native_sdpa_output() -> None:
    model = small_model()
    attention = model.blocks[0].attn
    attention.train()
    x = torch.randn(2, 8, 16)

    observed = _muonclip_attention_forward(attention, x)
    tracked = attention._muonclip_max_logits
    attention._muonclip_max_logits = None
    expected = attention.__class__.__dict__["forward"](attention, x)

    assert tracked is not None
    assert tracked.shape == (1,)
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-5)


def test_extension_validates_muonclip_config_in_isolated_process() -> None:
    code = """
from pathlib import Path
from rg_nanogpt_one_head.muonclip import install_muonclip_extension
install_muonclip_extension()
from rg_nanogpt_one_head.config import load_config, optimizer_profile, max_steps, lr_schedule_steps, warmup_steps
root = Path.cwd()
cfg = load_config(root / 'configs' / 'muonclip_reference.yaml')
p = optimizer_profile(cfg, 'muon_clip')
print(max_steps(cfg), lr_schedule_steps(cfg, p), warmup_steps(p, lr_schedule_steps(cfg, p)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=EXPERIMENT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().endswith("9766 9766 500")

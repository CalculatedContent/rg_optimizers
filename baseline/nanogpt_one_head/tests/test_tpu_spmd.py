import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from rg_nanogpt_one_head import tpu_spmd as spmd
from rg_nanogpt_one_head.model import GPT, GPTConfig
from rg_nanogpt_one_head.muonclip_resilient import _worker_command


def test_cpu_transfer_preserves_tied_weights_even_when_parameters_are_replaced():
    model = GPT(GPTConfig(vocab_size=32, block_size=8, n_embd=8))
    previous = torch.__future__.get_overwrite_module_params_on_conversion()
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(True)
        model.to(dtype=torch.float64)
        assert model.token_embedding.weight is model.lm_head.weight
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(previous)


def test_untied_model_stays_untied():
    model = GPT(GPTConfig(vocab_size=32, block_size=8, n_embd=8, tie_weights=False))
    model.to(dtype=torch.float64)
    assert model.token_embedding.weight is not model.lm_head.weight


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_spmd_rejects_wrong_accelerator(device):
    cfg = {"runtime": {"tpu_spmd": True}, "training": {"batch_size": 8}}
    with pytest.raises(ValueError, match="requires --device tpu"):
        spmd.initialize(cfg, device)


def test_spmd_rejects_indivisible_global_batch():
    cfg = {"runtime": {"tpu_spmd": True, "tpu_expected_chips": 4}, "training": {"batch_size": 2}}
    with pytest.raises(ValueError, match="divisible"):
        spmd.initialize(cfg, "tpu")


def test_spmd_rejects_silent_mode_switch(monkeypatch):
    monkeypatch.setenv("XLA_USE_SPMD", "1")
    with pytest.raises(RuntimeError, match="fresh process"):
        spmd.initialize({"runtime": {}}, "tpu")


@pytest.mark.parametrize("global_count,local_count,process_count,message", [
    (1, 1, 1, "Expected 4"), (8, 4, 2, "one host/process"),
])
def test_spmd_rejects_wrong_topology(global_count, local_count, process_count, message):
    xr = SimpleNamespace(use_spmd=lambda: None,
                         global_runtime_device_count=lambda: global_count,
                         addressable_runtime_device_count=lambda: local_count,
                         process_count=lambda: process_count)
    with pytest.raises(RuntimeError, match=message):
        spmd._initialize_mesh(xr, 4)


def test_supervisor_passes_tpu_device(tmp_path):
    args = SimpleNamespace(config=tmp_path / "c.yaml", seed=1337,
                           data_root=tmp_path / "data", results_root=tmp_path / "results", device="tpu")
    cmd = _worker_command(args)
    assert cmd[cmd.index("--device") + 1] == "tpu"


def test_longrun_global_batch_and_schedule():
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension
    install_muonclip_extension()
    from rg_nanogpt_one_head.config import load_config, tokens_per_step, max_steps, lr_schedule_steps, warmup_steps
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/muonclip_tpu_spmd_long.yaml")
    assert cfg["runtime"]["matmul_precision"] == "highest"
    assert cfg["weightwatcher"]["fix_fingers"] == "clip_xmax"
    assert cfg["weightwatcher"]["require_raw_alpha"] is True
    assert tokens_per_step(cfg) == 8192
    assert cfg["training"]["batch_size"] // cfg["runtime"]["tpu_expected_chips"] == 8
    assert max_steps(cfg) == 2150000
    profile = cfg["optimizer_profiles"]["muon_clip"]
    assert lr_schedule_steps(cfg, profile) == 2150000
    assert warmup_steps(profile, 2150000) == 2000
    smoke = load_config(root / "configs/muonclip_tpu_spmd_smoke.yaml")
    assert smoke["runtime"]["matmul_precision"] == "highest"
    assert max_steps(smoke) == 20


@pytest.mark.skipif(os.environ.get("RG_TEST_XLA_SPMD") != "1", reason="opt-in real XLA CPU test")
def test_real_four_device_xla_gradients_clipping_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PJRT_DEVICE": "CPU", "CPU_NUM_DEVICES": "4", "OMP_NUM_THREADS": "1"}
    environment.pop("XLA_USE_SPMD", None)
    result = subprocess.run([
        sys.executable, "-m", "rg_nanogpt_one_head.tpu_spmd_check", "--backend", "cpu",
        "--chips", "4", "--output", str(tmp_path / "report.json"),
    ], cwd=root, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"passed": true' in result.stdout

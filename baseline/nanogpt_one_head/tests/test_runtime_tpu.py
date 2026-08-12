from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

import rg_nanogpt_one_head.runtime as runtime
from rg_nanogpt_one_head.model import CausalSelfAttention, GPTConfig


class FakeTorchXLA:
    __version__ = torch.__version__
    sync_calls: list[bool] = []
    seeds: list[int] = []

    @staticmethod
    def device():
        return torch.device("xla")

    @classmethod
    def sync(cls, wait=False):
        cls.sync_calls.append(bool(wait))

    @classmethod
    def manual_seed(cls, seed, device=None):
        del device
        cls.seeds.append(int(seed))


class FakeXR:
    @staticmethod
    def device_type():
        return "TPU"

    @staticmethod
    def process_count():
        return 1

    @staticmethod
    def process_index():
        return 0

    @staticmethod
    def addressable_device_count():
        return 1


class FakeXM:
    rng_state = 123

    @classmethod
    def get_rng_state(cls, device=None):
        del device
        return cls.rng_state

    @classmethod
    def set_rng_state(cls, seed, device=None):
        del device
        cls.rng_state = int(seed)

    @staticmethod
    def mark_step():
        raise AssertionError("top-level torch_xla.sync should be used")


@pytest.fixture
def fake_xla(monkeypatch):
    FakeTorchXLA.sync_calls = []
    FakeTorchXLA.seeds = []
    FakeXM.rng_state = 123
    monkeypatch.setattr(runtime, "_tpu_hardware_hint", lambda: True)
    monkeypatch.setattr(
        runtime,
        "_load_xla",
        lambda required: (FakeTorchXLA, FakeXR, FakeXM),
    )
    return FakeTorchXLA, FakeXR, FakeXM


def test_remote_tpu_name_does_not_misclassify_a_mac(monkeypatch):
    monkeypatch.setenv("TPU_NAME", "remote-builder-tpu")
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")

    assert runtime._tpu_hardware_hint() is False


def test_auto_device_prefers_tpu_when_tpu_is_present(fake_xla):
    device = runtime.choose_device("auto")
    assert device.type == "xla"
    assert runtime.accelerator_name(device) == "tpu"


def test_xla_step_and_synchronize_use_correct_wait_semantics(fake_xla):
    device = torch.device("xla")
    runtime.mark_step(device)
    runtime.synchronize(device)

    assert FakeTorchXLA.sync_calls == [False, True]


def test_xla_seed_and_rng_roundtrip(fake_xla):
    device = torch.device("xla")
    runtime.seed_everything(17, device)
    state = runtime.capture_accelerator_rng_state(device)
    FakeXM.rng_state = 999
    runtime.restore_accelerator_rng_state(state, device)

    assert FakeTorchXLA.seeds == [17]
    assert FakeXM.rng_state == 123


def test_configure_runtime_rejects_multi_process_xla(
    fake_xla,
    monkeypatch,
):
    monkeypatch.setattr(
        runtime,
        "_xla_process_count",
        lambda xr: 2,
    )
    cfg = {"runtime": {"matmul_precision": "high"}}

    with pytest.raises(RuntimeError, match="one PyTorch/XLA process"):
        runtime.configure_runtime(torch.device("xla"), cfg)


def test_configure_runtime_rejects_unregistered_bf16(
    fake_xla,
    monkeypatch,
):
    monkeypatch.setenv("XLA_USE_BF16", "1")
    cfg = {"runtime": {"matmul_precision": "high"}}

    with pytest.raises(RuntimeError, match="reference protocol is float32"):
        runtime.configure_runtime(torch.device("xla"), cfg)


def test_cpu_tree_conversion_detaches_nested_tensors():
    value = {
        "a": torch.tensor([1.0], requires_grad=True),
        "b": [torch.tensor([2.0])],
        "c": (torch.tensor([3.0]),),
    }

    converted = runtime.tree_to_cpu(value)

    assert converted["a"].device.type == "cpu"
    assert converted["a"].requires_grad is False
    assert converted["b"][0].device.type == "cpu"
    assert converted["c"][0].device.type == "cpu"


def test_resolved_device_is_accepted_without_redetection(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "is_tpu_environment",
        lambda requested="auto": (_ for _ in ()).throw(
            AssertionError("resolved device should not be redetected")
        ),
    )
    device = torch.device("cpu")
    assert runtime.choose_device(device) is device


def test_tpu_math_attention_matches_reference_sdpa_on_cpu():
    module = CausalSelfAttention(
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
            dropout=0.0,
        )
    )
    generator = torch.Generator().manual_seed(11)
    q = torch.randn(2, 1, 8, 16, generator=generator)
    k = torch.randn(2, 1, 8, 16, generator=generator)
    v = torch.randn(2, 1, 8, 16, generator=generator)

    observed = module._xla_math_attention(q, k, v, dropout_p=0.0)
    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
    )

    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-5)

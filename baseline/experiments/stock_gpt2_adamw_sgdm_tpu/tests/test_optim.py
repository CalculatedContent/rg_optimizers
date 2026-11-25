"""Parity with the pinned magnitude graft and native AdamW; no TPU is simulated as real."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .. import optim
from ..optim import (GRAFT_SCOPE, OPTIMIZER_ID, REFERENCE_SHA256,
                     TensorLrAdamWSGDMGraft, make_optimizer)
from short_test.vendor.model import GPT, GPTConfig
from short_test_adamw_graft.graft import MagnitudeGraft


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def model(dtype=torch.float32, *, layers=2, seed=13):
    torch.manual_seed(seed)
    result = GPT(GPTConfig(n_layer=layers, n_head=2, n_embd=8, block_size=8,
                           vocab_size=17, dropout=0.0, bias=True)).to(dtype=dtype)
    result.lm_head.weight = result.transformer.wte.weight
    return result


def configuration(**overrides):
    values = dict(optimizer=OPTIMIZER_ID, graft_scope=GRAFT_SCOPE, peak_lr=0.001,
                  beta1=0.9, beta2=0.99, eps=1e-8, reference_lr_ratio=400.0,
                  norm_epsilon=1e-30, weight_decay=0.0, weight_decay_scope="matrix_only",
                  n_layer=1, n_head=2, n_embd=8, vocab_size=17, context_length=8,
                  microbatch_sequences=1, global_batch_sequences=1, grad_clip=1.0,
                  seed=42, schedule_factor=lambda step: 1.0)
    values.update(overrides)
    return SimpleNamespace(**values)


def native_adam(net, cfg):
    return torch.optim.AdamW([
        {"params": [p for p in net.parameters() if p.ndim >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in net.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ], lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
       foreach=False, fused=False)


def gradients(net, seed, *, missing=()):
    generator = torch.Generator().manual_seed(seed)
    for name, parameter in net.named_parameters():
        parameter.grad = None if name in missing else torch.randn(
            parameter.shape, generator=generator, dtype=parameter.dtype)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)


class RecordOperations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


def set_lr(optimizer, value, *, tensor=False, dtype=torch.float32):
    for group in optimizer.param_groups:
        group["lr"] = torch.tensor(value, dtype=dtype) if tensor else value


def check_no_scalar_extraction(optimizer):
    with RecordOperations() as observed:
        optimizer.step()
    assert not any("_local_scalar_dense" in name for name in observed.operations)


def test_canonical_reference_is_the_pinned_byte_identical_source():
    assert hashlib.sha256((REPO / "short_test_adamw_graft/graft.py").read_bytes()).hexdigest() == REFERENCE_SHA256


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ratio", [100.0, 400.0, 1000.0])
@pytest.mark.parametrize("decay", [0.0, 0.03])
def test_matches_canonical_graft_with_changing_lr_missing_gradients_and_decay(dtype, ratio, decay):
    cfg = configuration(reference_lr_ratio=ratio, weight_decay=decay)
    actual = model(dtype)
    expected = deepcopy(actual)
    optimizer = make_optimizer(actual, cfg)
    canonical = MagnitudeGraft(expected, native_adam(expected, cfg), reference_lr_ratio=ratio)
    assert optimizer.grafted_parameter_names == canonical.matrix_names
    tolerance = {"rtol": 2e-5, "atol": 2e-7} if dtype == torch.float32 else {"rtol": 1e-11, "atol": 1e-13}
    for index, lr in enumerate((0.001, 0.0005, 0.0, 0.002, 0.00008, 0.001)):
        missing = (optimizer.grafted_parameter_names[1], "transformer.ln_f.bias") if index in (1, 4) else ()
        gradients(actual, index + 25, missing=missing)
        for p, q in zip(actual.parameters(), expected.parameters()):
            q.grad = None if p.grad is None else p.grad.clone()
        set_lr(optimizer, lr, tensor=True, dtype=dtype)
        set_lr(canonical, lr)
        check_no_scalar_extraction(optimizer)
        canonical.step()
        for (name, p), (_, q) in zip(actual.named_parameters(), expected.named_parameters()):
            torch.testing.assert_close(p, q, **tolerance, msg=lambda message: name + ": " + message)
            if p in optimizer.state:
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[p][key], canonical.optimizer.state[q][key],
                                               rtol=0, atol=0)


@pytest.mark.parametrize("decay", [0.0, 0.1])
def test_target_is_uncorrected_sgdm_norm_and_direction_is_adam_not_sgdm(decay):
    cfg = configuration(weight_decay=decay)
    actual = model(torch.float64)
    native = deepcopy(actual)
    reference = deepcopy(actual)
    optimizer = make_optimizer(actual, cfg)
    adam = native_adam(native, cfg)
    sgd = torch.optim.SGD(reference.parameters(), lr=cfg.peak_lr * cfg.reference_lr_ratio,
                          momentum=cfg.beta1, dampening=0, nesterov=False, weight_decay=0,
                          foreach=False)
    grafted = set(optimizer.grafted_parameter_names)
    different_directions = []
    for index, lr in enumerate((0.001, 0.0004, 0.0017, 0.0002)):
        gradients(actual, 80 + index)
        native.load_state_dict(actual.state_dict())
        reference.load_state_dict(actual.state_dict())
        before = {name: p.detach().clone() for name, p in actual.named_parameters()}
        for p, q, r in zip(actual.parameters(), native.parameters(), reference.parameters()):
            q.grad = p.grad.clone()
            r.grad = p.grad.clone()
        set_lr(optimizer, lr, tensor=True, dtype=torch.float64)
        set_lr(adam, lr)
        set_lr(sgd, lr * cfg.reference_lr_ratio)
        optimizer.step()
        adam.step()
        sgd.step()
        for (name, p), q, r in zip(actual.named_parameters(), native.parameters(), reference.parameters()):
            if name not in grafted:
                torch.testing.assert_close(p, q, rtol=1e-11, atol=1e-13)
                continue
            decayed = before[name] * (1 - lr * decay)
            applied = p - decayed
            adam_delta = q - decayed
            momentum = sgd.state[r]["momentum_buffer"]
            target = lr * cfg.reference_lr_ratio * momentum.norm()
            expected = adam_delta * target / adam_delta.norm().clamp_min(cfg.norm_epsilon)
            torch.testing.assert_close(applied, expected, rtol=1e-10, atol=1e-13)
            torch.testing.assert_close(applied.norm(), target, rtol=1e-11, atol=1e-13)
            different_directions.append(not torch.allclose(
                applied / applied.norm(), -momentum / momentum.norm(), rtol=1e-3, atol=1e-5))
    assert any(different_directions)


def test_scope_is_48_physical_matrices_with_joint_qkv_and_tied_head_once():
    net = model(layers=12)
    optimizer = make_optimizer(net, configuration(n_layer=12, weight_decay=0.03))
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.optimizer_id == OPTIMIZER_ID
    assert optimizer.grafted_matrix_count == 48
    expected = {f"transformer.h.{i}.{projection}.weight" for i in range(12)
                for projection in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")}
    assert set(optimizer.grafted_parameter_names) == expected
    assert sum(name.endswith("attn.c_attn.weight") for name in expected) == 12
    assert net.lm_head.weight is net.transformer.wte.weight
    entries = [(p, group["weight_decay"]) for group in optimizer.param_groups for p in group["params"]]
    assert len(entries) == len({id(p) for p, _ in entries}) == len(list(net.parameters()))
    for parameter, decay in entries:
        assert decay == (0.03 if parameter.ndim >= 2 else 0.0)
    assert all(not any(part in name for part in ("wte", "wpe", "lm_head", "ln_"))
               for name in optimizer.grafted_parameter_names)


def test_independent_parameter_counters_advance_only_with_gradients():
    net = model()
    optimizer = make_optimizer(net, configuration(weight_decay=0.1))
    lookup = dict(net.named_parameters())
    first = lookup[optimizer.grafted_parameter_names[0]]
    second = lookup[optimizer.grafted_parameter_names[1]]
    first.grad = torch.ones_like(first)
    check_no_scalar_extraction(optimizer)
    assert first in optimizer.state and second not in optimizer.state
    first_moment = optimizer.state[first]["exp_avg"].clone()
    first_weight = first.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    second.grad = torch.zeros_like(second)
    check_no_scalar_extraction(optimizer)
    assert optimizer.state[first]["step"].item() == optimizer.state[second]["step"].item() == 1
    assert torch.equal(first, first_weight)
    assert torch.equal(optimizer.state[first]["exp_avg"], first_moment)
    first.grad = torch.zeros_like(first)
    check_no_scalar_extraction(optimizer)
    assert optimizer.state[first]["step"].item() == optimizer.state[second]["step"].item() == 2
    for parameter, state in optimizer.state.items():
        assert state["step"].shape == torch.Size([])
        assert state["step"].dtype == torch.float32
        assert state["step"].device == parameter.device


def test_zero_gradient_has_only_separate_decay_and_zero_lr_keeps_all_weights():
    cfg = configuration(weight_decay=0.1)
    net = model()
    optimizer = make_optimizer(net, cfg)
    initial = deepcopy(net.state_dict())
    for parameter in net.parameters():
        parameter.grad = torch.zeros_like(parameter)
    check_no_scalar_extraction(optimizer)
    for name, parameter in net.named_parameters():
        expected = initial[name] * (1 - cfg.peak_lr * cfg.weight_decay) if parameter.ndim >= 2 else initial[name]
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    gradients(net, 503)
    optimizer.step()
    before = deepcopy(net.state_dict())
    gradients(net, 504)
    set_lr(optimizer, 0.0, tensor=True)
    check_no_scalar_extraction(optimizer)
    for name, parameter in net.state_dict().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert all(state["step"].item() == 3 for state in optimizer.state.values())


def test_beta2_changes_direction_but_not_same_history_sgdm_target():
    one, two = model(torch.float64), model(torch.float64)
    first = TensorLrAdamWSGDMGraft(one, betas=(0.9, 0.9))
    second = TensorLrAdamWSGDMGraft(two, betas=(0.9, 0.99))
    name = first.grafted_parameter_names[0]
    for index in range(4):
        gradients(one, 600 + index)
        for p, q in zip(one.parameters(), two.parameters()):
            q.grad = p.grad.clone()
        old1 = dict(one.named_parameters())[name].detach().clone()
        old2 = dict(two.named_parameters())[name].detach().clone()
        first.step()
        second.step()
    delta1 = dict(one.named_parameters())[name] - old1
    delta2 = dict(two.named_parameters())[name] - old2
    torch.testing.assert_close(delta1.norm(), delta2.norm(), rtol=1e-12, atol=1e-14)
    assert not torch.allclose(delta1, delta2, rtol=1e-4, atol=1e-7)


def test_resume_is_exact_with_tensor_lrs_and_missing_gradient_history():
    cfg = configuration(weight_decay=0.03)
    original = model()
    optimizer = make_optimizer(original, cfg)
    for index in range(3):
        gradients(original, index + 100)
        set_lr(optimizer, (index + 1) * 0.0002, tensor=True)
        optimizer.step()
    restored = deepcopy(original)
    restored_optimizer = make_optimizer(restored, cfg)
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for index in range(3, 7):
        missing = (optimizer.grafted_parameter_names[0],) if index == 4 else ()
        gradients(original, index + 100, missing=missing)
        gradients(restored, index + 100, missing=missing)
        for current in (optimizer, restored_optimizer):
            set_lr(current, (index + 1) * 0.0002, tensor=True)
            check_no_scalar_extraction(current)
    for p, q in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[p][key], restored_optimizer.state[q][key], rtol=0, atol=0)
    assert restored_optimizer.graft_contract() == optimizer.graft_contract()


def test_load_moves_cpu_counters_and_learning_rates_to_parameter_device_without_tpu_claim():
    net = model()
    cfg = configuration()
    optimizer = make_optimizer(net, cfg)
    gradients(net, 18)
    set_lr(optimizer, 0.001, tensor=True)
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    # Meta placement exercises the device relocation path without running TPU.
    destination = deepcopy(net).to("meta")
    destination.lm_head.weight = destination.transformer.wte.weight
    restored = make_optimizer(destination, cfg)
    restored.load_state_dict(saved)
    for group in restored.param_groups:
        assert group["lr"].device.type == "meta"
        for parameter in group["params"]:
            for tensor in restored.state[parameter].values():
                assert tensor.device == parameter.device


@pytest.mark.parametrize("alteration", ["ratio", "scope", "beta", "inventory", "state_shape"])
def test_checkpoint_contract_refuses_a_different_rule_or_state_shape(alteration):
    net = model()
    optimizer = make_optimizer(net, configuration())
    gradients(net, 31)
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    if alteration == "ratio":
        saved["graft_contract"]["reference_lr_ratio"] = 100.0
    elif alteration == "scope":
        saved["graft_contract"]["graft_scope"] = "all_parameters"
    elif alteration == "beta":
        saved["param_groups"][0]["betas"] = (0.8, 0.99)
    elif alteration == "inventory":
        saved["graft_contract"]["parameter_inventory"][0]["name"] = "wrong"
    else:
        saved["state"][0]["exp_avg"] = torch.zeros(1)
    with pytest.raises(ValueError, match="Checkpoint"):
        make_optimizer(deepcopy(net), configuration()).load_state_dict(saved)


def test_fixed_beta1_cannot_change_mid_history():
    optimizer = make_optimizer(model(), configuration())
    optimizer.param_groups[0]["betas"] = (0.8, 0.99)
    with pytest.raises(ValueError, match="Fixed Adam"):
        optimizer.step()


def test_reference_ratio_cannot_change_mid_history():
    optimizer = make_optimizer(model(), configuration())
    optimizer.reference_lr_ratio = 100.0
    with pytest.raises(ValueError, match="graft contract changed"):
        optimizer.step()


def test_contract_inventory_is_detached_and_group_names_cannot_change():
    optimizer = make_optimizer(model(), configuration())
    contract = optimizer.graft_contract()
    contract["parameter_inventory"][0]["name"] = "changed"
    assert optimizer.graft_contract()["parameter_inventory"][0]["name"] != "changed"
    optimizer.param_groups[0]["parameter_names"][0] = "changed"
    with pytest.raises(ValueError, match="parameter-group scope changed"):
        optimizer.step()


def test_factory_rejects_a_different_optimizer_scope_or_partial_gpt():
    for settings in ({"optimizer": "sgdm"}, {"graft_scope": "all_matrices"},
                     {"weight_decay_scope": "all_parameters"}):
        with pytest.raises(ValueError):
            make_optimizer(model(), configuration(**settings))
    partial = model()
    del partial.transformer.h[0].attn.c_proj
    with pytest.raises(ValueError, match="all four physical matrices"):
        make_optimizer(partial, configuration())


@pytest.mark.parametrize("field,value", [("reference_lr_ratio", 0), ("norm_epsilon", 0),
                                        ("eps", 0), ("weight_decay", -1), ("reference_lr_ratio", float("inf"))])
def test_invalid_optimizer_scalars_fail_before_updates(field, value):
    with pytest.raises(ValueError):
        make_optimizer(model(), configuration(**{field: value}))


def test_closure_reenables_gradients_and_returns_loss():
    net = model()
    optimizer = make_optimizer(net, configuration())
    tokens = torch.arange(8).reshape(1, 8)

    def closure():
        optimizer.zero_grad()
        _, loss = net(tokens, tokens)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert torch.isfinite(loss)
    assert all(state["step"].item() == 1 for state in optimizer.state.values())


def simulated_probe(monkeypatch, compile_counts):
    """CPU numerics plus fake XLA telemetry test the gate, not TPU readiness."""
    from .. import preflight

    state = {"compile_samples": 0, "updates": 0, "marks": 0}
    xla = ModuleType("torch_xla")
    xla.debug = ModuleType("torch_xla.debug")
    xla.debug.metrics = ModuleType("torch_xla.debug.metrics")

    def metric_data(name):
        assert name == "CompileTime"
        count = state["compile_samples"]
        return None if count is None else (count, 0.0, [])

    xla.debug.metrics.metric_data = metric_data
    for name, module in (("torch_xla", xla), ("torch_xla.debug", xla.debug),
                         ("torch_xla.debug.metrics", xla.debug.metrics)):
        monkeypatch.setitem(sys.modules, name, module)

    class CpuRuntime:
        def __init__(self, requested):
            assert requested == "tpu"
            self.device = torch.device("cpu")

        def replicate(self, value):
            return value

        def put(self, value):
            return value

        def scalar(self, value):
            return torch.tensor(value)

        def autocast(self):
            return nullcontext()

        def attention(self, name):
            assert name == "flash"
            return None

        def step(self, wait=False):
            state["marks"] += 1
            if state["marks"] % 3 == 0:
                assert wait
                state["compile_samples"] = compile_counts[state["updates"]]
                state["updates"] += 1

    class ProbeGPT(GPT):
        def forward(self, x, y, return_logits=False):
            assert return_logits is False
            _, loss = super().forward(x, y)
            return None, loss

    def make_probe_model(config, *, seed, device):
        torch.manual_seed(seed)
        return ProbeGPT(config).to(device)

    model_module = SimpleNamespace(GPTConfig=GPTConfig, make_model=make_probe_model,
                                   configure_attention=lambda attention: None)
    return preflight._tpu_model_probe(configuration(), {"simulated_cpu_runtime": True},
                                      CpuRuntime, model_module, {"simulated": True})


def test_preflight_accepts_stable_graph_after_warmup_using_shared_optimizer(monkeypatch):
    result = simulated_probe(monkeypatch, [1, 2, 3, 3, 3])
    assert result["optimizer"] == "TensorLrAdamWSGDMGraft"
    assert result["dynamic_lr_graph_reuse"]["checked_updates"] == 2
    updates = result["optimizer_updates"]
    assert len({row["learning_rate"] for row in updates}) == 5
    assert [row["new_compilations"] for row in updates] == [1, 1, 1, 0, 0]
    assert all(row["optimizer_host_scalar_extractions"] == 0 for row in updates)


def test_preflight_refuses_recompilation_after_warmup(monkeypatch):
    with pytest.raises(RuntimeError, match="recompiled the warmed TPU model"):
        simulated_probe(monkeypatch, [1, 2, 3, 4, 5])


def test_preflight_refuses_missing_compile_telemetry(monkeypatch):
    with pytest.raises(RuntimeError, match="CompileTime telemetry unavailable"):
        simulated_probe(monkeypatch, [None])


def test_preflight_detects_native_adamw_host_step_extraction(monkeypatch):
    monkeypatch.setattr(optim, "make_optimizer", native_adam)
    with pytest.raises(RuntimeError, match="optimizer extracted a host scalar"):
        simulated_probe(monkeypatch, [1])

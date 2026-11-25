"""The scheduled-LR SGD update must retain SGD math without scalar extraction."""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from optim import TensorLrSGD, make_optimizer


class RecordOperations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


def parameters(dtype=torch.float64):
    return [torch.nn.Parameter(torch.tensor([[0.4, -0.2], [0.7, 0.1]], dtype=dtype)),
            torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=dtype))]


def groups(params, decay):
    return [{"params": [params[0]], "weight_decay": decay, "role": "matrices_and_embeddings"},
            {"params": [params[1]], "weight_decay": 0.0, "role": "bias_and_layernorm"}]


def set_gradients(params, step):
    for index, parameter in enumerate(params):
        parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
        parameter.grad = (parameter.grad - 1.1 + step * 0.2) * (index + 1)


@pytest.mark.parametrize("momentum", [0.0, 0.9, 0.98])
@pytest.mark.parametrize("decay", [0.0, 3e-5, 0.1])
@pytest.mark.parametrize("tensor_lr", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_native_sgd_for_changed_learning_rates_and_decay_groups(momentum, decay, tensor_lr, dtype):
    native_params, device_params = parameters(dtype), parameters(dtype)
    tolerances = {"rtol": 2e-6, "atol": 2e-7} if dtype == torch.float32 else {"rtol": 1e-13, "atol": 1e-14}
    native = torch.optim.SGD(groups(native_params, decay), lr=0.1,
                             momentum=momentum, foreach=False)
    device = TensorLrSGD(groups(device_params, decay), lr=0.1, momentum=momentum)
    for step, lr in enumerate([0.02, 0.1, 0.07, 0.003, 0.0, 0.03]):
        set_gradients(native_params, step)
        set_gradients(device_params, step)
        for group in native.param_groups:
            group["lr"] = lr
        for group in device.param_groups:
            group["lr"] = torch.tensor(lr, dtype=dtype) if tensor_lr else lr
        native.step()
        with RecordOperations() as observed:
            device.step()
        assert not any("_local_scalar_dense" in name for name in observed.operations)
        for expected, actual in zip(native_params, device_params):
            torch.testing.assert_close(actual, expected, **tolerances)
            if momentum:
                torch.testing.assert_close(device.state[actual]["momentum_buffer"],
                                           native.state[expected]["momentum_buffer"],
                                           **tolerances)


@pytest.mark.parametrize("source_class,target_class", [(torch.optim.SGD, TensorLrSGD),
                                                       (TensorLrSGD, torch.optim.SGD)])
def test_state_dict_continues_in_native_sgd_and_tensor_lr_sgd(source_class, target_class):
    source_params = parameters()
    source = source_class(groups(source_params, 3e-5), lr=0.02, momentum=0.98, foreach=False)
    for step in range(3):
        set_gradients(source_params, step)
        source.step()
    target_params = [torch.nn.Parameter(p.detach().clone()) for p in source_params]
    target = target_class(groups(target_params, 0.0), lr=0.3, momentum=0.0, foreach=False)
    target.load_state_dict(deepcopy(source.state_dict()))
    assert target.param_groups[0]["weight_decay"] == 3e-5
    assert target.param_groups[1]["weight_decay"] == 0.0
    assert target.param_groups[0]["momentum"] == 0.98
    for step in range(3, 7):
        set_gradients(source_params, step)
        set_gradients(target_params, step)
        source.step()
        target.step()
        for expected, actual in zip(source_params, target_params):
            torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
            torch.testing.assert_close(target.state[actual]["momentum_buffer"],
                                       source.state[expected]["momentum_buffer"],
                                       rtol=1e-13, atol=1e-14)


def test_factory_deduplicates_tied_weights_and_separates_decay():
    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8)
            self.head.weight = self.embedding.weight
            self.norm = torch.nn.LayerNorm(4)

    model = TiedModel()
    optimizer = make_optimizer(model, SimpleNamespace(peak_lr=0.03, momentum=0.98,
                                                      weight_decay=3e-5))
    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(optimizer, TensorLrSGD)
    by_parameter = {id(p): group["weight_decay"]
                    for group in optimizer.param_groups for p in group["params"]}
    assert len(by_parameter) == sum(len(group["params"]) for group in optimizer.param_groups)
    assert len(by_parameter) == len(list(model.parameters()))
    for parameter in model.parameters():
        assert by_parameter[id(parameter)] == (3e-5 if parameter.ndim >= 2 else 0.0)


def test_absent_gradient_skips_decay_and_momentum_and_closure_enables_grad():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = TensorLrSGD([parameter], lr=0.1, momentum=0.9, weight_decay=0.1)
    optimizer.step()
    assert parameter.item() == 2.0 and not optimizer.state

    def closure():
        optimizer.zero_grad()
        loss = parameter.square()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.item() == 4.0
    torch.testing.assert_close(parameter, torch.tensor(1.58))
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], torch.tensor(4.2))


@pytest.mark.parametrize("options", [{"dampening": 0.1}, {"nesterov": True},
                                     {"maximize": True}, {"foreach": True},
                                     {"differentiable": True}, {"fused": True}])
def test_rejects_other_sgd_variants(options):
    with pytest.raises(ValueError, match="TensorLrSGD requires"):
        TensorLrSGD(parameters(), **options)


def test_rejects_variant_loaded_or_edited_after_construction():
    optimizer = TensorLrSGD(parameters())
    optimizer.param_groups[0]["nesterov"] = True
    with pytest.raises(ValueError, match="TensorLrSGD requires"):
        optimizer.step()


def simulated_probe(monkeypatch, compile_counts):
    """CPU numerics plus fake XLA telemetry test the gate, not TPU readiness."""
    from config import smoke_config
    import preflight

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

    class TinyModel(torch.nn.Module):
        def __init__(self, config, *, seed, device):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.wte = torch.nn.Embedding(config.vocab_size, 4, device=device)
            self.lm_head = torch.nn.Linear(4, config.vocab_size, bias=False, device=device)
            self.lm_head.weight = self.transformer.wte.weight

        def forward(self, x, y, return_logits):
            assert return_logits is False
            logits = self.lm_head(self.transformer.wte(x))
            return None, torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())

    model_module = SimpleNamespace(GPTConfig=SimpleNamespace, make_model=TinyModel,
                                   configure_attention=lambda attention: None)
    torch.set_num_threads(1)
    return preflight._tpu_model_probe(smoke_config(), {"simulated_cpu_runtime": True},
                                      CpuRuntime, model_module, {"simulated": True})


def test_preflight_accepts_stable_graph_after_warmup_using_shared_optimizer(monkeypatch):
    result = simulated_probe(monkeypatch, [1, 2, 3, 3, 3])
    assert result["optimizer"] == "TensorLrSGD"
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


def test_preflight_detects_native_sgd_tensor_lr_host_extraction(monkeypatch):
    import optim
    monkeypatch.setattr(optim, "make_optimizer", lambda model, config:
                        torch.optim.SGD(model.parameters(), lr=config.peak_lr,
                                        momentum=config.momentum, foreach=False))
    with pytest.raises(RuntimeError, match="optimizer extracted a host scalar"):
        simulated_probe(monkeypatch, [1])

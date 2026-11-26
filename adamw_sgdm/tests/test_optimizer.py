"""Numerical contracts against independent PyTorch AdamW and SGD references.

Run with ``python -m unittest discover -s tests -v`` after installing the
package. These checks use CPU tensors; they do not certify accelerator runs.
"""

import copy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn

from rg_adamw_sgdm import AdamWSGDM, parameter_groups


def setUpModule():
    torch.set_num_threads(1)


def tiny_gpt(dtype=torch.float32):
    """A parameter tree with the historical names, no training dependency."""
    model = nn.Module()
    blocks = []
    for _ in range(4):
        block = nn.ModuleDict({
            "attn": nn.ModuleDict({"c_attn": nn.Linear(4, 12), "c_proj": nn.Linear(4, 4)}),
            "mlp": nn.ModuleDict({"c_fc": nn.Linear(4, 8), "c_proj": nn.Linear(8, 4)}),
            "ln_1": nn.LayerNorm(4),
        })
        blocks.append(block)
    model.transformer = nn.ModuleDict({
        "wte": nn.Embedding(7, 4),
        "wpe": nn.Embedding(8, 4),
        "h": nn.ModuleList(blocks),
        "ln_f": nn.LayerNorm(4),
    })
    model.lm_head = nn.Linear(4, 7, bias=False)
    model.lm_head.weight = model.transformer.wte.weight
    return model.to(dtype=dtype)


class OptimizerNumerics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7201)

    def assert_state_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_independent_adamw_direction_sgd_magnitude_and_group_schedules(self):
        """Each matrix has its own target norm, including unequal layer rates."""
        for dtype in (torch.float32, torch.float64):
            for amsgrad, maximize in ((False, False), (True, True)):
                with self.subTest(dtype=dtype, amsgrad=amsgrad, maximize=maximize):
                    params = [nn.Parameter(torch.randn(5, 7, dtype=dtype) * 0.1),
                              nn.Parameter(torch.randn(3, 6, dtype=dtype) * 0.1),
                              nn.Parameter(torch.randn(2, 5, dtype=dtype) * 0.1),
                              nn.Parameter(torch.randn(7, dtype=dtype) * 0.1)]
                    native_params = [nn.Parameter(p.detach().clone()) for p in params]
                    momentum_params = [nn.Parameter(p.detach().clone()) for p in params]
                    ratios = (40.0, 200.0, 800.0, 100.0)
                    decays = (0.03, 0.01, 0.0, 0.02)
                    groups = [{"params": [p], "reference_lr_ratio": ratio,
                               "weight_decay": decay}
                              for p, ratio, decay in zip(params, ratios, decays)]
                    opt = AdamWSGDM(groups, lr=0.001, betas=(0.9, 0.99),
                                    amsgrad=amsgrad, maximize=maximize)
                    native = torch.optim.AdamW(
                        [{"params": [p], "weight_decay": decay}
                         for p, decay in zip(native_params, decays)],
                        lr=0.001, betas=(0.9, 0.99), eps=1e-8,
                        amsgrad=amsgrad, maximize=maximize, foreach=False, fused=False)
                    sgd = torch.optim.SGD(momentum_params, lr=0.1, momentum=0.9,
                                          dampening=0, nesterov=False, weight_decay=0,
                                          maximize=maximize, foreach=False)
                    rtol, atol = ((3e-5, 2e-7) if dtype == torch.float32 else (2e-12, 2e-14))
                    for index, lr in enumerate((0.001, 0.0003, 0.0, 0.0007, 0.0001)):
                        before = [p.detach().clone() for p in params]
                        for p, q, r in zip(params, native_params, momentum_params):
                            grad = torch.randn_like(p) * (0.01 if index % 2 else 0.2)
                            p.grad = grad.clone()
                            q.grad = grad.clone()
                            r.grad = grad.clone()
                            with torch.no_grad():
                                q.copy_(p)
                                r.copy_(p)
                        for group in opt.param_groups:
                            group["lr"] = lr
                        for group in native.param_groups:
                            group["lr"] = lr
                        opt.step()
                        native.step()
                        sgd.step()
                        for j, (p, q, r) in enumerate(zip(params, native_params, momentum_params)):
                            state = native.state[q]
                            t = int(state["step"].item())
                            for key in ("exp_avg", "exp_avg_sq"):
                                torch.testing.assert_close(opt.state[p][key], state[key],
                                                           rtol=0, atol=0)
                            self.assertEqual(int(opt.state[p]["step"].item()), t)
                            self.assertEqual(opt.state[p]["step"].dtype, torch.int64)
                            self.assertEqual(opt.state[p]["step"].device, p.device)
                            if amsgrad:
                                torch.testing.assert_close(opt.state[p]["max_exp_avg_sq"],
                                                           state["max_exp_avg_sq"], rtol=0, atol=0)
                            if p.ndim < 2:
                                torch.testing.assert_close(p, q, rtol=rtol, atol=atol)
                                continue
                            variance = state["max_exp_avg_sq"] if amsgrad else state["exp_avg_sq"]
                            adam_delta = -lr * (state["exp_avg"] / (1 - 0.9**t)) / (
                                (variance / (1 - 0.99**t)).sqrt() + 1e-8)
                            target = lr * ratios[j] * sgd.state[r]["momentum_buffer"].norm()
                            grafted = adam_delta * (target / adam_delta.norm().clamp_min(1e-30))
                            decayed = before[j] * (1 - lr * decays[j])
                            torch.testing.assert_close(p, decayed + grafted, rtol=rtol, atol=atol)
                            if lr:
                                actual_delta = (p - decayed).double()
                                torch.testing.assert_close(actual_delta.norm(), target.double(),
                                                           rtol=rtol, atol=atol)
                                cosine = torch.nn.functional.cosine_similarity(
                                    actual_delta.flatten(), adam_delta.double().flatten(), dim=0)
                                torch.testing.assert_close(cosine, torch.ones_like(cosine),
                                                           rtol=0, atol=2e-7 if dtype == torch.float32 else 2e-12)
                            else:
                                torch.testing.assert_close(p, before[j], rtol=0, atol=0)

    def test_forced_graft_selection_and_float64_norms(self):
        vector = nn.Parameter(torch.randn(6))
        matrix = nn.Parameter(torch.randn(4, 6))
        reference = nn.Parameter(matrix.detach().clone())
        opt = AdamWSGDM([
            {"params": [vector], "graft": True, "reference_lr_ratio": 40},
            {"params": [matrix], "graft": False},
        ], lr=0.001, weight_decay=0, norm_dtype=torch.float64)
        native = torch.optim.AdamW([reference], lr=0.001, weight_decay=0, foreach=False, fused=False)
        old = vector.detach().clone()
        vector.grad = torch.tensor([0.1, -0.3, 0.5, 0.2, -0.8, 0.4])
        matrix.grad = torch.randn_like(matrix)
        reference.grad = matrix.grad.clone()
        opt.step()
        native.step()
        torch.testing.assert_close((vector - old).double().norm(),
                                   0.04 * vector.grad.double().norm(), rtol=3e-5, atol=1e-7)
        torch.testing.assert_close(matrix, reference, rtol=3e-5, atol=1e-7)

    def test_extreme_betas_epsilon_precision_and_downcast_norms(self):
        """Legal near-one betas must not round the bias correction to zero."""
        grad = torch.tensor([[0.1, -0.2, 0.0], [0.7, -0.1, 0.5]])
        for beta in (0.0, 0.99999999):
            for graft in (False, True):
                with self.subTest(beta=beta, graft=graft):
                    p = nn.Parameter(torch.zeros_like(grad))
                    reference = nn.Parameter(torch.zeros_like(grad, dtype=torch.float64))
                    opt = AdamWSGDM([p], lr=0.001, betas=(beta, beta),
                                    weight_decay=0, reference_lr_ratio=40, graft=graft)
                    native = torch.optim.AdamW([reference], lr=0.001, betas=(beta, beta),
                                               weight_decay=0, foreach=False, fused=False)
                    p.grad, reference.grad = grad.clone(), grad.double()
                    opt.step()
                    native.step()
                    expected = reference.detach().clone()
                    if graft:
                        # At update one the raw SGDM buffer is exactly g,
                        # independent of beta. The reference remains FP64.
                        target = 0.04 * grad.double().norm()
                        expected *= target / expected.norm()
                    self.assertTrue(torch.isfinite(p).all())
                    torch.testing.assert_close(p.double(), expected, rtol=8e-7, atol=2e-9)
        p = nn.Parameter(torch.zeros_like(grad, dtype=torch.float64))
        opt = AdamWSGDM([p], lr=0.001, weight_decay=0,
                        reference_lr_ratio=40, norm_dtype=torch.float32)
        p.grad = grad.double()
        opt.step()  # torch.linalg.vector_norm(dtype=...) alone cannot downcast.
        torch.testing.assert_close(p.norm(), 0.04 * grad.double().norm(),
                                   rtol=8e-7, atol=2e-9)
        for dtype, kwargs in ((torch.float32, {"eps": 1e-50}),
                              (torch.float32, {"norm_epsilon": 1e-50}),
                              (torch.float64, {"norm_epsilon": 1e-50,
                                               "norm_dtype": torch.float32})):
            with self.subTest(dtype=dtype, kwargs=kwargs), self.assertRaisesRegex(ValueError, "representable"):
                AdamWSGDM([nn.Parameter(torch.zeros(2, 3, dtype=dtype))], **kwargs)

    def test_zero_gradient_missing_gradient_and_residual_momentum(self):
        p = nn.Parameter(torch.ones(3, 4, dtype=torch.float64))
        q = nn.Parameter(torch.ones(3, 4, dtype=torch.float64))
        opt = AdamWSGDM([p, q], lr=0.01, weight_decay=0.2, reference_lr_ratio=2)
        p.grad = torch.zeros_like(p)
        opt.step()
        torch.testing.assert_close(p, torch.full_like(p, 0.998), rtol=0, atol=0)
        torch.testing.assert_close(q, torch.ones_like(q), rtol=0, atol=0)
        self.assertNotIn(q, opt.state)
        self.assertTrue(torch.isfinite(p).all())
        p.grad = torch.full_like(p, 0.2)
        opt.step()
        old = p.detach().clone()
        old_moment = opt.state[p]["exp_avg"].clone()
        p.grad.zero_()
        opt.step()
        torch.testing.assert_close(opt.state[p]["exp_avg"], old_moment * 0.9,
                                   rtol=2e-15, atol=2e-18)
        self.assertGreater((p - old * 0.998).norm().item(), 0)
        old, old_state = p.detach().clone(), copy.deepcopy(opt.state[p])
        opt.zero_grad(set_to_none=True)
        opt.step()
        torch.testing.assert_close(p, old, rtol=0, atol=0)
        self.assert_state_equal(opt.state[p], old_state)

    def test_checkpoint_resume_is_exact_and_rejects_other_optimizer_formats(self):
        params = [nn.Parameter(torch.randn(3, 5)), nn.Parameter(torch.randn(5))]
        opt = AdamWSGDM(params, lr=0.0005, amsgrad=True, reference_lr_ratio=40)
        for _ in range(4):
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()
        stream = io.BytesIO()
        torch.save({"parameters": [p.detach().clone() for p in params],
                    "optimizer": opt.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        resumed = [nn.Parameter(p.clone()) for p in checkpoint["parameters"]]
        restored = AdamWSGDM(resumed)
        restored.load_state_dict(checkpoint["optimizer"])
        for _ in range(3):
            for p, q in zip(params, resumed):
                grad = torch.randn_like(p)
                p.grad, q.grad = grad.clone(), grad.clone()
            opt.step()
            restored.step()
            for p, q in zip(params, resumed):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
                self.assert_state_equal(opt.state[p], restored.state[q])
        with self.assertRaises(ValueError):
            restored.load_state_dict({"optimizer": opt.state_dict(), "matrix_names": ()})
        native = torch.optim.AdamW(resumed)
        with self.assertRaises(ValueError):
            restored.load_state_dict(native.state_dict())

    def test_closure_and_standard_lr_scheduler(self):
        p = nn.Parameter(torch.tensor([[0.5, -0.7]], dtype=torch.float64))
        opt = AdamWSGDM([p], lr=0.001, reference_lr_ratio=10, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        calls = []

        def closure():
            self.assertTrue(torch.is_grad_enabled())
            opt.zero_grad(set_to_none=True)
            loss = p.square().sum()
            loss.backward()
            calls.append(loss)
            return loss

        with torch.no_grad():
            returned = opt.step(closure)
        self.assertIs(returned, calls[0])
        self.assertEqual(len(calls), 1)
        scheduler.step()
        self.assertEqual(opt.param_groups[0]["lr"], 0.0005)
        self.assertEqual(opt.param_groups[0]["reference_lr_ratio"], 10)
        self.assertLess(p.square().sum().item(), returned.item())

    def test_step_does_not_extract_device_scalars(self):
        p = nn.Parameter(torch.randn(3, 5))
        opt = AdamWSGDM([p], amsgrad=True)
        p.grad = torch.randn_like(p)
        message = "optimizer.step must not extract device tensors on the host"
        with mock.patch.object(torch.Tensor, "item", side_effect=AssertionError(message)), \
             mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError(message)), \
             mock.patch.object(torch.Tensor, "numpy", side_effect=AssertionError(message)), \
             mock.patch.object(torch.Tensor, "__float__", side_effect=AssertionError(message)), \
             mock.patch.object(torch.Tensor, "__bool__", side_effect=AssertionError(message)):
            opt.step()
            opt.step()
        self.assertEqual(opt.state[p]["step"].item(), 2)

    def test_invalid_groups_sparse_grads_and_changed_betas_fail_before_updates(self):
        for kwargs in ({"lr": -1}, {"lr": float("nan")}, {"lr": torch.tensor(0.01)},
                       {"betas": (0.9, 1)}, {"betas": (-0.1, 0.99)},
                       {"eps": 0}, {"weight_decay": -0.1},
                       {"reference_lr_ratio": 0}, {"reference_lr_ratio": float("inf")},
                       {"norm_epsilon": 0}, {"norm_dtype": torch.float16}, {"graft": "all"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                AdamWSGDM([nn.Parameter(torch.ones(2, 3))], **kwargs)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                AdamWSGDM([nn.Parameter(torch.ones(2, 3, dtype=dtype))])
        p, q = nn.Parameter(torch.ones(2, 3)), nn.Parameter(torch.ones(2, 3))
        opt = AdamWSGDM([{"params": [p]}, {"params": [q]}])
        p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
        opt.step()
        old = [p.detach().clone(), q.detach().clone()]
        old_states = [copy.deepcopy(opt.state[p]), copy.deepcopy(opt.state[q])]
        opt.param_groups[1]["betas"] = (0.8, 0.999)
        with self.assertRaisesRegex(ValueError, "fixed betas"):
            opt.step()
        for actual, value, state in zip((p, q), old, old_states):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
            self.assert_state_equal(opt.state[actual], state)
        opt.param_groups[1]["betas"] = (0.9, 0.999)
        q.grad = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 3))
        with self.assertRaisesRegex(RuntimeError, "sparse"):
            opt.step()
        for actual, value, state in zip((p, q), old, old_states):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
            self.assert_state_equal(opt.state[actual], state)

    def test_cpu_bfloat16_autocast_keeps_fp32_state_and_learns(self):
        model = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 4))
        opt = AdamWSGDM(parameter_groups(model, reference_lr_ratio=10), lr=0.001)
        x, labels = torch.randn(32, 16), torch.arange(32) % 4
        losses = []
        for _ in range(16):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(x), labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])
        for p in model.parameters():
            self.assertEqual(p.dtype, torch.float32)
            self.assertEqual(opt.state[p]["exp_avg"].dtype, torch.float32)
            self.assertEqual(opt.state[p]["exp_avg_sq"].dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware/runtime is unavailable")
    def test_cuda_optimizer_amp_gradscaler_and_overflow_skip(self):
        """Executed only on real CUDA; a skip is not accelerator validation."""
        cpu = nn.Parameter(torch.randn(5, 7))
        gpu = nn.Parameter(cpu.detach().cuda())
        cpu_opt = AdamWSGDM([cpu], reference_lr_ratio=40, amsgrad=True)
        gpu_opt = AdamWSGDM([gpu], reference_lr_ratio=40, amsgrad=True)
        for _ in range(3):
            grad = torch.randn_like(cpu)
            cpu.grad, gpu.grad = grad.clone(), grad.cuda()
            cpu_opt.step()
            gpu_opt.step()
        torch.testing.assert_close(gpu.cpu(), cpu, rtol=3e-5, atol=2e-7)
        for value in gpu_opt.state[gpu].values():
            self.assertEqual(value.device, gpu.device)

        model = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 4)).cuda()
        opt = AdamWSGDM(parameter_groups(model, reference_lr_ratio=10), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
        x, labels = torch.randn(32, 16).cuda(), (torch.arange(32) % 4).cuda()
        losses = []
        for _ in range(16):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = nn.functional.cross_entropy(model(x), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            self.assertTrue(torch.isfinite(loss))
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])
        for p in model.parameters():
            self.assertEqual(p.dtype, torch.float32)
            self.assertEqual(opt.state[p]["exp_avg"].dtype, torch.float32)

        # GradScaler must skip the entire custom step on nonfinite gradients.
        before = [p.detach().clone() for p in model.parameters()]
        states = [copy.deepcopy(opt.state[p]) for p in model.parameters()]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = nn.functional.cross_entropy(model(x), labels)
        scaler.scale(loss).backward()
        for p in model.parameters():
            p.grad.fill_(float("inf"))
        scaler.step(opt)
        scaler.update()
        for p, value, state in zip(model.parameters(), before, states):
            torch.testing.assert_close(p, value, rtol=0, atol=0)
            self.assert_state_equal(opt.state[p], state)

    def test_legacy_shakespeare_wrapper_parity(self):
        legacy_path = Path(__file__).resolve().parents[2] / "short_test_adamw_graft" / "graft.py"
        if not legacy_path.exists():
            self.skipTest("Legacy repository source is absent in the standalone installation")
        spec = importlib.util.spec_from_file_location("_legacy_shakespeare_graft", legacy_path)
        legacy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_module)
        model = tiny_gpt(dtype=torch.float64)
        reference = copy.deepcopy(model)
        opt = AdamWSGDM(parameter_groups(model, selection="shakespeare", weight_decay=0.03,
                                         reference_lr_ratio=400),
                         lr=0.001, betas=(0.9, 0.99))
        native = torch.optim.AdamW(
            parameter_groups(reference, selection="shakespeare", weight_decay=0.03),
            lr=0.001, betas=(0.9, 0.99), eps=1e-8, foreach=False, fused=False)
        legacy = legacy_module.MagnitudeGraft(reference, native, reference_lr_ratio=400)
        self.assertEqual(len(legacy.matrix_names), 16)
        for lr in (0.001, 0.0004, 0.0001):
            for group in opt.param_groups:
                group["lr"] = lr
            for group in native.param_groups:
                group["lr"] = lr
            for p, q in zip(model.parameters(), reference.parameters()):
                grad = torch.randn_like(p) * 0.01
                p.grad, q.grad = grad.clone(), grad.clone()
            opt.step()
            legacy.step()
            for p, q in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, rtol=2e-12, atol=2e-14)


class ParameterSelection(unittest.TestCase):
    def test_embedding_and_tied_weights_are_unique_and_ungrafted(self):
        model = tiny_gpt()
        model.transformer.ln_f.weight.requires_grad_(False)
        groups = parameter_groups(model, weight_decay=0.07, reference_lr_ratio=40)
        flat = [(name, p, group) for group in groups
                for name, p in zip(group["param_names"], group["params"])]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual({id(p) for _, p, _ in flat}, expected)
        self.assertEqual(len(flat), len(expected))
        embeddings = {id(model.transformer.wte.weight), id(model.transformer.wpe.weight)}
        for name, p, group in flat:
            self.assertEqual(group["reference_lr_ratio"], 40)
            self.assertEqual(group["weight_decay"], 0.07 if p.ndim >= 2 else 0.0)
            self.assertEqual(group["graft"], p.ndim >= 2 and id(p) not in embeddings, name)
        self.assertIs(model.lm_head.weight, model.transformer.wte.weight)

    def test_shakespeare_selects_exact_historical_sixteen_matrices(self):
        model = tiny_gpt()
        groups = parameter_groups(model, selection="shakespeare", weight_decay=0,
                                  reference_lr_ratio=400)
        grafted = {name for group in groups if group["graft"] for name in group["param_names"]}
        expected = {f"transformer.h.{block}.{projection}.weight" for block in range(4)
                    for projection in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")}
        self.assertEqual(grafted, expected)
        self.assertTrue(all(group["weight_decay"] == 0 for group in groups))
        with self.assertRaises(ValueError):
            parameter_groups(nn.Linear(4, 3), selection="shakespeare")
        with self.assertRaises(ValueError):
            parameter_groups(model, selection="unknown")
        model.requires_grad_(False)
        with self.assertRaises(ValueError):
            parameter_groups(model)


if __name__ == "__main__":
    unittest.main()

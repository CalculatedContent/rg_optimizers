"""Numerical checks for the update, independent of the training and data code."""

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "short_test" / "vendor"))
from model import GPT, GPTConfig

from graft import MagnitudeGraft


def make_model():
    return GPT(GPTConfig(n_layer=4, n_head=2, n_embd=16, block_size=8,
                         vocab_size=17, dropout=0.0, bias=True))


def adam(model):
    return torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.99),
                             eps=1e-8, weight_decay=0.03, foreach=False, fused=False)


def gradients(model, seed):
    rng = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=rng)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


class TestMagnitudeGraft(unittest.TestCase):
    def test_update_matches_independent_optimizers(self):
        torch.manual_seed(6)
        model = make_model()
        native = copy.deepcopy(model)
        reference = copy.deepcopy(model)
        graft = MagnitudeGraft(model, adam(model))
        native_opt = adam(native)
        sgd = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
        expected_names = {
            f"transformer.h.{i}.{projection}.weight" for i in range(4)
            for projection in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
        }
        self.assertEqual(set(graft.matrix_names), expected_names)
        for i, lr in enumerate((0.001, 0.0005, 0.0009)):
            gradients(model, i + 89)
            # Start native AdamW at the graft's actual position each step.
            # Its step includes both the bias-corrected direction and decay.
            native.load_state_dict(model.state_dict())
            reference.load_state_dict(model.state_dict())
            for p, q, r in zip(model.parameters(), native.parameters(), reference.parameters()):
                q.grad = p.grad.clone()
                r.grad = p.grad.clone()
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            graft.param_groups[0]["lr"] = native_opt.param_groups[0]["lr"] = lr
            sgd.param_groups[0]["lr"] = 100 * lr
            graft.step()
            native_opt.step()
            sgd.step()
            for (name, p), q, r in zip(model.named_parameters(), native.parameters(),
                                       reference.parameters()):
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(graft.optimizer.state[p][key],
                                               native_opt.state[q][key], rtol=0, atol=0)
                if name not in expected_names:
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                    continue
                decayed = before[name] * (1 - lr * 0.03)
                momentum_size = (100 * lr) * sgd.state[r]["momentum_buffer"].double().norm()
                delta = (p - decayed).double()
                direction = (q - decayed).double()
                expected = direction * (momentum_size / direction.norm())
                torch.testing.assert_close(delta, expected, rtol=5e-5, atol=2e-8)
                torch.testing.assert_close(delta.norm(), momentum_size, rtol=5e-5, atol=2e-8)

    def test_zero_missing_and_resume(self):
        torch.manual_seed(24)
        model = make_model()
        graft = MagnitudeGraft(model, adam(model))
        p, q = (dict(model.named_parameters())[n] for n in graft.matrix_names[:2])
        initial_p, initial_q = p.detach().clone(), q.detach().clone()
        p.grad = torch.zeros_like(p)
        graft.step()
        torch.testing.assert_close(p, initial_p * (1 - 0.001 * 0.03), rtol=0, atol=0)
        torch.testing.assert_close(q, initial_q, rtol=0, atol=0)
        self.assertNotIn(q, graft.optimizer.state)
        graft.zero_grad()
        self.assertIsNone(p.grad)
        gradients(model, 1)
        graft.step()
        restarted = copy.deepcopy(model)
        restored_opt = MagnitudeGraft(restarted, adam(restarted))
        restored_opt.load_state_dict(copy.deepcopy(graft.state_dict()))
        gradients(model, 2)
        gradients(restarted, 2)
        graft.step()
        restored_opt.step()
        for p, q in zip(model.parameters(), restarted.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            MagnitudeGraft(restarted, adam(restarted), reference_lr_ratio=5).load_state_dict(
                graft.state_dict())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

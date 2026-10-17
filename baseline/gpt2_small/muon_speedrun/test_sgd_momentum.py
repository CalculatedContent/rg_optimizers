"""CPU regression checks for the stock-GPT-2 momentum-SGD optimizer arm."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ['RG_GPT2_PROFILE'] = 'sgd_speedrun'

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optim import make_stock_optimizers, apply_update, optimizer_metadata
from benchmark_config import SGD_PEAK_LR, SGD_MOMENTUM, SGD_WEIGHT_DECAY, TOTAL_STEPS, WARMUP, WARMDOWN, lr_factor


class TinyTiedModel(torch.nn.Module):
    architecture_id = "not-needed-for-direct-stock-factory"

    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.transformer.wte = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.norm = torch.nn.LayerNorm(4)


class SGDTests(unittest.TestCase):
    def test_tied_parameter_once_and_classical_momentum(self):
        model = TinyTiedModel()
        muon, opt = make_stock_optimizers(model, SimpleNamespace(tpu=False), "sgd_momentum")
        self.assertIsNone(muon)
        self.assertIsInstance(opt, torch.optim.SGD)
        params = [p for g in opt.param_groups for p in g["params"]]
        self.assertEqual(len(params), len(set(map(id, params))))
        self.assertEqual(set(map(id, params)), set(map(id, model.parameters())))
        groups = {g['role']:g for g in opt.param_groups}
        self.assertEqual(groups['matrices']['weight_decay'], SGD_WEIGHT_DECAY)
        self.assertEqual(groups['bias_and_layernorm']['weight_decay'], 0.0)
        self.assertTrue(all(g["momentum"] == SGD_MOMENTUM for g in opt.param_groups))
        self.assertTrue(all(not g["nesterov"] for g in opt.param_groups))
        self.assertEqual(optimizer_metadata(opt)["class"], "torch.optim.SGD")
        p = model.norm.bias
        old = p.detach().clone()
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        apply_update(muon, opt, SimpleNamespace(scalar=lambda x: x), 0)
        self.assertTrue(torch.allclose(p, old - SGD_PEAK_LR * lr_factor(0)))
        previous = p.detach().clone()
        apply_update(muon, opt, SimpleNamespace(scalar=lambda x: x), 1)
        self.assertTrue(torch.allclose(p, previous - SGD_PEAK_LR * lr_factor(1) * (1 + SGD_MOMENTUM)))


    def test_speedrun_schedule(self):
        self.assertEqual(TOTAL_STEPS, 6200)
        self.assertAlmostEqual(lr_factor(0), 1/WARMUP)
        self.assertAlmostEqual(lr_factor(WARMUP-1), 1.0)
        self.assertAlmostEqual(lr_factor(TOTAL_STEPS-WARMDOWN), 1.0)
        self.assertEqual(lr_factor(TOTAL_STEPS), 0.0)


if __name__ == "__main__":
    unittest.main()

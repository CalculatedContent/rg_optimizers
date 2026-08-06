import unittest

import torch

from rg_baselines.config import BaselineConfig
from rg_baselines.model import MLP3
from rg_baselines.optimizers import (
    SGDMomentumMuon,
    build_optimizer,
    zeropower_via_newton_schulz_5,
)


class OptimizerTests(unittest.TestCase):
    def test_newton_schulz_preserves_shape_and_is_finite(self) -> None:
        torch.manual_seed(2)
        matrix = torch.randn(7, 11)
        result = zeropower_via_newton_schulz_5(matrix, steps=5)
        self.assertEqual(result.shape, matrix.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_muon_assignment_is_explicit(self) -> None:
        model = MLP3()
        config = BaselineConfig(optimizer="sgd_momentum_muon")
        optimizer = build_optimizer(model, config)
        self.assertIsInstance(optimizer, SGDMomentumMuon)
        self.assertEqual(optimizer.assignment["fc1.weight"], "muon")
        self.assertEqual(optimizer.assignment["fc2.weight"], "muon")
        self.assertEqual(optimizer.assignment["fc3.weight"], "sgd")
        self.assertEqual(optimizer.assignment["fc1.bias"], "sgd")

    def test_all_baselines_take_a_finite_step(self) -> None:
        torch.manual_seed(3)
        inputs = torch.randn(8, 1, 28, 28)
        targets = torch.randint(0, 10, (8,))
        for name in ("sgd_momentum", "adamw", "sgd_momentum_muon"):
            model = MLP3()
            optimizer = build_optimizer(model, BaselineConfig(optimizer=name))
            before = model.fc1.weight.detach().clone()
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(model.fc1.weight).all(), name)
            self.assertFalse(torch.equal(before, model.fc1.weight), name)


if __name__ == "__main__":
    unittest.main()

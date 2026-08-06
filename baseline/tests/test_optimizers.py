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

    def test_auxiliary_sgd_matches_torch_sgd(self) -> None:
        torch.manual_seed(5)
        model_custom = MLP3()
        model_torch = MLP3()
        model_torch.load_state_dict(model_custom.state_dict())
        custom = SGDMomentumMuon(
            model_custom.named_parameters(),
            muon_parameter_names=("fc1.weight", "fc2.weight"),
            muon_lr=0.02,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_weight_decay=0.0,
            newton_schulz_steps=5,
            muon_eps=1e-7,
            auxiliary_lr=0.05,
            auxiliary_momentum=0.9,
            auxiliary_dampening=0.0,
            auxiliary_nesterov=False,
            auxiliary_weight_decay=1e-4,
        )
        auxiliary_parameters = [
            parameter
            for name, parameter in model_torch.named_parameters()
            if name not in {"fc1.weight", "fc2.weight"}
        ]
        reference = torch.optim.SGD(
            auxiliary_parameters,
            lr=0.05,
            momentum=0.9,
            dampening=0.0,
            nesterov=False,
            weight_decay=1e-4,
        )
        custom_named = dict(model_custom.named_parameters())
        reference_named = dict(model_torch.named_parameters())
        for name in custom_named:
            gradient = torch.randn_like(custom_named[name])
            custom_named[name].grad = gradient.clone()
            if name not in {"fc1.weight", "fc2.weight"}:
                reference_named[name].grad = gradient.clone()
        custom.step()
        reference.step()
        for name in ("fc3.weight", "fc1.bias", "fc2.bias", "fc3.bias"):
            self.assertTrue(
                torch.allclose(custom_named[name], reference_named[name], atol=1e-7, rtol=1e-6),
                name,
            )

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

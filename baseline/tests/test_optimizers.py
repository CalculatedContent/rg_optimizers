import unittest

import torch

from rg_baselines.config import BaselineConfig
from rg_baselines.model import MLP3
from rg_baselines.optimizers import (
    MuonWithAuxAdamW,
    build_optimizer,
    scheduled_learning_rates,
    warmup_cosine_learning_rate,
    zeropower_via_newton_schulz_5,
)


class OptimizerTests(unittest.TestCase):
    def test_newton_schulz_preserves_shape_and_is_finite(self) -> None:
        torch.manual_seed(2)
        matrix = torch.randn(7, 11)
        result = zeropower_via_newton_schulz_5(matrix, steps=5)
        self.assertEqual(result.shape, matrix.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_muon_assignment_uses_auxiliary_adamw(self) -> None:
        model = MLP3()
        config = BaselineConfig(optimizer="sgd_momentum_muon")
        optimizer = build_optimizer(model, config)
        self.assertIsInstance(optimizer, MuonWithAuxAdamW)
        self.assertEqual(optimizer.assignment["fc1.weight"], "muon")
        self.assertEqual(optimizer.assignment["fc2.weight"], "muon")
        self.assertEqual(optimizer.assignment["fc3.weight"], "adamw")
        self.assertEqual(optimizer.assignment["fc1.bias"], "adamw")
        self.assertEqual(
            {group["kind"] for group in optimizer.param_groups},
            {"muon", "adamw_decay", "adamw_no_decay"},
        )

    def test_auxiliary_adamw_matches_torch_adamw_for_one_step(self) -> None:
        torch.manual_seed(5)
        model_custom = MLP3()
        model_reference = MLP3()
        model_reference.load_state_dict(model_custom.state_dict())
        custom = MuonWithAuxAdamW(
            model_custom.named_parameters(),
            muon_parameter_names=("fc1.weight", "fc2.weight"),
            muon_lr=0.02,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_weight_decay=0.01,
            newton_schulz_steps=5,
            muon_eps=1e-7,
            auxiliary_lr=3e-4,
            auxiliary_betas=(0.9, 0.95),
            auxiliary_eps=1e-8,
            auxiliary_weight_decay=0.01,
        )
        auxiliary_named = [
            (name, parameter)
            for name, parameter in model_reference.named_parameters()
            if name not in {"fc1.weight", "fc2.weight"}
        ]
        reference = torch.optim.AdamW(
            [
                {
                    "params": [parameter for _, parameter in auxiliary_named if parameter.ndim >= 2],
                    "weight_decay": 0.01,
                },
                {
                    "params": [parameter for _, parameter in auxiliary_named if parameter.ndim < 2],
                    "weight_decay": 0.0,
                },
            ],
            lr=3e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        custom_named = dict(model_custom.named_parameters())
        reference_named = dict(model_reference.named_parameters())
        for name in custom_named:
            gradient = torch.randn_like(custom_named[name])
            custom_named[name].grad = gradient.clone()
            if name not in {"fc1.weight", "fc2.weight"}:
                reference_named[name].grad = gradient.clone()
        custom.step()
        reference.step()
        for name in ("fc3.weight", "fc1.bias", "fc2.bias", "fc3.bias"):
            self.assertTrue(
                torch.allclose(custom_named[name], reference_named[name], atol=2e-7, rtol=2e-6),
                name,
            )

    def test_warmup_cosine_has_peak_and_nonzero_floor(self) -> None:
        values = [
            warmup_cosine_learning_rate(
                index,
                total_epochs=30,
                warmup_epochs=2,
                peak_lr=0.05,
                min_lr=5e-4,
            )
            for index in range(30)
        ]
        self.assertAlmostEqual(values[0], 0.025)
        self.assertAlmostEqual(values[1], 0.05)
        self.assertAlmostEqual(values[-1], 5e-4)
        self.assertTrue(all(value > 0 for value in values))

    def test_profiles_have_distinct_source_backed_schedules(self) -> None:
        sgd = scheduled_learning_rates(BaselineConfig(optimizer="sgd_momentum"), epoch_index=0)
        adamw = scheduled_learning_rates(BaselineConfig(optimizer="adamw"), epoch_index=0)
        muon = scheduled_learning_rates(BaselineConfig(optimizer="sgd_momentum_muon"), epoch_index=0)
        self.assertEqual(set(sgd), {"primary"})
        self.assertEqual(set(adamw), {"primary"})
        self.assertEqual(set(muon), {"primary", "auxiliary"})
        self.assertNotEqual(sgd["primary"], adamw["primary"])
        self.assertNotEqual(muon["primary"], muon["auxiliary"])

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

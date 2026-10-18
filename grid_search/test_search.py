"""Small scientific-contract checks, with no downloads or test-set access."""
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data import split_indices
from search import DEFAULTS, MLP, make_optimizer, parameter_grid, schedule, train_one


class ScientificContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_paired_bias_initialization(self):
        torch.manual_seed(401)
        biased = MLP(True)
        torch.manual_seed(401)
        unbiased = MLP(False)
        for name in ("fc1", "fc2", "fc3"):
            torch.testing.assert_close(getattr(biased, name).weight, getattr(unbiased, name).weight, rtol=0, atol=0)
            self.assertIsNone(getattr(unbiased, name).bias)

    def test_regularization_and_rates(self):
        model = MLP()
        c = {**DEFAULTS, "fc1_weight_decay": 0.001, "momentum": 0.95, "nesterov": True}
        opt = make_optimizer(model, c, [0.1, 0.2, 0.3])
        self.assertEqual([g["lr"] for g in opt.param_groups], [0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
        self.assertEqual([g["weight_decay"] for g in opt.param_groups], [0.001, 0, 0, 0, 0, 0])
        self.assertTrue(opt.defaults["nesterov"])
        self.assertEqual(opt.defaults["momentum"], 0.95)

    def test_schedule_and_split_match_protocol(self):
        steps = math.ceil(55000 / 256)
        self.assertEqual(schedule(0, steps, 120, 5, 0.1), 0.01)
        self.assertEqual(schedule(5 * steps - 1, steps, 120, 5, 0.1), 1.0)
        self.assertAlmostEqual(schedule(120 * steps - 1, steps, 120, 5, 0.1), 0.1)
        self.assertEqual(schedule(240 * steps, steps, 120, 5, 0.1), 0.1)
        labels = np.repeat(np.arange(10), 6000)
        train, valid = split_indices(labels)
        self.assertEqual((len(train), len(valid)), (55000, 5000))
        self.assertEqual(len(np.intersect1d(train, valid)), 0)
        np.testing.assert_array_equal(np.bincount(labels[valid]), np.full(10, 500))
        np.testing.assert_array_equal(train, split_indices(labels)[0])

    def test_grid_counts_and_invalid_keys(self):
        root = Path(__file__).parent / "configs"
        for name, count in (("mnist", 60), ("fashion_mnist", 60), ("mnist_high_dropout", 24),
                            ("layer_rates", 54), ("optimizer_sweep", 54)):
            self.assertEqual(len(list(parameter_grid(json.loads((root / f"{name}.json").read_text())))), count)
        with self.assertRaises(ValueError):
            list(parameter_grid({"momentumm": [0.9]}))

    def test_fixed_budget_and_interpolation(self):
        # Zero-error eval from epoch1 must not shorten a 12-epoch search.
        x, y = torch.zeros(8, 784), torch.zeros(8, dtype=torch.long)
        data = x, y, np.arange(6), np.arange(6, 8)
        c = {**DEFAULTS, "warmup_epochs": 0}
        metrics = dict(loss=0.1, errors=0, accuracy=1.0)
        with patch("search.evaluate", return_value=metrics), patch("search.alphas", return_value={
                "alpha_fc1": 1.8, "alpha_fc2": 2.0, "alpha_fc3": 2.2}):
            row = train_one("MNIST", c, 401, data, torch.device("cpu"), epochs=12)
        self.assertEqual(row["epoch"], 12)
        self.assertEqual(row["zero_error_streak"], 12)
        self.assertTrue(row["sustained_interpolation"])
        self.assertFalse(row["all_alphas_within_0_1"])
        self.assertAlmostEqual(row["alpha_mean"], 2.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

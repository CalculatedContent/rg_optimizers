from __future__ import annotations

import unittest

from ecs_trace_wall import BaseOptimizerConfig, ExperimentConfig, TraceWallConfig


class ConfigTests(unittest.TestCase):
    def test_repository_baseline_hyperparameters(self) -> None:
        adamw = BaseOptimizerConfig.adamw_baseline()
        self.assertEqual(adamw.name, "adamw")
        self.assertEqual(adamw.peak_learning_rate, 1e-3)
        self.assertEqual((adamw.beta1, adamw.beta2), (0.9, 0.999))
        self.assertEqual(adamw.eps, 1e-8)
        self.assertEqual(adamw.weight_decay, 1e-2)

        sgd = BaseOptimizerConfig.sgd_momentum_baseline()
        self.assertEqual(sgd.name, "sgd_momentum")
        self.assertEqual(sgd.peak_learning_rate, 5e-2)
        self.assertEqual(sgd.momentum, 0.9)
        self.assertEqual(sgd.dampening, 0.0)
        self.assertFalse(sgd.nesterov)
        self.assertEqual(sgd.weight_decay, 1e-4)

    def test_primary_experiment_defaults(self) -> None:
        config = ExperimentConfig(optimizer=BaseOptimizerConfig.adamw_baseline())
        config.validate()
        self.assertEqual(config.seeds, (1337, 2027, 31415))
        self.assertEqual(config.epochs, 20)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.corrections_per_epoch, 1)
        self.assertEqual(
            config.trace_wall.parameter_names,
            ("fc1.weight", "fc2.weight", "fc3.weight"),
        )
        self.assertEqual(config.trace_wall.probe_batch_size, 256)
        self.assertEqual(config.trace_wall.probe_batches_per_correction, 2)

    def test_invalid_probe_size_settings_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TraceWallConfig(probe_batch_size=0).validate()


if __name__ == "__main__":
    unittest.main()

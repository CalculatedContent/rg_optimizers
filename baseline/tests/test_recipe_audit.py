import unittest

from rg_baselines.config import BaselineConfig
from rg_baselines.nanochat_reference import NANOCHAT_COMMIT, NanoChatD12Config
from rg_baselines.vit_final import ViTBaselineConfig


class BaselineRecipeAuditTests(unittest.TestCase):
    def test_mnist_profiles_have_warmup_decay_and_nonzero_floors(self):
        for optimizer in (
            "sgd_momentum",
            "adamw",
            "sgd_momentum_muon",
        ):
            config = BaselineConfig(optimizer=optimizer)
            config.validate()
            self.assertEqual(config.schedule, "warmup_cosine")
            self.assertGreater(config.warmup_epochs, 0)
        muon = BaselineConfig(optimizer="sgd_momentum_muon")
        self.assertAlmostEqual(muon.muon_aux_learning_rate, 3e-4)
        self.assertEqual(
            (muon.muon_aux_beta1, muon.muon_aux_beta2),
            (0.9, 0.95),
        )

    def test_vit_recipe_contains_full_regularization_and_schedule_stack(self):
        config = ViTBaselineConfig()
        config.validate()
        self.assertEqual(config.recipe_version, 4)
        self.assertEqual(config.epochs, 300)
        self.assertEqual(config.validation_size, 5_000)
        self.assertEqual(config.dropout, 0.0)
        self.assertAlmostEqual(config.norm_eps, 1e-6)
        self.assertEqual(config.cooldown_epochs, 10)
        self.assertAlmostEqual(config.adamw_warmup_start_lr, 1e-6)
        self.assertGreater(config.drop_path, 0.0)
        self.assertGreater(config.mixup_alpha, 0.0)
        self.assertGreater(config.cutmix_alpha, 0.0)
        self.assertGreater(config.random_erasing_probability, 0.0)
        self.assertGreater(config.randaugment_ops, 0)
        self.assertGreater(config.sgd_min_lr, 0.0)
        self.assertGreater(config.adamw_min_lr, 0.0)
        self.assertGreater(config.muon_min_lr, 0.0)
        self.assertTrue(config.test_monitoring_only)

    def test_nanochat_remains_pinned_to_native_reference_recipe(self):
        config = NanoChatD12Config()
        config.validate()
        self.assertEqual(
            NANOCHAT_COMMIT,
            "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd",
        )
        self.assertEqual(config.depth, 12)
        self.assertEqual(config.model_dim, 768)
        self.assertEqual(config.warmup_steps, 40)
        self.assertAlmostEqual(config.warmdown_ratio, 0.65)
        self.assertAlmostEqual(config.final_lr_frac, 0.05)
        self.assertAlmostEqual(config.matrix_lr, 0.02)


if __name__ == "__main__":
    unittest.main()

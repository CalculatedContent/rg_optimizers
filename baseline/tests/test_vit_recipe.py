import tempfile
import unittest
from pathlib import Path

import torch

from rg_baselines.vit_cifar10 import (
    SmallViT,
    ViTBaselineConfig,
    _load_checkpoint,
    _save_checkpoint,
    build_optimizer,
    cosine_learning_rate,
    set_learning_rates,
)


class ViTRecipeTests(unittest.TestCase):
    def tiny_config(self):
        return ViTBaselineConfig(
            epochs=4,
            batch_size=4,
            validation_size=100,
            embed_dim=48,
            depth=2,
            num_heads=3,
            sgd_warmup_epochs=1,
            adamw_warmup_epochs=1,
            muon_warmup_epochs=1,
            ww_every=1,
            checkpoint_every=1,
        )

    def test_all_optimizer_steps_are_finite(self):
        config = self.tiny_config()
        inputs = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, 10, (2,))
        for name in ("sgd_momentum", "adamw", "muon"):
            model = SmallViT(config)
            optimizer = build_optimizer(model, name, config)
            lrs = set_learning_rates(optimizer, name, config, 0)
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            self.assertTrue(
                all(torch.isfinite(parameter).all() for parameter in model.parameters()),
                name,
            )
            self.assertGreater(lrs["primary"], 0.0)

    def test_schedule_reaches_nonzero_floor(self):
        config = self.tiny_config()
        value = cosine_learning_rate(
            config.epochs - 1,
            epochs=config.epochs,
            warmup_epochs=config.adamw_warmup_epochs,
            peak_lr=config.adamw_lr,
            min_lr=config.adamw_min_lr,
        )
        self.assertAlmostEqual(value, config.adamw_min_lr)

    def test_restart_checkpoint_restores_model_optimizer_and_generator(self):
        config = self.tiny_config()
        model = SmallViT(config)
        optimizer = build_optimizer(model, "muon", config)
        generator = torch.Generator().manual_seed(7)
        fingerprint = "unit-test"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _save_checkpoint(
                path,
                epoch=2,
                model=model,
                optimizer=optimizer,
                train_generator=generator,
                config=config,
                optimizer_name="muon",
                seed=7,
                best_validation_loss=1.23,
                fingerprint=fingerprint,
            )
            restored_model = SmallViT(config)
            restored_optimizer = build_optimizer(restored_model, "muon", config)
            restored_generator = torch.Generator().manual_seed(99)
            epoch, best = _load_checkpoint(
                path,
                model=restored_model,
                optimizer=restored_optimizer,
                train_generator=restored_generator,
                expected_fingerprint=fingerprint,
            )
            self.assertEqual(epoch, 2)
            self.assertAlmostEqual(best, 1.23)
            for left, right in zip(model.parameters(), restored_model.parameters(), strict=True):
                self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()

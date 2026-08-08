import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from rg_baselines import vit_cifar10 as core
from rg_baselines import vit_runtime as hardened
from rg_baselines.vit_final import (
    SmallViT,
    ViTBaselineConfig,
    _ensure_validation_best,
    cosine_learning_rate,
)


class FinalViTRecipeTests(unittest.TestCase):
    def tiny_config(self) -> ViTBaselineConfig:
        return ViTBaselineConfig(
            epochs=6,
            batch_size=4,
            validation_size=100,
            embed_dim=48,
            depth=2,
            num_heads=3,
            sgd_warmup_epochs=1,
            adamw_warmup_epochs=1,
            muon_warmup_epochs=1,
            cooldown_epochs=1,
            ww_min_evals=2,
            checkpoint_every=1,
        )

    def test_norm_epsilon_and_patch_initialization_match_final_recipe(self):
        torch.manual_seed(4)
        config = self.tiny_config()
        model = SmallViT(config)
        eps_values = {
            float(module.eps)
            for module in model.modules()
            if isinstance(module, nn.LayerNorm)
        }
        self.assertEqual(eps_values, {1e-6})
        # Fan-in Conv2d initialization is substantially wider than the 0.02
        # trunc-normal initialization used for transformer Linear matrices.
        patch_std = float(model.patch_embed.proj.weight.detach().std())
        qkv_std = float(model.blocks[0].attn.qkv.weight.detach().std())
        self.assertGreater(patch_std, 0.04)
        self.assertLess(qkv_std, 0.03)

    def test_schedule_starts_low_reaches_peak_and_has_cooldown(self):
        config = self.tiny_config()
        values = [
            cosine_learning_rate(
                epoch,
                epochs=config.epochs,
                warmup_epochs=config.adamw_warmup_epochs,
                cooldown_epochs=config.cooldown_epochs,
                warmup_start_lr=config.adamw_warmup_start_lr,
                peak_lr=config.adamw_lr,
                min_lr=config.adamw_min_lr,
            )
            for epoch in range(config.epochs)
        ]
        self.assertAlmostEqual(values[0], config.adamw_warmup_start_lr)
        self.assertAlmostEqual(values[1], config.adamw_lr)
        self.assertAlmostEqual(values[-1], config.adamw_min_lr)
        self.assertTrue(all(value > 0.0 for value in values))

    def test_existing_epoch_one_best_is_replaced_when_epoch_zero_is_selected(self):
        config = self.tiny_config()
        history = pd.DataFrame(
            [
                {
                    "epoch": 0,
                    "validation_loss": 0.5,
                    "test_loss": 0.6,
                    "test_accuracy": 0.2,
                },
                {
                    "epoch": 1,
                    "validation_loss": 0.7,
                    "test_loss": 0.8,
                    "test_accuracy": 0.1,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            model = SmallViT(config)
            optimizer = core.build_optimizer(model, "adamw", config)
            generator = torch.Generator().manual_seed(17)
            # Simulate the historical core-runner behavior: epoch one existed
            # even though epoch zero had the lower validation loss.
            core._save_checkpoint(
                run_dir / "checkpoint_best.pt",
                epoch=1,
                model=model,
                optimizer=optimizer,
                train_generator=generator,
                config=config,
                optimizer_name="adamw",
                seed=17,
                best_validation_loss=0.7,
                fingerprint=hardened._runtime_fingerprint("adamw", 17, config),
            )
            path = _ensure_validation_best(
                run_dir,
                history,
                optimizer_name="adamw",
                seed=17,
                config=config,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(int(payload["epoch"]), 0)
            self.assertAlmostEqual(float(payload["best_validation_loss"]), 0.5)


if __name__ == "__main__":
    unittest.main()

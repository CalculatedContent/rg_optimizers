import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from rg_baselines.vit_analysis import (
    summarize_layer_metric,
    terminal_summary,
    validation_selected_rows,
)
from rg_baselines.vit_cifar10 import ViTBaselineConfig
from rg_baselines.vit_runtime import _ensure_best_checkpoint


class ViTAnalysisTests(unittest.TestCase):
    def test_layer_ci_uses_three_runs_not_blocks_as_replicates(self):
        rows = []
        for seed, shift in ((17, -0.1), (29, 0.0), (43, 0.1)):
            for block in range(6):
                rows.append(
                    {
                        "optimizer": "adamw",
                        "seed": seed,
                        "matrix_name": f"L{block:02d}_W_Q",
                        "matrix_type": "W_Q",
                        "block": block,
                        "epoch": 1,
                        "alpha": 2.0 + shift + 0.01 * block,
                    }
                )
        summary = summarize_layer_metric(pd.DataFrame(rows), "alpha")
        self.assertEqual(len(summary), 6)
        self.assertTrue(summary["n"].eq(3).all())

    def test_pooling_multiple_blocks_under_one_matrix_name_is_rejected(self):
        rows = []
        for seed in (17, 29, 43):
            for duplicate in range(2):
                rows.append(
                    {
                        "optimizer": "adamw",
                        "seed": seed,
                        "matrix_name": "L00_W_Q",
                        "matrix_type": "W_Q",
                        "block": 0,
                        "epoch": 1,
                        "alpha": 2.0 + duplicate,
                    }
                )
        with self.assertRaisesRegex(RuntimeError, "pseudo-replication"):
            summarize_layer_metric(pd.DataFrame(rows), "alpha")

    def test_validation_selection_never_uses_test_metrics(self):
        rows = []
        for seed in (17, 29, 43):
            rows.extend(
                [
                    {
                        "optimizer": "adamw",
                        "seed": seed,
                        "epoch": 1,
                        "validation_loss": 0.8,
                        "validation_accuracy": 0.7,
                        "test_loss": 0.2,
                        "test_accuracy": 0.99,
                    },
                    {
                        "optimizer": "adamw",
                        "seed": seed,
                        "epoch": 2,
                        "validation_loss": 0.5,
                        "validation_accuracy": 0.8,
                        "test_loss": 0.9,
                        "test_accuracy": 0.5,
                    },
                ]
            )
        history = pd.DataFrame(rows)
        selected = validation_selected_rows(history)
        self.assertTrue(selected["epoch"].eq(2).all())
        summary = terminal_summary(history)
        selected_accuracy = summary[
            summary["checkpoint"].eq("validation_selected")
            & summary["metric"].eq("test_accuracy")
        ]
        self.assertAlmostEqual(float(selected_accuracy.iloc[0]["mean"]), 0.5)
        self.assertEqual(int(selected_accuracy.iloc[0]["n"]), 3)

    def test_epoch_zero_validation_optimum_gets_a_real_best_checkpoint(self):
        config = ViTBaselineConfig(
            epochs=2,
            validation_size=100,
            embed_dim=16,
            depth=1,
            num_heads=1,
            sgd_warmup_epochs=1,
            adamw_warmup_epochs=1,
            muon_warmup_epochs=1,
            ww_min_evals=2,
        )
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
            path = _ensure_best_checkpoint(
                run_dir,
                history,
                optimizer_name="adamw",
                seed=17,
                config=config,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(int(payload["epoch"]), 0)
            self.assertAlmostEqual(float(payload["best_validation_loss"]), 0.5)
            self.assertEqual(
                payload["fingerprint"],
                __import__("rg_baselines.vit_cifar10", fromlist=["_fingerprint"])._fingerprint(
                    "adamw", 17, config
                ),
            )


if __name__ == "__main__":
    unittest.main()

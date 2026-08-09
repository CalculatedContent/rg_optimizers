from __future__ import annotations

from dataclasses import asdict
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from rg_baselines.comparison import (
    LAYER_ORDER,
    OPTIMIZER_LABELS,
    OPTIMIZER_ORDER,
    _load_and_validate,
    run_baseline_comparison,
)
from rg_baselines.config import BaselineConfig


SEEDS = (1337, 2027, 31415)
EPOCHS = 3
OFFSETS = {
    "sgd_momentum": 0.00,
    "adamw": 0.01,
    "sgd_momentum_muon": 0.02,
}


def _touch(path: Path, content: bytes = b"placeholder") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_suite(root: Path) -> None:
    for optimizer in OPTIMIZER_ORDER:
        directory = root / optimizer
        directory.mkdir(parents=True, exist_ok=True)
        config = BaselineConfig(
            optimizer=optimizer,
            epochs=EPOCHS,
            sgd_warmup_epochs=1,
            muon_warmup_epochs=1,
        )
        config.validate()
        manifest = {
            "optimizer": optimizer,
            "optimizer_label": OPTIMIZER_LABELS[optimizer],
            "seeds": list(SEEDS),
            "replicate_count": len(SEEDS),
            "confidence": 0.95,
            "config_template": asdict(config),
        }
        (directory / "replicate_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        performance_rows = []
        spectral_rows = []
        offset = OFFSETS[optimizer]
        for seed_index, seed in enumerate(SEEDS):
            seed_effect = seed_index * 0.001
            for epoch in range(EPOCHS + 1):
                validation_loss = (
                    (1.0, 0.7, 0.5, 0.6)[epoch] - offset
                )
                train_accuracy = (
                    0.70 + 0.05 * epoch + offset + seed_effect
                )
                validation_accuracy = train_accuracy - 0.02
                test_accuracy = train_accuracy - 0.03
                train_loss = 1.2 - 0.2 * epoch - offset
                test_loss = train_loss + 0.08
                performance_rows.append(
                    {
                        "run": OPTIMIZER_LABELS[optimizer],
                        "optimizer": optimizer,
                        "optimizer_label": OPTIMIZER_LABELS[optimizer],
                        "seed": seed,
                        "epoch": epoch,
                        "global_step": 10 * epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "test_loss": test_loss,
                        "train_accuracy": train_accuracy,
                        "validation_accuracy": validation_accuracy,
                        "test_accuracy": test_accuracy,
                        "validation_accuracy_gap": (
                            train_accuracy - validation_accuracy
                        ),
                        "test_accuracy_gap": (
                            train_accuracy - test_accuracy
                        ),
                        "validation_loss_gap": (
                            validation_loss - train_loss
                        ),
                        "test_loss_gap": test_loss - train_loss,
                        "primary_lr": 0.01,
                        "test_monitoring_only": 1,
                    }
                )
                for layer_index, layer in enumerate(LAYER_ORDER):
                    spectral_rows.append(
                        {
                            "run": OPTIMIZER_LABELS[optimizer],
                            "optimizer": optimizer,
                            "optimizer_label": OPTIMIZER_LABELS[optimizer],
                            "seed": seed,
                            "epoch": epoch,
                            "global_step": 10 * epoch,
                            "layer_id": layer_index + 1,
                            "layer": layer,
                            "status": "ok",
                            "alpha": 2.0 + 0.1 * layer_index,
                            "num_traps": layer_index,
                            "detX_num": 4,
                            "num_pl_spikes": 2,
                            "ERG_gap": 2,
                            "m_midpoint": 3,
                            "trace_log_midpoint_per_eval": 0.0,
                            "trace_log_midpoint_total": 0.0,
                        }
                    )

            seed_dir = directory / "seeds" / f"seed_{seed}"
            required = (
                "final_state.pt",
                "checkpoint_latest.pt",
                "checkpoint_best.pt",
                "test_results.json",
                "manifest.json",
                "config.json",
                "performance_by_epoch.csv",
                "spectral_metrics_by_epoch_and_layer.csv",
                "esd_history.npz",
            )
            for filename in required:
                _touch(seed_dir / filename)
            for epoch in range(1, EPOCHS + 1):
                _touch(
                    seed_dir
                    / "checkpoints"
                    / f"epoch_{epoch:03d}.pt"
                )
            (seed_dir / "run_complete.json").write_text(
                json.dumps(
                    {
                        "completed": True,
                        "optimizer": optimizer,
                        "seed": seed,
                        "epochs": EPOCHS,
                        "best_validation_epoch": 2,
                    }
                ),
                encoding="utf-8",
            )

        pd.DataFrame(performance_rows).to_csv(
            directory / "performance_by_epoch_and_seed.csv",
            index=False,
        )
        pd.DataFrame(spectral_rows).to_csv(
            directory
            / "spectral_metrics_by_epoch_layer_and_seed.csv",
            index=False,
        )
        (directory / "performance_summary_95ci.csv").write_text(
            "placeholder\n", encoding="utf-8"
        )
        (directory / "spectral_summary_95ci.csv").write_text(
            "placeholder\n", encoding="utf-8"
        )


class MnistComparisonTests(unittest.TestCase):
    def test_synthetic_complete_suite_and_paired_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            output = Path(temporary) / "comparison"
            _write_suite(root)
            with mock.patch(
                "rg_baselines.comparison.plot_all_comparisons",
                return_value=(),
            ):
                result = run_baseline_comparison(
                    root,
                    output_dir=output,
                    show_plots=False,
                )

            self.assertEqual(result.seeds, SEEDS)
            self.assertEqual(result.epochs, EPOCHS)
            pd.testing.assert_frame_equal(
                result.paired_terminal_differences,
                result.paired_final_differences,
            )
            row = result.paired_terminal_differences.loc[
                result.paired_terminal_differences[
                    "checkpoint"
                ].eq("final")
                & result.paired_terminal_differences[
                    "optimizer_a"
                ].eq("adamw")
                & result.paired_terminal_differences[
                    "optimizer_b"
                ].eq("sgd_momentum")
                & result.paired_terminal_differences[
                    "metric"
                ].eq("test_accuracy")
            ].iloc[0]
            self.assertEqual(int(row["n"]), 3)
            self.assertAlmostEqual(
                float(row["mean_difference"]), 0.01
            )
            self.assertTrue(
                (
                    output
                    / "paired_terminal_differences_95ci.csv"
                ).is_file()
            )

    def test_duplicate_performance_grid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            _write_suite(root)
            path = (
                root
                / "adamw"
                / "performance_by_epoch_and_seed.csv"
            )
            frame = pd.read_csv(path)
            frame = pd.concat(
                [frame, frame.iloc[[0]]],
                ignore_index=True,
            )
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(
                RuntimeError,
                "incomplete or duplicate performance grid",
            ):
                _load_and_validate(root)


if __name__ == "__main__":
    unittest.main()

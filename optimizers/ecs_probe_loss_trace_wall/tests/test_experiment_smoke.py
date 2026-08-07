from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from ecs_trace_wall import (
    BaseOptimizerConfig,
    ExperimentConfig,
    TraceWallConfig,
    plot_all,
    run_paired_experiment,
)


class TinyMLP3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 7)
        self.fc2 = nn.Linear(7, 6)
        self.fc3 = nn.Linear(6, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_datasets() -> tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(501)
    teacher = torch.tensor(
        [[1.2, -0.4, 0.7, 0.1], [-0.3, 1.0, -0.2, 0.8], [0.2, -0.5, 1.1, -0.6]]
    )
    train_inputs = torch.randn(48, 4, generator=generator)
    test_inputs = torch.randn(24, 4, generator=generator)
    train_targets = (train_inputs @ teacher.T).argmax(dim=1)
    test_targets = (test_inputs @ teacher.T).argmax(dim=1)
    return (
        TensorDataset(train_inputs, train_targets),
        TensorDataset(test_inputs, test_targets),
    )


class ExperimentSmokeTests(unittest.TestCase):
    def test_paired_end_to_end_run_and_plots(self) -> None:
        train, test = make_datasets()
        trace = TraceWallConfig(
            parameter_names=("fc1.weight", "fc2.weight", "fc3.weight"),
            probe_batch_size=8,
            probe_batches_per_correction=1,
            correction_to_base_step_ratio=0.25,
            minimum_weight_fraction=1e-4,
            maximum_weight_fraction=2e-2,
            min_ecs_rank=2,
            strict=True,
        )
        config = ExperimentConfig(
            optimizer=BaseOptimizerConfig.sgd_momentum_baseline(),
            trace_wall=trace,
            seeds=(11, 29),
            epochs=2,
            batch_size=12,
            corrections_per_epoch=1,
            measure_weightwatcher=False,
            require_weightwatcher=False,
            save_epoch_checkpoints=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = run_paired_experiment(
                config,
                output_dir=output,
                device=torch.device("cpu"),
                train_dataset=train,
                test_dataset=test,
                progress=False,
                model_factory=TinyMLP3,
            )
            self.assertEqual(len(result.performance), 2 * 2 * 3)
            self.assertEqual(len(result.spectral), 2 * 2 * 3 * 3)
            self.assertFalse(result.corrections.empty)
            self.assertFalse(
                result.manifest["official_test_set_used_for_optimization"]
            )
            for run in result.manifest["seed_runs"]:
                self.assertEqual(
                    run["initial_checksum"], run["baseline_initial_checksum"]
                )
                self.assertEqual(
                    run["initial_checksum"], run["trace_wall_initial_checksum"]
                )
                self.assertEqual(
                    run["baseline_global_step"], run["trace_wall_global_step"]
                )

            applied = result.corrections.loc[result.corrections["applied"]]
            self.assertTrue(
                (applied["probe_loss_after"] <= applied["probe_loss_before"] + 1e-10).all()
            )
            self.assertTrue(
                (result.corrections["probe_examples"] == 8).all()
            )
            self.assertTrue(
                (result.corrections["projection_identity_error"] < 5e-5).all()
            )

            expected_outputs = {
                "performance_by_epoch_and_seed.csv",
                "spectral_metrics_by_epoch_layer_and_seed.csv",
                "trace_wall_corrections_by_step_layer_and_seed.csv",
                "performance_summary_95ci.csv",
                "spectral_summary_95ci.csv",
                "trace_wall_correction_summary.csv",
                "config.json",
                "paired_manifest.json",
            }
            self.assertTrue(
                expected_outputs.issubset({path.name for path in output.iterdir()})
            )
            for seed in config.seeds:
                checkpoint_dir = output / "seeds" / f"seed_{seed}" / "checkpoints"
                self.assertEqual(
                    len(list(checkpoint_dir.glob("baseline_epoch_*.pt"))),
                    config.epochs,
                )
                self.assertEqual(
                    len(list(checkpoint_dir.glob("trace_wall_epoch_*.pt"))),
                    config.epochs,
                )

            plot_dir = output / "plots"
            figures = plot_all(result, output_dir=plot_dir, show=False)
            self.assertGreaterEqual(len(figures), 5)
            self.assertTrue((plot_dir / "01_loss_95ci.png").is_file())
            self.assertTrue((plot_dir / "01_accuracy_95ci.png").is_file())
            self.assertTrue((plot_dir / "02_ecs_rank_95ci.png").is_file())
            self.assertTrue((plot_dir / "04_probe_loss_before_after.png").is_file())
            self.assertTrue((plot_dir / "05_correction_to_base_step_ratio.png").is_file())


if __name__ == "__main__":
    unittest.main()

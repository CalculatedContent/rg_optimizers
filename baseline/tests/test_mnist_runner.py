import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rg_baselines.config import BaselineConfig
from rg_baselines.diagnostics import SpectralCheckpoint
from rg_baselines.runner import run_baseline


class FakeMNIST(Dataset):
    def __init__(self, train: bool, transform=None):
        self.train = train
        self.transform = transform
        self.length = 100 if train else 20

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        tensor = torch.full((1, 28, 28), float(index % 17) / 17.0)
        return tensor, int(index % 10)


def fake_mnist_factory(root, train, download, transform):
    return FakeMNIST(train=train, transform=transform)


def fake_weightwatcher(model, *, run_label, epoch, global_step, **kwargs):
    rows = []
    for layer_id, layer in enumerate(("fc1", "fc2", "fc3"), start=1):
        rows.append(
            {
                "run": run_label,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "layer_id": layer_id,
                "layer": layer,
                "status": "ok",
                "alpha": 2.0 + 0.01 * layer_id,
                "num_traps": 0,
                "detX_num": 4,
                "num_pl_spikes": 2,
                "ERG_gap": 2,
                "m_midpoint": 3,
                "trace_log_midpoint_per_eval": 0.0,
                "trace_log_midpoint_total": 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    details = frame[
        ["run", "epoch", "global_step", "layer_id", "layer", "num_traps"]
    ].copy()
    return SpectralCheckpoint(
        details=details,
        metrics=frame,
        esd_arrays={
            f"epoch_{epoch:03d}_{layer}": np.asarray([1.0, 2.0])
            for layer in ("fc1", "fc2", "fc3")
        },
    )


class MNISTRunnerTests(unittest.TestCase):
    def test_two_epoch_run_writes_validation_selected_restart_artifacts(self):
        config = BaselineConfig(
            optimizer="adamw",
            epochs=2,
            batch_size=40,
            validation_size=20,
            adamw_warmup_epochs=1,
            save_epoch_checkpoints=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            with (
                mock.patch(
                    "rg_baselines.runner.datasets.MNIST",
                    side_effect=fake_mnist_factory,
                ),
                mock.patch(
                    "rg_baselines.runner.measure_weightwatcher_checkpoint",
                    side_effect=fake_weightwatcher,
                ),
                mock.patch(
                    "rg_baselines.runner.attach_correlation_traps",
                    side_effect=lambda checkpoint: checkpoint,
                ),
            ):
                result = run_baseline(
                    config,
                    data_dir=root / "data",
                    device=torch.device("cpu"),
                    output_dir=run_dir,
                    progress=False,
                )
                self.assertEqual(result.performance["epoch"].tolist(), [0, 1, 2])
                self.assertIn("validation_loss", result.performance)
                self.assertTrue(
                    result.performance["test_monitoring_only"].eq(1).all()
                )
                self.assertTrue((run_dir / "checkpoint_latest.pt").is_file())
                self.assertTrue((run_dir / "checkpoint_best.pt").is_file())
                self.assertTrue((run_dir / "run_complete.json").is_file())
                self.assertTrue((run_dir / "test_results.json").is_file())
                completion = json.loads(
                    (run_dir / "run_complete.json").read_text()
                )
                self.assertTrue(completion["completed"])
                self.assertIn(completion["best_validation_epoch"], (0, 1, 2))

                # Removing only completion/final markers exercises the
                # compatible latest-checkpoint resume path without retraining.
                (run_dir / "run_complete.json").unlink()
                (run_dir / "final_state.pt").unlink()
                resumed = run_baseline(
                    config,
                    data_dir=root / "data",
                    device=torch.device("cpu"),
                    output_dir=run_dir,
                    progress=False,
                    resume=True,
                )
                self.assertEqual(resumed.performance["epoch"].tolist(), [0, 1, 2])
                self.assertTrue((run_dir / "run_complete.json").is_file())
                self.assertTrue((run_dir / "final_state.pt").is_file())


if __name__ == "__main__":
    unittest.main()

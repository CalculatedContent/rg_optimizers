import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from rg_baselines.vit_final import ViTBaselineConfig, run_vit_baseline


class FakeCIFAR10(Dataset):
    def __init__(self, *, train: bool, transform=None):
        self.train = bool(train)
        self.transform = transform
        self.length = 96 if train else 24

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(index + (0 if self.train else 10_000))
        array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        image = Image.fromarray(array, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(index % 10)


def fake_cifar_factory(root, train, download, transform):
    return FakeCIFAR10(train=train, transform=transform)


def fake_weightwatcher_snapshot(model, epoch, config):
    rows = []
    for matrix_type in (
        "W_Q",
        "W_K",
        "W_V",
        "W_O",
        "W_MLP_IN",
        "W_MLP_OUT",
    ):
        rows.append(
            {
                "epoch": int(epoch),
                "matrix_name": f"L00_{matrix_type}",
                "matrix_type": matrix_type,
                "block": 0,
                "alpha": 2.0,
                "ERG_gap": 0,
                "num_traps": 0,
                "detX_num": 4,
                "num_pl_spikes": 4,
            }
        )
    return pd.DataFrame(rows)


class FinalViTRunnerTests(unittest.TestCase):
    def test_two_epoch_reference_run_writes_restart_and_selection_artifacts(self):
        config = ViTBaselineConfig(
            epochs=2,
            batch_size=8,
            validation_size=16,
            embed_dim=24,
            depth=1,
            num_heads=3,
            sgd_warmup_epochs=1,
            adamw_warmup_epochs=1,
            muon_warmup_epochs=1,
            cooldown_epochs=0,
            checkpoint_every=1,
            ww_min_evals=2,
            random_erasing_probability=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch(
                    "rg_baselines.vit_cifar10.datasets.CIFAR10",
                    side_effect=fake_cifar_factory,
                ),
                mock.patch(
                    "rg_baselines.vit_cifar10._ww_snapshot",
                    side_effect=fake_weightwatcher_snapshot,
                ),
            ):
                history, spectral = run_vit_baseline(
                    "adamw",
                    17,
                    data_dir=root / "data",
                    output_dir=root / "runs",
                    config=config,
                    device=torch.device("cpu"),
                    progress=False,
                    resume=True,
                )

            run_dir = root / "runs" / "adamw" / "seed_17"
            self.assertEqual(history["epoch"].astype(int).tolist(), [0, 1, 2])
            self.assertEqual(set(spectral["epoch"].astype(int)), {0, 1, 2})
            self.assertTrue(history["test_monitoring_only"].astype(int).eq(1).all())
            for filename in (
                "checkpoint_latest.pt",
                "checkpoint_best.pt",
                "history.csv",
                "weightwatcher_by_epoch_layer.csv",
                "test_results.json",
                "run_complete.json",
            ):
                self.assertTrue((run_dir / filename).is_file(), filename)

            completion = json.loads(
                (run_dir / "run_complete.json").read_text(encoding="utf-8")
            )
            self.assertTrue(completion["completed"])
            self.assertIn(completion["best_validation_epoch"], (0, 1, 2))
            best = torch.load(
                run_dir / "checkpoint_best.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                int(best["epoch"]), int(completion["best_validation_epoch"])
            )


if __name__ == "__main__":
    unittest.main()

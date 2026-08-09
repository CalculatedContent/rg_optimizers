from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rg_baselines.config import BaselineConfig
from rg_baselines.io_utils import atomic_csv, atomic_npz


class BaselineConfigCleanupTests(unittest.TestCase):
    def test_optional_limits_are_validated(self) -> None:
        invalid = (
            BaselineConfig(
                optimizer="adamw",
                train_eval_max_batches=0,
            ),
            BaselineConfig(
                optimizer="adamw",
                ww_min_evals=8,
                ww_max_evals=7,
            ),
            BaselineConfig(
                optimizer="adamw",
                ww_svd_method="",
            ),
            BaselineConfig(
                optimizer="sgd_momentum_muon",
                muon_parameter_names=(),
            ),
            BaselineConfig(
                optimizer="sgd_momentum_muon",
                muon_parameter_names=(
                    "fc1.weight",
                    "fc1.weight",
                ),
            ),
            BaselineConfig(
                optimizer="sgd_momentum_muon",
                muon_momentum=0.0,
                muon_nesterov=True,
            ),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()

    def test_seed_and_dampening_ranges_are_validated(self) -> None:
        for config in (
            BaselineConfig(optimizer="adamw", seed=-1),
            BaselineConfig(optimizer="adamw", split_seed=-1),
            BaselineConfig(
                optimizer="sgd_momentum",
                sgd_dampening=1.0,
                sgd_nesterov=False,
            ),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()


class AtomicPersistenceTests(unittest.TestCase):
    def test_csv_and_npz_replace_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "progress.csv"
            csv_path.write_text("old\n", encoding="utf-8")
            frame = pd.DataFrame(
                [
                    {"epoch": 0, "loss": 1.0},
                    {"epoch": 1, "loss": 0.5},
                ]
            )
            atomic_csv(frame, csv_path)
            pd.testing.assert_frame_equal(pd.read_csv(csv_path), frame)
            self.assertFalse(
                csv_path.with_suffix(".csv.tmp").exists()
            )

            npz_path = root / "history.npz"
            npz_path.write_bytes(b"old")
            arrays = {
                "epoch_000": np.asarray([1.0, 2.0]),
                "epoch_001": np.asarray([3.0, 4.0]),
            }
            atomic_npz(arrays, npz_path)
            with np.load(npz_path) as archive:
                self.assertEqual(set(archive.files), set(arrays))
                for name, expected in arrays.items():
                    np.testing.assert_array_equal(
                        archive[name], expected
                    )
            self.assertFalse(
                npz_path.with_suffix(".npz.tmp").exists()
            )


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import numpy as np
import pandas as pd

from rg_baselines.statistics import (
    require_complete_summary,
    student_t_critical_95,
    summarize_numeric_metrics,
)


class ReplicateStatisticsTests(unittest.TestCase):
    def test_three_seed_student_t_interval(self) -> None:
        frame = pd.DataFrame(
            {
                "epoch": [0, 0, 0],
                "seed": [1, 2, 3],
                "test_accuracy": [1.0, 2.0, 3.0],
            }
        )
        summary = summarize_numeric_metrics(
            frame,
            group_columns=("epoch",),
            metrics=("test_accuracy",),
        )
        row = summary.iloc[0]
        expected_half_width = student_t_critical_95(3) / math.sqrt(3.0)
        self.assertEqual(int(row["n"]), 3)
        self.assertAlmostEqual(float(row["mean"]), 2.0, places=12)
        self.assertAlmostEqual(float(row["std"]), 1.0, places=12)
        self.assertAlmostEqual(
            float(row["ci_half_width"]),
            expected_half_width,
            places=12,
        )
        self.assertAlmostEqual(
            float(row["ci_low"]),
            2.0 - expected_half_width,
            places=12,
        )
        self.assertAlmostEqual(
            float(row["ci_high"]),
            2.0 + expected_half_width,
            places=12,
        )

    def test_summary_is_per_epoch_and_layer(self) -> None:
        rows = []
        for seed, offset in ((11, -0.1), (22, 0.0), (33, 0.1)):
            for epoch in (0, 1):
                for layer in ("fc1", "fc2", "fc3"):
                    rows.append(
                        {
                            "run": "baseline",
                            "optimizer": "sgd_momentum",
                            "layer": layer,
                            "epoch": epoch,
                            "seed": seed,
                            "alpha": 2.0 + offset + 0.01 * epoch,
                            "ERG_gap": 10.0 + offset,
                        }
                    )
        summary = summarize_numeric_metrics(
            pd.DataFrame(rows),
            group_columns=("run", "optimizer", "layer", "epoch"),
            metrics=("alpha", "ERG_gap"),
        )
        self.assertEqual(len(summary), 3 * 2 * 2)
        self.assertTrue((summary["n"] == 3).all())
        self.assertTrue(np.isfinite(summary["ci_half_width"]).all())
        require_complete_summary(
            summary,
            expected_replicates=3,
            required_metrics=("alpha", "ERG_gap"),
        )

    def test_duplicate_or_missing_replicate_is_detected(self) -> None:
        summary = pd.DataFrame(
            {
                "metric": ["test_accuracy"],
                "epoch": [0],
                "n": [2],
                "mean": [0.9],
                "sem": [0.01],
                "ci_half_width": [0.04],
                "ci_low": [0.86],
                "ci_high": [0.94],
            }
        )
        with self.assertRaises(RuntimeError):
            require_complete_summary(
                summary,
                expected_replicates=3,
                required_metrics=("test_accuracy",),
            )


if __name__ == "__main__":
    unittest.main()

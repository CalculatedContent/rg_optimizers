import unittest

import pandas as pd

from rg_baselines.config import BaselineConfig
from rg_baselines.diagnostics import SpectralCheckpoint
from rg_baselines.trap_metrics import (
    attach_correlation_traps,
    correlation_trap_count_from_row,
)


class CorrelationTrapTests(unittest.TestCase):
    def test_canonical_num_traps(self):
        self.assertEqual(correlation_trap_count_from_row(pd.Series({"num_traps": 3})), 3)

    def test_randomized_mp_alias(self):
        self.assertEqual(
            correlation_trap_count_from_row(pd.Series({"rand_num_spikes": 4.0})), 4
        )

    def test_aliases_must_agree(self):
        with self.assertRaises(ValueError):
            correlation_trap_count_from_row(
                pd.Series({"num_traps": 2, "rand_num_spikes": 3})
            )

    def test_negative_or_fractional_counts_rejected(self):
        for value in (-1, 1.5):
            with self.assertRaises(ValueError):
                correlation_trap_count_from_row(pd.Series({"num_traps": value}))

    def test_randomized_analysis_is_required(self):
        config = BaselineConfig(optimizer="adamw", ww_randomize=False)
        with self.assertRaises(ValueError):
            config.validate()

    def test_attach_traps_by_layer_id(self):
        checkpoint = SpectralCheckpoint(
            details=pd.DataFrame(
                {
                    "layer_id": [1, 2, 3],
                    "num_traps": [0, 2, 1],
                    "rand_num_spikes": [0, 2, 1],
                }
            ),
            metrics=pd.DataFrame(
                {
                    "layer_id": [1, 2, 3],
                    "status": ["ok", "ok", "ok"],
                    "layer": ["fc1", "fc2", "fc3"],
                }
            ),
            esd_arrays={},
        )
        result = attach_correlation_traps(checkpoint)
        self.assertEqual(result.metrics["num_traps"].tolist(), [0, 2, 1])
        self.assertTrue(result.metrics["num_traps_source"].str.contains("randomize=True").all())


if __name__ == "__main__":
    unittest.main()

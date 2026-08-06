import unittest

import numpy as np

from rg_baselines.diagnostics import spectral_metrics_from_esd


class SpectralMetricsTests(unittest.TestCase):
    def test_original_boundaries_and_midpoint(self) -> None:
        raw = np.arange(1.0, 11.0)
        normalized = raw * (len(raw) / raw.sum())
        metrics = spectral_metrics_from_esd(
            raw,
            normalized,
            detx_num=8,
            num_pl_spikes=4,
            erg_gap=4,
        )
        self.assertEqual(metrics["m_midpoint"], 6)
        self.assertEqual(metrics["ERG_gap"], 4)
        self.assertAlmostEqual(metrics["rescaled_eigenvalue_sum"], 10.0)
        self.assertAlmostEqual(metrics["rescale_sum_minus_num_eigenvalues"], 0.0)
        self.assertGreater(metrics["midpoint_energy_fraction"], 0.5)

    def test_gap_mismatch_is_rejected(self) -> None:
        raw = np.arange(1.0, 11.0)
        normalized = raw * (len(raw) / raw.sum())
        with self.assertRaises(ValueError):
            spectral_metrics_from_esd(
                raw,
                normalized,
                detx_num=8,
                num_pl_spikes=4,
                erg_gap=3,
            )


if __name__ == "__main__":
    unittest.main()

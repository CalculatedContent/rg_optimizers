import unittest

import numpy as np

from rg_baselines.diagnostics import (
    clean_positive_eigenvalues,
    spectral_metrics_from_esd,
)


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
            expected_dimension=len(raw),
        )
        self.assertEqual(metrics["m_midpoint"], 6)
        self.assertEqual(metrics["ERG_gap"], 4)
        self.assertAlmostEqual(metrics["rescaled_eigenvalue_sum"], 10.0)
        self.assertAlmostEqual(
            metrics["rescale_sum_minus_num_eigenvalues"],
            0.0,
        )
        self.assertGreater(metrics["midpoint_energy_fraction"], 0.5)

    def test_trace_log_matches_analytic_top_spectrum_value(self) -> None:
        raw = np.asarray([1.0, 2.0, 4.0, 8.0])
        normalized = raw * (len(raw) / raw.sum())
        metrics = spectral_metrics_from_esd(
            raw,
            normalized,
            detx_num=4,
            num_pl_spikes=2,
            erg_gap=2,
            expected_dimension=4,
        )

        retained = normalized[::-1][:3]
        expected_total = float(np.sum(np.log(retained)))
        expected_per_eval = float(np.mean(np.log(retained)))
        self.assertEqual(metrics["m_midpoint"], 3)
        self.assertAlmostEqual(
            metrics["trace_log_midpoint_total"],
            expected_total,
        )
        self.assertAlmostEqual(
            metrics["trace_log_midpoint_per_eval"],
            expected_per_eval,
        )
        self.assertAlmostEqual(
            metrics["geometric_mean_midpoint"],
            float(np.exp(expected_per_eval)),
        )
        self.assertAlmostEqual(
            metrics["trace_log_midpoint_total"],
            3.0 * metrics["trace_log_midpoint_per_eval"],
        )

    def test_gap_mismatch_is_rejected(self) -> None:
        raw = np.arange(1.0, 11.0)
        normalized = raw * (len(raw) / raw.sum())
        with self.assertRaisesRegex(ValueError, "ERG_gap audit failed"):
            spectral_metrics_from_esd(
                raw,
                normalized,
                detx_num=8,
                num_pl_spikes=4,
                erg_gap=3,
                expected_dimension=len(raw),
            )

    def test_out_of_range_boundaries_are_rejected(self) -> None:
        raw = np.arange(1.0, 6.0)
        normalized = raw * (len(raw) / raw.sum())
        for field, detx_num, num_pl_spikes in (
            ("detX_num", 6, 2),
            ("num_pl_spikes", 4, 0),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    spectral_metrics_from_esd(
                        raw,
                        normalized,
                        detx_num=detx_num,
                        num_pl_spikes=num_pl_spikes,
                        erg_gap=detx_num - num_pl_spikes,
                        expected_dimension=len(raw),
                    )

    def test_rank_deficient_full_esd_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-deficient ESD"):
            clean_positive_eigenvalues(
                [0.0, 1.0, 2.0],
                expected_dimension=3,
            )

    def test_incomplete_full_esd_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ESD dimension mismatch"):
            clean_positive_eigenvalues(
                [1.0, 2.0],
                expected_dimension=3,
            )

    def test_incorrect_weightwatcher_normalization_is_rejected(self) -> None:
        raw = np.asarray([1.0, 2.0, 3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "normalization audit failed"):
            spectral_metrics_from_esd(
                raw,
                raw,
                detx_num=4,
                num_pl_spikes=2,
                erg_gap=2,
                expected_dimension=4,
            )


if __name__ == "__main__":
    unittest.main()

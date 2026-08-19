from __future__ import annotations

import unittest

import pandas as pd

from rg_baselines.tangent_rg.reporting import qualify_fixed_point


class ReportingTests(unittest.TestCase):
    def test_qualification_requires_persistence_and_fit_quality(self):
        rows = []
        for step in range(1, 6):
            rows.append(
                {
                    "optimizer": "muon",
                    "seed": 1337,
                    "layer": "fc1.weight",
                    "step": step,
                    "alpha": 2.05,
                    "ks_D": 0.08,
                    "n_tail": 20,
                    "trace_log_per_eval": 0.02,
                    "support_selected_from_same_trace_log": False,
                }
            )
        result = qualify_fixed_point(pd.DataFrame(rows))
        self.assertTrue(bool(result.iloc[0]["fixed_point_qualified"]))

        bad = pd.DataFrame(rows)
        bad.loc[bad["step"] == 5, "ks_D"] = 0.9
        result = qualify_fixed_point(bad, required_fraction=1.0)
        self.assertFalse(bool(result.iloc[0]["fixed_point_qualified"]))

    def test_same_curve_support_is_rejected(self):
        frame = pd.DataFrame(
            [
                {
                    "optimizer": "muon",
                    "seed": 1337,
                    "layer": "fc1.weight",
                    "step": 1,
                    "alpha": 2.0,
                    "ks_D": 0.01,
                    "n_tail": 20,
                    "trace_log_per_eval": 0.0,
                    "support_selected_from_same_trace_log": True,
                }
            ]
        )
        with self.assertRaises(ValueError):
            qualify_fixed_point(frame, persistence_measurements=1)


if __name__ == "__main__":
    unittest.main()

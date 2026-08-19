from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from rg_baselines.tangent_rg.powerlaw_fit import (
    amplitude_fit_to_energy,
    fit_clipping_sensitivity,
    fit_powerlaw,
    qualify_replicated_group_fits,
)
from rg_baselines.tangent_rg.trace_log import (
    nearest_trace_log_zero,
    trace_log_at_rank,
    trace_log_passes,
)


class _FakeDistribution:
    alpha = 3.0
    sigma = 0.4
    D = 0.07


class _FakeFit:
    calls: list[tuple[np.ndarray, dict]] = []

    def __init__(self, data, **kwargs):
        self.__class__.calls.append((np.asarray(data), dict(kwargs)))
        if "xmin" in kwargs or "xmax" in kwargs:
            raise AssertionError("the suite must let powerlaw.Fit select xmin")
        self.xmin = float(np.asarray(data)[2])
        self.power_law = _FakeDistribution()


class PowerLawContractTests(unittest.TestCase):
    def setUp(self):
        _FakeFit.calls.clear()
        self.fake_module = types.ModuleType("powerlaw")
        self.fake_module.Fit = _FakeFit

    def test_package_selects_xmin_and_clipping_is_only_sensitivity(self):
        values = np.geomspace(1.0, 1_000.0, 24)
        with mock.patch.dict(sys.modules, {"powerlaw": self.fake_module}):
            frame = fit_clipping_sensitivity(
                values,
                top_k_values=(0, 1, 3),
                minimum_tail=5,
                operator_kind="test_operator",
                map_definition="x -> x",
            )
        self.assertEqual(list(frame["clip_top_k"]), [0, 1, 3])
        self.assertEqual(frame.iloc[0]["selection_role"], "primary")
        self.assertTrue((frame.iloc[1:]["selection_role"] == "sensitivity_only").all())
        self.assertFalse(frame["sensitivity_selected"].any())
        self.assertEqual(len(_FakeFit.calls), 3)
        for _, kwargs in _FakeFit.calls:
            self.assertEqual(kwargs, {"discrete": False, "verbose": False})

    def test_amplitude_energy_transform_does_not_refit(self):
        with mock.patch.dict(sys.modules, {"powerlaw": self.fake_module}):
            row = fit_powerlaw(
                np.geomspace(1.0, 100.0, 20),
                minimum_tail=4,
                operator_kind="test_operator",
                map_definition="x -> x",
            )
        call_count = len(_FakeFit.calls)
        energy = amplitude_fit_to_energy(row)
        self.assertEqual(len(_FakeFit.calls), call_count)
        self.assertAlmostEqual(energy["alpha"], 2.0)
        self.assertAlmostEqual(energy["sigma"], 0.2)
        self.assertAlmostEqual(energy["xmin"], row["xmin"] ** 2)
        self.assertEqual(energy["ks_D"], row["ks_D"])
        self.assertEqual(energy["n_tail"], row["n_tail"])

    def test_two_mode_spectrum_is_retained_as_failed_row_not_an_exception(self):
        with mock.patch.dict(sys.modules, {"powerlaw": self.fake_module}):
            row = fit_powerlaw(
                [1.0, 2.0],
                minimum_tail=8,
                operator_kind="low_rank_test_operator",
                map_definition="two supported modes",
            )
        self.assertFalse(row["fit_ok"])
        self.assertEqual(row["n_used"], 2)
        self.assertIn("minimum_tail=8", row["warning"])
        self.assertEqual(_FakeFit.calls, [])

    def test_replicated_esd_clips_whole_groups_and_uses_group_tail_gate(self):
        groups = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
        multiplicity = 3
        expanded = np.repeat(groups, multiplicity)
        with mock.patch.dict(sys.modules, {"powerlaw": self.fake_module}):
            amplitudes = fit_clipping_sensitivity(
                expanded,
                top_k_values=(0, multiplicity),
                minimum_tail=5,
                operator_kind="replicated_test_operator",
                map_definition="five groups repeated three times",
            )
        energies = pd.DataFrame(
            [amplitude_fit_to_energy(row) for row in amplitudes.to_dict("records")]
        )
        qualified = qualify_replicated_group_fits(
            pd.concat([amplitudes, energies], ignore_index=True),
            groups,
            group_multiplicity=multiplicity,
            minimum_tail_groups=5,
        )
        primary = qualified[qualified["clip_top_k"].eq(0)]
        clipped = qualified[qualified["clip_top_k"].eq(multiplicity)]
        self.assertTrue(primary["fit_ok"].all())
        self.assertEqual(set(primary["tail_group_count"]), {5})
        self.assertFalse(clipped["fit_ok"].any())
        self.assertEqual(set(clipped["clip_group_count"]), {1})
        self.assertEqual(set(clipped["tail_group_count"]), {4})
        self.assertTrue(clipped["mode_level_fit_ok_before_group_gate"].all())
        self.assertTrue(
            clipped["warning"].str.contains("minimum_tail_groups=5").all()
        )

        malformed = amplitudes.copy()
        malformed.loc[1, "clip_top_k"] = 1
        with self.assertRaises(ValueError):
            qualify_replicated_group_fits(
                malformed,
                groups,
                group_multiplicity=multiplicity,
                minimum_tail_groups=2,
            )

        inconsistent = amplitudes.iloc[[0]].copy()
        inconsistent.loc[:, "n_tail"] = inconsistent["n_tail"] - 1
        with self.assertRaises(ValueError):
            qualify_replicated_group_fits(
                inconsistent,
                groups,
                group_multiplicity=multiplicity,
                minimum_tail_groups=2,
            )

        one_group = qualified.iloc[[0]].copy()
        one_group.loc[:, "xmin"] = 1.0
        one_group.loc[:, "n_total"] = multiplicity
        one_group.loc[:, "n_used"] = multiplicity
        one_group.loc[:, "n_tail"] = multiplicity
        audited = qualify_replicated_group_fits(
            one_group,
            [1.0],
            group_multiplicity=multiplicity,
            minimum_tail_groups=2,
        )
        self.assertEqual(int(audited.iloc[0]["tail_group_count"]), 1)
        self.assertFalse(bool(audited.iloc[0]["fit_ok"]))


class TraceLogContractTests(unittest.TestCase):
    def test_independent_support_can_pass(self):
        row = trace_log_at_rank(
            [0.5, 2.0],
            rank=2,
            normalization_dimension=2.5,
            rank_source="powerlaw_tail_count",
        )
        self.assertAlmostEqual(row["trace_log_total"], 0.0, places=12)
        self.assertTrue(trace_log_passes(row, tolerance_per_eval=1e-9))

    def test_same_curve_nearest_zero_is_never_certificate(self):
        row = nearest_trace_log_zero(
            [0.5, 2.0],
            normalization_dimension=2.5,
        )
        self.assertTrue(row["support_selected_from_same_trace_log"])
        self.assertFalse(trace_log_passes(row, tolerance_per_eval=1.0))


if __name__ == "__main__":
    unittest.main()

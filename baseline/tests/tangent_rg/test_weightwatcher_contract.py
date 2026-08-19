from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from rg_baselines.tangent_rg.training import _joined_spectral_table
from rg_baselines.tangent_rg.weightwatcher_fit import (
    WEIGHTWATCHER_EFFECTIVE_TAIL_SUPPORT_SOURCE,
    WEIGHTWATCHER_INFERRED_INTERNAL_SLICE_FALLBACK_SUPPORT_SOURCE,
    WEIGHTWATCHER_INTERNAL_SLICE_SENSITIVITY_SOURCE,
    WEIGHTWATCHER_REPORTED_FINGER_SENSITIVITY_SOURCE,
    WeightWatcherMeasurement,
    analyze_weightwatcher_dual,
    validate_weightwatcher_measurement,
    weightwatcher_trace_log_rows,
)


class _FakeArray:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def named_parameters(self):
        yield "fc1.weight", _FakeArray(np.diag([1.0, 2.0, 4.0, 5.0]))
        yield "fc2.weight", _FakeArray(np.diag([1.0, 3.0, 4.0, 6.0]))
        yield "fc3.weight", _FakeArray(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))


class _FakeWatcher:
    calls: list[dict] = []

    def __init__(self, model):
        self.model = model

    def analyze(
        self,
        *,
        plot,
        randomize,
        min_evals,
        savefig,
        vectors,
        start_ids,
        ERG,
        fix_fingers,
        max_fingers,
        svd_method,
    ):
        self.__class__.calls.append(
            {
                "ERG": ERG,
                "randomize": randomize,
                "fix_fingers": fix_fingers,
                "max_fingers": max_fingers,
            }
        )
        alpha = 2.1 if fix_fingers else 3.4
        clipped = fix_fingers == "clip_xmax"
        return pd.DataFrame(
            {
                "longname": ["fc1", "fc2", "fc3"],
                "alpha": [alpha] * 3,
                "sigma": [0.2] * 3,
                "D": [0.08] * 3,
                "xmin": [1.0] * 3,
                # The pinned 0.7.7 wheel returns the endpoint of the actual
                # sliced fit.  Two modes lie above it even though the public
                # num_fingers field reports one (idx-1).
                "xmax": [4.0, 9.0, 4.0] if clipped else [25.0, 36.0, 4.0],
                "num_pl_spikes": [4, 4, 2],
                "num_evals": [4, 4, 2],
                "detX_num": [4, 4, 2],
                "ERG_gap": [0, 0, 0],
                "num_traps": [0, 0, 0],
                "num_fingers": [1, 1, 0] if clipped else [0, 0, 0],
                "status": ["success"] * 3,
            }
        )


class _FakeWatcherNoXmax(_FakeWatcher):
    def analyze(self, **kwargs):
        details = super().analyze(**kwargs)
        if kwargs.get("fix_fingers") == "clip_xmax":
            details["xmax"] = None
        return details


class WeightWatcherContractTests(unittest.TestCase):
    def test_dual_fit_propagates_clip_xmax_without_hiding_raw(self):
        _FakeWatcher.calls.clear()
        fake_module = types.ModuleType("weightwatcher")
        fake_module.WeightWatcher = _FakeWatcher
        with mock.patch.dict(sys.modules, {"weightwatcher": fake_module}):
            measurement = analyze_weightwatcher_dual(
                _FakeModel(), max_fingers=7, min_evals=2
            )
        self.assertEqual(len(measurement.details), 6)
        self.assertEqual(set(measurement.details["fit_variant"]), {"raw", "clip_xmax"})
        self.assertEqual(
            [call["fix_fingers"] for call in _FakeWatcher.calls],
            [False, "clip_xmax"],
        )
        self.assertTrue(all(call["ERG"] for call in _FakeWatcher.calls))
        self.assertTrue(all(call["randomize"] for call in _FakeWatcher.calls))
        self.assertEqual(_FakeWatcher.calls[1]["max_fingers"], 7)
        clipped = measurement.details[measurement.details["fit_variant"] == "clip_xmax"]
        self.assertTrue((clipped["selection_role"] == "preregistered_primary").all())
        self.assertTrue(clipped["low_rank_warning"].sum() == 1)
        self.assertEqual(list(clipped["num_pl_spikes"]), [4, 4, 2])
        self.assertEqual(list(clipped["pl_support_rank"]), [4, 4, 2])
        self.assertEqual(list(clipped["n_tail"]), [2, 2, 2])
        np.testing.assert_allclose(clipped["tail_fraction"], [0.5, 0.5, 1.0])
        np.testing.assert_allclose(clipped["backend_xmax"], [4.0, 9.0, 4.0])
        np.testing.assert_allclose(clipped["n_fingers_removed"], [2, 2, 0])
        np.testing.assert_allclose(clipped["num_fingers_reported"], [1, 1, 0])
        np.testing.assert_allclose(
            clipped["num_fingers_inferred_internal_slice"], [2, 2, 0]
        )
        self.assertTrue(
            clipped["effective_tail_support_source"]
            .eq(WEIGHTWATCHER_EFFECTIVE_TAIL_SUPPORT_SOURCE)
            .all()
        )
        self.assertTrue(clipped["fit_ok"].all())

        raw = measurement.details[measurement.details["fit_variant"] == "raw"]
        self.assertEqual(list(raw["n_tail"]), [4, 4, 2])
        validation = validate_weightwatcher_measurement(measurement)
        self.assertTrue(validation.primary_ok)
        self.assertEqual(validation.raw_audit_warnings, ())

    def test_trace_uses_effective_post_clip_tail_and_precise_source(self):
        fake_module = types.ModuleType("weightwatcher")
        fake_module.WeightWatcher = _FakeWatcher
        with mock.patch.dict(sys.modules, {"weightwatcher": fake_module}):
            measurement = analyze_weightwatcher_dual(_FakeModel(), min_evals=2)
        traces = weightwatcher_trace_log_rows(measurement)
        primary = traces[
            traces["support_rank_source"].eq(
                WEIGHTWATCHER_EFFECTIVE_TAIL_SUPPORT_SOURCE
            )
        ].sort_values("layer")
        self.assertEqual(len(primary), 3)
        self.assertEqual(list(primary["effective_fit_tail_rank"]), [2, 2, 2])
        self.assertEqual(list(primary["pl_support_rank_before_finger_clip"]), [4, 4, 2])
        self.assertEqual(
            list(primary["support_window_start_descending_zero_based"]), [2, 2, 0]
        )
        self.assertFalse(primary["support_selected_from_same_trace_log"].any())
        self.assertTrue(primary["certification_eligible"].all())

        reported = traces[
            traces["support_rank_source"].eq(
                WEIGHTWATCHER_REPORTED_FINGER_SENSITIVITY_SOURCE
            )
        ].sort_values("layer")
        inferred = traces[
            traces["support_rank_source"].eq(
                WEIGHTWATCHER_INTERNAL_SLICE_SENSITIVITY_SOURCE
            )
        ].sort_values("layer")
        self.assertEqual(list(reported["support_rank"]), [3, 3])
        self.assertEqual(list(inferred["support_rank"]), [2, 2])
        self.assertTrue(reported["sensitivity_only"].all())
        self.assertTrue(inferred["sensitivity_only"].all())
        self.assertFalse(reported["certification_eligible"].any())
        self.assertFalse(inferred["certification_eligible"].any())

        raw_traces = weightwatcher_trace_log_rows(
            measurement, fit_variant="raw"
        ).sort_values(["layer", "support_rank_source"])
        raw_midpoint = raw_traces[
            raw_traces["support_rank_source"].eq("weightwatcher_midpoint")
        ].sort_values("layer")
        self.assertEqual(len(raw_midpoint), 3)
        self.assertEqual(list(raw_midpoint["support_rank"]), [4, 4, 2])
        self.assertEqual(list(raw_midpoint["normalization_dimension"]), [4.0, 4.0, 2.0])
        self.assertTrue(raw_midpoint["support_selected_from_same_trace_log"].all())

        metadata = {
            "optimizer": "muon",
            "seed": 1337,
            "epoch": 10,
            "global_step": 4300,
        }
        fits = measurement.details.copy()
        for key, value in metadata.items():
            fits[key] = value
        joined = _joined_spectral_table(
            {
                "weightwatcher": fits,
                "trace_log": weightwatcher_trace_log_rows(
                    measurement, metadata=metadata
                ),
            }
        )
        joined_primary = joined[joined["fit_variant"].eq("clip_xmax")]
        self.assertTrue(joined_primary["trace_log_per_eval"].notna().all())
        self.assertTrue(
            joined_primary["support_rank_source"].eq(
                WEIGHTWATCHER_EFFECTIVE_TAIL_SUPPORT_SOURCE
            ).all()
        )

    def test_trace_window_excludes_a_huge_removed_top_finger(self):
        values = np.array([1.0, 2.0, 1.0e12])
        details = pd.DataFrame(
            [
                {
                    "layer": "fc1.weight",
                    "fit_variant": "clip_xmax",
                    "n_tail": 2,
                    "n_tail_fit": 2,
                    "num_fingers": 1,
                    "num_fingers_reported": 1,
                    "num_fingers_inferred_internal_slice": 2,
                    "n_fingers_removed": 1,
                    "pl_support_rank": 3,
                    "reported_count_tail_sensitivity": 2,
                    "inferred_internal_slice_tail_sensitivity": 1,
                    "backend_xmax": 2.0,
                    "removed_above_backend_xmax": 1,
                    "reported_minus_endpoint_removed": 0,
                    "inferred_slice_minus_endpoint_removed": 1,
                    "primary_tail_membership_source": (
                        "exact_saved_esd_membership_in_[xmin,backend_xmax]"
                    ),
                    "effective_tail_support_source": (
                        WEIGHTWATCHER_EFFECTIVE_TAIL_SUPPORT_SOURCE
                    ),
                    "detX_num": 2,
                }
            ]
        )
        measurement = WeightWatcherMeasurement(details, {"fc1.weight": values})
        traces = weightwatcher_trace_log_rows(measurement)
        primary = traces[
            traces["support_rank_source"].eq(
                WEIGHTWATCHER_EFFECTIVE_TAIL_SUPPORT_SOURCE
            )
        ].iloc[0]
        normalized_descending = (values * (3.0 / values.sum()))[::-1]
        expected = float(np.log(normalized_descending[1:3]).sum())
        naive_top_two = float(np.log(normalized_descending[:2]).sum())
        self.assertAlmostEqual(float(primary["trace_log_total"]), expected)
        self.assertNotAlmostEqual(float(primary["trace_log_total"]), naive_top_two)
        self.assertEqual(
            int(primary["support_window_start_descending_zero_based"]), 1
        )
        self.assertEqual(int(primary["support_window_end_descending_exclusive"]), 3)

    def test_missing_backend_xmax_uses_inferred_internal_slice_fallback(self):
        fake_module = types.ModuleType("weightwatcher")
        fake_module.WeightWatcher = _FakeWatcherNoXmax
        with mock.patch.dict(sys.modules, {"weightwatcher": fake_module}):
            measurement = analyze_weightwatcher_dual(_FakeModel(), min_evals=2)
        clipped = measurement.details[
            measurement.details["fit_variant"].eq("clip_xmax")
        ].sort_values("layer")
        self.assertEqual(list(clipped["n_tail"]), [2, 2, 2])
        self.assertEqual(list(clipped["n_fingers_removed"]), [2, 2, 0])
        self.assertTrue(clipped["backend_xmax"].isna().all())
        self.assertTrue(
            clipped["effective_tail_support_source"]
            .eq(WEIGHTWATCHER_INFERRED_INTERNAL_SLICE_FALLBACK_SUPPORT_SOURCE)
            .all()
        )
        traces = weightwatcher_trace_log_rows(measurement)
        primary = traces[
            traces["support_rank_source"].eq(
                WEIGHTWATCHER_INFERRED_INTERNAL_SLICE_FALLBACK_SUPPORT_SOURCE
            )
        ]
        self.assertEqual(len(primary), 3)
        self.assertFalse(primary["sensitivity_only"].any())
        inferred_sensitivity = traces[
            traces["support_rank_source"].eq(
                WEIGHTWATCHER_INTERNAL_SLICE_SENSITIVITY_SOURCE
            )
        ]
        self.assertTrue(inferred_sensitivity.empty)

    def test_fit_failures_reject_qualification_but_not_acquisition(self):
        fake_module = types.ModuleType("weightwatcher")
        fake_module.WeightWatcher = _FakeWatcher
        with mock.patch.dict(sys.modules, {"weightwatcher": fake_module}):
            measurement = analyze_weightwatcher_dual(_FakeModel(), min_evals=2)

        raw_failed = measurement.details.copy()
        selector = raw_failed["fit_variant"].eq("raw") & raw_failed["layer"].eq(
            "fc1.weight"
        )
        raw_failed.loc[selector, "alpha"] = np.nan
        raw_failed.loc[selector, "fit_ok"] = False
        raw_report = validate_weightwatcher_measurement(
            WeightWatcherMeasurement(raw_failed, measurement.esds)
        )
        self.assertTrue(raw_report.primary_ok)
        self.assertTrue(raw_report.acquisition_usable)
        self.assertTrue(raw_report.raw_audit_warnings)

        primary_failed = measurement.details.copy()
        selector = primary_failed["fit_variant"].eq("clip_xmax") & primary_failed[
            "layer"
        ].eq("fc1.weight")
        primary_failed.loc[selector, "num_fingers"] = 99
        primary_failed.loc[selector, "n_fingers_removed"] = 99
        primary_failed.loc[selector, "finger_count_valid"] = False
        primary_failed.loc[selector, "fit_ok"] = False
        primary_report = validate_weightwatcher_measurement(
            WeightWatcherMeasurement(primary_failed, measurement.esds)
        )
        self.assertFalse(primary_report.primary_ok)
        self.assertTrue(primary_report.acquisition_usable)
        self.assertTrue(primary_report.primary_fit_failures)
        self.assertTrue(
            any("num_fingers" in message for message in primary_report.primary_errors)
        )

    def test_duplicate_or_missing_primary_layer_is_rejected(self):
        fake_module = types.ModuleType("weightwatcher")
        fake_module.WeightWatcher = _FakeWatcher
        with mock.patch.dict(sys.modules, {"weightwatcher": fake_module}):
            measurement = analyze_weightwatcher_dual(_FakeModel(), min_evals=2)
        details = measurement.details
        duplicate = pd.concat(
            [
                details,
                details[
                    details["fit_variant"].eq("clip_xmax")
                    & details["layer"].eq("fc1.weight")
                ],
            ],
            ignore_index=True,
        )
        report = validate_weightwatcher_measurement(
            WeightWatcherMeasurement(duplicate, measurement.esds)
        )
        self.assertFalse(report.acquisition_usable)
        self.assertTrue(
            any("expected exactly one row" in message for message in report.primary_errors)
        )


if __name__ == "__main__":
    unittest.main()

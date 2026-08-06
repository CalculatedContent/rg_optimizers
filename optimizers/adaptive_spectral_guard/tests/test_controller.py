import unittest

import pandas as pd

from adaptive_spectral_guard import GuardConfig, preset_policies
from adaptive_spectral_guard.controller import AdaptiveSpectralController


def row(epoch, alpha, midpoint=100, erg_gap=0, overlap=1.0, beta=0.0):
    return {
        "parameter_name": "fc1.weight",
        "epoch": epoch,
        "status": "ok",
        "alpha": alpha,
        "alpha_source": "WeightWatcher",
        "ERG_gap": erg_gap,
        "ERG_gap_source": "WeightWatcher",
        "detX_num": midpoint,
        "num_pl_spikes": midpoint - erg_gap,
        "m_midpoint": midpoint,
        "boundary_overlap_ratio": overlap,
        "beta_E_midpoint": beta,
        "scale_balance_reliable": True,
    }


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = AdaptiveSpectralController(
            GuardConfig(policies=preset_policies("adaptive"))
        )

    def test_hysteresis_and_alpha_trend(self):
        self.controller.update_from_weightwatcher(
            pd.DataFrame([row(0, 2.40)])
        )
        self.assertEqual(
            self.controller.get_state("fc1.weight").regime,
            "off",
        )

        self.controller.update_from_weightwatcher(
            pd.DataFrame([row(1, 2.25)])
        )
        state = self.controller.get_state("fc1.weight")
        self.assertEqual(state.regime, "weak")
        self.assertGreater(state.effective_gain, 0.0)

        self.controller.update_from_weightwatcher(
            pd.DataFrame([row(2, 1.95, beta=0.2)])
        )
        state = self.controller.get_state("fc1.weight")
        self.assertEqual(state.regime, "strong")
        self.assertTrue(state.shape_active)

        self.controller.update_from_weightwatcher(
            pd.DataFrame([row(3, 2.20)])
        )
        self.assertEqual(
            self.controller.get_state("fc1.weight").regime,
            "weak",
        )
        self.controller.update_from_weightwatcher(
            pd.DataFrame([row(4, 2.21)])
        )
        self.assertEqual(
            self.controller.get_state("fc1.weight").regime,
            "off",
        )

    def test_unstable_support_reduces_confidence(self):
        self.controller.update_from_weightwatcher(
            pd.DataFrame([row(0, 2.0, midpoint=100)])
        )
        first = self.controller.get_state("fc1.weight").confidence
        self.controller.update_from_weightwatcher(
            pd.DataFrame([row(1, 1.95, midpoint=40)])
        )
        second = self.controller.get_state("fc1.weight").confidence
        self.assertLess(second, first)

    def test_task_conflict_throttles_layer(self):
        self.controller.update_from_weightwatcher(
            pd.DataFrame([row(0, 1.95)])
        )
        before = self.controller.get_state("fc1.weight").effective_gain
        stats = pd.DataFrame(
            {
                "parameter": ["fc1.weight"] * 10,
                "task_conflict_ratio_pre": [1.0] * 10,
            }
        )
        self.controller.observe_task_feedback(stats)
        state = self.controller.get_state("fc1.weight")
        self.assertLess(state.task_throttle, 1.0)
        self.assertLess(state.effective_gain, before)


if __name__ == "__main__":
    unittest.main()

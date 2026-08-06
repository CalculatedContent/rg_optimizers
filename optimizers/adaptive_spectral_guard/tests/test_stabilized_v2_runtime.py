import unittest

import pandas as pd

from adaptive_spectral_guard import (
    STABILIZED_V2_API,
    assert_stabilized_v2_controller_frame,
    build_stabilized_v2_configuration,
    run_stabilized_v2_preflight,
)


class StabilizedV2RuntimeTests(unittest.TestCase):
    def test_preflight_keeps_volume_active_and_vetoes_shape(self):
        configuration = build_stabilized_v2_configuration(epochs=3)
        frame = run_stabilized_v2_preflight(configuration.guard)
        row = frame.loc[frame["parameter"].eq("fc1.weight")].iloc[0]
        self.assertGreaterEqual(STABILIZED_V2_API, 2)
        self.assertEqual(row["regime"], "strong")
        self.assertGreater(float(row["volume_effective_gain"]), 0.0)
        self.assertEqual(float(row["shape_effective_gain"]), 0.0)
        self.assertNotEqual(row["reason"], "low ECS confidence")

    def test_v1_controller_frame_is_rejected(self):
        configuration = build_stabilized_v2_configuration(epochs=3)
        frame = pd.DataFrame(
            [
                {
                    "parameter": "fc1.weight",
                    "regime": "off",
                    "reason": "low ECS confidence",
                    "alpha": 1.59,
                    "raw_confidence": 0.10,
                    "smoothed_confidence": 0.17,
                    "volume_confidence": 0.0,
                    "shape_confidence": 0.0,
                    "volume_effective_gain": 0.0,
                    "shape_effective_gain": 0.0,
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "V1 all-or-nothing"):
            assert_stabilized_v2_controller_frame(frame, configuration.guard)


if __name__ == "__main__":
    unittest.main()

import unittest
from dataclasses import replace

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_spectral_guard import (
    AdaptiveSpectralController,
    AdaptiveSpectralGuard,
    ControllerConfig,
    GuardConfig,
    preset_policies,
)


def ww_row(
    *,
    epoch: int,
    alpha: float,
    midpoint: int,
    detx: int,
    mpl: int,
    beta: float,
):
    return {
        "parameter_name": "fc1.weight",
        "epoch": epoch,
        "status": "ok",
        "alpha": alpha,
        "alpha_source": "WeightWatcher",
        "ERG_gap": detx - mpl,
        "ERG_gap_source": "WeightWatcher",
        "detX_num": detx,
        "num_pl_spikes": mpl,
        "m_midpoint": midpoint,
        "boundary_overlap_ratio": min(detx, mpl) / max(detx, mpl),
        "beta_E_midpoint": beta,
        "scale_balance_reliable": True,
    }


def stabilized_config():
    return GuardConfig(
        controller=ControllerConfig(
            confidence_ema_decay=0.67,
            separate_channel_confidence=True,
            volume_confidence_floor_below_boundary=0.25,
            volume_confidence_floor_alpha=2.05,
            shape_min_confidence=0.15,
            shape_raw_confidence_floor=0.05,
            shape_requires_alpha_boundary=True,
        ),
        policies=preset_policies("stabilized"),
    )


class StabilizedV2Tests(unittest.TestCase):
    def test_low_raw_confidence_does_not_disable_volume_below_boundary(self):
        controller = AdaptiveSpectralController(stabilized_config())
        controller.update_from_weightwatcher(
            pd.DataFrame(
                [
                    ww_row(
                        epoch=0,
                        alpha=2.12,
                        midpoint=238,
                        detx=254,
                        mpl=223,
                        beta=0.14,
                    )
                ]
            )
        )
        controller.update_from_weightwatcher(
            pd.DataFrame(
                [
                    ww_row(
                        epoch=1,
                        alpha=2.01,
                        midpoint=170,
                        detx=244,
                        mpl=96,
                        beta=0.19,
                    )
                ]
            )
        )
        state = controller.get_state("fc1.weight")
        self.assertNotEqual(state.regime, "off")
        self.assertGreaterEqual(state.volume_confidence, 0.25)
        self.assertGreater(state.volume_effective_gain, 0.0)
        self.assertEqual(state.shape_confidence, 0.0)
        self.assertFalse(state.shape_active)

    def test_confidence_ema_dampens_one_checkpoint_collapse(self):
        controller = AdaptiveSpectralController(stabilized_config())
        controller.update_from_weightwatcher(
            pd.DataFrame(
                [
                    ww_row(
                        epoch=0,
                        alpha=2.12,
                        midpoint=220,
                        detx=240,
                        mpl=210,
                        beta=0.10,
                    )
                ]
            )
        )
        first = controller.get_state("fc1.weight")
        controller.update_from_weightwatcher(
            pd.DataFrame(
                [
                    ww_row(
                        epoch=1,
                        alpha=1.99,
                        midpoint=80,
                        detx=220,
                        mpl=70,
                        beta=0.20,
                    )
                ]
            )
        )
        second = controller.get_state("fc1.weight")
        self.assertLess(second.raw_confidence, first.raw_confidence)
        self.assertGreater(second.smoothed_confidence, second.raw_confidence)

    def test_stabilized_shape_cap_and_deadband_are_applied(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 32, bias=False)

            def forward(self, x):
                return self.fc1(x)

        torch.manual_seed(13)
        model = TinyModel()
        config = stabilized_config()
        policies = dict(config.policies)
        policies["fc1.weight"] = replace(
            policies["fc1.weight"],
            cadence=1,
            min_shape_retained=10,
            min_shape_decades=0.05,
        )
        config = GuardConfig(
            controller=config.controller,
            policies=policies,
        )
        optimizer = AdaptiveSpectralGuard(
            torch.optim.SGD(model.parameters(), lr=0.02),
            model.named_parameters(),
            config=config,
        )
        optimizer.update_from_weightwatcher(
            pd.DataFrame(
                [
                    ww_row(
                        epoch=1,
                        alpha=1.90,
                        midpoint=24,
                        detx=24,
                        mpl=24,
                        beta=0.20,
                    )
                ]
            )
        )

        x = torch.randn(16, 64)
        y = torch.randn(16, 32)
        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        optimizer.step()
        stats = pd.DataFrame(optimizer.pop_step_stats())

        self.assertEqual(len(stats), 1)
        row = stats.iloc[0]
        self.assertAlmostEqual(float(row["beta_E_deadband"]), 0.05, places=6)
        self.assertLessEqual(float(row["shape_correction_ratio"]), 0.020001)
        self.assertLessEqual(float(row["task_conflict_ratio_post"]), 1e-6)


if __name__ == "__main__":
    unittest.main()

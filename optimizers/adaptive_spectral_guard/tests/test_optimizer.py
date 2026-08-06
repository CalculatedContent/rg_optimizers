import unittest
from dataclasses import replace

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_spectral_guard import (
    GuardConfig,
    default_layer_policies,
)
from adaptive_spectral_guard.optimizer import AdaptiveSpectralGuard


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        return self.fc1(x)


def ww_state(alpha=1.90):
    return pd.DataFrame(
        [
            {
                "parameter_name": "fc1.weight",
                "epoch": 1,
                "status": "ok",
                "alpha": alpha,
                "alpha_source": "WeightWatcher",
                "ERG_gap": 0,
                "ERG_gap_source": "WeightWatcher",
                "detX_num": 24,
                "num_pl_spikes": 24,
                "m_midpoint": 24,
                "boundary_overlap_ratio": 1.0,
                "beta_E_midpoint": 0.2,
                "scale_balance_reliable": True,
            }
        ]
    )


class OptimizerTests(unittest.TestCase):
    def test_layer_cap_and_loss_neutral_safeguard(self):
        torch.manual_seed(11)
        model = TinyModel()
        policies = default_layer_policies()
        policies["fc1.weight"] = replace(
            policies["fc1.weight"],
            cadence=1,
            weak_gain=1.0,
            strong_gain=1.0,
            volume_max_ratio=0.10,
            shape_max_ratio=0.05,
            combined_max_ratio=0.12,
            min_shape_retained=10,
            min_shape_decades=0.05,
        )
        config = GuardConfig(policies=policies)
        base = torch.optim.SGD(model.parameters(), lr=0.02)
        optimizer = AdaptiveSpectralGuard(
            base,
            model.named_parameters(),
            config=config,
        )
        optimizer.update_from_weightwatcher(ww_state())

        x = torch.randn(16, 64)
        target = torch.randn(16, 32)
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        stats = pd.DataFrame(optimizer.pop_step_stats())

        self.assertEqual(len(stats), 1)
        row = stats.iloc[0]
        self.assertLessEqual(
            float(row["combined_correction_ratio"]),
            0.120001,
        )
        self.assertLessEqual(
            float(row["task_conflict_ratio_post"]),
            1e-6,
        )
        self.assertEqual(row["parameter"], "fc1.weight")
        self.assertEqual(row["regime"], "strong")

    def test_disabled_layer_is_not_prepared(self):
        torch.manual_seed(3)
        model = TinyModel()
        policies = default_layer_policies()
        policies["fc1.weight"] = replace(
            policies["fc1.weight"],
            enabled=False,
        )
        optimizer = AdaptiveSpectralGuard(
            torch.optim.SGD(model.parameters(), lr=0.01),
            model.named_parameters(),
            config=GuardConfig(policies=policies),
        )
        optimizer.update_from_weightwatcher(ww_state())
        x = torch.randn(8, 64)
        y = torch.randn(8, 32)
        optimizer.zero_grad()
        F.mse_loss(model(x), y).backward()
        optimizer.step()
        self.assertEqual(optimizer.pop_step_stats(), [])


if __name__ == "__main__":
    unittest.main()

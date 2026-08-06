import unittest

import torch

from adaptive_spectral_guard.geometry import (
    loss_neutralize,
    spectral_geometry,
)


class SpectralGeometryTests(unittest.TestCase):
    def test_beta_direction_is_trace_orthogonal_and_reduces_beta(self):
        torch.manual_seed(7)
        weight = torch.randn(64, 32)
        geometry = spectral_geometry(
            weight,
            volume_rank=20,
            shape_rank=24,
            n_shells=5,
            min_shape_retained=10,
            min_shape_decades=0.10,
        )
        self.assertTrue(geometry.beta_reliable)
        self.assertGreater(float(geometry.beta_E), 0.0)
        self.assertLess(
            abs(geometry.trace_beta_inner_product_after),
            1e-5,
        )

        correction = (
            -0.05
            * float(geometry.beta_E)
            / geometry.gradient_norm_sq_beta
            * geometry.beta_gradient
        )
        after = spectral_geometry(
            weight + correction,
            volume_rank=20,
            shape_rank=24,
            n_shells=5,
            min_shape_retained=10,
            min_shape_decades=0.10,
        )
        self.assertLess(float(after.beta_E), float(geometry.beta_E))
        self.assertLess(
            abs(
                float(after.trace_residual)
                - float(geometry.trace_residual)
            ),
            2e-3,
        )

    def test_loss_neutralization_removes_harmful_component(self):
        gradient = torch.tensor([1.0, 0.0])
        base_delta = torch.tensor([-1.0, 0.0])
        attempted = torch.tensor([0.5, 1.0])
        corrected, stats = loss_neutralize(
            attempted,
            gradient,
            base_delta,
        )
        self.assertTrue(stats["loss_neutral_applied"])
        self.assertGreater(stats["task_conflict_ratio_pre"], 0.0)
        self.assertLessEqual(
            stats["task_conflict_ratio_post"],
            1e-9,
        )
        self.assertAlmostEqual(float(corrected[0]), 0.0, places=6)
        self.assertAlmostEqual(float(corrected[1]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

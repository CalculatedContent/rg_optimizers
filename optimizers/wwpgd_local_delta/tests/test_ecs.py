import unittest

import torch

from wwpgd_local_delta.ecs import (
    damp_delta_outside_ecs,
    local_ecs_geometry,
    select_self_consistent_ecs,
    split_delta_by_ecs,
)


class ECSTests(unittest.TestCase):
    def test_self_consistent_scan_returns_valid_rank(self):
        s = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5])
        scan = select_self_consistent_ecs(s, min_retained=2)
        self.assertGreaterEqual(scan.rank, 2)
        self.assertLessEqual(scan.rank, 5)
        self.assertGreaterEqual(scan.normalization_dimension, scan.rank)

    def test_fraction_zero_is_identity(self):
        torch.manual_seed(0)
        w = torch.randn(6, 5)
        delta = torch.randn(6, 5)
        result = damp_delta_outside_ecs(
            delta, w, correction_fraction=0.0, min_retained=2
        )
        self.assertTrue(torch.allclose(result.corrected_delta, delta, atol=1e-5))
        self.assertAlmostEqual(result.removed_fraction_of_base, 0.0, places=6)
        self.assertAlmostEqual(result.observed_orthogonal_damping, 1.0, places=5)

    def test_tall_matrix_projects_on_right(self):
        torch.manual_seed(1)
        w = torch.randn(8, 6)
        delta = torch.randn(8, 6)
        geometry = local_ecs_geometry(w, min_retained=2, max_retained=2)
        self.assertFalse(geometry.transposed)
        self.assertEqual(geometry.projection_side, "right")
        result = damp_delta_outside_ecs(
            delta, w, correction_fraction=1.0, min_retained=2, max_retained=2
        )
        _, orth = split_delta_by_ecs(result.corrected_delta, geometry)
        self.assertLess(float(torch.linalg.vector_norm(orth)), 1e-4)

    def test_wide_matrix_uses_tall_orientation_and_projects_on_left(self):
        torch.manual_seed(2)
        w = torch.randn(6, 8)
        delta = torch.randn(6, 8)
        geometry = local_ecs_geometry(w, min_retained=2, max_retained=2)
        self.assertTrue(geometry.transposed)
        self.assertEqual(geometry.projection_side, "left")
        result = damp_delta_outside_ecs(
            delta, w, correction_fraction=1.0, min_retained=2, max_retained=2
        )
        _, orth = split_delta_by_ecs(result.corrected_delta, geometry)
        self.assertLess(float(torch.linalg.vector_norm(orth)), 1e-4)

    def test_fraction_half_damps_orthogonal_component_by_half(self):
        torch.manual_seed(3)
        w = torch.randn(10, 7)
        delta = torch.randn(10, 7)
        result = damp_delta_outside_ecs(
            delta, w, correction_fraction=0.5, min_retained=3, max_retained=3
        )
        self.assertAlmostEqual(result.observed_orthogonal_damping, 0.5, places=5)
        self.assertLess(result.damping_error, 1e-5)
        self.assertAlmostEqual(
            result.removed_fraction_of_base,
            0.5 * result.orthogonal_fraction,
            places=5,
        )
        self.assertLess(result.pythagorean_error, 1e-5)
        self.assertLess(result.correction_identity_error, 1e-6)


if __name__ == "__main__":
    unittest.main()

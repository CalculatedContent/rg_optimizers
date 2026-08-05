import math
import unittest

import torch

from rg_trace_log.controller import (
    correct_trace_log_component,
    trace_log_geometry,
)


class TraceLogControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.weight = torch.randn(9, 6, dtype=torch.float64) + 0.1
        self.delta = 1e-3 * torch.randn_like(self.weight)

    def test_tangent_projection_removes_first_order_drift(self):
        geometry = trace_log_geometry(
            self.weight, 4, normalization="raw", ridge_relative=0.0
        )
        result = correct_trace_log_component(
            self.delta,
            geometry,
            mode="tangent",
            correction_scale=1.0,
            max_correction_ratio=None,
        )
        drift = torch.sum(geometry.gradient * result.corrected_delta).item()
        self.assertAlmostEqual(drift, 0.0, places=10)

    def test_tracking_matches_linearized_target(self):
        gamma = 0.2
        geometry = trace_log_geometry(
            self.weight, 5, normalization="raw", ridge_relative=0.0
        )
        result = correct_trace_log_component(
            self.delta,
            geometry,
            mode="tracking",
            gamma=gamma,
            correction_scale=1.0,
            max_correction_ratio=None,
        )
        target = -gamma * geometry.residual.item()
        self.assertAlmostEqual(result.corrected_drift, target, places=8)

    def test_one_sided_leaves_expansion_unchanged(self):
        geometry = trace_log_geometry(
            self.weight, 4, normalization="raw", ridge_relative=0.0
        )
        expansion = geometry.gradient.clone()
        result = correct_trace_log_component(
            expansion,
            geometry,
            mode="one_sided",
            correction_scale=1.0,
            max_correction_ratio=None,
        )
        self.assertFalse(result.applied)
        self.assertTrue(torch.equal(result.corrected_delta, expansion))

    def test_one_sided_cancels_contraction(self):
        geometry = trace_log_geometry(
            self.weight, 4, normalization="raw", ridge_relative=0.0
        )
        contraction = -geometry.gradient.clone()
        result = correct_trace_log_component(
            contraction,
            geometry,
            mode="one_sided",
            correction_scale=1.0,
            max_correction_ratio=None,
        )
        drift = torch.sum(geometry.gradient * result.corrected_delta).item()
        self.assertAlmostEqual(drift, 0.0, places=9)

    def test_weightwatcher_gradient_is_radially_orthogonal(self):
        geometry = trace_log_geometry(
            self.weight, 4, normalization="weightwatcher", ridge_relative=0.0
        )
        radial = torch.sum(geometry.gradient * self.weight).item()
        self.assertAlmostEqual(radial, 0.0, places=9)

    def test_transposed_matrix_preserves_shape(self):
        wide = torch.randn(5, 11, dtype=torch.float64)
        geometry = trace_log_geometry(
            wide, 3, normalization="raw", ridge_relative=0.0
        )
        self.assertEqual(tuple(geometry.gradient.shape), tuple(wide.shape))
        self.assertTrue(geometry.transposed)

    def test_finite_difference_matches_raw_gradient(self):
        geometry = trace_log_geometry(
            self.weight, 4, normalization="raw", ridge_relative=0.0
        )
        direction = torch.randn_like(self.weight)
        direction = direction / torch.linalg.vector_norm(direction)
        epsilon = 1e-6
        plus = trace_log_geometry(
            self.weight + epsilon * direction,
            4,
            normalization="raw",
            ridge_relative=0.0,
        ).residual.item()
        minus = trace_log_geometry(
            self.weight - epsilon * direction,
            4,
            normalization="raw",
            ridge_relative=0.0,
        ).residual.item()
        finite_difference = (plus - minus) / (2.0 * epsilon)
        analytic = torch.sum(geometry.gradient * direction).item()
        self.assertTrue(math.isfinite(finite_difference))
        self.assertAlmostEqual(finite_difference, analytic, places=5)


if __name__ == "__main__":
    unittest.main()

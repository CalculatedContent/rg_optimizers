import math
import unittest

import torch

from rg_sc_trace_log.geometry import (
    adaptive_trace_log_geometry,
    correct_trace_log_component,
)


class AdaptiveGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.weight = torch.randn(9, 6, dtype=torch.float64) + 0.1
        self.delta = 1e-3 * torch.randn_like(self.weight)

    def test_cached_normalization_gradient_is_radially_orthogonal(self) -> None:
        geometry = adaptive_trace_log_geometry(
            self.weight,
            fixed_ecs_rank=4,
            fixed_normalization_dimension=5.0,
            normalization_response="frozen",
            ridge_relative=0.0,
        )
        radial = torch.sum(geometry.gradient * self.weight).item()
        self.assertAlmostEqual(radial, 0.0, places=8)
        self.assertTrue(geometry.normalization_dimension_is_cached)

    def test_finite_difference_matches_cached_gradient(self) -> None:
        direction = torch.randn_like(self.weight)
        direction /= torch.linalg.vector_norm(direction)
        epsilon = 1e-6
        kwargs = dict(
            fixed_ecs_rank=4,
            fixed_normalization_dimension=5.0,
            normalization_response="frozen",
            ridge_relative=0.0,
        )
        geometry = adaptive_trace_log_geometry(self.weight, **kwargs)
        plus = adaptive_trace_log_geometry(
            self.weight + epsilon * direction, **kwargs
        ).residual.item()
        minus = adaptive_trace_log_geometry(
            self.weight - epsilon * direction, **kwargs
        ).residual.item()
        finite_difference = (plus - minus) / (2.0 * epsilon)
        analytic = torch.sum(geometry.gradient * direction).item()
        self.assertTrue(math.isfinite(finite_difference))
        self.assertAlmostEqual(finite_difference, analytic, places=5)

    def test_finite_difference_matches_live_pr_gradient(self) -> None:
        direction = torch.randn_like(self.weight)
        direction /= torch.linalg.vector_norm(direction)
        epsilon = 1e-6
        kwargs = dict(
            fixed_ecs_rank=4,
            normalization_response="differentiated",
            effective_rank_method="participation_ratio",
            ridge_relative=0.0,
        )
        geometry = adaptive_trace_log_geometry(self.weight, **kwargs)
        plus = adaptive_trace_log_geometry(
            self.weight + epsilon * direction, **kwargs
        ).residual.item()
        minus = adaptive_trace_log_geometry(
            self.weight - epsilon * direction, **kwargs
        ).residual.item()
        finite_difference = (plus - minus) / (2.0 * epsilon)
        analytic = torch.sum(geometry.gradient * direction).item()
        self.assertAlmostEqual(finite_difference, analytic, places=5)
        self.assertGreater(geometry.normalization_gradient_norm_sq, 0.0)

    def test_one_sided_leaves_expansion_unchanged(self) -> None:
        geometry = adaptive_trace_log_geometry(
            self.weight,
            fixed_ecs_rank=4,
            fixed_normalization_dimension=5.0,
            ridge_relative=0.0,
        )
        expansion = geometry.gradient.clone()
        result = correct_trace_log_component(
            expansion,
            geometry,
            mode="one_sided",
            max_correction_ratio=None,
        )
        self.assertFalse(result.applied)
        self.assertTrue(torch.equal(result.corrected_delta, expansion))

    def test_one_sided_cancels_contraction(self) -> None:
        geometry = adaptive_trace_log_geometry(
            self.weight,
            fixed_ecs_rank=4,
            fixed_normalization_dimension=5.0,
            ridge_relative=0.0,
        )
        contraction = -geometry.gradient.clone()
        result = correct_trace_log_component(
            contraction,
            geometry,
            mode="one_sided",
            max_correction_ratio=None,
        )
        drift = torch.sum(geometry.gradient * result.corrected_delta).item()
        self.assertAlmostEqual(drift, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()

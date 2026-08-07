from __future__ import annotations

import unittest

import numpy as np
import torch

from ecs_trace_wall import (
    compute_ecs_svd,
    project_gradient_to_ecs,
    select_self_consistent_ecs,
)


class ECSTests(unittest.TestCase):
    def test_selection_is_scale_invariant(self) -> None:
        eigenvalues = np.geomspace(1e-4, 20.0, 37)
        first = select_self_consistent_ecs(eigenvalues, min_rank=2)
        second = select_self_consistent_ecs(7.3e-19 * eigenvalues, min_rank=2)
        self.assertEqual(first.rank, second.rank)
        self.assertAlmostEqual(first.fractional_rank, second.fractional_rank, places=9)
        self.assertAlmostEqual(
            first.normalization_dimension,
            second.normalization_dimension,
            places=9,
        )
        self.assertAlmostEqual(first.trace_log, second.trace_log, places=9)

    def test_truncated_svd_has_selected_rank(self) -> None:
        generator = torch.Generator().manual_seed(17)
        weight = torch.randn(13, 9, generator=generator)
        state = compute_ecs_svd(weight, min_rank=2, svd_device="cpu")
        numerical_rank = int(torch.linalg.matrix_rank(state.truncated_weight).item())
        self.assertEqual(numerical_rank, state.rank)
        self.assertLessEqual(state.rank, min(weight.shape))
        self.assertTrue(torch.isfinite(state.truncated_weight).all())

    def test_core_projection_is_idempotent_and_inside_ecs(self) -> None:
        generator = torch.Generator().manual_seed(19)
        weight = torch.randn(11, 7, generator=generator)
        gradient = torch.randn(11, 7, generator=generator)
        state = compute_ecs_svd(weight, min_rank=2, svd_device="cpu")
        projected = project_gradient_to_ecs(gradient, state, mode="core")
        projected_twice = project_gradient_to_ecs(projected, state, mode="core")
        self.assertTrue(torch.allclose(projected, projected_twice, atol=1e-5, rtol=1e-5))

        left_residual = projected - state.left_ecs @ (state.left_ecs.T @ projected)
        right_residual = projected - (projected @ state.right_ecs) @ state.right_ecs.T
        self.assertLess(float(torch.linalg.vector_norm(left_residual)), 1e-4)
        self.assertLess(float(torch.linalg.vector_norm(right_residual)), 1e-4)

    def test_rank_m_tangent_projection_is_idempotent(self) -> None:
        generator = torch.Generator().manual_seed(23)
        weight = torch.randn(12, 8, generator=generator)
        gradient = torch.randn(12, 8, generator=generator)
        state = compute_ecs_svd(weight, min_rank=2, svd_device="cpu")
        projected = project_gradient_to_ecs(
            gradient, state, mode="rank_m_tangent"
        )
        projected_twice = project_gradient_to_ecs(
            projected, state, mode="rank_m_tangent"
        )
        self.assertTrue(torch.allclose(projected, projected_twice, atol=2e-5, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()

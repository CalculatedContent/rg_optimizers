import unittest

import torch

from rg_spectral_flow.flow import (
    centered_log_eigenvalue_shape,
    collapse_potential_from_shape,
    effective_rank_from_shape,
    remove_trivial_branch_component,
    trivial_branch_flow_vector,
)


class SpectralFlowTests(unittest.TestCase):
    def test_centered_shape_is_scale_invariant(self) -> None:
        s = torch.tensor([5.0, 3.0, 2.0, 1.0], dtype=torch.float64)
        first = centered_log_eigenvalue_shape(s, 4)
        second = centered_log_eigenvalue_shape(17.0 * s, 4)
        self.assertTrue(torch.allclose(first, second, atol=1e-12, rtol=0.0))
        self.assertAlmostEqual(float(first.sum()), 0.0, places=12)

    def test_pr_flow_vector_matches_finite_difference(self) -> None:
        shape = torch.tensor([1.1, 0.2, -0.4, -0.9], dtype=torch.float64)
        shape -= shape.mean()
        direction = torch.tensor([0.4, -0.2, 0.1, -0.3], dtype=torch.float64)
        direction -= direction.mean()
        direction /= torch.linalg.vector_norm(direction)
        vector = trivial_branch_flow_vector(shape, potential="participation_ratio")
        epsilon = 1e-6
        plus = collapse_potential_from_shape(
            shape + epsilon * direction,
            potential="participation_ratio",
        )
        minus = collapse_potential_from_shape(
            shape - epsilon * direction,
            potential="participation_ratio",
        )
        finite = float((plus - minus) / (2.0 * epsilon))
        analytic = float(torch.dot(vector, direction))
        self.assertAlmostEqual(finite, analytic, places=6)
        self.assertAlmostEqual(float(vector.sum()), 0.0, places=12)

    def test_toward_collapse_component_is_removed(self) -> None:
        before_s = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
        base_s = torch.tensor([4.6, 2.9, 1.8, 0.8], dtype=torch.float64)
        before = torch.diag(before_s)
        base = torch.diag(base_s)
        result = remove_trivial_branch_component(
            before,
            base,
            4,
            potential="participation_ratio",
            projection_strength=1.0,
            max_abs_log_eigenvalue_correction=None,
            max_correction_ratio=None,
        )
        self.assertTrue(result.applied)
        self.assertGreater(result.geometry.base_flow_component, 0.0)
        self.assertAlmostEqual(result.corrected_flow_component, 0.0, places=8)
        self.assertGreaterEqual(
            result.effective_rank_corrected,
            result.geometry.effective_rank_base,
        )
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(result.corrected_weight)),
            float(torch.linalg.vector_norm(base)),
            places=10,
        )

    def test_away_from_collapse_is_unchanged(self) -> None:
        before_s = torch.tensor([5.0, 2.5, 1.2, 0.6], dtype=torch.float64)
        base_s = torch.tensor([4.2, 2.8, 1.5, 0.9], dtype=torch.float64)
        before = torch.diag(before_s)
        base = torch.diag(base_s)
        result = remove_trivial_branch_component(
            before,
            base,
            4,
            potential="participation_ratio",
            max_abs_log_eigenvalue_correction=None,
            max_correction_ratio=None,
        )
        self.assertFalse(result.applied)
        self.assertLessEqual(result.geometry.base_projection_coefficient, 0.0)
        self.assertTrue(torch.equal(result.corrected_weight, base))

    def test_power_law_move_below_alpha_two_is_reduced(self) -> None:
        m = 128
        ranks = torch.arange(1, m + 1, dtype=torch.float64)
        before = torch.diag(ranks.pow(-0.5))  # lambda_k ~ k^-1, alpha = 2
        base = torch.diag(ranks.pow(-0.55))   # lambda_k ~ k^-1.1, alpha < 2
        result = remove_trivial_branch_component(
            before,
            base,
            m,
            max_abs_log_eigenvalue_correction=None,
            max_correction_ratio=None,
        )
        self.assertTrue(result.applied)

        def fitted_alpha(weight: torch.Tensor) -> float:
            s = torch.linalg.svdvals(weight)
            x = torch.log(ranks)
            y = torch.log(s.square())
            slope = torch.sum((x - x.mean()) * (y - y.mean())) / torch.sum(
                (x - x.mean()).square()
            )
            q = -float(slope)
            return 1.0 + 1.0 / q

        alpha_base = fitted_alpha(base)
        alpha_corrected = fitted_alpha(result.corrected_weight)
        self.assertLess(alpha_base, 2.0)
        self.assertGreater(alpha_corrected, alpha_base)
        self.assertLessEqual(alpha_corrected, 2.0 + 1e-10)

    def test_rank_one_limit_has_larger_collapse_potential(self) -> None:
        uniform = torch.zeros(8, dtype=torch.float64)
        collapsed = torch.tensor([8.0] + [-8.0] * 7, dtype=torch.float64)
        c_uniform = collapse_potential_from_shape(uniform)
        c_collapsed = collapse_potential_from_shape(collapsed)
        r_uniform = effective_rank_from_shape(uniform)
        r_collapsed = effective_rank_from_shape(collapsed)
        self.assertGreater(float(c_collapsed), float(c_uniform))
        self.assertLess(float(r_collapsed), float(r_uniform))


if __name__ == "__main__":
    unittest.main()

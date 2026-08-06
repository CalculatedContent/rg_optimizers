import unittest

import numpy as np

from rg_sc_trace_log.ecs import (
    self_consistent_candidate_scan,
    solve_self_consistent_ecs,
)


class SelfConsistentECSTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evals = np.sort(np.exp(np.linspace(-4.0, 3.0, 128)))

    def test_gamma_one_restores_full_dimension(self) -> None:
        scan = self_consistent_candidate_scan(
            self.evals,
            method="participation_ratio",
            gamma=1.0,
        )
        self.assertTrue(
            np.allclose(scan["normalization_dimension"], len(self.evals))
        )

    def test_solution_is_invariant_to_global_scale(self) -> None:
        first = solve_self_consistent_ecs(
            self.evals,
            method="participation_ratio",
            gamma=0.0,
        )
        second = solve_self_consistent_ecs(
            17.5 * self.evals,
            method="participation_ratio",
            gamma=0.0,
        )
        self.assertEqual(first.ecs_rank, second.ecs_rank)
        self.assertAlmostEqual(
            first.normalization_dimension,
            second.normalization_dimension,
            places=10,
        )
        self.assertAlmostEqual(
            first.trace_log_per_eval,
            second.trace_log_per_eval,
            places=10,
        )

    def test_selected_support_reproduces_its_own_detx_rank(self) -> None:
        solution = solve_self_consistent_ecs(
            self.evals,
            method="participation_ratio",
            gamma=0.0,
        )
        self.assertEqual(solution.fixed_point_error_nearest, 0)
        self.assertLess(abs(solution.trace_log_per_eval), 0.02)

    def test_literal_dimension_equal_rank_has_no_positive_residual(self) -> None:
        # This is the AM-GM obstruction described in the source notebook.
        spectral_sum = float(np.sum(self.evals))
        values_desc = self.evals[::-1]
        cumulative_log = np.cumsum(np.log(values_desc))
        ranks = np.arange(1, len(self.evals) + 1)
        residual = cumulative_log / ranks + np.log(ranks / spectral_sum)
        self.assertLessEqual(float(np.max(residual)), 1e-12)


if __name__ == "__main__":
    unittest.main()

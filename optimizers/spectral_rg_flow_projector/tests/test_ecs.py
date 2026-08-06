import unittest

import numpy as np

from rg_spectral_flow.ecs import candidate_arrays, solve_self_consistent_ecs


class ECSTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evals = np.sort(np.exp(np.linspace(-4.0, 3.0, 128)))

    def test_gamma_one_restores_full_dimension(self) -> None:
        arrays = candidate_arrays(self.evals, gamma=1.0)
        self.assertTrue(
            np.allclose(arrays["normalization_dimension"], len(self.evals))
        )

    def test_solution_is_scale_invariant(self) -> None:
        first = solve_self_consistent_ecs(self.evals, gamma=0.0)
        second = solve_self_consistent_ecs(23.0 * self.evals, gamma=0.0)
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


if __name__ == "__main__":
    unittest.main()

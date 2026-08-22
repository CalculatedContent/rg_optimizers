from __future__ import annotations

import unittest

import numpy as np

from portfolio_pgd import (
    ConstraintSet,
    PGDOptions,
    PortfolioProblem,
    factor_covariance,
    solve_pgd,
    solve_quadratic_kkt,
    solve_scipy_slsqp,
)


class QuadraticPortfolioTests(unittest.TestCase):
    def setUp(self) -> None:
        n = 18
        covariance, loadings = factor_covariance(n, 3, seed=101, specific_risk=0.12)
        rng = np.random.default_rng(102)
        alpha = rng.normal(scale=0.025, size=n)
        previous = rng.normal(scale=0.01, size=n)
        q_diag = 0.5 + rng.random(n)
        self.problem = PortfolioProblem(
            alpha=alpha,
            covariance=covariance,
            previous_holdings=previous,
            risk_aversion=2.5,
            quadratic_cost_matrix=q_diag,
            quadratic_cost_aversion=0.8,
        )
        factor_direction = loadings[:, 0]
        factor_direction = factor_direction - np.mean(factor_direction)
        self.constraints = ConstraintSet(
            n,
            equality_matrix=np.vstack([np.ones(n), factor_direction]),
            equality_target=np.array([1.0, 0.0]),
        )

    def test_pgd_matches_exact_kkt_holdings_and_objective(self) -> None:
        exact = solve_quadratic_kkt(self.problem, self.constraints)
        result = solve_pgd(
            self.problem,
            self.constraints,
            options=PGDOptions(max_iterations=20_000, tolerance=2.0e-9),
        )
        self.assertTrue(result.converged, result.status)
        self.assertLess(result.max_constraint_violation, 1.0e-8)
        self.assertLess(np.linalg.norm(result.holdings - exact.holdings), 2.0e-7)
        self.assertAlmostEqual(result.objective, exact.objective, places=10)

    def test_pgd_and_kkt_match_scipy_slsqp(self) -> None:
        exact = solve_quadratic_kkt(self.problem, self.constraints)
        standard = solve_scipy_slsqp(self.problem, self.constraints)
        self.assertTrue(standard.success, standard.message)
        self.assertLess(np.linalg.norm(standard.holdings - exact.holdings), 2.0e-6)
        self.assertAlmostEqual(standard.objective, exact.objective, places=9)

    def test_objective_gradient(self) -> None:
        rng = np.random.default_rng(18)
        h = rng.normal(size=self.problem.dimension)
        direction = rng.normal(size=self.problem.dimension)
        direction /= np.linalg.norm(direction)
        epsilon = 1.0e-6
        finite_difference = (
            self.problem.value(h + epsilon * direction)
            - self.problem.value(h - epsilon * direction)
        ) / (2.0 * epsilon)
        analytic = float(self.problem.gradient(h) @ direction)
        self.assertAlmostEqual(finite_difference, analytic, places=7)

    def test_progress_callback_reports_initial_and_final_state(self) -> None:
        states = []
        result = solve_pgd(
            self.problem,
            self.constraints,
            options=PGDOptions(
                max_iterations=20_000,
                tolerance=2.0e-9,
                progress_interval=25,
            ),
            progress_callback=states.append,
        )
        self.assertTrue(result.converged)
        self.assertGreaterEqual(len(states), 2)
        self.assertEqual(states[0].iteration, 0)
        self.assertEqual(states[-1].iteration, result.iterations)
        self.assertAlmostEqual(states[-1].objective, result.objective, places=14)
        self.assertLess(states[-1].max_constraint_violation, 1.0e-8)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from portfolio_pgd import (
    ConstraintSet,
    PGDOptions,
    PortfolioProblem,
    PowerLawCost,
    SmoothAbsoluteCost,
    capped_long_only_portfolio,
    factor_covariance,
    solve_pgd,
    solve_scipy_slsqp,
)


class NonlinearCostTests(unittest.TestCase):
    def test_power_law_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(200)
        trades = rng.normal(scale=0.1, size=15)
        cost = PowerLawCost(eta=np.linspace(0.01, 0.04, 15), p=1.5, epsilon=2.0e-4)
        analytic = cost.gradient(trades)
        epsilon = 1.0e-7
        finite_difference = np.empty_like(trades)
        for index in range(trades.size):
            perturbation = np.zeros_like(trades)
            perturbation[index] = epsilon
            finite_difference[index] = (
                cost.value(trades + perturbation) - cost.value(trades - perturbation)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-6, atol=2.0e-8)

    def test_smoothed_absolute_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(201)
        trades = rng.normal(scale=0.05, size=10)
        cost = SmoothAbsoluteCost(rate=0.003, epsilon=1.0e-3)
        direction = rng.normal(size=10)
        direction /= np.linalg.norm(direction)
        epsilon = 1.0e-7
        finite_difference = (
            cost.value(trades + epsilon * direction)
            - cost.value(trades - epsilon * direction)
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(finite_difference, float(cost.gradient(trades) @ direction), places=8)

    def test_nonlinear_pgd_matches_scipy(self) -> None:
        n = 14
        covariance, _ = factor_covariance(n, 3, seed=210, specific_risk=0.18)
        rng = np.random.default_rng(211)
        previous = capped_long_only_portfolio(n, cap=0.14, seed=212)
        problem = PortfolioProblem(
            alpha=rng.normal(scale=0.035, size=n),
            covariance=covariance,
            previous_holdings=previous,
            risk_aversion=1.8,
            quadratic_cost_matrix=0.15 + rng.random(n),
            quadratic_cost_aversion=0.25,
            nonlinear_cost=PowerLawCost(
                eta=0.01 + 0.015 * rng.random(n), p=1.5, epsilon=1.0e-3
            ),
        )
        constraints = ConstraintSet(
            n,
            equality_matrix=np.ones((1, n)),
            equality_target=np.array([1.0]),
            lower_bounds=0.0,
            upper_bounds=0.16,
        )
        result = solve_pgd(
            problem,
            constraints,
            options=PGDOptions(max_iterations=20_000, tolerance=5.0e-8),
        )
        standard = solve_scipy_slsqp(problem, constraints)
        self.assertTrue(result.converged, result.status)
        self.assertTrue(standard.success, standard.message)
        self.assertLess(result.max_constraint_violation, 2.0e-8)
        self.assertLess(abs(result.objective - standard.objective), 5.0e-8)
        self.assertLess(np.linalg.norm(result.holdings - standard.holdings), 3.0e-4)


if __name__ == "__main__":
    unittest.main()

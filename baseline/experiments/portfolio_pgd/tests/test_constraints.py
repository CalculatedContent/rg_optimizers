from __future__ import annotations

import unittest

import numpy as np

from portfolio_pgd import (
    ConstraintSet,
    PGDOptions,
    PortfolioProblem,
    PowerLawCost,
    capped_long_only_portfolio,
    factor_covariance,
    sector_membership,
    solve_pgd,
    solve_scipy_slsqp,
)


def realistic_case(n: int = 16):
    rng = np.random.default_rng(301)
    covariance, loadings = factor_covariance(n, 3, seed=302, specific_risk=0.20)
    previous = capped_long_only_portfolio(n, cap=0.12, seed=303)
    sectors = sector_membership(n, 4)
    previous_sector = sectors @ previous
    lower_sector = np.maximum(0.10, previous_sector - 0.05)
    upper_sector = np.minimum(0.40, previous_sector + 0.05)
    a_ub = np.vstack([sectors, -sectors])
    b_ub = np.concatenate([upper_sector, -lower_sector])
    factor = loadings[:, 0] - np.mean(loadings[:, 0])
    equality = np.vstack([np.ones(n), factor])
    target = np.array([1.0, float(factor @ previous)])
    constraints = ConstraintSet(
        n,
        equality_matrix=equality,
        equality_target=target,
        inequality_matrix=a_ub,
        inequality_upper=b_ub,
        lower_bounds=0.0,
        upper_bounds=0.14,
        turnover_limit=0.24,
        turnover_center=previous,
    )
    problem = PortfolioProblem(
        alpha=rng.normal(scale=0.04, size=n),
        covariance=covariance,
        previous_holdings=previous,
        risk_aversion=1.6,
        quadratic_cost_matrix=0.2 + rng.random(n),
        quadratic_cost_aversion=0.2,
        nonlinear_cost=PowerLawCost(eta=0.008, p=1.5, epsilon=1.0e-3),
    )
    return problem, constraints, sectors


class ConstraintProjectionTests(unittest.TestCase):
    def test_dykstra_projection_satisfies_realistic_intersection(self) -> None:
        problem, constraints, _ = realistic_case()
        rng = np.random.default_rng(304)
        projected = constraints.project(rng.normal(scale=0.5, size=problem.dimension))
        self.assertLess(constraints.max_violation(projected), 2.0e-8)
        reprojection = constraints.project(projected)
        np.testing.assert_allclose(projected, reprojection, atol=2.0e-8, rtol=0.0)

    def test_realistic_constraints_pgd_matches_standard_solver(self) -> None:
        problem, constraints, _ = realistic_case()
        result = solve_pgd(
            problem,
            constraints,
            options=PGDOptions(
                max_iterations=20_000,
                tolerance=1.0e-7,
                projection_tolerance=2.0e-10,
            ),
        )
        standard = solve_scipy_slsqp(problem, constraints, tolerance=1.0e-10)
        self.assertTrue(result.converged, result.status)
        self.assertTrue(standard.success, standard.message)
        self.assertLess(result.max_constraint_violation, 2.0e-7)
        self.assertLess(constraints.max_violation(standard.holdings), 2.0e-6)
        self.assertLess(abs(result.objective - standard.objective), 1.0e-5)

    def test_long_short_gross_exposure_projection(self) -> None:
        n = 12
        rng = np.random.default_rng(305)
        beta = rng.normal(size=n)
        constraints = ConstraintSet(
            n,
            equality_matrix=np.vstack([np.ones(n), beta]),
            equality_target=np.array([0.0, 0.0]),
            lower_bounds=-0.20,
            upper_bounds=0.20,
            gross_exposure_limit=1.0,
        )
        projected = constraints.project(rng.normal(size=n))
        self.assertLess(constraints.max_violation(projected), 2.0e-8)
        self.assertLessEqual(np.sum(np.abs(projected)), 1.0 + 2.0e-8)


if __name__ == "__main__":
    unittest.main()

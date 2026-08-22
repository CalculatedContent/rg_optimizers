"""Run a constrained portfolio optimization and print audit diagnostics."""

from __future__ import annotations

import numpy as np

from portfolio_pgd import (
    ConstraintSet,
    PGDOptions,
    PortfolioProblem,
    PowerLawCost,
    capped_long_only_portfolio,
    factor_covariance,
    solve_pgd,
)


def main() -> None:
    n = 30
    covariance, _ = factor_covariance(n, 4, seed=700)
    rng = np.random.default_rng(701)
    previous = capped_long_only_portfolio(n, cap=0.06, seed=702)
    problem = PortfolioProblem(
        alpha=rng.normal(scale=0.025, size=n),
        covariance=covariance,
        previous_holdings=previous,
        risk_aversion=2.0,
        quadratic_cost_matrix=0.3 + rng.random(n),
        quadratic_cost_aversion=0.4,
        nonlinear_cost=PowerLawCost(eta=0.01, p=1.5, epsilon=1.0e-3),
    )
    constraints = ConstraintSet(
        n,
        equality_matrix=np.ones((1, n)),
        equality_target=np.array([1.0]),
        lower_bounds=0.0,
        upper_bounds=0.075,
        turnover_limit=0.20,
        turnover_center=previous,
    )
    result = solve_pgd(problem, constraints, options=PGDOptions(tolerance=1.0e-8))
    print(f"status:                  {result.status}")
    print(f"iterations:              {result.iterations}")
    print(f"utility:                 {result.utility:.10f}")
    print(f"two-way turnover:        {np.sum(np.abs(result.trades)):.10f}")
    print(f"projected-gradient norm: {result.projected_gradient_norm:.3e}")
    print(f"constraint violation:    {result.max_constraint_violation:.3e}")


if __name__ == "__main__":
    main()

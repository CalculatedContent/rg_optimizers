"""Generate the checked-in notebooks using only the Python standard library."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    (NOTEBOOK_DIR / filename).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


COMMON_SETUP = r'''from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.float_format", lambda value: f"{value:,.8g}")'''


def build_quadratic() -> None:
    cells = [
        markdown(r'''# Quadratic portfolio construction: PGD versus exact and standard solvers

This notebook implements the quadratic model

$$
\min_h\;\frac{\lambda}{2}h^\top Vh-\alpha^\top h
+\frac{\theta}{2}(h-h_-)^\top Q(h-h_-)
\quad\text{subject to}\quad C^\top h=c.
$$

We compare three independent solution routes:

1. projected gradient descent with exact affine projection;
2. the exact equality-constrained KKT system;
3. SciPy's standard SLSQP solver.

The numerical assertions at the end turn the notebook into an executable validation document.'''),
        markdown(r'''## PGD iteration

With negative utility denoted by $F(h)$,

$$
\nabla F(h)=\lambda Vh+\theta Q(h-h_-)-\alpha.
$$

The Euclidean affine projection is

$$
\Pi(z)=z-C(C^\top C)^{\dagger}(C^\top z-c).
$$

The implementation uses a projected-majorization line search, records the projected-gradient norm,
and refuses to return a portfolio whose constraints exceed the requested tolerance.'''),
        code(COMMON_SETUP),
        code(r'''from portfolio_pgd import (
    ConstraintSet,
    PGDOptions,
    PortfolioProblem,
    factor_covariance,
    solve_pgd,
    solve_quadratic_kkt,
    solve_scipy_slsqp,
)

n_assets = 36
rng = np.random.default_rng(1201)
covariance, loadings = factor_covariance(n_assets, 4, seed=1202, specific_risk=0.12)
alpha = rng.normal(scale=0.025, size=n_assets)
previous = rng.normal(scale=0.01, size=n_assets)
q_diagonal = 0.4 + rng.random(n_assets)

problem = PortfolioProblem(
    alpha=alpha,
    covariance=covariance,
    previous_holdings=previous,
    risk_aversion=2.25,
    quadratic_cost_matrix=q_diagonal,
    quadratic_cost_aversion=0.75,
)

# Full investment and one factor-exposure target.
factor_direction = loadings[:, 0] - np.mean(loadings[:, 0])
C_transpose = np.vstack([np.ones(n_assets), factor_direction])
targets = np.array([1.0, 0.0])
constraints = ConstraintSet(
    n_assets,
    equality_matrix=C_transpose,
    equality_target=targets,
)
print("Minimum covariance eigenvalue:", np.linalg.eigvalsh(covariance).min())'''),
        code(r'''pgd = solve_pgd(
    problem,
    constraints,
    options=PGDOptions(max_iterations=25_000, tolerance=2e-9),
)
kkt = solve_quadratic_kkt(problem, constraints)
slsqp = solve_scipy_slsqp(problem, constraints)

comparison = pd.DataFrame(
    {
        "objective": [pgd.objective, kkt.objective, slsqp.objective],
        "utility": [pgd.utility, -kkt.objective, -slsqp.objective],
        "distance_to_KKT": [
            np.linalg.norm(pgd.holdings - kkt.holdings),
            0.0,
            np.linalg.norm(slsqp.holdings - kkt.holdings),
        ],
        "max_constraint_violation": [
            constraints.max_violation(pgd.holdings),
            constraints.max_violation(kkt.holdings),
            constraints.max_violation(slsqp.holdings),
        ],
    },
    index=["PGD", "Exact KKT", "SciPy SLSQP"],
)
print(comparison.to_string())
print(f"\nPGD status={pgd.status}; iterations={pgd.iterations}; SLSQP={slsqp.message}")'''),
        markdown(r'''## Convergence audit

For a convex quadratic, the KKT objective is the global minimum of the estimated problem. The first
panel plots the PGD objective gap; the second shows the norm of the projected-gradient mapping.'''),
        code(r'''history = pd.DataFrame(pgd.history)
objective_gap = np.maximum(history["objective"] - kkt.objective, 1e-18)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(history["iteration"], objective_gap)
axes[0].set(title="Objective gap to exact KKT", xlabel="Iteration", ylabel="F(h) - F(h*)")
valid = history["projected_gradient_norm"].notna()
axes[1].semilogy(
    history.loc[valid, "iteration"],
    np.maximum(history.loc[valid, "projected_gradient_norm"], 1e-18),
)
axes[1].set(title="First-order residual", xlabel="Iteration", ylabel="Projected-gradient norm")
plt.tight_layout()
plt.show()'''),
        markdown(r'''## Alpha, persistence, and constraint decomposition

Let $H=\lambda V+\theta Q$ and $b=\alpha+\theta Qh_-$. The unconstrained target is $H^{-1}b$.
The exact constrained solution decomposes as

$$
h^*=\underbrace{H^{-1}\alpha}_{\text{alpha}}
+\underbrace{H^{-1}\theta Qh_-}_{\text{persistence}}
-\underbrace{H^{-1}C\mu}_{\text{constraint correction}}.
$$'''),
        code(r'''H = problem.quadratic_hessian
Q = np.asarray(problem.quadratic_cost_matrix)
A = np.asarray(constraints.equality_matrix)
c = np.asarray(constraints.equality_target)

alpha_component = np.linalg.solve(H, alpha)
persistence_component = np.linalg.solve(
    H, problem.quadratic_cost_aversion * Q @ previous
)
unconstrained = alpha_component + persistence_component
H_inv_A_T = np.linalg.solve(H, A.T)
mu = np.linalg.solve(A @ H_inv_A_T, A @ unconstrained - c)
constraint_component = -H_inv_A_T @ mu
reconstructed = alpha_component + persistence_component + constraint_component

decomposition = pd.DataFrame(
    {
        "L2 norm": [
            np.linalg.norm(alpha_component),
            np.linalg.norm(persistence_component),
            np.linalg.norm(constraint_component),
            np.linalg.norm(reconstructed),
        ],
        "net exposure": [
            np.sum(alpha_component),
            np.sum(persistence_component),
            np.sum(constraint_component),
            np.sum(reconstructed),
        ],
    },
    index=["alpha", "persistence", "constraint correction", "total"],
)
print(decomposition.to_string())
print("Reconstruction error:", np.linalg.norm(reconstructed - kkt.holdings))'''),
        markdown("## Executable acceptance tests"),
        code(r'''assert pgd.converged
assert slsqp.success
assert constraints.max_violation(pgd.holdings) < 1e-8
assert np.linalg.norm(pgd.holdings - kkt.holdings) < 5e-7
assert np.linalg.norm(slsqp.holdings - kkt.holdings) < 5e-6
assert np.linalg.norm(reconstructed - kkt.holdings) < 1e-10
print("All quadratic validation checks passed.")'''),
    ]
    write_notebook("01_quadratic_pgd_vs_standard_solvers.ipynb", cells)


def build_nonlinear() -> None:
    cells = [
        markdown(r'''# Nonlinear convex transaction costs

This notebook replaces the quadratic-only trading model with a separable power-law impact cost

$$
c(t)=\sum_i\eta_i\left[(t_i^2+\epsilon^2)^{p/2}-\epsilon^p\right],
\qquad t=h-h_-.
$$

For $p=3/2$ this is a smooth approximation to square-root-impact total cost. The smoothing parameter
$\epsilon>0$ avoids the divergent curvature of $|t|^{3/2}$ at zero without changing convexity.'''),
        code(COMMON_SETUP),
        code(r'''from portfolio_pgd import (
    ConstraintSet,
    PGDOptions,
    PortfolioProblem,
    PowerLawCost,
    capped_long_only_portfolio,
    factor_covariance,
    solve_pgd,
    solve_scipy_slsqp,
)

n_assets = 30
rng = np.random.default_rng(2201)
covariance, _ = factor_covariance(n_assets, 4, seed=2202, specific_risk=0.15)
previous = capped_long_only_portfolio(n_assets, cap=0.055, seed=2203)
eta = 0.008 + 0.012 * rng.random(n_assets)

cost = PowerLawCost(eta=eta, p=1.5, epsilon=1e-3)
problem = PortfolioProblem(
    alpha=rng.normal(scale=0.03, size=n_assets),
    covariance=covariance,
    previous_holdings=previous,
    risk_aversion=1.8,
    quadratic_cost_matrix=0.2 + rng.random(n_assets),
    quadratic_cost_aversion=0.25,
    nonlinear_cost=cost,
)
constraints = ConstraintSet(
    n_assets,
    equality_matrix=np.ones((1, n_assets)),
    equality_target=np.array([1.0]),
    lower_bounds=0.0,
    upper_bounds=0.075,
)

pgd = solve_pgd(
    problem,
    constraints,
    options=PGDOptions(max_iterations=25_000, tolerance=5e-8),
)
slsqp = solve_scipy_slsqp(problem, constraints)
print(f"PGD: {pgd.status} in {pgd.iterations} iterations")
print(f"SLSQP: success={slsqp.success}; {slsqp.message}")'''),
        markdown(r'''## Independent solver comparison

The nonlinear objective is convex but no longer quadratic. Therefore the notebook compares PGD to
SciPy SLSQP rather than to a linear KKT solve.'''),
        code(r'''comparison = pd.DataFrame(
    {
        "objective": [pgd.objective, slsqp.objective],
        "utility": [pgd.utility, -slsqp.objective],
        "turnover": [np.sum(np.abs(pgd.trades)), np.sum(np.abs(slsqp.holdings - previous))],
        "distance_to_SLSQP": [np.linalg.norm(pgd.holdings - slsqp.holdings), 0.0],
        "constraint_violation": [
            constraints.max_violation(pgd.holdings),
            constraints.max_violation(slsqp.holdings),
        ],
    },
    index=["PGD", "SciPy SLSQP"],
)
print(comparison.to_string())'''),
        markdown(r'''## Cost, marginal cost, and curvature

The gradient enters PGD directly. The Hessian diagonal is used only to initialize a conservative
step; the majorization line search supplies the actual global safeguard.'''),
        code(r'''trade_grid = np.linspace(-0.08, 0.08, 401)
unit_cost = PowerLawCost(eta=1.0, p=1.5, epsilon=1e-3)
cost_values = np.array([unit_cost.value(np.array([trade])) for trade in trade_grid])
marginal = np.array([unit_cost.gradient(np.array([trade]))[0] for trade in trade_grid])
curvature = np.array([unit_cost.hessian_diag(np.array([trade]))[0] for trade in trade_grid])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(trade_grid, cost_values)
axes[0].set(title="Power-law cost", xlabel="Trade", ylabel="Cost / eta")
axes[1].plot(trade_grid, marginal)
axes[1].set(title="Marginal cost", xlabel="Trade", ylabel="dc/dt / eta")
axes[2].plot(trade_grid, curvature)
axes[2].set(title="Local curvature", xlabel="Trade", ylabel="d²c/dt² / eta")
plt.tight_layout()
plt.show()'''),
        markdown("## Convergence and the realized trade distribution"),
        code(r'''history = pd.DataFrame(pgd.history)
valid = history["projected_gradient_norm"].notna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(
    history.loc[valid, "iteration"],
    np.maximum(history.loc[valid, "projected_gradient_norm"], 1e-18),
)
axes[0].set(title="Projected-gradient residual", xlabel="Iteration", ylabel="Norm")
axes[1].bar(np.arange(n_assets), pgd.trades)
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].set(title="Optimal trades", xlabel="Asset", ylabel="Weight change")
plt.tight_layout()
plt.show()'''),
        markdown(r'''## Sensitivity to the impact exponent

Holding every other input fixed, we resolve the portfolio for several convex exponents. This is an
algorithmic comparison—not a claim that any one exponent is universally appropriate.'''),
        code(r'''rows = []
for exponent in [1.25, 1.5, 2.0, 3.0]:
    exponent_problem = PortfolioProblem(
        alpha=problem.alpha,
        covariance=problem.covariance,
        previous_holdings=previous,
        risk_aversion=problem.risk_aversion,
        quadratic_cost_matrix=problem.quadratic_cost_matrix,
        quadratic_cost_aversion=problem.quadratic_cost_aversion,
        nonlinear_cost=PowerLawCost(eta=eta, p=exponent, epsilon=1e-3),
    )
    solved = solve_pgd(
        exponent_problem,
        constraints,
        options=PGDOptions(max_iterations=25_000, tolerance=2e-7),
    )
    rows.append(
        {
            "p": exponent,
            "converged": solved.converged,
            "utility": solved.utility,
            "turnover": np.sum(np.abs(solved.trades)),
            "max_abs_trade": np.max(np.abs(solved.trades)),
            "iterations": solved.iterations,
        }
    )
sensitivity = pd.DataFrame(rows).set_index("p")
print(sensitivity.to_string())'''),
        markdown("## Executable acceptance tests"),
        code(r'''assert pgd.converged
assert slsqp.success
assert constraints.max_violation(pgd.holdings) < 2e-8
assert abs(pgd.objective - slsqp.objective) < 2e-7
assert np.linalg.norm(pgd.holdings - slsqp.holdings) < 8e-4
assert sensitivity["converged"].all()
print("All nonlinear-cost validation checks passed.")'''),
    ]
    write_notebook("02_nonlinear_transaction_costs.ipynb", cells)


def build_realistic() -> None:
    cells = [
        markdown(r'''# Realistic institutional constraints

This notebook combines the smooth nonlinear objective with a realistic long-only mandate:

- fully invested;
- one fixed factor exposure;
- long-only and per-name caps;
- sector lower and upper bounds;
- hard two-way turnover.

The PGD projection is the Euclidean projection onto the **joint intersection**. Dykstra's algorithm
cycles over analytic projectors while retaining correction terms, so this is not naive sequential
clipping.'''),
        code(COMMON_SETUP),
        code(r'''from portfolio_pgd import (
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

n_assets = 40
n_sectors = 5
rng = np.random.default_rng(3301)
covariance, loadings = factor_covariance(n_assets, 5, seed=3302, specific_risk=0.18)
previous = capped_long_only_portfolio(n_assets, cap=0.045, seed=3303)
sectors = sector_membership(n_assets, n_sectors)
previous_sector = sectors @ previous

# Sector bands are centered on the existing portfolio, making feasibility explicit.
sector_lower = np.maximum(0.12, previous_sector - 0.035)
sector_upper = np.minimum(0.30, previous_sector + 0.035)
A_ub = np.vstack([sectors, -sectors])
b_ub = np.concatenate([sector_upper, -sector_lower])

factor_direction = loadings[:, 0] - np.mean(loadings[:, 0])
A_eq = np.vstack([np.ones(n_assets), factor_direction])
b_eq = np.array([1.0, float(factor_direction @ previous)])

constraints = ConstraintSet(
    n_assets,
    equality_matrix=A_eq,
    equality_target=b_eq,
    inequality_matrix=A_ub,
    inequality_upper=b_ub,
    lower_bounds=0.0,
    upper_bounds=0.06,
    turnover_limit=0.22,
    turnover_center=previous,
)

problem = PortfolioProblem(
    alpha=rng.normal(scale=0.035, size=n_assets),
    covariance=covariance,
    previous_holdings=previous,
    risk_aversion=1.6,
    quadratic_cost_matrix=0.2 + rng.random(n_assets),
    quadratic_cost_aversion=0.25,
    nonlinear_cost=PowerLawCost(eta=0.006 + 0.006 * rng.random(n_assets), p=1.5, epsilon=1e-3),
)
print("Starting portfolio violations:", constraints.violations(previous))'''),
        code(r'''pgd = solve_pgd(
    problem,
    constraints,
    options=PGDOptions(
        max_iterations=30_000,
        tolerance=1e-7,
        projection_tolerance=2e-10,
    ),
)
slsqp = solve_scipy_slsqp(problem, constraints, tolerance=1e-10)

comparison = pd.DataFrame(
    {
        "objective": [pgd.objective, slsqp.objective],
        "utility": [pgd.utility, -slsqp.objective],
        "turnover": [np.sum(np.abs(pgd.trades)), np.sum(np.abs(slsqp.holdings - previous))],
        "distance_to_SLSQP": [np.linalg.norm(pgd.holdings - slsqp.holdings), 0.0],
        "constraint_violation": [
            constraints.max_violation(pgd.holdings),
            constraints.max_violation(slsqp.holdings),
        ],
    },
    index=["PGD", "SciPy SLSQP"],
)
print(comparison.to_string())
print(f"\nPGD status={pgd.status}; iterations={pgd.iterations}; SLSQP={slsqp.message}")'''),
        markdown("## Constraint audit"),
        code(r'''audit = pd.DataFrame(
    {
        "PGD violation": constraints.violations(pgd.holdings),
        "SLSQP violation": constraints.violations(slsqp.holdings),
    }
)
print(audit.to_string())

sector_exposure = sectors @ pgd.holdings
sector_table = pd.DataFrame(
    {
        "lower": sector_lower,
        "previous": previous_sector,
        "optimized": sector_exposure,
        "upper": sector_upper,
    },
    index=[f"Sector {index}" for index in range(n_sectors)],
)
print("\nSector exposures:\n", sector_table.to_string())'''),
        code(r'''fig, axes = plt.subplots(1, 2, figsize=(13, 4))
asset_index = np.arange(n_assets)
axes[0].plot(asset_index, previous, "o-", label="Previous", markersize=3)
axes[0].plot(asset_index, pgd.holdings, "o-", label="Optimized", markersize=3)
axes[0].axhline(0.06, color="black", linestyle="--", linewidth=1, label="Per-name cap")
axes[0].set(title="Holdings", xlabel="Asset", ylabel="Weight")
axes[0].legend()
axes[1].bar(asset_index, pgd.trades)
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].set(title=f"Trades (L1={np.sum(np.abs(pgd.trades)):.4f})", xlabel="Asset", ylabel="Weight change")
plt.tight_layout()
plt.show()

sector_table.plot(kind="bar", figsize=(11, 4))
plt.title("Sector exposure audit")
plt.ylabel("Portfolio weight")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()'''),
        markdown("## Optimization convergence"),
        code(r'''history = pd.DataFrame(pgd.history)
valid = history["projected_gradient_norm"].notna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["iteration"], history["objective"])
axes[0].axhline(slsqp.objective, color="black", linestyle="--", label="SLSQP")
axes[0].set(title="Objective", xlabel="Iteration", ylabel="Negative utility")
axes[0].legend()
axes[1].semilogy(
    history.loc[valid, "iteration"],
    np.maximum(history.loc[valid, "projected_gradient_norm"], 1e-18),
)
axes[1].set(title="Projected-gradient residual", xlabel="Iteration", ylabel="Norm")
plt.tight_layout()
plt.show()'''),
        markdown(r'''## Long-short extension

The same projection engine can construct a dollar-neutral, beta-neutral portfolio with per-name and
gross-exposure limits. This remains convex; gross exposure is an $L_1$ ball centered at zero.'''),
        code(r'''long_short_constraints = ConstraintSet(
    n_assets,
    equality_matrix=np.vstack([np.ones(n_assets), loadings[:, 1]]),
    equality_target=np.array([0.0, 0.0]),
    lower_bounds=-0.08,
    upper_bounds=0.08,
    gross_exposure_limit=1.0,
)
long_short_problem = PortfolioProblem(
    alpha=2.5 * problem.alpha,
    covariance=problem.covariance,
    previous_holdings=np.zeros(n_assets),
    risk_aversion=0.9,
    quadratic_cost_matrix=np.ones(n_assets),
    quadratic_cost_aversion=0.08,
)
long_short = solve_pgd(
    long_short_problem,
    long_short_constraints,
    options=PGDOptions(max_iterations=25_000, tolerance=1e-8),
)
print(
    pd.Series(
        {
            "status": long_short.status,
            "net exposure": np.sum(long_short.holdings),
            "gross exposure": np.sum(np.abs(long_short.holdings)),
            "beta exposure": loadings[:, 1] @ long_short.holdings,
            "maximum position": np.max(np.abs(long_short.holdings)),
            "constraint violation": long_short.max_constraint_violation,
        }
    ).to_string()
)'''),
        markdown("## Executable acceptance tests"),
        code(r'''assert pgd.converged
assert slsqp.success
assert constraints.max_violation(pgd.holdings) < 2e-7
assert constraints.max_violation(slsqp.holdings) < 2e-6
assert abs(pgd.objective - slsqp.objective) < 2e-5
assert np.sum(np.abs(pgd.trades)) <= 0.22 + 2e-7
assert long_short.converged
assert long_short_constraints.max_violation(long_short.holdings) < 2e-7
print("All realistic-constraint validation checks passed.")'''),
    ]
    write_notebook("03_realistic_constraints.ipynb", cells)


def main() -> None:
    build_quadratic()
    build_nonlinear()
    build_realistic()
    print(f"Wrote notebooks to {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()

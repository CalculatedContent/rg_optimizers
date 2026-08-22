# Portfolio PGD Optimizer Experiment

A tested projected-gradient implementation for the portfolio-construction models in
*Portfolio Construction and Information Flow: Transaction Costs, Constraints, and Forecast
Persistence*.

This experiment lives inside `CalculatedContent/rg_optimizers` at
`baseline/experiments/portfolio_pgd`. Runtime environments, logs, metrics, and generated outputs
are kept under literal `/tmp` on the Mac.

## Complete Mac run from a fresh checkout

```bash
cd /tmp
git clone https://github.com/CalculatedContent/rg_optimizers.git
cd /tmp/rg_optimizers

bash baseline/experiments/portfolio_pgd/scripts/setup_mac.sh
bash baseline/experiments/portfolio_pgd/scripts/run_portfolio_pgd_experiment.sh
```

The foreground campaign runner prints every stage and command, streams PGD progress records with
objective, projected-gradient norm, step size, constraint violation, and elapsed time, and saves the
same terminal log under:

```text
/tmp/portfolio-pgd-runs/<run-id>/run.log
```

Machine-readable output is written beside it under `metrics/`: per-scenario histories, holdings,
trades, `summary.csv`, and `summary.json`. The runner never backgrounds, detaches, replaces the
calling shell, or sends process-killing signals.

The package solves

\[
\min_h\;
\frac{\lambda}{2}h^\top Vh-\alpha^\top h
+\frac{\theta}{2}(h-h_-)^\top Q(h-h_-)
+c(h-h_-)
\]

over an intersection of convex portfolio constraints. Here, \(h_-\) is the pre-trade portfolio and
\(c\) may be a smooth square-root/power-law or smoothed bid-ask cost.

## What is included

- Projected gradient descent with a majorization line search and convergence diagnostics.
- Exact KKT solver for the equality-constrained quadratic case.
- Independent SciPy SLSQP benchmark solver. Turnover and gross exposure use exact lifted linear
  formulations rather than nonsmooth finite differences.
- Dykstra projection over analytic projectors for:
  - full-investment, factor-neutrality, and other affine equalities;
  - sector, industry, and other linear exposure bounds;
  - long-only/short and per-name bounds;
  - hard \(L_1\) turnover;
  - hard gross exposure.
- Convex power-law impact and smoothed bid-ask transaction costs.
- Ten automated tests covering gradients, progress callbacks, exact quadratic solutions, nonlinear costs, and
  realistic institutional constraint intersections.
- Three detailed Jupyter notebooks.

## Standalone installation

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

The optimizer itself requires only NumPy and SciPy. The `notebook` extra adds JupyterLab,
Matplotlib, and pandas.

## Run the tests

The tests use the standard-library `unittest` runner, so pytest is not required:

```bash
python tests/run_tests.py
```

or:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Open the notebooks

```bash
jupyter notebook notebooks
```

Run them in order:

1. `01_quadratic_pgd_vs_standard_solvers.ipynb`
2. `02_nonlinear_transaction_costs.ipynb`
3. `03_realistic_constraints.ipynb`

Each notebook contains numerical assertions and can be run from a fresh kernel.

## Minimal example

```python
import numpy as np
from portfolio_pgd import ConstraintSet, PGDOptions, PortfolioProblem, solve_pgd

n = 20
rng = np.random.default_rng(7)
raw = rng.normal(size=(n, n))
covariance = raw @ raw.T / n + 0.1 * np.eye(n)

problem = PortfolioProblem(
    alpha=rng.normal(scale=0.02, size=n),
    covariance=covariance,
    previous_holdings=np.full(n, 1.0 / n),
    risk_aversion=2.0,
    quadratic_cost_matrix=np.ones(n),
    quadratic_cost_aversion=0.5,
)

constraints = ConstraintSet(
    n,
    equality_matrix=np.ones((1, n)),
    equality_target=np.array([1.0]),
    lower_bounds=0.0,
    upper_bounds=0.10,
    turnover_limit=0.25,
    turnover_center=problem.previous_holdings,
)

result = solve_pgd(problem, constraints, options=PGDOptions(tolerance=1e-8))
print(result.status, result.utility, result.max_constraint_violation)
```

## Solver conventions

- The implementation **minimizes negative utility**. `result.utility` is the economically familiar
  maximized quantity; `result.objective` is its negative.
- All matrices use the holdings convention \(A_{eq}h=b_{eq}\) and
  \(A_{ub}h\le b_{ub}\).
- Turnover is two-way turnover \(\lVert h-h_-\rVert_1\). Divide by two externally if your reporting
  convention defines one-way turnover.
- Power-law costs with \(1<p<2\) have singular curvature at zero. For numerical work, set a small
  positive `epsilon`; the notebooks use `1e-3`.
- Cardinality, minimum lots, fixed ticket costs, and integer positions are nonconvex and are outside
  this package's global-convergence guarantees.

## Numerical method

For the quadratic case, the Hessian is

\[
H=\lambda V+\theta Q\succ0,
\]

so the solution is unique. The package validates PGD against both the exact KKT system and SciPy's
SLSQP implementation. For nonlinear smooth convex costs, PGD uses the projected-majorization test

\[
F(h^+)\le F(h)+\nabla F(h)^\top(h^+-h)+\frac{\lVert h^+-h\rVert^2}{2\eta}
\]

to backtrack to a safe step.

Projection onto a constraint intersection is performed by Dykstra's algorithm. Every constituent
set has an analytic Euclidean projector; failure to reach the requested feasibility tolerance raises
`ProjectionError` instead of silently returning an infeasible portfolio.

## Scope

This is a research-quality reference implementation intended for reproducible experiments and as a
clear foundation for a production service. A production deployment should additionally integrate
data validation, sparse matrix representations, warm starts across rebalance dates, monitoring,
and organization-specific trading/risk controls.

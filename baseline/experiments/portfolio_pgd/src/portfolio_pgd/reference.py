"""Exact quadratic and SciPy reference solvers used for validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, minimize

from .constraints import ConstraintSet
from .problem import PortfolioProblem

FloatArray = NDArray[np.float64]


@dataclass
class ReferenceResult:
    holdings: FloatArray
    objective: float
    success: bool
    message: str
    multipliers: FloatArray | None = None
    raw_result: Any | None = None


def solve_quadratic_kkt(
    problem: PortfolioProblem,
    constraints: ConstraintSet | None = None,
) -> ReferenceResult:
    """Solve the quadratic, equality-constrained problem through its KKT system."""
    constraints = constraints or ConstraintSet(problem.dimension)
    if problem.nonlinear_cost is not None:
        raise ValueError("KKT solver supports quadratic costs only")
    if not constraints.has_only_equalities:
        raise ValueError("KKT solver supports affine equality constraints only")
    hessian = problem.quadratic_hessian
    linear = problem.quadratic_linear_term
    a_eq = np.asarray(constraints.equality_matrix)
    b_eq = np.asarray(constraints.equality_target)
    if a_eq.shape[0] == 0:
        holdings = np.linalg.solve(hessian, linear)
        multipliers = np.zeros(0, dtype=float)
    else:
        kkt = np.block(
            [
                [hessian, a_eq.T],
                [a_eq, np.zeros((a_eq.shape[0], a_eq.shape[0]))],
            ]
        )
        rhs = np.concatenate([linear, b_eq])
        try:
            solution = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            solution, residuals, rank, _ = np.linalg.lstsq(kkt, rhs, rcond=1.0e-13)
            if rank < kkt.shape[0] and residuals.size and float(np.max(residuals)) > 1.0e-14:
                raise ValueError("singular or inconsistent KKT system")
        holdings = solution[: problem.dimension]
        multipliers = solution[problem.dimension :]
    return ReferenceResult(
        holdings=np.asarray(holdings, dtype=float),
        objective=problem.value(holdings),
        success=True,
        message="exact KKT solution",
        multipliers=np.asarray(multipliers, dtype=float),
    )


def solve_scipy_slsqp(
    problem: PortfolioProblem,
    constraints: ConstraintSet | None = None,
    *,
    initial_holdings: ArrayLike | None = None,
    max_iterations: int = 5_000,
    tolerance: float = 1.0e-11,
) -> ReferenceResult:
    """Solve the same problem with SciPy SLSQP as an independent benchmark."""
    constraints = constraints or ConstraintSet(problem.dimension)
    initial = problem.previous_holdings if initial_holdings is None else np.asarray(initial_holdings, dtype=float)
    initial = constraints.project(initial, tolerance=1.0e-10, max_cycles=10_000)
    # Lift L1 constraints with auxiliary variables.  This makes the reference
    # problem smooth with purely linear constraints instead of asking SLSQP to
    # finite-difference an absolute value at zero.
    n = problem.dimension
    turnover_slice: slice | None = None
    gross_slice: slice | None = None
    total_dimension = n
    if constraints.turnover_limit is not None:
        turnover_slice = slice(total_dimension, total_dimension + n)
        total_dimension += n
    if constraints.gross_exposure_limit is not None:
        gross_slice = slice(total_dimension, total_dimension + n)
        total_dimension += n

    lifted_initial = np.zeros(total_dimension, dtype=float)
    lifted_initial[:n] = initial
    if turnover_slice is not None:
        lifted_initial[turnover_slice] = np.abs(initial - np.asarray(constraints.turnover_center))
    if gross_slice is not None:
        lifted_initial[gross_slice] = np.abs(initial)

    def lifted_value(vector: FloatArray) -> float:
        return problem.value(vector[:n])

    def lifted_gradient(vector: FloatArray) -> FloatArray:
        gradient = np.zeros(total_dimension, dtype=float)
        gradient[:n] = problem.gradient(vector[:n])
        return gradient

    equality_rows: list[FloatArray] = []
    equality_targets: list[FloatArray] = []
    inequality_rows: list[FloatArray] = []
    inequality_targets: list[FloatArray] = []
    a_eq = np.asarray(constraints.equality_matrix)
    b_eq = np.asarray(constraints.equality_target)
    if a_eq.shape[0]:
        lifted = np.zeros((a_eq.shape[0], total_dimension), dtype=float)
        lifted[:, :n] = a_eq
        equality_rows.append(lifted)
        equality_targets.append(b_eq)
    a_ub = np.asarray(constraints.inequality_matrix)
    b_ub = np.asarray(constraints.inequality_upper)
    if a_ub.shape[0]:
        lifted = np.zeros((a_ub.shape[0], total_dimension), dtype=float)
        lifted[:, :n] = a_ub
        inequality_rows.append(lifted)
        inequality_targets.append(b_ub)
    if turnover_slice is not None:
        center = np.asarray(constraints.turnover_center)
        radius = float(constraints.turnover_limit)
        positive = np.zeros((n, total_dimension), dtype=float)
        positive[:, :n] = np.eye(n)
        positive[:, turnover_slice] = -np.eye(n)
        negative = np.zeros((n, total_dimension), dtype=float)
        negative[:, :n] = -np.eye(n)
        negative[:, turnover_slice] = -np.eye(n)
        total = np.zeros((1, total_dimension), dtype=float)
        total[:, turnover_slice] = 1.0
        inequality_rows.extend([positive, negative, total])
        inequality_targets.extend([center, -center, np.array([radius])])
    if gross_slice is not None:
        radius = float(constraints.gross_exposure_limit)
        positive = np.zeros((n, total_dimension), dtype=float)
        positive[:, :n] = np.eye(n)
        positive[:, gross_slice] = -np.eye(n)
        negative = np.zeros((n, total_dimension), dtype=float)
        negative[:, :n] = -np.eye(n)
        negative[:, gross_slice] = -np.eye(n)
        total = np.zeros((1, total_dimension), dtype=float)
        total[:, gross_slice] = 1.0
        inequality_rows.extend([positive, negative, total])
        inequality_targets.extend([np.zeros(n), np.zeros(n), np.array([radius])])

    scipy_constraints: list[dict[str, Any]] = []
    if equality_rows:
        lifted_a_eq = np.vstack(equality_rows)
        lifted_b_eq = np.concatenate(equality_targets)
        scipy_constraints.append(
            {
                "type": "eq",
                "fun": lambda y, a=lifted_a_eq, b=lifted_b_eq: a @ y - b,
                "jac": lambda y, a=lifted_a_eq: a,
            }
        )
    if inequality_rows:
        lifted_a_ub = np.vstack(inequality_rows)
        lifted_b_ub = np.concatenate(inequality_targets)
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda y, a=lifted_a_ub, b=lifted_b_ub: b - a @ y,
                "jac": lambda y, a=lifted_a_ub: -a,
            }
        )

    lower = np.full(total_dimension, -np.inf, dtype=float)
    upper = np.full(total_dimension, np.inf, dtype=float)
    lower[:n] = np.asarray(constraints.lower_bounds)
    upper[:n] = np.asarray(constraints.upper_bounds)
    if turnover_slice is not None:
        lower[turnover_slice] = 0.0
    if gross_slice is not None:
        lower[gross_slice] = 0.0
    result = minimize(
        lifted_value,
        lifted_initial,
        jac=lifted_gradient,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=scipy_constraints,
        options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
    )
    holdings = np.asarray(result.x[:n], dtype=float)
    return ReferenceResult(
        holdings=holdings,
        objective=problem.value(holdings),
        success=bool(result.success),
        message=str(result.message),
        raw_result=result,
    )

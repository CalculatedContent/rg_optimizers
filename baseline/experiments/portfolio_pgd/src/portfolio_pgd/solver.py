"""Projected-gradient solver with majorization backtracking and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constraints import ConstraintSet
from .problem import PortfolioProblem

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PGDOptions:
    max_iterations: int = 10_000
    tolerance: float = 1.0e-8
    step_size: float | None = None
    use_backtracking: bool = True
    backtracking_factor: float = 0.5
    step_growth: float = 1.15
    minimum_step: float = 1.0e-16
    maximum_step: float = 1.0e6
    max_backtracking_steps: int = 60
    projection_tolerance: float = 1.0e-10
    projection_max_cycles: int = 5_000
    record_every: int = 1
    progress_interval: int = 50

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.step_size is not None and self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must lie in (0, 1)")
        if self.step_growth < 1.0:
            raise ValueError("step_growth must be at least one")
        if self.record_every <= 0:
            raise ValueError("record_every must be positive")
        if self.progress_interval <= 0:
            raise ValueError("progress_interval must be positive")


@dataclass(frozen=True)
class ProgressState:
    """Immutable progress record emitted by a running PGD solve."""

    iteration: int
    objective: float
    utility: float
    projected_gradient_norm: float
    step_size: float
    max_constraint_violation: float
    elapsed_seconds: float


@dataclass
class SolverResult:
    holdings: FloatArray
    trades: FloatArray
    objective: float
    utility: float
    converged: bool
    status: str
    iterations: int
    projected_gradient_norm: float
    max_constraint_violation: float
    history: dict[str, list[float]] = field(default_factory=dict)


def solve_pgd(
    problem: PortfolioProblem,
    constraints: ConstraintSet | None = None,
    *,
    initial_holdings: ArrayLike | None = None,
    options: PGDOptions | None = None,
    progress_callback: Callable[[ProgressState], None] | None = None,
) -> SolverResult:
    """Solve a smooth convex portfolio problem by projected gradient descent."""
    started = perf_counter()
    options = options or PGDOptions()
    constraints = constraints or ConstraintSet(problem.dimension)
    if constraints.dimension != problem.dimension:
        raise ValueError("problem and constraints have different dimensions")
    x0 = problem.previous_holdings if initial_holdings is None else np.asarray(initial_holdings, dtype=float)
    x = constraints.project(
        x0,
        tolerance=options.projection_tolerance,
        max_cycles=options.projection_max_cycles,
    )
    objective = problem.value(x)
    if not np.isfinite(objective):
        raise FloatingPointError("initial objective is not finite")

    if options.step_size is None:
        step = 1.0 / problem.local_lipschitz_bound(x)
    else:
        step = float(options.step_size)
    step = float(np.clip(step, options.minimum_step, options.maximum_step))

    history: dict[str, list[float]] = {
        "iteration": [0.0],
        "objective": [objective],
        "utility": [-objective],
        "projected_gradient_norm": [np.nan],
        "step_size": [step],
        "max_constraint_violation": [constraints.max_violation(x)],
    }
    converged = False
    status = "maximum_iterations_reached"
    projected_gradient_norm = np.inf
    last_callback_iteration = -1

    if progress_callback is not None:
        progress_callback(
            ProgressState(
                iteration=0,
                objective=float(objective),
                utility=float(-objective),
                projected_gradient_norm=float("nan"),
                step_size=step,
                max_constraint_violation=constraints.max_violation(x),
                elapsed_seconds=perf_counter() - started,
            )
        )
        last_callback_iteration = 0

    for iteration in range(1, options.max_iterations + 1):
        gradient = problem.gradient(x)
        if np.any(~np.isfinite(gradient)):
            raise FloatingPointError("objective gradient is not finite")

        trial_step = (
            min(step * options.step_growth, options.maximum_step)
            if options.use_backtracking
            else step
        )
        candidate = x
        candidate_objective = objective
        accepted = False
        for _ in range(options.max_backtracking_steps):
            candidate = constraints.project(
                x - trial_step * gradient,
                tolerance=options.projection_tolerance,
                max_cycles=options.projection_max_cycles,
            )
            displacement = candidate - x
            candidate_objective = problem.value(candidate)
            majorizer = (
                objective
                + float(gradient @ displacement)
                + 0.5 * float(displacement @ displacement) / trial_step
            )
            slack = 1.0e-13 * max(1.0, abs(objective), abs(candidate_objective))
            if (
                not options.use_backtracking
                or candidate_objective <= majorizer + slack
            ):
                accepted = True
                break
            trial_step *= options.backtracking_factor
            if trial_step < options.minimum_step:
                break
        if not accepted:
            status = "line_search_failed"
            break

        projected_gradient_norm = float(np.linalg.norm(x - candidate) / trial_step)
        x = candidate
        objective = candidate_objective
        step = trial_step
        violation = constraints.max_violation(x)

        if iteration % options.record_every == 0 or iteration == options.max_iterations:
            history["iteration"].append(float(iteration))
            history["objective"].append(float(objective))
            history["utility"].append(float(-objective))
            history["projected_gradient_norm"].append(projected_gradient_norm)
            history["step_size"].append(step)
            history["max_constraint_violation"].append(violation)

        should_report = iteration % options.progress_interval == 0

        scale = max(1.0, float(np.linalg.norm(x)))
        if projected_gradient_norm <= options.tolerance * scale:
            converged = True
            status = "converged"
            should_report = True
            if history["iteration"][-1] != float(iteration):
                history["iteration"].append(float(iteration))
                history["objective"].append(float(objective))
                history["utility"].append(float(-objective))
                history["projected_gradient_norm"].append(projected_gradient_norm)
                history["step_size"].append(step)
                history["max_constraint_violation"].append(violation)
        if progress_callback is not None and should_report:
            progress_callback(
                ProgressState(
                    iteration=iteration,
                    objective=float(objective),
                    utility=float(-objective),
                    projected_gradient_norm=projected_gradient_norm,
                    step_size=step,
                    max_constraint_violation=violation,
                    elapsed_seconds=perf_counter() - started,
                )
            )
            last_callback_iteration = iteration
        if converged:
            break

    iterations = iteration
    if progress_callback is not None and last_callback_iteration != iterations:
        progress_callback(
            ProgressState(
                iteration=iterations,
                objective=float(objective),
                utility=float(-objective),
                projected_gradient_norm=projected_gradient_norm,
                step_size=step,
                max_constraint_violation=constraints.max_violation(x),
                elapsed_seconds=perf_counter() - started,
            )
        )
    return SolverResult(
        holdings=np.asarray(x, dtype=float),
        trades=np.asarray(x - problem.previous_holdings, dtype=float),
        objective=float(objective),
        utility=float(-objective),
        converged=converged,
        status=status,
        iterations=iterations,
        projected_gradient_norm=projected_gradient_norm,
        max_constraint_violation=constraints.max_violation(x),
        history=history,
    )

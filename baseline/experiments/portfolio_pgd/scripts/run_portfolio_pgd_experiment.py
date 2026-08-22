#!/usr/bin/env python3
"""Run the complete portfolio-PGD validation campaign with visible progress logging."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = EXPERIMENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from portfolio_pgd import (  # noqa: E402
    ConstraintSet,
    PGDOptions,
    PortfolioProblem,
    PowerLawCost,
    ProgressState,
    capped_long_only_portfolio,
    factor_covariance,
    sector_membership,
    solve_pgd,
    solve_quadratic_kkt,
    solve_scipy_slsqp,
)

LOGGER = logging.getLogger("portfolio_pgd.experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quadratic, nonlinear-cost, and realistic-constraint PGD benchmarks."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-iterations", type=int, default=30_000)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    return parser.parse_args()


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def stage(number: int, total: int, title: str) -> None:
    LOGGER.info("=" * 72)
    LOGGER.info("STAGE %d/%d %s", number, total, title)
    LOGGER.info("=" * 72)


def progress_logger(name: str):
    def report(state: ProgressState) -> None:
        residual = (
            "initial"
            if not np.isfinite(state.projected_gradient_norm)
            else f"{state.projected_gradient_norm:.3e}"
        )
        LOGGER.info(
            "PROGRESS scenario=%-11s iter=%6d objective=% .10e utility=% .10e "
            "pg_norm=%s step=%.3e violation=%.3e elapsed=%.2fs",
            name,
            state.iteration,
            state.objective,
            state.utility,
            residual,
            state.step_size,
            state.max_constraint_violation,
            state.elapsed_seconds,
        )

    return report


def write_history(path: Path, history: dict[str, list[float]]) -> None:
    columns = list(history)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*(history[column] for column in columns)))


def solve_and_compare(
    name: str,
    problem: PortfolioProblem,
    constraints: ConstraintSet,
    options: PGDOptions,
    output_dir: Path,
    *,
    exact_quadratic: bool,
    objective_gap_tolerance: float,
    holdings_distance_tolerance: float,
) -> dict[str, object]:
    LOGGER.info(
        "START scenario=%s assets=%d initial_objective=% .10e initial_violation=%.3e",
        name,
        problem.dimension,
        problem.value(constraints.project(problem.previous_holdings)),
        constraints.max_violation(constraints.project(problem.previous_holdings)),
    )
    started = perf_counter()
    pgd = solve_pgd(
        problem,
        constraints,
        options=options,
        progress_callback=progress_logger(name),
    )
    LOGGER.info("REFERENCE scenario=%s solver=scipy_slsqp status=running", name)
    slsqp = solve_scipy_slsqp(problem, constraints, tolerance=1.0e-10)
    if not slsqp.success:
        raise RuntimeError(f"SciPy SLSQP failed for {name}: {slsqp.message}")

    kkt = solve_quadratic_kkt(problem, constraints) if exact_quadratic else None
    reference = kkt if kkt is not None else slsqp
    holdings_distance = float(np.linalg.norm(pgd.holdings - reference.holdings))
    objective_gap = float(pgd.objective - reference.objective)
    turnover = float(np.sum(np.abs(pgd.trades)))
    elapsed = perf_counter() - started
    LOGGER.info(
        "RESULT scenario=%s status=%s iterations=%d objective=% .10e reference=% .10e "
        "objective_gap=% .3e holdings_distance=%.3e turnover=%.6f violation=%.3e elapsed=%.2fs",
        name,
        pgd.status,
        pgd.iterations,
        pgd.objective,
        reference.objective,
        objective_gap,
        holdings_distance,
        turnover,
        pgd.max_constraint_violation,
        elapsed,
    )
    if not pgd.converged:
        raise RuntimeError(f"PGD failed to converge for {name}: {pgd.status}")
    if pgd.max_constraint_violation > 2.0e-7:
        raise RuntimeError(f"constraint violation too large for {name}")
    if abs(objective_gap) > objective_gap_tolerance:
        raise RuntimeError(
            f"objective gap too large for {name}: {objective_gap:.3e} > "
            f"{objective_gap_tolerance:.3e}"
        )
    if holdings_distance > holdings_distance_tolerance:
        raise RuntimeError(
            f"holdings distance too large for {name}: {holdings_distance:.3e} > "
            f"{holdings_distance_tolerance:.3e}"
        )

    write_history(output_dir / f"{name}_history.csv", pgd.history)
    np.savetxt(output_dir / f"{name}_holdings.csv", pgd.holdings, delimiter=",")
    np.savetxt(output_dir / f"{name}_trades.csv", pgd.trades, delimiter=",")
    return {
        "scenario": name,
        "converged": pgd.converged,
        "status": pgd.status,
        "iterations": pgd.iterations,
        "objective": pgd.objective,
        "utility": pgd.utility,
        "reference_objective": reference.objective,
        "objective_gap": objective_gap,
        "holdings_distance_to_reference": holdings_distance,
        "turnover": turnover,
        "projected_gradient_norm": pgd.projected_gradient_norm,
        "max_constraint_violation": pgd.max_constraint_violation,
        "elapsed_seconds": elapsed,
        "reference_solver": "exact_kkt" if kkt is not None else "scipy_slsqp",
        "slsqp_message": slsqp.message,
    }


def quadratic_case(seed: int):
    n = 36
    covariance, loadings = factor_covariance(n, 4, seed=seed + 1, specific_risk=0.12)
    rng = np.random.default_rng(seed + 2)
    previous = rng.normal(scale=0.01, size=n)
    problem = PortfolioProblem(
        alpha=rng.normal(scale=0.025, size=n),
        covariance=covariance,
        previous_holdings=previous,
        risk_aversion=2.25,
        quadratic_cost_matrix=0.4 + rng.random(n),
        quadratic_cost_aversion=0.75,
    )
    factor = loadings[:, 0] - np.mean(loadings[:, 0])
    constraints = ConstraintSet(
        n,
        equality_matrix=np.vstack([np.ones(n), factor]),
        equality_target=np.array([1.0, 0.0]),
    )
    return problem, constraints


def nonlinear_case(seed: int):
    n = 30
    covariance, _ = factor_covariance(n, 4, seed=seed + 11, specific_risk=0.15)
    rng = np.random.default_rng(seed + 12)
    previous = capped_long_only_portfolio(n, cap=0.055, seed=seed + 13)
    problem = PortfolioProblem(
        alpha=rng.normal(scale=0.03, size=n),
        covariance=covariance,
        previous_holdings=previous,
        risk_aversion=1.8,
        quadratic_cost_matrix=0.2 + rng.random(n),
        quadratic_cost_aversion=0.25,
        nonlinear_cost=PowerLawCost(
            eta=0.008 + 0.012 * rng.random(n), p=1.5, epsilon=1.0e-3
        ),
    )
    constraints = ConstraintSet(
        n,
        equality_matrix=np.ones((1, n)),
        equality_target=np.array([1.0]),
        lower_bounds=0.0,
        upper_bounds=0.075,
    )
    return problem, constraints


def realistic_case(seed: int):
    n = 40
    covariance, loadings = factor_covariance(n, 5, seed=seed + 21, specific_risk=0.18)
    rng = np.random.default_rng(seed + 22)
    previous = capped_long_only_portfolio(n, cap=0.045, seed=seed + 23)
    sectors = sector_membership(n, 5)
    previous_sector = sectors @ previous
    sector_lower = np.maximum(0.12, previous_sector - 0.035)
    sector_upper = np.minimum(0.30, previous_sector + 0.035)
    factor = loadings[:, 0] - np.mean(loadings[:, 0])
    constraints = ConstraintSet(
        n,
        equality_matrix=np.vstack([np.ones(n), factor]),
        equality_target=np.array([1.0, float(factor @ previous)]),
        inequality_matrix=np.vstack([sectors, -sectors]),
        inequality_upper=np.concatenate([sector_upper, -sector_lower]),
        lower_bounds=0.0,
        upper_bounds=0.06,
        turnover_limit=0.22,
        turnover_center=previous,
    )
    problem = PortfolioProblem(
        alpha=rng.normal(scale=0.035, size=n),
        covariance=covariance,
        previous_holdings=previous,
        risk_aversion=1.6,
        quadratic_cost_matrix=0.2 + rng.random(n),
        quadratic_cost_aversion=0.25,
        nonlinear_cost=PowerLawCost(
            eta=0.006 + 0.006 * rng.random(n), p=1.5, epsilon=1.0e-3
        ),
    )
    return problem, constraints


def main() -> int:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_stages = 5
    started = perf_counter()

    stage(1, total_stages, "runtime and configuration preflight")
    LOGGER.info("experiment_dir=%s", EXPERIMENT_DIR)
    LOGGER.info("output_dir=%s", args.output_dir)
    LOGGER.info("python=%s", sys.version.replace("\n", " "))
    LOGGER.info("platform=%s", platform.platform())
    LOGGER.info("numpy=%s scipy=%s", np.__version__, scipy.__version__)
    LOGGER.info(
        "seed=%d tolerance=%.3e max_iterations=%d log_every=%d",
        args.seed,
        args.tolerance,
        args.max_iterations,
        args.log_every,
    )

    options = PGDOptions(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        projection_tolerance=2.0e-10,
        progress_interval=args.log_every,
    )
    summaries = []
    cases = [
        (2, "quadratic", quadratic_case(args.seed), True, 1.0e-9, 1.0e-6),
        (3, "nonlinear", nonlinear_case(args.seed), False, 1.0e-6, 1.0e-3),
        (4, "realistic", realistic_case(args.seed), False, 2.0e-5, 2.0e-3),
    ]
    for stage_number, name, (problem, constraints), exact, objective_tol, distance_tol in cases:
        stage(stage_number, total_stages, f"{name} portfolio solve")
        summaries.append(
            solve_and_compare(
                name,
                problem,
                constraints,
                options,
                args.output_dir,
                exact_quadratic=exact,
                objective_gap_tolerance=objective_tol,
                holdings_distance_tolerance=distance_tol,
            )
        )

    stage(5, total_stages, "write manifest and final acceptance summary")
    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "portfolio_pgd",
        "seed": args.seed,
        "options": asdict(options),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scenarios": summaries,
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for summary in summaries:
        LOGGER.info(
            "ACCEPT scenario=%s converged=%s objective_gap=% .3e distance=%.3e violation=%.3e",
            summary["scenario"],
            summary["converged"],
            summary["objective_gap"],
            summary["holdings_distance_to_reference"],
            summary["max_constraint_violation"],
        )
    LOGGER.info("COMPLETE output_dir=%s elapsed=%.2fs", args.output_dir, manifest["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

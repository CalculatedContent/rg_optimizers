"""Projected-gradient portfolio optimization."""

from .constraints import ConstraintSet, ProjectionError
from .costs import PowerLawCost, SmoothAbsoluteCost, TransactionCost
from .problem import PortfolioProblem
from .reference import ReferenceResult, solve_quadratic_kkt, solve_scipy_slsqp
from .solver import PGDOptions, ProgressState, SolverResult, solve_pgd
from .synthetic import capped_long_only_portfolio, factor_covariance, sector_membership

__all__ = [
    "ConstraintSet",
    "PGDOptions",
    "PortfolioProblem",
    "PowerLawCost",
    "ProjectionError",
    "ProgressState",
    "ReferenceResult",
    "SmoothAbsoluteCost",
    "SolverResult",
    "TransactionCost",
    "capped_long_only_portfolio",
    "factor_covariance",
    "sector_membership",
    "solve_pgd",
    "solve_quadratic_kkt",
    "solve_scipy_slsqp",
]

__version__ = "1.0.0"

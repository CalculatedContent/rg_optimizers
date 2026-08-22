"""Portfolio objective and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .costs import TransactionCost

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PortfolioProblem:
    r"""Single-period convex portfolio-construction problem.

    The minimized objective is

    .. math::

        \tfrac{\lambda}{2}h^T Vh - \alpha^T h
        + \tfrac{\theta}{2}(h-h_-)^TQ(h-h_-)
        + c(h-h_-).
    """

    alpha: ArrayLike
    covariance: ArrayLike
    previous_holdings: ArrayLike
    risk_aversion: float = 1.0
    quadratic_cost_matrix: ArrayLike | None = None
    quadratic_cost_aversion: float = 0.0
    nonlinear_cost: TransactionCost | None = None

    def __post_init__(self) -> None:
        alpha = np.asarray(self.alpha, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        previous = np.asarray(self.previous_holdings, dtype=float)
        if alpha.ndim != 1:
            raise ValueError("alpha must be one-dimensional")
        n = alpha.size
        if covariance.shape != (n, n):
            raise ValueError(f"covariance must have shape ({n}, {n})")
        if previous.shape != (n,):
            raise ValueError(f"previous_holdings must have shape ({n},)")
        if self.risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be strictly positive")
        if self.quadratic_cost_aversion < 0.0:
            raise ValueError("quadratic_cost_aversion must be nonnegative")
        if np.any(~np.isfinite(alpha)) or np.any(~np.isfinite(covariance)):
            raise ValueError("problem data must be finite")
        if not np.allclose(covariance, covariance.T, atol=1.0e-12):
            raise ValueError("covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be positive definite") from exc

        if self.quadratic_cost_matrix is None:
            q = np.zeros((n, n), dtype=float)
        else:
            raw_q = np.asarray(self.quadratic_cost_matrix, dtype=float)
            if raw_q.ndim == 1:
                if raw_q.shape != (n,):
                    raise ValueError(f"quadratic cost diagonal must have shape ({n},)")
                q = np.diag(raw_q)
            else:
                q = raw_q
            if q.shape != (n, n):
                raise ValueError(f"quadratic_cost_matrix must have shape ({n}, {n})")
            if not np.allclose(q, q.T, atol=1.0e-12):
                raise ValueError("quadratic_cost_matrix must be symmetric")
            if np.min(np.linalg.eigvalsh(q)) < -1.0e-12:
                raise ValueError("quadratic_cost_matrix must be positive semidefinite")

        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "previous_holdings", previous)
        object.__setattr__(self, "quadratic_cost_matrix", q)

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.alpha).size)

    @property
    def quadratic_hessian(self) -> FloatArray:
        return (
            self.risk_aversion * np.asarray(self.covariance)
            + self.quadratic_cost_aversion * np.asarray(self.quadratic_cost_matrix)
        )

    @property
    def quadratic_linear_term(self) -> FloatArray:
        return np.asarray(self.alpha) + self.quadratic_cost_aversion * (
            np.asarray(self.quadratic_cost_matrix) @ np.asarray(self.previous_holdings)
        )

    def value(self, holdings: ArrayLike) -> float:
        h = np.asarray(holdings, dtype=float)
        if h.shape != (self.dimension,):
            raise ValueError(f"holdings must have shape ({self.dimension},)")
        trades = h - np.asarray(self.previous_holdings)
        value = (
            0.5 * self.risk_aversion * float(h @ np.asarray(self.covariance) @ h)
            - float(np.asarray(self.alpha) @ h)
            + 0.5
            * self.quadratic_cost_aversion
            * float(trades @ np.asarray(self.quadratic_cost_matrix) @ trades)
        )
        if self.nonlinear_cost is not None:
            value += self.nonlinear_cost.value(trades)
        return float(value)

    def utility(self, holdings: ArrayLike) -> float:
        """Return utility, the negative of the minimized objective."""
        return -self.value(holdings)

    def gradient(self, holdings: ArrayLike) -> FloatArray:
        h = np.asarray(holdings, dtype=float)
        trades = h - np.asarray(self.previous_holdings)
        gradient = (
            self.risk_aversion * (np.asarray(self.covariance) @ h)
            - np.asarray(self.alpha)
            + self.quadratic_cost_aversion
            * (np.asarray(self.quadratic_cost_matrix) @ trades)
        )
        if self.nonlinear_cost is not None:
            gradient = gradient + self.nonlinear_cost.gradient(trades)
        return np.asarray(gradient, dtype=float)

    def local_lipschitz_bound(self, holdings: ArrayLike) -> float:
        """Return a conservative infinity-norm bound on local Hessian size."""
        h = np.asarray(holdings, dtype=float)
        base = self.quadratic_hessian
        bound = float(np.max(np.sum(np.abs(base), axis=1)))
        if self.nonlinear_cost is not None:
            trades = h - np.asarray(self.previous_holdings)
            diagonal = self.nonlinear_cost.hessian_diag(trades)
            finite = diagonal[np.isfinite(diagonal)]
            if finite.size:
                bound += float(max(0.0, np.max(finite)))
        return max(bound, np.finfo(float).eps)

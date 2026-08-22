"""Convex portfolio constraints and Euclidean projection by Dykstra's algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class ProjectionError(RuntimeError):
    """Raised when the supplied convex constraints appear infeasible or fail to project."""


def _vector(value: ArrayLike | None, size: int, name: str, default: float) -> FloatArray:
    if value is None:
        return np.full(size, default, dtype=float)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must be scalar or have shape ({size},)")
    return array


def project_l1_ball(vector: ArrayLike, radius: float, center: ArrayLike | None = None) -> FloatArray:
    """Project a vector onto ``{x: ||x-center||_1 <= radius}``."""
    x = np.asarray(vector, dtype=float)
    if radius < 0.0:
        raise ValueError("radius must be nonnegative")
    c = np.zeros_like(x) if center is None else np.asarray(center, dtype=float)
    if c.shape != x.shape:
        raise ValueError("center and vector must have the same shape")
    shifted = x - c
    absolute = np.abs(shifted)
    if float(np.sum(absolute)) <= radius:
        return x.copy()
    if radius == 0.0:
        return c.copy()
    ordered = np.sort(absolute)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, ordered.size + 1)
    active = np.nonzero(ordered - (cumulative - radius) / indices > 0.0)[0]
    if active.size == 0:
        return c.copy()
    rho = int(active[-1])
    threshold = (cumulative[rho] - radius) / float(rho + 1)
    return c + np.sign(shifted) * np.maximum(absolute - threshold, 0.0)


@dataclass(frozen=True)
class ConstraintSet:
    """Intersection of common convex institutional portfolio constraints.

    Supported constraints are affine equalities, linear inequalities, per-name
    lower/upper bounds, turnover around a reference portfolio, and gross
    exposure.  Projection onto their intersection uses Dykstra's algorithm;
    individual projections are analytic.
    """

    dimension: int
    equality_matrix: ArrayLike | None = None
    equality_target: ArrayLike | None = None
    inequality_matrix: ArrayLike | None = None
    inequality_upper: ArrayLike | None = None
    lower_bounds: ArrayLike | None = None
    upper_bounds: ArrayLike | None = None
    turnover_limit: float | None = None
    turnover_center: ArrayLike | None = None
    gross_exposure_limit: float | None = None

    def __post_init__(self) -> None:
        n = int(self.dimension)
        if n <= 0:
            raise ValueError("dimension must be positive")

        if self.equality_matrix is None:
            a_eq = np.zeros((0, n), dtype=float)
            b_eq = np.zeros(0, dtype=float)
        else:
            a_eq = np.atleast_2d(np.asarray(self.equality_matrix, dtype=float))
            if a_eq.shape[1] != n:
                raise ValueError(f"equality_matrix must have {n} columns")
            if self.equality_target is None:
                raise ValueError("equality_target is required with equality_matrix")
            b_eq = np.atleast_1d(np.asarray(self.equality_target, dtype=float))
            if b_eq.shape != (a_eq.shape[0],):
                raise ValueError("equality_target has incompatible shape")

        if self.inequality_matrix is None:
            a_ub = np.zeros((0, n), dtype=float)
            b_ub = np.zeros(0, dtype=float)
        else:
            a_ub = np.atleast_2d(np.asarray(self.inequality_matrix, dtype=float))
            if a_ub.shape[1] != n:
                raise ValueError(f"inequality_matrix must have {n} columns")
            if self.inequality_upper is None:
                raise ValueError("inequality_upper is required with inequality_matrix")
            b_ub = np.atleast_1d(np.asarray(self.inequality_upper, dtype=float))
            if b_ub.shape != (a_ub.shape[0],):
                raise ValueError("inequality_upper has incompatible shape")

        lower = _vector(self.lower_bounds, n, "lower_bounds", -np.inf)
        upper = _vector(self.upper_bounds, n, "upper_bounds", np.inf)
        if np.any(lower > upper):
            raise ValueError("lower_bounds must not exceed upper_bounds")

        if self.turnover_limit is not None and self.turnover_limit < 0.0:
            raise ValueError("turnover_limit must be nonnegative")
        if self.turnover_limit is not None:
            if self.turnover_center is None:
                raise ValueError("turnover_center is required with turnover_limit")
            center = np.asarray(self.turnover_center, dtype=float)
            if center.shape != (n,):
                raise ValueError(f"turnover_center must have shape ({n},)")
        else:
            center = np.zeros(n, dtype=float)
        if self.gross_exposure_limit is not None and self.gross_exposure_limit < 0.0:
            raise ValueError("gross_exposure_limit must be nonnegative")

        for name, array in (("equality_matrix", a_eq), ("equality_target", b_eq),
                            ("inequality_matrix", a_ub), ("inequality_upper", b_ub)):
            if np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must contain finite values")

        object.__setattr__(self, "dimension", n)
        object.__setattr__(self, "equality_matrix", a_eq)
        object.__setattr__(self, "equality_target", b_eq)
        object.__setattr__(self, "inequality_matrix", a_ub)
        object.__setattr__(self, "inequality_upper", b_ub)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(self, "turnover_center", center)

    @property
    def has_only_equalities(self) -> bool:
        return bool(
            np.asarray(self.inequality_matrix).shape[0] == 0
            and np.all(np.isneginf(np.asarray(self.lower_bounds)))
            and np.all(np.isposinf(np.asarray(self.upper_bounds)))
            and self.turnover_limit is None
            and self.gross_exposure_limit is None
        )

    def _projectors(self):
        projectors = []
        a_eq = np.asarray(self.equality_matrix)
        b_eq = np.asarray(self.equality_target)
        if a_eq.shape[0]:
            gram_pinv = np.linalg.pinv(a_eq @ a_eq.T, rcond=1.0e-13)
            correction = a_eq.T @ gram_pinv

            def affine(x, a=a_eq, b=b_eq, k=correction):
                return x - k @ (a @ x - b)

            projectors.append(affine)

        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        if np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)):
            projectors.append(lambda x, lo=lower, hi=upper: np.clip(x, lo, hi))

        a_ub = np.asarray(self.inequality_matrix)
        b_ub = np.asarray(self.inequality_upper)
        for row, bound in zip(a_ub, b_ub):
            norm_squared = float(row @ row)
            if norm_squared == 0.0:
                if bound < 0.0:
                    raise ProjectionError("infeasible zero-row inequality")
                continue

            def halfspace(x, a=row.copy(), b=float(bound), denom=norm_squared):
                excess = float(a @ x - b)
                return x if excess <= 0.0 else x - (excess / denom) * a

            projectors.append(halfspace)

        if self.turnover_limit is not None:
            center = np.asarray(self.turnover_center)
            radius = float(self.turnover_limit)
            projectors.append(lambda x, r=radius, c=center: project_l1_ball(x, r, c))

        if self.gross_exposure_limit is not None:
            radius = float(self.gross_exposure_limit)
            projectors.append(lambda x, r=radius: project_l1_ball(x, r))
        return projectors

    def project(
        self,
        vector: ArrayLike,
        *,
        tolerance: float = 1.0e-10,
        max_cycles: int = 5_000,
        check_feasible: bool = True,
    ) -> FloatArray:
        """Return the Euclidean projection onto the full constraint intersection."""
        z = np.asarray(vector, dtype=float)
        if z.shape != (self.dimension,):
            raise ValueError(f"vector must have shape ({self.dimension},)")
        projectors = self._projectors()
        if not projectors:
            return z.copy()
        x = z.copy()
        corrections = [np.zeros_like(x) for _ in projectors]
        converged = False
        for cycle in range(max_cycles):
            before = x.copy()
            for index, projector in enumerate(projectors):
                shifted = x + corrections[index]
                projected = np.asarray(projector(shifted), dtype=float)
                corrections[index] = shifted - projected
                x = projected
            scale = max(1.0, float(np.linalg.norm(x, ord=np.inf)))
            delta = float(np.linalg.norm(x - before, ord=np.inf))
            if delta <= tolerance * scale and self.max_violation(x) <= max(10.0 * tolerance, 1.0e-11):
                converged = True
                break
        violation = self.max_violation(x)
        if check_feasible and (not converged or violation > max(100.0 * tolerance, 1.0e-8)):
            raise ProjectionError(
                f"projection failed after {max_cycles} cycles; maximum violation={violation:.3e}. "
                "The constraints may be infeasible."
            )
        return x

    def violations(self, holdings: ArrayLike) -> dict[str, float]:
        h = np.asarray(holdings, dtype=float)
        if h.shape != (self.dimension,):
            raise ValueError(f"holdings must have shape ({self.dimension},)")
        a_eq = np.asarray(self.equality_matrix)
        a_ub = np.asarray(self.inequality_matrix)
        values: dict[str, float] = {
            "equality": float(np.max(np.abs(a_eq @ h - np.asarray(self.equality_target))))
            if a_eq.shape[0]
            else 0.0,
            "linear_inequality": float(
                max(0.0, np.max(a_ub @ h - np.asarray(self.inequality_upper)))
            )
            if a_ub.shape[0]
            else 0.0,
            "lower_bound": float(max(0.0, np.max(np.asarray(self.lower_bounds) - h))),
            "upper_bound": float(max(0.0, np.max(h - np.asarray(self.upper_bounds)))),
            "turnover": 0.0,
            "gross_exposure": 0.0,
        }
        if self.turnover_limit is not None:
            values["turnover"] = float(
                max(0.0, np.sum(np.abs(h - np.asarray(self.turnover_center))) - self.turnover_limit)
            )
        if self.gross_exposure_limit is not None:
            values["gross_exposure"] = float(
                max(0.0, np.sum(np.abs(h)) - self.gross_exposure_limit)
            )
        return values

    def max_violation(self, holdings: ArrayLike) -> float:
        return max(self.violations(holdings).values())

"""Tall/wide Stiefel tangent geometry and Muon-source perturbations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .polar import polar_factor, polar_frechet_derivative


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StiefelProjectionRecord:
    tangent: FloatArray
    normal: FloatArray
    orientation: str
    factor_constraint_residual: float
    tangent_constraint_residual: float
    reconstruction_residual: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class MuonPolarPerturbationRecord:
    base_factor: FloatArray
    plus_factor: FloatArray
    minus_factor: FloatArray
    central_response: FloatArray
    analytic_response: FloatArray
    ambient_stiefel_projection: FloatArray
    source_singular_values: FloatArray
    epsilon: float
    central_vs_analytic_relative_error: float
    polar_response_tangent_residual: float
    ambient_projection_relative_difference: float
    orientation: str
    operator_kind: str
    map_definition: str


def sym(matrix: ArrayLike) -> FloatArray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("sym expects a square matrix")
    return 0.5 * (array + array.T)


def _matrices(factor: ArrayLike, ambient: ArrayLike) -> tuple[FloatArray, FloatArray]:
    q = np.asarray(factor, dtype=np.float64)
    z = np.asarray(ambient, dtype=np.float64)
    if q.ndim != 2 or z.ndim != 2 or q.shape != z.shape:
        raise ValueError("factor and ambient must be same-shaped matrices")
    if min(q.shape) < 1 or not np.all(np.isfinite(q)) or not np.all(np.isfinite(z)):
        raise ValueError("factor and ambient must be finite and non-empty")
    return q, z


def project_stiefel_tangent(
    factor: ArrayLike,
    ambient: ArrayLike,
    *,
    validate_factor: bool = True,
    factor_tolerance: float = 1e-7,
) -> StiefelProjectionRecord:
    """Orthogonally project an ambient matrix onto a Stiefel tangent space.

    For a tall column-Stiefel factor, ``Q.T Q=I`` and

    ``Pi_Q(Z) = Z - Q sym(Q.T Z)``.

    A wide row-Stiefel factor is handled by transposition, equivalently

    ``Pi_Q(Z) = Z - sym(Z Q.T) Q``.
    """

    q, z = _matrices(factor, ambient)
    rows, columns = q.shape
    if rows >= columns:
        identity = np.eye(columns, dtype=np.float64)
        factor_error = q.T @ q - identity
        normal = q @ sym(q.T @ z)
        tangent = z - normal
        tangent_error = q.T @ tangent + tangent.T @ q
        orientation = "column_stiefel"
        definition = "Pi_Q(Z)=Z-Q sym(Q^T Z), for Q^T Q=I"
    else:
        identity = np.eye(rows, dtype=np.float64)
        factor_error = q @ q.T - identity
        normal = sym(z @ q.T) @ q
        tangent = z - normal
        tangent_error = tangent @ q.T + q @ tangent.T
        orientation = "row_stiefel"
        definition = "Pi_Q(Z)=Z-sym(Z Q^T) Q, for Q Q^T=I"

    factor_residual = float(np.linalg.norm(factor_error, ord="fro"))
    if validate_factor and factor_residual > float(factor_tolerance):
        raise ValueError(
            "factor is not on the requested Stiefel manifold: "
            f"constraint residual={factor_residual:.3e}"
        )
    tangent_residual = float(
        np.linalg.norm(tangent_error, ord="fro")
        / max(np.linalg.norm(tangent, ord="fro"), 1.0)
    )
    reconstruction_residual = float(
        np.linalg.norm(z - tangent - normal, ord="fro")
    )
    return StiefelProjectionRecord(
        tangent=tangent,
        normal=normal,
        orientation=orientation,
        factor_constraint_residual=factor_residual,
        tangent_constraint_residual=tangent_residual,
        reconstruction_residual=reconstruction_residual,
        operator_kind="orthogonal_stiefel_tangent_projection",
        map_definition=definition,
    )


def muon_polar_source_perturbation(
    source: ArrayLike,
    perturbation: ArrayLike,
    *,
    epsilon: float | None = None,
    rcond: float | None = None,
) -> MuonPolarPerturbationRecord:
    """Probe how an ideal Muon polar source responds to a perturbation.

    This isolates the exact polar map ``P(M)`` that Newton--Schulz approximates.
    It is neither a model-training Jacobian nor a claim that the finite-step
    Newton--Schulz implementation equals this map exactly.
    """

    matrix = np.asarray(source, dtype=np.float64)
    direction = np.asarray(perturbation, dtype=np.float64)
    if matrix.ndim != 2 or direction.shape != matrix.shape:
        raise ValueError("source and perturbation must be same-shaped matrices")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(direction)):
        raise ValueError("source and perturbation must be finite")
    direction_norm = float(np.linalg.norm(direction, ord="fro"))
    if direction_norm == 0.0:
        raise ValueError("perturbation must be nonzero")

    if epsilon is None:
        epsilon = float(
            np.cbrt(np.finfo(np.float64).eps)
            * max(np.linalg.norm(matrix, ord="fro"), 1.0)
            / direction_norm
        )
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    base = polar_factor(matrix)
    plus = polar_factor(matrix + float(epsilon) * direction)
    minus = polar_factor(matrix - float(epsilon) * direction)
    central = (plus - minus) / (2.0 * float(epsilon))
    analytic_record = polar_frechet_derivative(
        matrix,
        direction,
        rcond=rcond,
    )
    analytic = analytic_record.derivative
    ambient_projection = project_stiefel_tangent(base, direction).tangent

    analytic_norm = float(np.linalg.norm(analytic, ord="fro"))
    finite_difference_error = float(
        np.linalg.norm(central - analytic, ord="fro")
        / max(analytic_norm, np.finfo(np.float64).tiny)
    )
    projection_difference = float(
        np.linalg.norm(analytic - ambient_projection, ord="fro")
        / max(analytic_norm, np.finfo(np.float64).tiny)
    )
    return MuonPolarPerturbationRecord(
        base_factor=base,
        plus_factor=plus,
        minus_factor=minus,
        central_response=central,
        analytic_response=analytic,
        ambient_stiefel_projection=ambient_projection,
        source_singular_values=np.linalg.svd(matrix, compute_uv=False),
        epsilon=float(epsilon),
        central_vs_analytic_relative_error=finite_difference_error,
        polar_response_tangent_residual=(
            analytic_record.tangent_constraint_residual
        ),
        ambient_projection_relative_difference=projection_difference,
        orientation=analytic_record.orientation,
        operator_kind="ideal_muon_polar_source_perturbation",
        map_definition=(
            "central and analytic directional derivatives of M -> P(M)=UV^T; "
            "the exact polar source is an idealization of Muon Newton-Schulz"
        ),
    )


# Descriptive alias used by some experiment cells.
stiefel_tangent_projection = project_stiefel_tangent


__all__ = [
    "MuonPolarPerturbationRecord",
    "StiefelProjectionRecord",
    "muon_polar_source_perturbation",
    "project_stiefel_tangent",
    "stiefel_tangent_projection",
    "sym",
]

"""Gauge-aligned core and Grassmann spectra for rectangular matrix flows.

A full-rank matrix W in R^{m x n} is decomposed into a square invertible core
and an orthonormal basis for its row space (m <= n) or column space (m > n).
Successive bases are aligned with the orthogonal Procrustes solution before the
square relative-flow operator is formed. The remaining subspace motion is
reported through principal angles.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _as_matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
    matrix = value.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape={tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _validate_pair(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    rank_rtol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = _as_matrix(previous, name="previous")
    right = _as_matrix(current, name="current")
    if left.shape != right.shape:
        raise ValueError(
            "successive matrices must have the same shape, got "
            f"{tuple(left.shape)} and {tuple(right.shape)}"
        )
    if not math.isfinite(float(rank_rtol)) or not 0.0 < rank_rtol < 1.0:
        raise ValueError("rank_rtol must be finite and lie in (0, 1)")
    return left, right


def _full_rank_svd(
    matrix: torch.Tensor,
    *,
    rank_rtol: float,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    left, singular_values, right_h = torch.linalg.svd(
        matrix, full_matrices=False
    )
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    threshold = float(rank_rtol) * largest
    if smallest <= threshold:
        numerical_rank = int(
            torch.count_nonzero(singular_values > threshold).item()
        )
        raise ValueError(
            f"{name} is not numerically full rank: rank={numerical_rank}, "
            f"expected={min(matrix.shape)}, smallest={smallest:.3e}, "
            f"threshold={threshold:.3e}"
        )
    condition = largest / smallest
    return left, singular_values, right_h, condition


def _align_subspace_basis(
    previous_basis: torch.Tensor,
    current_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align current_basis to previous_basis by orthogonal Procrustes.

    The bases have shape ambient x rank and orthonormal columns. If
    previous_basis.T @ current_basis = L diag(cos(theta)) R.T, the minimizing
    right action is R L.T.
    """

    overlap = previous_basis.T @ current_basis
    left, cosines, right_h = torch.linalg.svd(overlap, full_matrices=False)
    alignment = right_h.T @ left.T
    aligned = current_basis @ alignment
    cosines = cosines.clamp(0.0, 1.0)
    principal_angles = torch.acos(cosines)
    return aligned, principal_angles, cosines, alignment


def principal_angles_sorted(angles: torch.Tensor) -> torch.Tensor:
    values = angles.detach().to(device="cpu", dtype=torch.float64).flatten()
    return torch.sort(values).values


def aligned_core_flow_operator(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    rank_rtol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return the square core flow and principal angles for a full-rank pair.

    For a wide matrix, W = B V.T with an invertible m x m core B and a row-space
    basis V in St(n, m). The current V is Procrustes-aligned to the previous V,
    and J_core = B_current_aligned B_previous^{-1}.

    For a tall matrix, W = U B with an invertible n x n core B and a column-space
    basis U in St(m, n). After alignment,
    J_core = B_previous^{-1} B_current_aligned.

    For a square full-rank matrix, this reduces (up to roundoff) to
    J_core = W_current W_previous^{-1}, with no nontrivial angular sector.
    """

    old, new = _validate_pair(previous, current, rank_rtol=rank_rtol)
    rows, columns = old.shape
    rank = min(rows, columns)

    old_u, _, old_vh, old_condition = _full_rank_svd(
        old, rank_rtol=rank_rtol, name="previous"
    )
    new_u, _, new_vh, new_condition = _full_rank_svd(
        new, rank_rtol=rank_rtol, name="current"
    )

    if rows <= columns:
        old_basis = old_vh.T
        new_basis = new_vh.T
        aligned_basis, angles, cosines, alignment = _align_subspace_basis(
            old_basis, new_basis
        )
        old_core = old @ old_basis
        new_core = new @ aligned_basis
        operator = torch.linalg.solve(old_core.T, new_core.T).T
        subspace = "row"
        ambient_dimension = columns
    else:
        old_basis = old_u
        new_basis = new_u
        aligned_basis, angles, cosines, alignment = _align_subspace_basis(
            old_basis, new_basis
        )
        old_core = old_basis.T @ old
        new_core = aligned_basis.T @ new
        operator = torch.linalg.solve(old_core, new_core)
        subspace = "column"
        ambient_dimension = rows

    forced_intersection = max(0, 2 * rank - ambient_dimension)
    metadata: dict[str, Any] = {
        "shape": (rows, columns),
        "rank": rank,
        "subspace": subspace,
        "ambient_dimension": ambient_dimension,
        "forced_intersection_dimension": forced_intersection,
        "maximum_angular_modes": rank - forced_intersection,
        "previous_condition_number": old_condition,
        "current_condition_number": new_condition,
        "cosines": cosines,
        "alignment": alignment,
    }
    return operator, principal_angles_sorted(angles), metadata


def squared_singular_value_spectrum(matrix: torch.Tensor) -> np.ndarray:
    value = _as_matrix(matrix, name="matrix")
    eigenvalues = torch.linalg.svdvals(value).square().numpy()
    eigenvalues = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > 0.0)]
    return np.sort(eigenvalues)


def core_log_flow_spectrum(
    operator: torch.Tensor,
    *,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """Return |log sigma(J_core)^2| after removing identity modes."""

    if not math.isfinite(float(zero_tol)) or zero_tol < 0.0:
        raise ValueError("zero_tol must be finite and nonnegative")
    eigenvalues = squared_singular_value_spectrum(operator)
    deviations = np.abs(np.log(eigenvalues))
    return np.sort(deviations[deviations > float(zero_tol)])


def grassmann_angular_spectrum(
    principal_angles: torch.Tensor | np.ndarray,
    *,
    forced_intersection_dimension: int = 0,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """Return theta^2 after removing dimension-forced and numerical zeros."""

    angles = np.asarray(
        torch.as_tensor(principal_angles, dtype=torch.float64).cpu(), dtype=float
    ).reshape(-1)
    angles = np.sort(angles[np.isfinite(angles) & (angles >= 0.0)])
    forced = int(forced_intersection_dimension)
    if forced < 0 or forced > angles.size:
        raise ValueError(
            "forced_intersection_dimension must lie between zero and the "
            "number of principal angles"
        )
    if not math.isfinite(float(zero_tol)) or zero_tol < 0.0:
        raise ValueError("zero_tol must be finite and nonnegative")
    values = np.square(angles[forced:])
    return np.sort(values[values > float(zero_tol)])


def rectangular_flow_spectra(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    rank_rtol: float = 1e-10,
    log_zero_tol: float = 1e-12,
    angle_zero_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compute aligned core and Grassmann spectra for one matrix step."""

    operator, angles, metadata = aligned_core_flow_operator(
        previous, current, rank_rtol=rank_rtol
    )
    core_eigenvalues = squared_singular_value_spectrum(operator)
    core_log = core_log_flow_spectrum(operator, zero_tol=log_zero_tol)
    angular = grassmann_angular_spectrum(
        angles,
        forced_intersection_dimension=int(
            metadata["forced_intersection_dimension"]
        ),
        zero_tol=angle_zero_tol,
    )
    return {
        "core_operator": operator,
        "core_eigenvalues": core_eigenvalues,
        "core_log_deviation": core_log,
        "principal_angles": angles.numpy(),
        "angular_eigenvalues": angular,
        **metadata,
    }


__all__ = [
    "aligned_core_flow_operator",
    "core_log_flow_spectrum",
    "grassmann_angular_spectrum",
    "principal_angles_sorted",
    "rectangular_flow_spectra",
    "squared_singular_value_spectrum",
]

"""Polar-factor geometry for local Muon-flow experiments.

The routines in this module differentiate the *matrix polar map*.  They do not
differentiate a loss, optimizer, or training trajectory.  That distinction is
carried explicitly in every diagnostic record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PolarDecompositionRecord:
    factor: FloatArray
    singular_values: FloatArray
    numerical_rank: int
    rank_tolerance: float
    full_rank: bool
    orientation: str
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class PolarFrechetRecord:
    derivative: FloatArray
    factor: FloatArray
    singular_values: FloatArray
    orientation: str
    tangent_constraint_residual: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class PolarPullbackSpectrumRecord:
    singular_amplitudes: FloatArray
    jt_j_nonzero_eigenvalues: FloatArray
    zero_count: int
    input_dimension: int
    derivative_rank: int
    singular_values: FloatArray
    mode_labels: tuple[str, ...]
    orientation: str
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class CentralDifferenceJacobianRecord:
    jacobian: FloatArray
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    coordinate_steps: FloatArray
    singular_values: FloatArray
    jt_j_eigenvalues: FloatArray
    numerical_rank: int
    rank_relative_tolerance: float
    rank_absolute_tolerance: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class MuonNewtonSchulzRecord:
    output: FloatArray
    ideal_polar_factor: FloatArray
    input_frobenius_norm: float
    relative_error_to_ideal_polar: float
    transposed_for_iteration: bool
    steps: int
    eps: float
    coefficients: tuple[float, float, float]
    operator_kind: str
    map_definition: str


def _as_matrix(value: ArrayLike, *, name: str = "matrix") -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {array.shape}")
    if min(array.shape) < 1:
        raise ValueError(f"{name} must have non-zero dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _rank_tolerance(
    singular_values: FloatArray,
    shape: tuple[int, int],
    rcond: float | None,
) -> float:
    largest = float(singular_values[0]) if singular_values.size else 0.0
    if rcond is None:
        return float(max(shape) * np.finfo(np.float64).eps * max(largest, 1.0))
    if rcond < 0.0:
        raise ValueError("rcond must be non-negative")
    return float(rcond * max(largest, 1.0))


def polar_factor(matrix: ArrayLike) -> FloatArray:
    """Return the thin-SVD polar factor ``U @ Vh``.

    At a rank-deficient input this is a conventional SVD representative; the
    polar factor is not locally unique there.  Derivative routines therefore
    require full rank separately.
    """

    work = _as_matrix(matrix)
    left, _, right_h = np.linalg.svd(work, full_matrices=False)
    return left @ right_h


def polar_decomposition(
    matrix: ArrayLike,
    *,
    rcond: float | None = None,
) -> PolarDecompositionRecord:
    work = _as_matrix(matrix)
    left, singular_values, right_h = np.linalg.svd(work, full_matrices=False)
    tolerance = _rank_tolerance(singular_values, work.shape, rcond)
    rank = int(np.count_nonzero(singular_values > tolerance))
    rows, columns = work.shape
    return PolarDecompositionRecord(
        factor=left @ right_h,
        singular_values=singular_values,
        numerical_rank=rank,
        rank_tolerance=tolerance,
        full_rank=rank == min(rows, columns),
        orientation="column_stiefel" if rows >= columns else "row_stiefel",
        operator_kind="rectangular_polar_factor",
        map_definition="P(A)=U V^T for the thin SVD A=U diag(s) V^T",
    )


def polar_frechet_derivative(
    matrix: ArrayLike,
    direction: ArrayLike,
    *,
    rcond: float | None = None,
) -> PolarFrechetRecord:
    """Evaluate the full-rank Frechet derivative of the polar map.

    For ``F=U.T @ E @ V``, the in-support skew term is

    ``K_ij = (F_ij-F_ji)/(s_i+s_j)``.

    Tall and wide matrices additionally have, respectively, column-normal and
    row-normal terms with amplitudes ``1/s_i``.
    """

    work = _as_matrix(matrix)
    perturbation = _as_matrix(direction, name="direction")
    if perturbation.shape != work.shape:
        raise ValueError("matrix and direction must have identical shapes")

    left, singular_values, right_h = np.linalg.svd(work, full_matrices=False)
    tolerance = _rank_tolerance(singular_values, work.shape, rcond)
    if np.any(singular_values <= tolerance):
        raise np.linalg.LinAlgError(
            "the polar Frechet derivative requires full row/column rank"
        )

    right = right_h.T
    core_direction = left.T @ perturbation @ right
    denominator = singular_values[:, None] + singular_values[None, :]
    skew = (core_direction - core_direction.T) / denominator
    core = left @ skew @ right.T

    rows, columns = work.shape
    inverse_s = 1.0 / singular_values
    if rows >= columns:
        perpendicular = perturbation - left @ (left.T @ perturbation)
        complement = (
            (perpendicular @ right) * inverse_s[None, :]
        ) @ right.T
        derivative = core + complement
        factor = left @ right.T
        constraint = factor.T @ derivative + derivative.T @ factor
        orientation = "column_stiefel"
    else:
        perpendicular = perturbation - (perturbation @ right) @ right.T
        complement = left @ (
            inverse_s[:, None] * (left.T @ perpendicular)
        )
        derivative = core + complement
        factor = left @ right.T
        constraint = derivative @ factor.T + factor @ derivative.T
        orientation = "row_stiefel"

    residual = float(
        np.linalg.norm(constraint, ord="fro")
        / max(np.linalg.norm(derivative, ord="fro"), 1.0)
    )
    return PolarFrechetRecord(
        derivative=derivative,
        factor=factor,
        singular_values=singular_values,
        orientation=orientation,
        tangent_constraint_residual=residual,
        operator_kind="polar_map_frechet_derivative",
        map_definition=(
            "D P_A[E], the derivative of P(A)=U V^T; this is not a "
            "training-map Jacobian"
        ),
    )


def polar_pullback_spectrum(
    matrix: ArrayLike,
    *,
    rcond: float | None = None,
) -> PolarPullbackSpectrumRecord:
    """Return the exact nonzero spectrum of ``(D P)^* (D P)``.

    The singular amplitudes of ``D P`` are ``2/(s_i+s_j)`` for each
    in-support skew pair and ``1/s_i`` for every rectangular complement
    direction.  The remaining ``r(r+1)/2`` input directions are exact zeros.
    """

    work = _as_matrix(matrix)
    singular_values = np.linalg.svd(work, compute_uv=False)
    tolerance = _rank_tolerance(singular_values, work.shape, rcond)
    if np.any(singular_values <= tolerance):
        raise np.linalg.LinAlgError(
            "the exact polar derivative spectrum requires full rank"
        )

    rows, columns = work.shape
    rank = min(rows, columns)
    amplitudes: list[float] = []
    labels: list[str] = []
    for first in range(rank):
        for second in range(first + 1, rank):
            amplitudes.append(
                2.0 / float(singular_values[first] + singular_values[second])
            )
            labels.append(f"skew[{first},{second}]")

    complement_multiplicity = abs(rows - columns)
    complement_name = "column_complement" if rows > columns else "row_complement"
    for index, value in enumerate(singular_values):
        for copy_index in range(complement_multiplicity):
            amplitudes.append(1.0 / float(value))
            labels.append(f"{complement_name}[{index},{copy_index}]")

    order = np.argsort(np.asarray(amplitudes, dtype=np.float64))[::-1]
    amplitude_array = np.asarray(amplitudes, dtype=np.float64)[order]
    ordered_labels = tuple(labels[index] for index in order)
    input_dimension = int(rows * columns)
    derivative_rank = int(amplitude_array.size)
    zero_count = input_dimension - derivative_rank
    expected_zeros = rank * (rank + 1) // 2
    if zero_count != expected_zeros:
        raise RuntimeError("internal polar-spectrum dimension audit failed")

    return PolarPullbackSpectrumRecord(
        singular_amplitudes=amplitude_array,
        jt_j_nonzero_eigenvalues=amplitude_array**2,
        zero_count=zero_count,
        input_dimension=input_dimension,
        derivative_rank=derivative_rank,
        singular_values=singular_values,
        mode_labels=ordered_labels,
        orientation="column_stiefel" if rows >= columns else "row_stiefel",
        operator_kind="polar_frechet_pullback_gram_spectrum",
        map_definition=(
            "nonzero eigenvalues of (D P_A)^*(D P_A), with exact nullity "
            "r(r+1)/2; this is polar-map geometry, not an optimizer Jacobian"
        ),
    )


def muon_quintic_orthogonalizer(
    update: ArrayLike,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> FloatArray:
    """Pure-NumPy form of ``zeropower_via_newton_schulz_5``.

    This matches the repository Muon kernel algebra: transpose a tall source,
    Frobenius-normalize it, and apply the quintic coefficients
    ``(3.4445, -4.7750, 2.0315)`` for the requested number of iterations.
    """

    matrix = _as_matrix(update, name="update")
    if int(steps) < 1 or eps <= 0.0:
        raise ValueError("steps and eps must be positive")
    transposed = matrix.shape[0] > matrix.shape[1]
    work = matrix.T.copy() if transposed else matrix.copy()
    norm = float(np.linalg.norm(work, ord="fro"))
    work /= max(norm, float(eps))
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = work @ work.T
        work = a * work + (b * gram + c * (gram @ gram)) @ work
    return work.T if transposed else work


def muon_newton_schulz_map(
    update: ArrayLike,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> MuonNewtonSchulzRecord:
    """Evaluate the actual finite-iteration Muon orthogonalizer map."""

    matrix = _as_matrix(update, name="update")
    output = muon_quintic_orthogonalizer(matrix, steps=steps, eps=eps)
    ideal = polar_factor(matrix)
    ideal_norm = float(np.linalg.norm(ideal, ord="fro"))
    return MuonNewtonSchulzRecord(
        output=output,
        ideal_polar_factor=ideal,
        input_frobenius_norm=float(np.linalg.norm(matrix, ord="fro")),
        relative_error_to_ideal_polar=float(
            np.linalg.norm(output - ideal, ord="fro")
            / max(ideal_norm, np.finfo(np.float64).tiny)
        ),
        transposed_for_iteration=matrix.shape[0] > matrix.shape[1],
        steps=int(steps),
        eps=float(eps),
        coefficients=(3.4445, -4.7750, 2.0315),
        operator_kind="actual_finite_iteration_muon_quintic_orthogonalizer",
        map_definition=(
            "transpose tall M; X0=M/max(||M||_F,eps); repeat "
            "X<-3.4445X+(-4.7750XX^T+2.0315(XX^T)^2)X; transpose back. "
            "This is the implemented finite map, distinct from ideal P(M)."
        ),
    )


def central_difference_jacobian(
    map_fn: Callable[[FloatArray], ArrayLike],
    point: ArrayLike,
    *,
    epsilon: float | None = None,
    max_input_dimension: int = 256,
    rank_rtol: float = 1e-7,
    rank_atol: float = 0.0,
    operator_kind: str = "numerical_central_difference_jacobian",
    map_definition: str = "user-supplied map",
) -> CentralDifferenceJacobianRecord:
    """Materialize a small explicit Jacobian with central differences."""

    base = np.asarray(point, dtype=np.float64)
    if not np.all(np.isfinite(base)):
        raise ValueError("point contains non-finite values")
    dimension = int(base.size)
    if dimension < 1:
        raise ValueError("point must be non-empty")
    if dimension > int(max_input_dimension):
        raise ValueError(
            f"explicit Jacobian dimension {dimension} exceeds "
            f"max_input_dimension={max_input_dimension}"
        )
    if epsilon is None:
        epsilon = float(
            np.cbrt(np.finfo(np.float64).eps)
            * max(1.0, np.linalg.norm(base.ravel()) / np.sqrt(dimension))
        )
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if rank_rtol < 0.0 or rank_atol < 0.0:
        raise ValueError("rank tolerances must be non-negative")

    output = np.asarray(map_fn(base.copy()), dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ValueError("map_fn returned non-finite values at the base point")
    jacobian = np.empty((output.size, dimension), dtype=np.float64)
    steps = np.empty(dimension, dtype=np.float64)
    flat = base.ravel()
    for index in range(dimension):
        step = float(epsilon * max(1.0, abs(float(flat[index]))))
        steps[index] = step
        plus = base.copy().ravel()
        minus = base.copy().ravel()
        plus[index] += step
        minus[index] -= step
        plus_value = np.asarray(map_fn(plus.reshape(base.shape)), dtype=np.float64)
        minus_value = np.asarray(map_fn(minus.reshape(base.shape)), dtype=np.float64)
        if plus_value.shape != output.shape or minus_value.shape != output.shape:
            raise ValueError("map_fn output shape changed under perturbation")
        jacobian[:, index] = (
            plus_value.ravel() - minus_value.ravel()
        ) / (2.0 * step)

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    largest_jacobian_singular = (
        float(singular_values[0]) if singular_values.size else 0.0
    )
    rank_tolerance = max(
        float(rank_atol),
        float(rank_rtol) * largest_jacobian_singular,
    )
    return CentralDifferenceJacobianRecord(
        jacobian=jacobian,
        input_shape=tuple(base.shape),
        output_shape=tuple(output.shape),
        coordinate_steps=steps,
        singular_values=singular_values,
        jt_j_eigenvalues=np.sort(
            np.maximum(np.linalg.eigvalsh(jacobian.T @ jacobian), 0.0)
        )[::-1],
        numerical_rank=int(np.count_nonzero(singular_values > rank_tolerance)),
        rank_relative_tolerance=float(rank_rtol),
        rank_absolute_tolerance=float(rank_tolerance),
        operator_kind=str(operator_kind),
        map_definition=str(map_definition),
    )


def explicit_polar_jacobian(
    matrix: ArrayLike,
    *,
    epsilon: float | None = None,
    max_input_dimension: int = 256,
    rank_rtol: float = 1e-7,
    rank_atol: float = 0.0,
) -> CentralDifferenceJacobianRecord:
    """Numerically materialize the polar-map Jacobian for a small matrix."""

    work = _as_matrix(matrix)
    return central_difference_jacobian(
        polar_factor,
        work,
        epsilon=epsilon,
        max_input_dimension=max_input_dimension,
        rank_rtol=rank_rtol,
        rank_atol=rank_atol,
        operator_kind="explicit_numerical_polar_map_jacobian",
        map_definition=(
            "central-difference Jacobian of P(A)=U V^T; explicitly not a "
            "Jacobian of the neural-network training map"
        ),
    )


def explicit_muon_newton_schulz_jacobian(
    update: ArrayLike,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    difference_epsilon: float | None = None,
    max_input_dimension: int = 256,
    rank_rtol: float = 1e-7,
    rank_atol: float = 0.0,
) -> CentralDifferenceJacobianRecord:
    """Materialize the small Jacobian of the finite Muon quintic map."""

    matrix = _as_matrix(update, name="update")
    return central_difference_jacobian(
        lambda value: muon_quintic_orthogonalizer(
            value,
            steps=steps,
            eps=eps,
        ),
        matrix,
        epsilon=difference_epsilon,
        max_input_dimension=max_input_dimension,
        rank_rtol=rank_rtol,
        rank_atol=rank_atol,
        operator_kind="explicit_numerical_finite_muon_quintic_map_jacobian",
        map_definition=(
            "central-difference Jacobian of the actual configured finite "
            "Newton-Schulz quintic orthogonalizer; compare against, but do not "
            "identify with, the ideal polar-map derivative"
        ),
    )


# Name mirroring the PyTorch kernel for direct cross-checks.
zeropower_via_newton_schulz_5_numpy = muon_quintic_orthogonalizer


__all__ = [
    "CentralDifferenceJacobianRecord",
    "MuonNewtonSchulzRecord",
    "PolarDecompositionRecord",
    "PolarFrechetRecord",
    "PolarPullbackSpectrumRecord",
    "central_difference_jacobian",
    "explicit_polar_jacobian",
    "explicit_muon_newton_schulz_jacobian",
    "muon_newton_schulz_map",
    "muon_quintic_orthogonalizer",
    "polar_decomposition",
    "polar_factor",
    "polar_frechet_derivative",
    "polar_pullback_spectrum",
    "zeropower_via_newton_schulz_5_numpy",
]

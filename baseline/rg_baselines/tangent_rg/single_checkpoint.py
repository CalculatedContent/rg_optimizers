"""Single-checkpoint spectral maps and calibrated local-map contracts.

Weights alone determine algebraic spectral derivatives.  They do not determine
the local training map: that additionally requires a batch/loss definition and
the complete optimizer state.  The calibrated interfaces below enforce those
inputs rather than silently calling a weight-only object a training Jacobian.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GramTranslationRecord:
    gram: FloatArray
    translated_gram: FloatArray
    gram_eigenvalues: FloatArray
    translated_eigenvalues: FloatArray
    lambda_min: float
    side: str
    numerical_rank: int
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class NormalizedGramRecord:
    value: FloatArray
    eigenvalues: FloatArray
    side: str
    normalization_dimension: int
    frobenius_norm_sq: float
    trace_audit: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class NormalizedGramJVPRecord:
    value: FloatArray
    jvp: FloatArray
    side: str
    normalization_dimension: int
    scale_direction_residual: float
    trace_derivative_residual: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class NormalizedGramSpectrumRecord:
    singular_amplitudes: FloatArray
    jt_j_nonzero_eigenvalues: FloatArray
    zero_count: int
    input_dimension: int
    derivative_rank: int
    side: str
    matrix_singular_values: FloatArray
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class SingleCheckpointJacobianSpectrumRecord:
    singular_amplitudes: FloatArray
    jt_j_nonzero_eigenvalues: FloatArray
    zero_count: int
    input_dimension: int
    derivative_rank: int
    side: str
    matrix_singular_values: FloatArray
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class CenteredLogGramRecord:
    value: FloatArray
    normalized_gram: FloatArray
    normalized_gram_eigenvalues: FloatArray
    side: str
    normalization_dimension: int
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class CenteredLogGramJVPRecord:
    value: FloatArray
    jvp: FloatArray
    side: str
    normalization_dimension: int
    scale_direction_residual: float
    trace_derivative_residual: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class CenteredLogSingularRecord:
    value: FloatArray
    singular_values: FloatArray
    mean_log_singular_value: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class CenteredLogSingularJVPRecord:
    value: FloatArray
    jvp: FloatArray
    scale_direction_residual: float
    sum_derivative_residual: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class CenteredLogSingularFlowRecord:
    singular_values: FloatArray
    instantaneous_log_singular_rates: FloatArray
    centered_log_singular_rates: FloatArray
    mean_log_singular_rate: float
    retained_rank: int
    dropped_singular_directions: int
    delta_s: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class CalibratedTrainingMapContract:
    map_definition: str
    requires_batch: bool
    requires_optimizer_state: bool
    weight_only: bool
    operator_kind: str


@dataclass(frozen=True)
class CalibratedTrainingMapDerivativeRecord:
    base_output: Any
    directional_derivative: Any
    epsilon: float | None
    backend: str
    batch_supplied: bool
    optimizer_state_supplied: bool
    weight_only: bool
    operator_kind: str
    map_definition: str


def _matrix(value: ArrayLike, *, name: str = "weight") -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _side_and_gram(weight: FloatArray, side: str) -> tuple[str, FloatArray]:
    rows, columns = weight.shape
    selected = str(side).lower()
    if selected == "auto":
        selected = "right" if rows >= columns else "left"
    if selected in {"column", "columns"}:
        selected = "right"
    if selected in {"row", "rows"}:
        selected = "left"
    if selected == "right":
        return selected, weight.T @ weight
    if selected == "left":
        return selected, weight @ weight.T
    raise ValueError("side must be auto, right/column, or left/row")


def _full_rank_svd(
    matrix: FloatArray,
    *,
    rcond: float | None,
    map_name: str,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0])
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative")
    tolerance = (
        float(rcond) * max(largest, 1.0)
        if rcond is not None
        else max(matrix.shape)
        * np.finfo(np.float64).eps
        * max(largest, 1.0)
    )
    if np.any(singular_values <= tolerance):
        raise np.linalg.LinAlgError(
            f"{map_name} requires full rank on the smaller matrix side"
        )
    return left, singular_values, right_h, float(tolerance)


def _positive_singular_amplitudes(operator: FloatArray) -> FloatArray:
    amplitudes = np.linalg.svd(operator, compute_uv=False)
    tolerance = (
        max(operator.shape)
        * np.finfo(np.float64).eps
        * max(float(amplitudes[0]), 1.0)
    )
    return amplitudes[amplitudes > tolerance]


def _spectrum_singular_values(
    matrix: FloatArray,
    *,
    precomputed_singular_values: ArrayLike | None,
    rcond: float | None,
    map_name: str,
) -> tuple[FloatArray, float]:
    if precomputed_singular_values is None:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    else:
        singular_values = np.asarray(
            precomputed_singular_values,
            dtype=np.float64,
        )
        expected = min(matrix.shape)
        if singular_values.shape != (expected,):
            raise ValueError(
                f"precomputed_singular_values must have shape {(expected,)}"
            )
        if not np.all(np.isfinite(singular_values)):
            raise ValueError("precomputed_singular_values contain non-finite values")
        if np.any(np.diff(singular_values) > 0.0):
            raise ValueError("precomputed_singular_values must be non-increasing")
    largest = float(singular_values[0])
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative")
    tolerance = (
        float(rcond) * max(largest, 1.0)
        if rcond is not None
        else max(matrix.shape)
        * np.finfo(np.float64).eps
        * max(largest, 1.0)
    )
    if np.any(singular_values <= tolerance):
        raise np.linalg.LinAlgError(
            f"{map_name} requires full rank on the smaller matrix side"
        )
    return singular_values, float(tolerance)


def _require_simple_singular_values(
    singular_values: FloatArray,
    *,
    tolerance: float,
    map_name: str,
) -> None:
    if singular_values.size > 1 and np.any(
        np.abs(np.diff(singular_values)) <= tolerance
    ):
        raise np.linalg.LinAlgError(
            f"{map_name} is not differentiable at repeated singular values"
        )


def _log_divided_difference(first: float, second: float) -> float:
    scale = max(abs(first), abs(second), 1.0)
    if abs(first - second) <= 32.0 * np.finfo(np.float64).eps * scale:
        return 2.0 / (first + second)
    return float((np.log(first) - np.log(second)) / (first - second))


def gram_translation_quotient(
    weight: ArrayLike,
    *,
    side: str = "auto",
    eigenvalue_tolerance: float | None = None,
) -> GramTranslationRecord:
    """Remove the scalar-identity component ``lambda_min(G) I`` from a Gram map."""

    matrix = _matrix(weight)
    selected, gram = _side_and_gram(matrix, side)
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    lambda_min = float(eigenvalues[0])
    tolerance = (
        float(eigenvalue_tolerance)
        if eigenvalue_tolerance is not None
        else gram.shape[0]
        * np.finfo(np.float64).eps
        * max(float(eigenvalues[-1]), 1.0)
    )
    if tolerance < 0.0:
        raise ValueError("eigenvalue_tolerance must be non-negative")
    if lambda_min < 0.0 and abs(lambda_min) <= tolerance:
        lambda_min = 0.0
    if lambda_min < -tolerance:
        raise RuntimeError("Gram matrix has a materially negative eigenvalue")
    translated = gram - lambda_min * np.eye(gram.shape[0])
    translated_eigenvalues = eigenvalues - lambda_min
    return GramTranslationRecord(
        gram=gram,
        translated_gram=translated,
        gram_eigenvalues=eigenvalues[::-1],
        translated_eigenvalues=translated_eigenvalues[::-1],
        lambda_min=lambda_min,
        side=selected,
        numerical_rank=int(np.count_nonzero(translated_eigenvalues > tolerance)),
        operator_kind="single_checkpoint_gram_translation_quotient",
        map_definition=(
            "Q_G(W)=G(W)-lambda_min(G(W)) I on the smaller Gram side; "
            "an algebraic translation quotient, not a training Jacobian"
        ),
    )


def normalized_gram_map(
    weight: ArrayLike,
    *,
    side: str = "auto",
) -> NormalizedGramRecord:
    """Return the scale-free Gram map ``d G / ||W||_F^2`` with trace ``d``."""

    matrix = _matrix(weight)
    selected, gram = _side_and_gram(matrix, side)
    frobenius_sq = float(np.sum(matrix * matrix))
    if frobenius_sq <= 0.0:
        raise ValueError("normalized Gram map is undefined at the zero matrix")
    dimension = int(gram.shape[0])
    value = (float(dimension) / frobenius_sq) * gram
    value = 0.5 * (value + value.T)
    return NormalizedGramRecord(
        value=value,
        eigenvalues=np.linalg.eigvalsh(value)[::-1],
        side=selected,
        normalization_dimension=dimension,
        frobenius_norm_sq=frobenius_sq,
        trace_audit=float(np.trace(value)),
        operator_kind="single_checkpoint_normalized_gram_map",
        map_definition=(
            "F(W)=d G(W)/||W||_F^2, where d is the selected Gram dimension; "
            "F(cW)=F(W) and trace(F)=d"
        ),
    )


def normalized_gram_jvp(
    weight: ArrayLike,
    direction: ArrayLike,
    *,
    side: str = "auto",
) -> NormalizedGramJVPRecord:
    """Analytic Jacobian-vector product of the normalized Gram map."""

    matrix = _matrix(weight)
    tangent = _matrix(direction, name="direction")
    if tangent.shape != matrix.shape:
        raise ValueError("weight and direction must have identical shapes")
    normalized = normalized_gram_map(matrix, side=side)
    selected, gram = _side_and_gram(matrix, normalized.side)
    frobenius_sq = normalized.frobenius_norm_sq
    radial_inner = float(np.sum(matrix * tangent))
    if selected == "right":
        differential_gram = tangent.T @ matrix + matrix.T @ tangent
    else:
        differential_gram = tangent @ matrix.T + matrix @ tangent.T
    dimension = normalized.normalization_dimension
    differential = float(dimension) * (
        differential_gram / frobenius_sq
        - gram * (2.0 * radial_inner / frobenius_sq**2)
    )
    differential = 0.5 * (differential + differential.T)

    # This independently evaluates the exact scale-null direction D F[W]=0.
    if selected == "right":
        scale_dg = 2.0 * gram
    else:
        scale_dg = 2.0 * gram
    scale_jvp = float(dimension) * (
        scale_dg / frobenius_sq
        - gram * (2.0 * frobenius_sq / frobenius_sq**2)
    )
    return NormalizedGramJVPRecord(
        value=normalized.value,
        jvp=differential,
        side=selected,
        normalization_dimension=dimension,
        scale_direction_residual=float(np.linalg.norm(scale_jvp, ord="fro")),
        trace_derivative_residual=float(abs(np.trace(differential))),
        operator_kind="analytic_normalized_gram_map_jvp",
        map_definition=(
            "D[d G/||W||_F^2]_W[Z], an algebraic map JVP with exact scale "
            "null direction; not the neural-network training-map Jacobian"
        ),
    )


def normalized_gram_analytic_spectrum(
    weight: ArrayLike,
    *,
    side: str = "auto",
    rcond: float | None = None,
    precomputed_singular_values: ArrayLike | None = None,
) -> NormalizedGramSpectrumRecord:
    """Exact nonzero singular amplitudes of the normalized-Gram derivative.

    For full rank on the smaller Gram side, off-diagonal amplitudes are

    ``(d/S) sqrt(2 (s_i^2+s_j^2))``.

    The remaining diagonal amplitudes are the nonzero singular values of

    ``2(d/S)[diag(s)-outer(s^2,s)/S]``.
    """

    matrix = _matrix(weight)
    selected, gram = _side_and_gram(matrix, side)
    singular_values, _ = _spectrum_singular_values(
        matrix,
        precomputed_singular_values=precomputed_singular_values,
        rcond=rcond,
        map_name="analytic normalized-Gram spectrum",
    )
    dimension = int(gram.shape[0])
    if singular_values.size != dimension:
        raise RuntimeError("selected Gram side is not the smaller matrix side")
    frobenius_sq = float(np.sum(singular_values**2))
    coefficient = float(dimension) / frobenius_sq
    first, second = np.triu_indices(dimension, k=1)
    pair_amplitudes = coefficient * np.sqrt(
        2.0 * (singular_values[first] ** 2 + singular_values[second] ** 2)
    )
    diagonal_operator = 2.0 * coefficient * (
        np.diag(singular_values)
        - np.outer(singular_values**2, singular_values) / frobenius_sq
    )
    diagonal_amplitudes = np.linalg.svd(
        diagonal_operator,
        compute_uv=False,
    )
    diagonal_tolerance = (
        max(diagonal_operator.shape)
        * np.finfo(np.float64).eps
        * max(float(diagonal_amplitudes[0]), 1.0)
    )
    diagonal_amplitudes = diagonal_amplitudes[
        diagonal_amplitudes > diagonal_tolerance
    ]
    amplitude_array = np.sort(
        np.concatenate([pair_amplitudes, diagonal_amplitudes])
    )[::-1]
    expected_rank = dimension * (dimension + 1) // 2 - 1
    if int(amplitude_array.size) != expected_rank:
        raise RuntimeError("normalized-Gram analytic rank audit failed")
    input_dimension = int(matrix.size)
    return NormalizedGramSpectrumRecord(
        singular_amplitudes=amplitude_array,
        jt_j_nonzero_eigenvalues=amplitude_array**2,
        zero_count=input_dimension - expected_rank,
        input_dimension=input_dimension,
        derivative_rank=expected_rank,
        side=selected,
        matrix_singular_values=singular_values,
        operator_kind="normalized_gram_derivative_analytic_pullback_spectrum",
        map_definition=(
            "exact nonzero spectrum of (D F_W)^*D F_W for "
            "F(W)=dG/||W||_F^2; zeros include scale, rotations, and Gram-side "
            "complements; this is not a training Jacobian"
        ),
    )


def centered_log_gram_map(
    weight: ArrayLike,
    *,
    side: str = "auto",
    rcond: float | None = None,
) -> CenteredLogGramRecord:
    """Single-checkpoint trace-log quotient of the normalized smaller Gram.

    The map is ``L(W)=log(F(W))-tr(log(F(W))) I/d`` with
    ``F(W)=d G(W)/||W||_F^2``.  Centering removes the uniform scale mode
    exactly; unlike a list of log singular values, the matrix logarithm also
    retains the response of the Gram eigenvectors.
    """

    matrix = _matrix(weight)
    _full_rank_svd(
        matrix,
        rcond=rcond,
        map_name="centered log-Gram map",
    )
    normalized = normalized_gram_map(matrix, side=side)
    eigenvalues, eigenvectors = np.linalg.eigh(normalized.value)
    if np.any(eigenvalues <= 0.0):
        raise np.linalg.LinAlgError(
            "centered log-Gram map requires a positive-definite smaller Gram"
        )
    log_eigenvalues = np.log(eigenvalues)
    centered = log_eigenvalues - float(np.mean(log_eigenvalues))
    value = (eigenvectors * centered[None, :]) @ eigenvectors.T
    value = 0.5 * (value + value.T)
    return CenteredLogGramRecord(
        value=value,
        normalized_gram=normalized.value,
        normalized_gram_eigenvalues=eigenvalues[::-1],
        side=normalized.side,
        normalization_dimension=normalized.normalization_dimension,
        operator_kind="single_checkpoint_centered_log_normalized_gram_map",
        map_definition=(
            "L(W)=log(d G(W)/||W||_F^2)-tr(log(d G(W)/||W||_F^2))I/d "
            "on the smaller Gram side; a scale-free candidate RG map "
            "defined by one checkpoint"
        ),
    )


def centered_log_gram_jvp(
    weight: ArrayLike,
    direction: ArrayLike,
    *,
    side: str = "auto",
    rcond: float | None = None,
) -> CenteredLogGramJVPRecord:
    """Analytic Frechet derivative of :func:`centered_log_gram_map`."""

    matrix = _matrix(weight)
    tangent = _matrix(direction, name="direction")
    if tangent.shape != matrix.shape:
        raise ValueError("weight and direction must have identical shapes")
    mapped = centered_log_gram_map(matrix, side=side, rcond=rcond)
    gram_jvp = normalized_gram_jvp(matrix, tangent, side=mapped.side)
    eigenvalues, eigenvectors = np.linalg.eigh(mapped.normalized_gram)
    rotated = eigenvectors.T @ gram_jvp.jvp @ eigenvectors
    divided = np.empty((eigenvalues.size, eigenvalues.size), dtype=np.float64)
    for first in range(eigenvalues.size):
        for second in range(eigenvalues.size):
            divided[first, second] = _log_divided_difference(
                float(eigenvalues[first]),
                float(eigenvalues[second]),
            )
    differential = eigenvectors @ (divided * rotated) @ eigenvectors.T
    differential -= (
        float(np.trace(differential)) / mapped.normalization_dimension
    ) * np.eye(mapped.normalization_dimension)
    differential = 0.5 * (differential + differential.T)

    scale_gram_jvp = normalized_gram_jvp(matrix, matrix, side=mapped.side)
    scale_rotated = eigenvectors.T @ scale_gram_jvp.jvp @ eigenvectors
    scale_differential = eigenvectors @ (divided * scale_rotated) @ eigenvectors.T
    scale_differential -= (
        float(np.trace(scale_differential)) / mapped.normalization_dimension
    ) * np.eye(mapped.normalization_dimension)
    return CenteredLogGramJVPRecord(
        value=mapped.value,
        jvp=differential,
        side=mapped.side,
        normalization_dimension=mapped.normalization_dimension,
        scale_direction_residual=float(np.linalg.norm(scale_differential)),
        trace_derivative_residual=float(abs(np.trace(differential))),
        operator_kind="analytic_centered_log_gram_candidate_rg_jvp",
        map_definition=(
            "D L_W[Z] for L(W)=log(F(W))-tr(log(F(W)))I/d and "
            "F(W)=dG(W)/||W||_F^2; an actual Jacobian-vector product of "
            "the declared single-checkpoint candidate RG map"
        ),
    )


def centered_log_gram_analytic_spectrum(
    weight: ArrayLike,
    *,
    side: str = "auto",
    rcond: float | None = None,
    precomputed_singular_values: ArrayLike | None = None,
) -> SingleCheckpointJacobianSpectrumRecord:
    """Exact nonzero singular spectrum of the centered log-Gram Jacobian."""

    matrix = _matrix(weight)
    singular_values, _ = _spectrum_singular_values(
        matrix,
        precomputed_singular_values=precomputed_singular_values,
        rcond=rcond,
        map_name="centered log-Gram Jacobian spectrum",
    )
    selected, gram = _side_and_gram(matrix, side)
    dimension = int(gram.shape[0])
    squared = singular_values**2
    first, second = np.triu_indices(dimension, k=1)
    difference = squared[first] - squared[second]
    scale = np.maximum.reduce(
        [
            np.abs(squared[first]),
            np.abs(squared[second]),
            np.ones_like(difference),
        ]
    )
    repeated = np.abs(difference) <= (
        32.0 * np.finfo(np.float64).eps * scale
    )
    divided = np.empty_like(difference)
    divided[repeated] = 2.0 / (
        squared[first[repeated]] + squared[second[repeated]]
    )
    divided[~repeated] = (
        np.log(squared[first[~repeated]])
        - np.log(squared[second[~repeated]])
    ) / difference[~repeated]
    pair_amplitudes = (
        np.sqrt(2.0)
        * np.abs(divided)
        * np.sqrt(squared[first] + squared[second])
    )
    centering = np.eye(dimension) - np.ones((dimension, dimension)) / dimension
    diagonal_operator = 2.0 * centering @ np.diag(1.0 / singular_values)
    diagonal_amplitudes = _positive_singular_amplitudes(diagonal_operator)
    amplitude_array = np.sort(
        np.concatenate([pair_amplitudes, diagonal_amplitudes])
    )[::-1]
    expected_rank = dimension * (dimension - 1) // 2 + dimension - 1
    if amplitude_array.size != expected_rank:
        raise RuntimeError("centered log-Gram analytic rank audit failed")
    input_dimension = int(matrix.size)
    return SingleCheckpointJacobianSpectrumRecord(
        singular_amplitudes=amplitude_array,
        jt_j_nonzero_eigenvalues=amplitude_array**2,
        zero_count=input_dimension - expected_rank,
        input_dimension=input_dimension,
        derivative_rank=expected_rank,
        side=selected,
        matrix_singular_values=singular_values,
        operator_kind="centered_log_gram_candidate_rg_jacobian_exact_spectrum",
        map_definition=(
            "exact nonzero spectrum of (D L_W)^*D L_W for the declared "
            "single-checkpoint map L(W)=log(dG/||W||_F^2)-tr(log(dG/||W||_F^2))I/d; "
            "includes diagonal radial and off-diagonal Gram-eigenvector modes"
        ),
    )


def centered_log_singular_map(
    weight: ArrayLike,
    *,
    rcond: float | None = None,
) -> CenteredLogSingularRecord:
    """Return centered log singular values from one checkpoint."""

    matrix = _matrix(weight)
    _, singular_values, _, _ = _full_rank_svd(
        matrix,
        rcond=rcond,
        map_name="centered log-singular map",
    )
    logged = np.log(singular_values)
    mean_log = float(np.mean(logged))
    return CenteredLogSingularRecord(
        value=logged - mean_log,
        singular_values=singular_values,
        mean_log_singular_value=mean_log,
        operator_kind="single_checkpoint_centered_log_singular_radial_map",
        map_definition=(
            "R(W)=log(s(W))-mean(log(s(W))) on the full smaller-side singular "
            "support; a scale-free radial candidate RG coordinate map"
        ),
    )


def centered_log_singular_jvp(
    weight: ArrayLike,
    direction: ArrayLike,
    *,
    rcond: float | None = None,
) -> CenteredLogSingularJVPRecord:
    """Jacobian-vector product of the centered log-singular radial map."""

    matrix = _matrix(weight)
    tangent = _matrix(direction, name="direction")
    if tangent.shape != matrix.shape:
        raise ValueError("weight and direction must have identical shapes")
    left, singular_values, right_h, tolerance = _full_rank_svd(
        matrix,
        rcond=rcond,
        map_name="centered log-singular JVP",
    )
    _require_simple_singular_values(
        singular_values,
        tolerance=tolerance,
        map_name="centered log-singular JVP",
    )
    raw = np.diag(left.T @ tangent @ right_h.T) / singular_values
    differential = raw - float(np.mean(raw))
    mapped = centered_log_singular_map(matrix, rcond=rcond)
    scale_raw = np.diag(left.T @ matrix @ right_h.T) / singular_values
    scale_differential = scale_raw - float(np.mean(scale_raw))
    return CenteredLogSingularJVPRecord(
        value=mapped.value,
        jvp=differential,
        scale_direction_residual=float(np.linalg.norm(scale_differential)),
        sum_derivative_residual=float(abs(np.sum(differential))),
        operator_kind="analytic_centered_log_singular_candidate_rg_jvp",
        map_definition=(
            "D R_W[Z] for R(W)=log(s(W))-mean(log(s(W))); an actual "
            "Jacobian-vector product of the single-checkpoint radial map"
        ),
    )


def centered_log_singular_analytic_spectrum(
    weight: ArrayLike,
    *,
    rcond: float | None = None,
    precomputed_singular_values: ArrayLike | None = None,
) -> SingleCheckpointJacobianSpectrumRecord:
    """Exact nonzero spectrum of the centered log-singular Jacobian."""

    matrix = _matrix(weight)
    singular_values, tolerance = _spectrum_singular_values(
        matrix,
        precomputed_singular_values=precomputed_singular_values,
        rcond=rcond,
        map_name="centered log-singular Jacobian spectrum",
    )
    _require_simple_singular_values(
        singular_values,
        tolerance=tolerance,
        map_name="centered log-singular Jacobian spectrum",
    )
    rank = int(singular_values.size)
    centering = np.eye(rank) - np.ones((rank, rank)) / rank
    radial_operator = centering @ np.diag(1.0 / singular_values)
    amplitudes = _positive_singular_amplitudes(radial_operator)
    expected_rank = rank - 1
    if amplitudes.size != expected_rank:
        raise RuntimeError("centered log-singular analytic rank audit failed")
    input_dimension = int(matrix.size)
    return SingleCheckpointJacobianSpectrumRecord(
        singular_amplitudes=amplitudes,
        jt_j_nonzero_eigenvalues=amplitudes**2,
        zero_count=input_dimension - expected_rank,
        input_dimension=input_dimension,
        derivative_rank=expected_rank,
        side="thin_svd_radial_coordinates",
        matrix_singular_values=singular_values,
        operator_kind="centered_log_singular_candidate_rg_jacobian_exact_spectrum",
        map_definition=(
            "exact nonzero spectrum of (D R_W)^*D R_W for the declared "
            "single-checkpoint radial map R(W)=log(s)-mean(log(s)); uniform "
            "scale and all singular-vector directions are exact null modes"
        ),
    )


def centered_log_singular_flow(
    weight: ArrayLike,
    captured_update: ArrayLike,
    *,
    delta_s: float = 1.0,
    rcond: float | None = None,
) -> CenteredLogSingularFlowRecord:
    """Essential instantaneous log-singular flow supplied by an update ``Z``.

    ``dot(log s_i)=u_i.T (Z/delta_s) v_i / s_i``.  Subtracting its mean removes
    the uniform radial/scale direction.  A checkpoint without ``Z`` cannot
    provide this observable.
    """

    matrix = _matrix(weight)
    update = _matrix(captured_update, name="captured_update")
    if update.shape != matrix.shape:
        raise ValueError("weight and captured_update must have identical shapes")
    ds = float(delta_s)
    if not np.isfinite(ds) or ds == 0.0:
        raise ValueError("delta_s must be finite and nonzero")
    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0])
    tolerance = (
        float(rcond) * max(largest, 1.0)
        if rcond is not None
        else max(matrix.shape)
        * np.finfo(np.float64).eps
        * max(largest, 1.0)
    )
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative")
    retained = singular_values > tolerance
    if not np.any(retained):
        raise np.linalg.LinAlgError("weight has no retained singular support")
    u = left[:, retained]
    s = singular_values[retained]
    v = right_h[retained, :].T
    velocity = update / ds
    mode_velocity = np.sum((velocity @ v) * u, axis=0)
    log_rates = mode_velocity / s
    mean_rate = float(np.mean(log_rates))
    centered = log_rates - mean_rate
    return CenteredLogSingularFlowRecord(
        singular_values=s,
        instantaneous_log_singular_rates=log_rates,
        centered_log_singular_rates=centered,
        mean_log_singular_rate=mean_rate,
        retained_rank=int(s.size),
        dropped_singular_directions=int(singular_values.size - s.size),
        delta_s=ds,
        operator_kind="essential_centered_log_singular_flow_with_captured_update",
        map_definition=(
            "u_i^T (Z/delta_s) v_i / s_i minus its retained-mode mean; "
            "requires captured Z and is not available from W alone"
        ),
    )


def calibrated_training_map_contract(
    map_definition: str,
) -> CalibratedTrainingMapContract:
    definition = str(map_definition).strip()
    if not definition:
        raise ValueError("map_definition must explicitly define the training map")
    return CalibratedTrainingMapContract(
        map_definition=definition,
        requires_batch=True,
        requires_optimizer_state=True,
        weight_only=False,
        operator_kind="calibrated_training_map_local_derivative_contract",
    )


def _require_calibration(
    *,
    batch: Any,
    optimizer_state: Any,
    map_definition: str,
) -> CalibratedTrainingMapContract:
    if batch is None:
        raise ValueError("a concrete calibration batch is required")
    if optimizer_state is None:
        raise ValueError("the complete optimizer state is required")
    return calibrated_training_map_contract(map_definition)


def calibrated_training_map_finite_difference(
    map_fn: Callable[[FloatArray, Any, Any], ArrayLike],
    weight: ArrayLike,
    direction: ArrayLike,
    *,
    batch: Any,
    optimizer_state: Any,
    map_definition: str,
    epsilon: float | None = None,
) -> CalibratedTrainingMapDerivativeRecord:
    """Central-difference JVP of an explicitly calibrated training map.

    ``map_fn`` must be a pure callable with signature
    ``map_fn(weight, batch, optimizer_state)``.  Deep copies of batch and state
    are supplied to each evaluation to prevent optimizer mutation from
    contaminating the central difference.
    """

    contract = _require_calibration(
        batch=batch,
        optimizer_state=optimizer_state,
        map_definition=map_definition,
    )
    matrix = _matrix(weight)
    tangent = _matrix(direction, name="direction")
    if tangent.shape != matrix.shape:
        raise ValueError("weight and direction must have identical shapes")
    direction_norm = float(np.linalg.norm(tangent, ord="fro"))
    if direction_norm == 0.0:
        raise ValueError("direction must be nonzero")
    if epsilon is None:
        epsilon = float(
            np.cbrt(np.finfo(np.float64).eps)
            * max(np.linalg.norm(matrix, ord="fro"), 1.0)
            / direction_norm
        )
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    def evaluate(value: FloatArray) -> FloatArray:
        result = map_fn(
            value,
            copy.deepcopy(batch),
            copy.deepcopy(optimizer_state),
        )
        array = np.asarray(result, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("calibrated training map returned non-finite values")
        return array

    base = evaluate(matrix.copy())
    plus = evaluate(matrix + float(epsilon) * tangent)
    minus = evaluate(matrix - float(epsilon) * tangent)
    if plus.shape != base.shape or minus.shape != base.shape:
        raise ValueError("calibrated training map output shape changed")
    derivative = (plus - minus) / (2.0 * float(epsilon))
    return CalibratedTrainingMapDerivativeRecord(
        base_output=base,
        directional_derivative=derivative,
        epsilon=float(epsilon),
        backend="numpy_central_difference",
        batch_supplied=True,
        optimizer_state_supplied=True,
        weight_only=False,
        operator_kind="calibrated_training_map_finite_difference_jvp_not_w_only",
        map_definition=contract.map_definition,
    )


def calibrated_training_map_jvp(
    map_fn: Callable[..., Any],
    weight: Any,
    direction: Any,
    *,
    batch: Any,
    optimizer_state: Any,
    map_definition: str,
    backend: str = "finite_difference",
    epsilon: float | None = None,
) -> CalibratedTrainingMapDerivativeRecord:
    """Evaluate a calibrated local-map JVP with NumPy FD or lazy PyTorch AD."""

    selected = str(backend).lower()
    if selected in {"finite_difference", "numpy", "fd"}:
        return calibrated_training_map_finite_difference(
            map_fn,
            weight,
            direction,
            batch=batch,
            optimizer_state=optimizer_state,
            map_definition=map_definition,
            epsilon=epsilon,
        )
    if selected not in {"torch", "autograd", "torch_autograd"}:
        raise ValueError("backend must be finite_difference or torch")

    contract = _require_calibration(
        batch=batch,
        optimizer_state=optimizer_state,
        map_definition=map_definition,
    )
    # Torch is intentionally imported only inside this optional calibrated path.
    import torch

    if not torch.is_tensor(weight) or not torch.is_tensor(direction):
        raise TypeError("torch backend requires tensor weight and direction")
    if weight.shape != direction.shape:
        raise ValueError("weight and direction must have identical shapes")

    def wrapped(candidate):
        return map_fn(candidate, batch, optimizer_state)

    base, derivative = torch.autograd.functional.jvp(
        wrapped,
        (weight,),
        (direction,),
        create_graph=False,
        strict=True,
    )
    return CalibratedTrainingMapDerivativeRecord(
        base_output=base.detach().cpu().numpy(),
        directional_derivative=derivative.detach().cpu().numpy(),
        epsilon=None,
        backend="torch_autograd_jvp",
        batch_supplied=True,
        optimizer_state_supplied=True,
        weight_only=False,
        operator_kind="calibrated_training_map_autograd_jvp_not_w_only",
        map_definition=contract.map_definition,
    )


# Short aliases used in exploratory notebooks.
normalized_gram_spectrum = normalized_gram_analytic_spectrum
essential_centered_log_singular_flow = centered_log_singular_flow


__all__ = [
    "CalibratedTrainingMapContract",
    "CalibratedTrainingMapDerivativeRecord",
    "CenteredLogGramJVPRecord",
    "CenteredLogGramRecord",
    "CenteredLogSingularJVPRecord",
    "CenteredLogSingularRecord",
    "CenteredLogSingularFlowRecord",
    "GramTranslationRecord",
    "NormalizedGramJVPRecord",
    "NormalizedGramRecord",
    "NormalizedGramSpectrumRecord",
    "SingleCheckpointJacobianSpectrumRecord",
    "calibrated_training_map_contract",
    "calibrated_training_map_finite_difference",
    "calibrated_training_map_jvp",
    "centered_log_singular_flow",
    "centered_log_gram_analytic_spectrum",
    "centered_log_gram_jvp",
    "centered_log_gram_map",
    "centered_log_singular_analytic_spectrum",
    "centered_log_singular_jvp",
    "centered_log_singular_map",
    "essential_centered_log_singular_flow",
    "gram_translation_quotient",
    "normalized_gram_analytic_spectrum",
    "normalized_gram_jvp",
    "normalized_gram_map",
    "normalized_gram_spectrum",
]

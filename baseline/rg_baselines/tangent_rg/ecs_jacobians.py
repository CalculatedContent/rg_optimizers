"""Exact single-checkpoint ECS Jacobians requested by the RG test suite.

Every spectrum in this module is the nonzero singular spectrum of the
derivative of a declared map.  Squaring those amplitudes gives the nonzero
eigenvalues of ``J* J`` in the same energy convention as a WeightWatcher ESD.
The checkpoint-dependent maps freeze their SVD frame at the named checkpoint;
that anchoring is part of the map definition and is never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ECSJacobianMapRecord:
    value: FloatArray
    operator_kind: str
    map_definition: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ECSJacobianJVPRecord:
    value: FloatArray
    jvp: FloatArray
    operator_kind: str
    map_definition: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ECSJacobianSpectrumRecord:
    singular_amplitudes: FloatArray
    jt_j_nonzero_eigenvalues: FloatArray
    zero_count: int
    input_dimension: int
    output_dimension: int
    derivative_rank: int
    retained_rank: int | None
    outer_rank: int | None
    operator_kind: str
    map_definition: str
    parameters: Mapping[str, Any]


def _matrix(value: ArrayLike, *, name: str = "weight") -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _wide_matrix(value: ArrayLike, *, name: str = "weight") -> FloatArray:
    result = _matrix(value, name=name)
    if result.shape[0] >= result.shape[1]:
        raise ValueError("the ECS row-space Jacobians require a wide matrix")
    return result


def _relative_tolerance(scale: float, dimension: int, rcond: float | None) -> float:
    if rcond is not None and (not np.isfinite(rcond) or rcond < 0.0):
        raise ValueError("rcond must be finite and non-negative")
    return (
        float(rcond) * float(scale)
        if rcond is not None
        else float(dimension) * np.finfo(np.float64).eps * float(scale)
    )


def _frames(
    weight: FloatArray,
    *,
    rcond: float | None,
) -> tuple[FloatArray, FloatArray, FloatArray, int, float]:
    left, singular_values, right_h = np.linalg.svd(weight, full_matrices=True)
    tolerance = _relative_tolerance(
        float(singular_values[0]), max(weight.shape), rcond
    )
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    return left, singular_values, right_h.T, numerical_rank, tolerance


def _ranks(
    retained_rank: int,
    outer_rank: int,
    *,
    maximum_rank: int,
) -> tuple[int, int]:
    k = int(retained_rank)
    q = int(outer_rank)
    if float(retained_rank) != float(k) or float(outer_rank) != float(q):
        raise ValueError("retained_rank and outer_rank must be integers")
    if not (1 <= k < q <= int(maximum_rank)):
        raise ValueError(
            "ranks must satisfy 1 <= retained_rank < outer_rank "
            "<= checkpoint numerical rank"
        )
    return k, q


def _positive_amplitudes(values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    result = result[np.isfinite(result) & (result > 0.0)]
    if result.size == 0:
        return result
    tolerance = (
        max(1, result.size)
        * np.finfo(np.float64).eps
        * float(np.max(result))
    )
    return np.sort(result[result > tolerance])[::-1]


def _trace_free(value: FloatArray) -> FloatArray:
    dimension = int(value.shape[0])
    return value - (float(np.trace(value)) / dimension) * np.eye(dimension)


def _symmetric_log(value: FloatArray) -> FloatArray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (value + value.T))
    if np.any(eigenvalues <= 0.0):
        raise np.linalg.LinAlgError("matrix logarithm requires positive eigenvalues")
    return (eigenvectors * np.log(eigenvalues)[None, :]) @ eigenvectors.T


def _log_divided_difference(first: float, second: float) -> float:
    scale = max(abs(first), abs(second), np.finfo(np.float64).tiny)
    if abs(first - second) <= 64.0 * np.finfo(np.float64).eps * scale:
        return 2.0 / (first + second)
    return float((np.log(first) - np.log(second)) / (first - second))


def _frechet_log(spd: FloatArray, direction: FloatArray) -> FloatArray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (spd + spd.T))
    if np.any(eigenvalues <= 0.0):
        raise np.linalg.LinAlgError("log Frechet derivative requires SPD base")
    rotated = eigenvectors.T @ (0.5 * (direction + direction.T)) @ eigenvectors
    coefficients = np.empty((eigenvalues.size, eigenvalues.size), dtype=np.float64)
    for first in range(eigenvalues.size):
        for second in range(eigenvalues.size):
            coefficients[first, second] = _log_divided_difference(
                float(eigenvalues[first]), float(eigenvalues[second])
            )
    return eigenvectors @ (coefficients * rotated) @ eigenvectors.T


def _outer_basis(
    anchor: FloatArray,
    outer_rank: int,
    *,
    rcond: float | None,
) -> tuple[FloatArray, FloatArray, FloatArray, int, float]:
    left, singular_values, right, numerical_rank, tolerance = _frames(
        anchor, rcond=rcond
    )
    q = int(outer_rank)
    if not 1 <= q <= numerical_rank:
        raise ValueError("outer_rank must lie in the checkpoint numerical row rank")
    return left, singular_values, right[:, :q], numerical_rank, tolerance


def _outer_gram(candidate: FloatArray, outer_basis: FloatArray) -> FloatArray:
    projected = candidate @ outer_basis
    gram = projected.T @ projected
    return 0.5 * (gram + gram.T)


def _outer_dgram(
    anchor: FloatArray,
    direction: FloatArray,
    outer_basis: FloatArray,
) -> FloatArray:
    base = anchor @ outer_basis
    tangent = direction @ outer_basis
    result = base.T @ tangent + tangent.T @ base
    return 0.5 * (result + result.T)


def _trace_free_spectral_amplitudes(
    singular_values: FloatArray,
    pair_coefficients: FloatArray,
    diagonal_coefficients: FloatArray,
) -> FloatArray:
    dimension = int(singular_values.size)
    first, second = np.triu_indices(dimension, k=1)
    pair = np.sqrt(
        2.0
        * (singular_values[first] ** 2 + singular_values[second] ** 2)
    ) * np.abs(pair_coefficients)
    centering = np.eye(dimension) - np.ones((dimension, dimension)) / dimension
    diagonal_map = centering @ np.diag(
        2.0 * singular_values * diagonal_coefficients
    )
    diagonal = np.linalg.svd(diagonal_map, compute_uv=False)
    return _positive_amplitudes(np.concatenate([pair, diagonal]))


def gap_aware_projector_map(
    anchor_weight: ArrayLike,
    candidate_weight: ArrayLike,
    *,
    retained_rank: int,
    outer_rank: int,
    rcond: float | None = None,
) -> ECSJacobianMapRecord:
    """Evaluate the anchored core/shell cross block of the hard projector."""

    anchor = _wide_matrix(anchor_weight, name="anchor_weight")
    candidate = _wide_matrix(candidate_weight, name="candidate_weight")
    if candidate.shape != anchor.shape:
        raise ValueError("anchor_weight and candidate_weight must have equal shape")
    _, singular_values, right, numerical_rank, tolerance = _frames(
        anchor, rcond=rcond
    )
    k, q = _ranks(retained_rank, outer_rank, maximum_rank=numerical_rank)
    squared = singular_values**2
    gap_tolerance = _relative_tolerance(
        float(squared[0]), max(anchor.shape), rcond
    )
    if float(squared[k - 1] - squared[k]) <= gap_tolerance:
        raise np.linalg.LinAlgError("top-k row projector has no differentiable gap")
    candidate_right = np.linalg.svd(candidate, full_matrices=True)[2].T
    candidate_projector = candidate_right[:, :k] @ candidate_right[:, :k].T
    retained = right[:, :k]
    shell = right[:, k:q]
    cross = shell.T @ candidate_projector @ retained
    value = shell @ cross @ retained.T + retained @ cross.T @ shell.T
    return ECSJacobianMapRecord(
        value=value,
        operator_kind="anchored_gap_aware_grassmann_projector_cross_block_map",
        map_definition=(
            "F_W(X)=V_c[V_c^T P_k(X)V_k]V_k^T + transpose, with the "
            "checkpoint V_k,V_c frozen and P_k(X) the exact top-k row projector"
        ),
        parameters={
            "retained_rank": k,
            "outer_rank": q,
            "boundary_gap": float(squared[k - 1] - squared[k]),
            "rank_tolerance": float(tolerance),
        },
    )


def gap_aware_projector_jvp(
    weight: ArrayLike,
    direction: ArrayLike,
    *,
    retained_rank: int,
    outer_rank: int,
    rcond: float | None = None,
) -> ECSJacobianJVPRecord:
    matrix = _wide_matrix(weight)
    tangent = _matrix(direction, name="direction")
    if tangent.shape != matrix.shape:
        raise ValueError("weight and direction must have equal shape")
    _, singular_values, right, numerical_rank, _ = _frames(matrix, rcond=rcond)
    k, q = _ranks(retained_rank, outer_rank, maximum_rank=numerical_rank)
    squared = singular_values**2
    gaps = squared[:k][None, :] - squared[k:q][:, None]
    gap_tolerance = _relative_tolerance(
        float(squared[0]), max(matrix.shape), rcond
    )
    if np.any(gaps <= gap_tolerance):
        raise np.linalg.LinAlgError("retained/shell projector gaps are singular")
    dgram = matrix.T @ tangent + tangent.T @ matrix
    retained = right[:, :k]
    shell = right[:, k:q]
    coordinate = (shell.T @ dgram @ retained) / gaps
    jvp = shell @ coordinate @ retained.T + retained @ coordinate.T @ shell.T
    mapped = gap_aware_projector_map(
        matrix,
        matrix,
        retained_rank=k,
        outer_rank=q,
        rcond=rcond,
    )
    return ECSJacobianJVPRecord(
        value=mapped.value,
        jvp=jvp,
        operator_kind="gap_aware_grassmann_projector_jacobian_jvp",
        map_definition=(
            "D F_W(W)[E] with K_ai=(v_a^T(W^TE+E^TW)v_i)/"
            "(sigma_i^2-sigma_a^2) and output V_c K V_k^T+transpose"
        ),
        parameters={"retained_rank": k, "outer_rank": q},
    )


def gap_aware_projector_spectrum(
    weight: ArrayLike,
    *,
    retained_rank: int,
    outer_rank: int,
    rcond: float | None = None,
    precomputed_singular_values: ArrayLike | None = None,
) -> ECSJacobianSpectrumRecord:
    matrix = _wide_matrix(weight)
    if precomputed_singular_values is None:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    else:
        singular_values = np.asarray(precomputed_singular_values, dtype=np.float64)
        if singular_values.shape != (matrix.shape[0],):
            raise ValueError("precomputed singular-value shape mismatch")
    tolerance = _relative_tolerance(
        float(singular_values[0]), max(matrix.shape), rcond
    )
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    k, q = _ranks(retained_rank, outer_rank, maximum_rank=numerical_rank)
    squared = singular_values**2
    gaps = squared[:k][None, :] - squared[k:q][:, None]
    if np.any(gaps <= _relative_tolerance(float(squared[0]), max(matrix.shape), rcond)):
        raise np.linalg.LinAlgError("retained/shell projector gaps are singular")
    amplitudes = (
        np.sqrt(2.0)
        * np.sqrt(squared[:k][None, :] + squared[k:q][:, None])
        / gaps
    ).reshape(-1)
    amplitudes = np.sort(amplitudes)[::-1]
    rank = int(k * (q - k))
    return ECSJacobianSpectrumRecord(
        singular_amplitudes=amplitudes,
        jt_j_nonzero_eigenvalues=amplitudes**2,
        zero_count=int(matrix.size - rank),
        input_dimension=int(matrix.size),
        output_dimension=int(matrix.shape[1] ** 2),
        derivative_rank=rank,
        retained_rank=k,
        outer_rank=q,
        operator_kind="gap_aware_grassmann_projector_jacobian_exact_spectrum",
        map_definition=(
            "exact spectrum of the anchored restricted DP_k cross block; "
            "j_ia=sqrt(2(sigma_i^2+sigma_a^2))/|sigma_i^2-sigma_a^2|"
        ),
        parameters={
            "minimum_core_shell_gap": float(np.min(gaps)),
            "metric": "full reconstructed projector Frobenius norm",
        },
    )


def _logistic(values: FloatArray, center: float, temperature: float) -> FloatArray:
    scaled = (values - float(center)) / float(temperature)
    result = np.empty_like(scaled)
    positive = scaled >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-scaled[positive]))
    exponential = np.exp(scaled[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _logistic_derivative(value: float, center: float, temperature: float) -> float:
    probability = float(_logistic(np.asarray([value]), center, temperature)[0])
    return probability * (1.0 - probability) / float(temperature)


def _soft_divided_difference(
    first: float,
    second: float,
    *,
    center: float,
    temperature: float,
) -> float:
    scale = max(abs(first), abs(second), abs(center), np.finfo(np.float64).tiny)
    if abs(first - second) <= 64.0 * np.finfo(np.float64).eps * scale:
        return _logistic_derivative(
            0.5 * (first + second), center, temperature
        )
    values = _logistic(np.asarray([first, second]), center, temperature)
    return float((values[0] - values[1]) / (first - second))


def soft_ecs_projector_map(
    weight: ArrayLike,
    *,
    lambda_center: float,
    temperature: float,
) -> ECSJacobianMapRecord:
    matrix = _wide_matrix(weight)
    if not np.isfinite(lambda_center) or lambda_center < 0.0:
        raise ValueError("lambda_center must be finite and non-negative")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    gram = matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    filtered = _logistic(eigenvalues, lambda_center, temperature)
    value = (eigenvectors * filtered[None, :]) @ eigenvectors.T
    return ECSJacobianMapRecord(
        value=0.5 * (value + value.T),
        operator_kind="soft_ecs_logistic_projector_map",
        map_definition="P_tau(W)=f_tau(W^TW), f_tau(lambda)=sigmoid((lambda-lambda_c)/tau)",
        parameters={
            "lambda_center": float(lambda_center),
            "temperature": float(temperature),
        },
    )


def soft_ecs_projector_jvp(
    weight: ArrayLike,
    direction: ArrayLike,
    *,
    lambda_center: float,
    temperature: float,
) -> ECSJacobianJVPRecord:
    matrix = _wide_matrix(weight)
    tangent = _matrix(direction, name="direction")
    if tangent.shape != matrix.shape:
        raise ValueError("weight and direction must have equal shape")
    mapped = soft_ecs_projector_map(
        matrix, lambda_center=lambda_center, temperature=temperature
    )
    gram = matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    rotated = eigenvectors.T @ (matrix.T @ tangent + tangent.T @ matrix) @ eigenvectors
    coefficients = np.empty((eigenvalues.size, eigenvalues.size), dtype=np.float64)
    for first in range(eigenvalues.size):
        for second in range(eigenvalues.size):
            coefficients[first, second] = _soft_divided_difference(
                float(eigenvalues[first]),
                float(eigenvalues[second]),
                center=lambda_center,
                temperature=temperature,
            )
    jvp = eigenvectors @ (coefficients * rotated) @ eigenvectors.T
    return ECSJacobianJVPRecord(
        value=mapped.value,
        jvp=0.5 * (jvp + jvp.T),
        operator_kind="soft_ecs_logistic_projector_jacobian_jvp",
        map_definition=(
            "D f_tau(G)[dG] with the exact first divided difference of the "
            "logistic spectral filter"
        ),
        parameters=mapped.parameters,
    )


def soft_ecs_projector_spectrum(
    weight: ArrayLike,
    *,
    lambda_center: float,
    temperature: float,
    rcond: float | None = None,
    precomputed_singular_values: ArrayLike | None = None,
) -> ECSJacobianSpectrumRecord:
    matrix = _wide_matrix(weight)
    singular_values = (
        np.linalg.svd(matrix, compute_uv=False)
        if precomputed_singular_values is None
        else np.asarray(precomputed_singular_values, dtype=np.float64)
    )
    if singular_values.shape != (matrix.shape[0],):
        raise ValueError("precomputed singular-value shape mismatch")
    tolerance = _relative_tolerance(
        float(singular_values[0]), max(matrix.shape), rcond
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    active = singular_values[:rank]
    squared = active**2
    diagonal = np.asarray(
        [
            2.0
            * sigma
            * abs(_logistic_derivative(lam, lambda_center, temperature))
            for sigma, lam in zip(active, squared)
        ]
    )
    first, second = np.triu_indices(rank, k=1)
    pair_coefficients = np.asarray(
        [
            abs(
                _soft_divided_difference(
                    float(squared[i]),
                    float(squared[j]),
                    center=lambda_center,
                    temperature=temperature,
                )
            )
            for i, j in zip(first, second)
        ]
    )
    pairs = np.sqrt(2.0 * (squared[first] + squared[second])) * pair_coefficients
    null_count = int(matrix.shape[1] - rank)
    active_null = np.asarray([], dtype=np.float64)
    if null_count:
        null_coefficients = np.asarray(
            [
                abs(
                    _soft_divided_difference(
                        float(lam),
                        0.0,
                        center=lambda_center,
                        temperature=temperature,
                    )
                )
                for lam in squared
            ]
        )
        active_null = np.repeat(
            np.sqrt(2.0) * active * null_coefficients,
            null_count,
        )
    amplitudes = _positive_amplitudes(
        np.concatenate([diagonal, pairs, active_null])
    )
    derivative_rank = int(amplitudes.size)
    return ECSJacobianSpectrumRecord(
        singular_amplitudes=amplitudes,
        jt_j_nonzero_eigenvalues=amplitudes**2,
        zero_count=int(matrix.size - derivative_rank),
        input_dimension=int(matrix.size),
        output_dimension=int(matrix.shape[1] ** 2),
        derivative_rank=derivative_rank,
        retained_rank=None,
        outer_rank=None,
        operator_kind="soft_ecs_logistic_projector_jacobian_exact_spectrum",
        map_definition=(
            "exact nonzero spectrum of D[f_tau(W^TW)], including active-active "
            "and active-null row-space modes"
        ),
        parameters={
            "lambda_center": float(lambda_center),
            "temperature": float(temperature),
            "checkpoint_numerical_rank": rank,
            "right_null_dimension": null_count,
        },
    )


def outer_trace_free_log_gram_map(
    anchor_weight: ArrayLike,
    candidate_weight: ArrayLike,
    *,
    outer_rank: int,
    rcond: float | None = None,
) -> ECSJacobianMapRecord:
    anchor = _wide_matrix(anchor_weight, name="anchor_weight")
    candidate = _wide_matrix(candidate_weight, name="candidate_weight")
    if candidate.shape != anchor.shape:
        raise ValueError("anchor_weight and candidate_weight must have equal shape")
    _, singular_values, basis, numerical_rank, tolerance = _outer_basis(
        anchor, outer_rank, rcond=rcond
    )
    gram = _outer_gram(candidate, basis)
    value = _trace_free(_symmetric_log(gram))
    return ECSJacobianMapRecord(
        value=value,
        operator_kind="anchored_outer_ecs_trace_free_log_gram_map",
        map_definition=(
            "L_W(X)=Pi_tf log(V_o^T X^T X V_o), with the checkpoint top-q "
            "outer ECS basis V_o frozen"
        ),
        parameters={
            "outer_rank": int(outer_rank),
            "checkpoint_numerical_rank": numerical_rank,
            "minimum_outer_singular_value": float(singular_values[int(outer_rank) - 1]),
            "rank_tolerance": float(tolerance),
        },
    )


def outer_trace_free_log_gram_jvp(
    weight: ArrayLike,
    direction: ArrayLike,
    *,
    outer_rank: int,
    rcond: float | None = None,
) -> ECSJacobianJVPRecord:
    matrix = _wide_matrix(weight)
    tangent = _matrix(direction, name="direction")
    if tangent.shape != matrix.shape:
        raise ValueError("weight and direction must have equal shape")
    _, _, basis, _, _ = _outer_basis(matrix, outer_rank, rcond=rcond)
    gram = _outer_gram(matrix, basis)
    jvp = _trace_free(_frechet_log(gram, _outer_dgram(matrix, tangent, basis)))
    mapped = outer_trace_free_log_gram_map(
        matrix, matrix, outer_rank=outer_rank, rcond=rcond
    )
    return ECSJacobianJVPRecord(
        value=mapped.value,
        jvp=jvp,
        operator_kind="outer_ecs_trace_free_log_gram_jacobian_jvp",
        map_definition=(
            "Pi_tf L_log,G_o[V_o^T(W^TE+E^TW)V_o] in the frozen checkpoint outer ECS"
        ),
        parameters=mapped.parameters,
    )


def outer_trace_free_log_gram_spectrum(
    weight: ArrayLike,
    *,
    outer_rank: int,
    rcond: float | None = None,
    precomputed_singular_values: ArrayLike | None = None,
) -> ECSJacobianSpectrumRecord:
    matrix = _wide_matrix(weight)
    singular_values = (
        np.linalg.svd(matrix, compute_uv=False)
        if precomputed_singular_values is None
        else np.asarray(precomputed_singular_values, dtype=np.float64)
    )
    if singular_values.shape != (matrix.shape[0],):
        raise ValueError("precomputed singular-value shape mismatch")
    numerical_rank = int(
        np.count_nonzero(
            singular_values
            > _relative_tolerance(float(singular_values[0]), max(matrix.shape), rcond)
        )
    )
    q = int(outer_rank)
    if not 2 <= q <= numerical_rank:
        raise ValueError("outer trace-free log spectrum requires 2 <= q <= rank")
    selected = singular_values[:q]
    squared = selected**2
    first, second = np.triu_indices(q, k=1)
    pair_coefficients = np.asarray(
        [
            _log_divided_difference(float(squared[i]), float(squared[j]))
            for i, j in zip(first, second)
        ]
    )
    diagonal_coefficients = 1.0 / squared
    amplitudes = _trace_free_spectral_amplitudes(
        selected, pair_coefficients, diagonal_coefficients
    )
    expected_rank = q * (q - 1) // 2 + q - 1
    if amplitudes.size != expected_rank:
        raise RuntimeError("outer trace-free log-Gram rank audit failed")
    return ECSJacobianSpectrumRecord(
        singular_amplitudes=amplitudes,
        jt_j_nonzero_eigenvalues=amplitudes**2,
        zero_count=int(matrix.size - expected_rank),
        input_dimension=int(matrix.size),
        output_dimension=int(q * q),
        derivative_rank=expected_rank,
        retained_rank=None,
        outer_rank=q,
        operator_kind="outer_ecs_trace_free_log_gram_jacobian_exact_spectrum",
        map_definition=(
            "exact nonzero spectrum of Pi_tf L_log,G_o o D G_o in the "
            "checkpoint-frozen top-q right singular basis"
        ),
        parameters={"outer_rank": q, "scale_direction_is_null": True},
    )


def outer_resolvent_map(
    anchor_weight: ArrayLike,
    candidate_weight: ArrayLike,
    *,
    outer_rank: int,
    z: float,
    trace_free: bool = True,
    rcond: float | None = None,
) -> ECSJacobianMapRecord:
    anchor = _wide_matrix(anchor_weight, name="anchor_weight")
    candidate = _wide_matrix(candidate_weight, name="candidate_weight")
    if candidate.shape != anchor.shape:
        raise ValueError("anchor_weight and candidate_weight must have equal shape")
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("resolvent z must be finite and positive")
    _, _, basis, _, _ = _outer_basis(anchor, outer_rank, rcond=rcond)
    gram = _outer_gram(candidate, basis)
    value = np.linalg.inv(gram + float(z) * np.eye(int(outer_rank)))
    if trace_free:
        value = _trace_free(value)
    return ECSJacobianMapRecord(
        value=0.5 * (value + value.T),
        operator_kind=(
            "anchored_outer_ecs_trace_free_resolvent_map"
            if trace_free
            else "anchored_outer_ecs_resolvent_map"
        ),
        map_definition=(
            "R_W,z(X)=(V_o^T X^T X V_o+zI)^-1 with checkpoint V_o frozen"
            + (" and Pi_tf applied" if trace_free else "")
        ),
        parameters={
            "outer_rank": int(outer_rank),
            "z": float(z),
            "trace_free": bool(trace_free),
        },
    )


def outer_resolvent_jvp(
    weight: ArrayLike,
    direction: ArrayLike,
    *,
    outer_rank: int,
    z: float,
    trace_free: bool = True,
    rcond: float | None = None,
) -> ECSJacobianJVPRecord:
    matrix = _wide_matrix(weight)
    tangent = _matrix(direction, name="direction")
    if tangent.shape != matrix.shape:
        raise ValueError("weight and direction must have equal shape")
    _, _, basis, _, _ = _outer_basis(matrix, outer_rank, rcond=rcond)
    gram = _outer_gram(matrix, basis)
    resolvent = np.linalg.inv(gram + float(z) * np.eye(int(outer_rank)))
    jvp = -resolvent @ _outer_dgram(matrix, tangent, basis) @ resolvent
    if trace_free:
        jvp = _trace_free(jvp)
    mapped = outer_resolvent_map(
        matrix,
        matrix,
        outer_rank=outer_rank,
        z=z,
        trace_free=trace_free,
        rcond=rcond,
    )
    return ECSJacobianJVPRecord(
        value=mapped.value,
        jvp=0.5 * (jvp + jvp.T),
        operator_kind=(
            "outer_ecs_trace_free_resolvent_jacobian_jvp"
            if trace_free
            else "outer_ecs_resolvent_jacobian_jvp"
        ),
        map_definition="-Pi_tf[R_z V_o^T(W^TE+E^TW)V_o R_z]" if trace_free else "-R_z V_o^T(W^TE+E^TW)V_o R_z",
        parameters=mapped.parameters,
    )


def outer_resolvent_spectrum(
    weight: ArrayLike,
    *,
    outer_rank: int,
    z: float,
    trace_free: bool = True,
    rcond: float | None = None,
    precomputed_singular_values: ArrayLike | None = None,
) -> ECSJacobianSpectrumRecord:
    matrix = _wide_matrix(weight)
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("resolvent z must be finite and positive")
    singular_values = (
        np.linalg.svd(matrix, compute_uv=False)
        if precomputed_singular_values is None
        else np.asarray(precomputed_singular_values, dtype=np.float64)
    )
    numerical_rank = int(
        np.count_nonzero(
            singular_values
            > _relative_tolerance(float(singular_values[0]), max(matrix.shape), rcond)
        )
    )
    q = int(outer_rank)
    if not 1 <= q <= numerical_rank:
        raise ValueError("outer_rank must lie in checkpoint numerical rank")
    selected = singular_values[:q]
    inverse = 1.0 / (selected**2 + float(z))
    first, second = np.triu_indices(q, k=1)
    pairs = (
        np.sqrt(2.0 * (selected[first] ** 2 + selected[second] ** 2))
        * inverse[first]
        * inverse[second]
    )
    diagonal_values = 2.0 * selected * inverse**2
    if trace_free:
        centering = np.eye(q) - np.ones((q, q)) / q
        diagonal = np.linalg.svd(
            centering @ np.diag(diagonal_values), compute_uv=False
        )
    else:
        diagonal = diagonal_values
    amplitudes = _positive_amplitudes(np.concatenate([pairs, diagonal]))
    expected_rank = q * (q - 1) // 2 + q - int(trace_free)
    if amplitudes.size != expected_rank:
        raise RuntimeError("outer resolvent rank audit failed")
    return ECSJacobianSpectrumRecord(
        singular_amplitudes=amplitudes,
        jt_j_nonzero_eigenvalues=amplitudes**2,
        zero_count=int(matrix.size - expected_rank),
        input_dimension=int(matrix.size),
        output_dimension=int(q * q),
        derivative_rank=expected_rank,
        retained_rank=None,
        outer_rank=q,
        operator_kind=(
            "outer_ecs_trace_free_resolvent_jacobian_exact_spectrum"
            if trace_free
            else "outer_ecs_resolvent_jacobian_exact_spectrum"
        ),
        map_definition=(
            "exact spectrum of E -> -R_z[V_o^T(W^TE+E^TW)V_o]R_z"
            + (" followed by Pi_tf" if trace_free else "")
        ),
        parameters={"outer_rank": q, "z": float(z), "trace_free": trace_free},
    )


def _feshbach_components(
    candidate: FloatArray,
    outer_basis: FloatArray,
    retained_rank: int,
    z: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    if not np.isfinite(z):
        raise ValueError("Feshbach spectral parameter z must be finite")
    gram = _outer_gram(candidate, outer_basis)
    k = int(retained_rank)
    core = gram[:k, :k]
    coupling = gram[:k, k:]
    shell = gram[k:, k:]
    resolvent = np.linalg.inv(shell - float(z) * np.eye(shell.shape[0]))
    effective = core - coupling @ resolvent @ coupling.T
    return core, coupling, shell, resolvent, 0.5 * (effective + effective.T)


def feshbach_trace_free_log_map(
    anchor_weight: ArrayLike,
    candidate_weight: ArrayLike,
    *,
    retained_rank: int,
    outer_rank: int,
    z: float,
    rcond: float | None = None,
) -> ECSJacobianMapRecord:
    anchor = _wide_matrix(anchor_weight, name="anchor_weight")
    candidate = _wide_matrix(candidate_weight, name="candidate_weight")
    if candidate.shape != anchor.shape:
        raise ValueError("anchor_weight and candidate_weight must have equal shape")
    _, _, basis, numerical_rank, _ = _outer_basis(anchor, outer_rank, rcond=rcond)
    k, q = _ranks(retained_rank, outer_rank, maximum_rank=numerical_rank)
    _, coupling, _, _, effective = _feshbach_components(candidate, basis, k, z)
    value = _trace_free(_symmetric_log(effective))
    return ECSJacobianMapRecord(
        value=value,
        operator_kind="anchored_feshbach_trace_free_log_effective_core_map",
        map_definition=(
            "F_W,z(X)=Pi_tf log[A-B(C-zI)^-1B^T] in the checkpoint-frozen "
            "outer ECS basis split after retained rank k"
        ),
        parameters={
            "retained_rank": k,
            "outer_rank": q,
            "z": float(z),
            "base_or_candidate_coupling_norm": float(np.linalg.norm(coupling)),
        },
    )


def feshbach_trace_free_log_jvp(
    weight: ArrayLike,
    direction: ArrayLike,
    *,
    retained_rank: int,
    outer_rank: int,
    z: float,
    rcond: float | None = None,
) -> ECSJacobianJVPRecord:
    matrix = _wide_matrix(weight)
    tangent = _matrix(direction, name="direction")
    if tangent.shape != matrix.shape:
        raise ValueError("weight and direction must have equal shape")
    _, _, basis, numerical_rank, _ = _outer_basis(matrix, outer_rank, rcond=rcond)
    k, q = _ranks(retained_rank, outer_rank, maximum_rank=numerical_rank)
    _, coupling, _, resolvent, effective = _feshbach_components(matrix, basis, k, z)
    differential = _outer_dgram(matrix, tangent, basis)
    dcore = differential[:k, :k]
    dcoupling = differential[:k, k:]
    dshell = differential[k:, k:]
    deffective = (
        dcore
        - dcoupling @ resolvent @ coupling.T
        - coupling @ resolvent @ dcoupling.T
        + coupling @ resolvent @ dshell @ resolvent @ coupling.T
    )
    jvp = _trace_free(_frechet_log(effective, deffective))
    mapped = feshbach_trace_free_log_map(
        matrix,
        matrix,
        retained_rank=k,
        outer_rank=q,
        z=z,
        rcond=rcond,
    )
    return ECSJacobianJVPRecord(
        value=mapped.value,
        jvp=jvp,
        operator_kind="feshbach_trace_free_log_effective_core_jacobian_jvp",
        map_definition=(
            "Pi_tf L_log,G_eff[dA-dB R B^T-B R dB^T+B R dC R B^T]"
        ),
        parameters=mapped.parameters,
    )


def feshbach_trace_free_log_spectrum(
    weight: ArrayLike,
    *,
    retained_rank: int,
    outer_rank: int,
    z: float,
    rcond: float | None = None,
) -> ECSJacobianSpectrumRecord:
    """Exact spectrum at the checkpoint SVD frame.

    In that frame ``B=0`` exactly (up to SVD roundoff), so every term carrying
    the shell resolvent vanishes at first order.  The Jacobian is therefore the
    trace-free log-Gram Jacobian of the retained core.  Recording this collapse
    is the scientifically correct single-checkpoint Feshbach test.
    """

    matrix = _wide_matrix(weight)
    _, singular_values, basis, numerical_rank, tolerance = _outer_basis(
        matrix, outer_rank, rcond=rcond
    )
    k, q = _ranks(retained_rank, outer_rank, maximum_rank=numerical_rank)
    _, coupling, _, _, _ = _feshbach_components(matrix, basis, k, z)
    coupling_norm = float(np.linalg.norm(coupling))
    gram_scale = float(singular_values[0] ** 2)
    if coupling_norm > 256.0 * np.finfo(np.float64).eps * gram_scale:
        raise RuntimeError("checkpoint SVD basis did not diagonalize the outer Gram")
    if k < 2:
        amplitudes = np.asarray([], dtype=np.float64)
    else:
        selected = singular_values[:k]
        squared = selected**2
        first, second = np.triu_indices(k, k=1)
        coefficients = np.asarray(
            [
                _log_divided_difference(float(squared[i]), float(squared[j]))
                for i, j in zip(first, second)
            ]
        )
        amplitudes = _trace_free_spectral_amplitudes(
            selected, coefficients, 1.0 / squared
        )
    expected_rank = k * (k - 1) // 2 + max(0, k - 1)
    if amplitudes.size != expected_rank:
        raise RuntimeError("Feshbach trace-free log-core rank audit failed")
    return ECSJacobianSpectrumRecord(
        singular_amplitudes=amplitudes,
        jt_j_nonzero_eigenvalues=amplitudes**2,
        zero_count=int(matrix.size - expected_rank),
        input_dimension=int(matrix.size),
        output_dimension=int(k * k),
        derivative_rank=expected_rank,
        retained_rank=k,
        outer_rank=q,
        operator_kind="feshbach_trace_free_log_effective_core_jacobian_exact_spectrum",
        map_definition=(
            "exact checkpoint-SVD-gauge derivative of Pi_tf log[A-B(C-zI)^-1B^T]; "
            "B=0 at the base point, so shell-downfolding terms vanish at first order"
        ),
        parameters={
            "z": float(z),
            "base_coupling_norm": coupling_norm,
            "first_order_shell_downfolding_active": False,
            "rank_tolerance": float(tolerance),
        },
    )


__all__ = [
    "ECSJacobianJVPRecord",
    "ECSJacobianMapRecord",
    "ECSJacobianSpectrumRecord",
    "feshbach_trace_free_log_jvp",
    "feshbach_trace_free_log_map",
    "feshbach_trace_free_log_spectrum",
    "gap_aware_projector_jvp",
    "gap_aware_projector_map",
    "gap_aware_projector_spectrum",
    "outer_resolvent_jvp",
    "outer_resolvent_map",
    "outer_resolvent_spectrum",
    "outer_trace_free_log_gram_jvp",
    "outer_trace_free_log_gram_map",
    "outer_trace_free_log_gram_spectrum",
    "soft_ecs_projector_jvp",
    "soft_ecs_projector_map",
    "soft_ecs_projector_spectrum",
]

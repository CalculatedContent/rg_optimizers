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
) -> NormalizedGramSpectrumRecord:
    """Exact nonzero singular amplitudes of the normalized-Gram derivative.

    For full rank on the smaller Gram side, off-diagonal amplitudes are

    ``(d/S) sqrt(2 (s_i^2+s_j^2))``.

    The remaining diagonal amplitudes are the nonzero singular values of

    ``2(d/S)[diag(s)-outer(s^2,s)/S]``.
    """

    matrix = _matrix(weight)
    selected, gram = _side_and_gram(matrix, side)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    dimension = int(gram.shape[0])
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
    if singular_values.size != dimension or np.any(singular_values <= tolerance):
        raise np.linalg.LinAlgError(
            "analytic normalized-Gram spectrum requires full rank on the "
            "selected smaller Gram side"
        )
    frobenius_sq = float(np.sum(singular_values**2))
    coefficient = float(dimension) / frobenius_sq
    amplitudes: list[float] = []
    for first in range(dimension):
        for second in range(first + 1, dimension):
            amplitudes.append(
                coefficient
                * np.sqrt(
                    2.0
                    * (
                        singular_values[first] ** 2
                        + singular_values[second] ** 2
                    )
                )
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
    amplitudes.extend(
        float(value)
        for value in diagonal_amplitudes
        if value > diagonal_tolerance
    )
    amplitude_array = np.sort(np.asarray(amplitudes, dtype=np.float64))[::-1]
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
    "CenteredLogSingularFlowRecord",
    "GramTranslationRecord",
    "NormalizedGramJVPRecord",
    "NormalizedGramRecord",
    "NormalizedGramSpectrumRecord",
    "calibrated_training_map_contract",
    "calibrated_training_map_finite_difference",
    "calibrated_training_map_jvp",
    "centered_log_singular_flow",
    "essential_centered_log_singular_flow",
    "gram_translation_quotient",
    "normalized_gram_analytic_spectrum",
    "normalized_gram_jvp",
    "normalized_gram_map",
    "normalized_gram_spectrum",
]

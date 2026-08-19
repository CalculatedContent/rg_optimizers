"""Two-checkpoint observables with conservative scientific labels.

Two saved matrices determine finite transfers and secants.  They do not, by
themselves, determine the Jacobian of a training vector field.  All records in
this module make that distinction explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .polar import polar_factor


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FiniteDifferenceBetaRecord:
    beta_surrogate: FloatArray
    delta_s: float
    delta_norm: float
    beta_norm: float
    relative_delta_norm: float
    is_jacobian: bool
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class ConditionedSquareTransferRecord:
    operator: FloatArray | None
    available: bool
    condition_number: float
    maximum_condition_number: float
    minimum_singular_value: float
    numerical_rank: int
    relative_reconstruction_residual: float
    unavailable_reason: str | None
    is_training_jacobian: bool
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class AlignedRectangularTransferRecord:
    """Finite rectangular transfer and its gauge-aligned rank-r core.

    ``operator`` is the minimum-Frobenius-norm ambient solution of the exact
    checkpoint interpolation problem when it is materialized.  The supported
    singular values are computed without forming that potentially large,
    structurally rank-deficient ambient matrix.  ``core_operator`` is a
    separate quotient candidate obtained after Procrustes-aligning the active
    row or column subspaces; neither object is a training-map Jacobian.
    """

    operator: FloatArray | None
    operator_materialized: bool
    core_operator: FloatArray | None
    available: bool
    orientation: str
    matrix_shape: tuple[int, int]
    effective_rank: int
    ambient_operator_dimension: int
    structural_zero_count: int
    numerical_rank0: int
    numerical_rank1: int
    rank_rtol: float
    rank_threshold0: float
    rank_threshold1: float
    condition_number0: float
    condition_number1: float
    maximum_condition_number: float
    relative_reconstruction_residual: float
    unsupported_source_action_residual: float
    core_reconstruction_residual: float
    subspace_alignment_residual: float
    forced_intersection_zeros: int
    principal_angles: FloatArray
    principal_angle_rates: FloatArray
    supported_transfer_singular_values: FloatArray
    supported_transfer_log_rates: FloatArray
    supported_transfer_rate_amplitudes: FloatArray
    core_singular_values: FloatArray
    core_log_rates: FloatArray
    core_rate_amplitudes: FloatArray
    nonpositive_supported_count: int
    nonpositive_core_count: int
    delta_s: float
    unavailable_reason: str | None
    is_training_jacobian: bool
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class GeneralizedGramRateRecord:
    side: str
    gram0_eigenvalues: FloatArray
    generalized_eigenvalue_ratios: FloatArray
    gram_log_rates: FloatArray
    radial_log_rates: FloatArray
    radial_rate_amplitudes: FloatArray
    retained_rank: int
    dropped_initial_null_directions: int
    target_energy_in_initial_nullspace: float
    nonpositive_ratio_count: int
    delta_s: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class GrassmannSectorRateRecord:
    sector: str
    ambient_dimension: int
    rank0: int
    rank1: int
    compared_dimension: int
    unmatched_dimensions: int
    forced_intersection_zeros: int
    observed_zero_angles: int
    additional_zero_angles: int
    principal_angles: FloatArray
    geodesic_rates: FloatArray
    chordal_rates: FloatArray
    chart_rates: FloatArray
    chart_infinite_count: int
    geodesic_distance: float
    chordal_distance: float
    chart_distance: float
    geodesic_rate: float
    chordal_rate: float
    chart_rate: float
    delta_s: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class GrassmannFlowRecord:
    column: GrassmannSectorRateRecord
    row: GrassmannSectorRateRecord
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class RelativePolarAngularRecord:
    relative_map: FloatArray
    relative_polar: FloatArray
    relative_rank: int
    twist_unique: bool
    orientation: str
    tilt_cosines: FloatArray
    tilt_angles: FloatArray
    tilt_geodesic_rates: FloatArray
    tilt_chordal_rates: FloatArray
    tilt_projective_rates: FloatArray
    tilt_forced_intersection_zeros: int
    tilt_zero_atoms: int
    tilt_endpoint_atoms: int
    twist_eigenvalues: FloatArray
    twist_angles: FloatArray
    twist_geodesic_rates: FloatArray
    twist_chordal_rates: FloatArray
    twist_projective_rates: FloatArray
    twist_zero_atoms: int
    twist_endpoint_atoms: int
    endpoint_tolerance: float
    delta_s: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class RelativePolarInvarianceRecord:
    passed: bool
    maximum_absolute_spectral_error: float
    atom_counts_match: bool
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class TwoCheckpointAnalysisRecord:
    beta: FiniteDifferenceBetaRecord
    square_transfer: ConditionedSquareTransferRecord
    rectangular_transfer: AlignedRectangularTransferRecord
    radial: GeneralizedGramRateRecord
    grassmann: GrassmannFlowRecord
    relative_polar: RelativePolarAngularRecord
    operator_kind: str
    map_definition: str


def _matrix_pair(
    first: ArrayLike,
    second: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    w0 = np.asarray(first, dtype=np.float64)
    w1 = np.asarray(second, dtype=np.float64)
    if w0.ndim != 2 or w1.ndim != 2 or w0.shape != w1.shape:
        raise ValueError("W0 and W1 must be same-shaped matrices")
    if min(w0.shape) < 1:
        raise ValueError("checkpoint matrices must be non-empty")
    if not np.all(np.isfinite(w0)) or not np.all(np.isfinite(w1)):
        raise ValueError("checkpoint matrices contain non-finite values")
    return w0, w1


def _nonzero_delta(delta_s: float) -> float:
    value = float(delta_s)
    if not np.isfinite(value) or value == 0.0:
        raise ValueError("delta_s must be finite and nonzero")
    return value


def finite_difference_beta(
    W0: ArrayLike,
    W1: ArrayLike,
    delta_s: float,
) -> FiniteDifferenceBetaRecord:
    """Return the checkpoint secant ``(W1-W0)/delta_s``.

    This object is deliberately called a beta *surrogate*, never a Jacobian.
    """

    w0, w1 = _matrix_pair(W0, W1)
    ds = _nonzero_delta(delta_s)
    delta = w1 - w0
    beta = delta / ds
    delta_norm = float(np.linalg.norm(delta, ord="fro"))
    return FiniteDifferenceBetaRecord(
        beta_surrogate=beta,
        delta_s=ds,
        delta_norm=delta_norm,
        beta_norm=float(np.linalg.norm(beta, ord="fro")),
        relative_delta_norm=delta_norm
        / max(float(np.linalg.norm(w0, ord="fro")), np.finfo(np.float64).tiny),
        is_jacobian=False,
        operator_kind="two_checkpoint_finite_difference_beta_surrogate",
        map_definition=(
            "beta_secant=(W1-W0)/delta_s; a secant estimate of local flow, "
            "not d beta/dW and not a Jacobian"
        ),
    )


def conditioned_square_transfer(
    W0: ArrayLike,
    W1: ArrayLike,
    *,
    maximum_condition_number: float = 1e8,
    rcond: float | None = None,
) -> ConditionedSquareTransferRecord:
    """Form ``J_transfer=W1 W0^{-1}`` only for a conditioned square ``W0``.

    The symbol is a checkpoint transfer operator.  It is not the Jacobian of
    the optimizer or RG beta function.
    """

    w0, w1 = _matrix_pair(W0, W1)
    if maximum_condition_number <= 1.0:
        raise ValueError("maximum_condition_number must exceed one")
    if w0.shape[0] != w0.shape[1]:
        return ConditionedSquareTransferRecord(
            operator=None,
            available=False,
            condition_number=np.inf,
            maximum_condition_number=float(maximum_condition_number),
            minimum_singular_value=np.nan,
            numerical_rank=min(w0.shape),
            relative_reconstruction_residual=np.nan,
            unavailable_reason="W0 is rectangular; no inverse-based square transfer",
            is_training_jacobian=False,
            operator_kind="unavailable_square_checkpoint_transfer",
            map_definition=(
                "J_transfer=W1 W0^{-1} only for conditioned square W0; "
                "never interpreted as a training Jacobian"
            ),
        )

    singular_values = np.linalg.svd(w0, compute_uv=False)
    largest = float(singular_values[0])
    tolerance = (
        float(rcond) * max(largest, 1.0)
        if rcond is not None
        else max(w0.shape)
        * np.finfo(np.float64).eps
        * max(largest, 1.0)
    )
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative")
    rank = int(np.count_nonzero(singular_values > tolerance))
    smallest = float(singular_values[-1])
    condition = float(largest / smallest) if smallest > 0.0 else np.inf
    reason = None
    if rank != w0.shape[0]:
        reason = "W0 is numerically rank deficient"
    elif not np.isfinite(condition) or condition > maximum_condition_number:
        reason = "W0 exceeds the configured condition-number threshold"

    if reason is not None:
        return ConditionedSquareTransferRecord(
            operator=None,
            available=False,
            condition_number=condition,
            maximum_condition_number=float(maximum_condition_number),
            minimum_singular_value=smallest,
            numerical_rank=rank,
            relative_reconstruction_residual=np.nan,
            unavailable_reason=reason,
            is_training_jacobian=False,
            operator_kind="rejected_ill_conditioned_square_checkpoint_transfer",
            map_definition=(
                "J_transfer=W1 W0^{-1}, rejected unless W0 is full-rank and "
                "conditioned; it is not a training Jacobian"
            ),
        )

    # Solve X W0 = W1 without explicitly materializing W0^{-1}.
    operator = np.linalg.solve(w0.T, w1.T).T
    residual = float(
        np.linalg.norm(operator @ w0 - w1, ord="fro")
        / max(np.linalg.norm(w1, ord="fro"), np.finfo(np.float64).tiny)
    )
    return ConditionedSquareTransferRecord(
        operator=operator,
        available=True,
        condition_number=condition,
        maximum_condition_number=float(maximum_condition_number),
        minimum_singular_value=smallest,
        numerical_rank=rank,
        relative_reconstruction_residual=residual,
        unavailable_reason=None,
        is_training_jacobian=False,
        operator_kind="conditioned_square_checkpoint_left_transfer_not_jacobian",
        map_definition=(
            "J_transfer=W1 W0^{-1}, defined by W1=J_transfer W0; a finite "
            "checkpoint transfer operator, not d beta/dW"
        ),
    )


def _svd_rank_diagnostics(
    singular_values: FloatArray,
    shape: tuple[int, int],
    rcond: float | None,
) -> tuple[int, float, float, float]:
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative")
    rank_rtol = (
        float(rcond)
        if rcond is not None
        else max(shape) * np.finfo(np.float64).eps
    )
    largest = float(singular_values[0])
    threshold = rank_rtol * max(largest, 1.0)
    rank = int(np.count_nonzero(singular_values > threshold))
    smallest = float(singular_values[-1])
    condition = float(largest / smallest) if smallest > 0.0 else np.inf
    return rank, rank_rtol, threshold, condition


def _empty_aligned_rectangular_transfer(
    *,
    shape: tuple[int, int],
    orientation: str,
    rank0: int,
    rank1: int,
    rank_rtol: float,
    threshold0: float,
    threshold1: float,
    condition0: float,
    condition1: float,
    maximum_condition_number: float,
    delta_s: float,
    reason: str,
) -> AlignedRectangularTransferRecord:
    empty = np.asarray([], dtype=np.float64)
    effective_rank = min(shape)
    ambient = shape[0] if orientation.startswith("left") else shape[1]
    return AlignedRectangularTransferRecord(
        operator=None,
        operator_materialized=False,
        core_operator=None,
        available=False,
        orientation=orientation,
        matrix_shape=shape,
        effective_rank=effective_rank,
        ambient_operator_dimension=ambient,
        structural_zero_count=ambient - effective_rank,
        numerical_rank0=rank0,
        numerical_rank1=rank1,
        rank_rtol=rank_rtol,
        rank_threshold0=threshold0,
        rank_threshold1=threshold1,
        condition_number0=condition0,
        condition_number1=condition1,
        maximum_condition_number=float(maximum_condition_number),
        relative_reconstruction_residual=np.nan,
        unsupported_source_action_residual=np.nan,
        core_reconstruction_residual=np.nan,
        subspace_alignment_residual=np.nan,
        forced_intersection_zeros=0,
        principal_angles=empty,
        principal_angle_rates=empty,
        supported_transfer_singular_values=empty,
        supported_transfer_log_rates=empty,
        supported_transfer_rate_amplitudes=empty,
        core_singular_values=empty,
        core_log_rates=empty,
        core_rate_amplitudes=empty,
        nonpositive_supported_count=0,
        nonpositive_core_count=0,
        delta_s=delta_s,
        unavailable_reason=reason,
        is_training_jacobian=False,
        operator_kind="rejected_rectangular_checkpoint_transfer_not_jacobian",
        map_definition=(
            "minimum-norm exact rectangular checkpoint transfer plus an "
            "orthogonal-Procrustes-aligned rank-r core; rejected by the "
            "registered rank/conditioning gate and never a training Jacobian"
        ),
    )


def aligned_rectangular_transfer(
    W0: ArrayLike,
    W1: ArrayLike,
    delta_s: float,
    *,
    maximum_condition_number: float = 1e8,
    rcond: float | None = None,
    materialize_operator: bool = False,
) -> AlignedRectangularTransferRecord:
    """Return an exact supported rectangular transfer and aligned core.

    For a tall full-column-rank matrix the ambient transfer is the unique
    minimum-Frobenius-norm left solution ``L=W1 pinv(W0)`` to ``L W0=W1``.
    For a wide full-row-rank matrix it is the analogous right solution
    ``R=pinv(W0) W1`` to ``W0 R=W1``.  The ambient operator has structural
    zeros outside the rank-r source support, so its nonzero singular values
    are evaluated from thin factors without materializing it.

    A second, explicitly distinct candidate quotients common subspace gauge:
    the active column bases (tall) or row bases (wide) are orthogonally
    Procrustes-aligned and an ``r x r`` transfer is solved between the aligned
    coordinate cores.  Both constructions are finite checkpoint maps, not
    derivatives of the optimizer or beta field.
    """

    w0, w1 = _matrix_pair(W0, W1)
    ds = _nonzero_delta(delta_s)
    if maximum_condition_number <= 1.0:
        raise ValueError("maximum_condition_number must exceed one")

    u0, singular0, vh0 = np.linalg.svd(w0, full_matrices=False)
    u1, singular1, vh1 = np.linalg.svd(w1, full_matrices=False)
    rank0, rank_rtol, threshold0, condition0 = _svd_rank_diagnostics(
        singular0,
        w0.shape,
        rcond,
    )
    rank1, rank_rtol1, threshold1, condition1 = _svd_rank_diagnostics(
        singular1,
        w1.shape,
        rcond,
    )
    if rank_rtol1 != rank_rtol:
        raise RuntimeError("same-shaped matrices produced inconsistent rank rtol")

    rows, columns = w0.shape
    effective_rank = min(rows, columns)
    orientation = (
        "left_minimum_norm_L_equals_W1_pinv_W0"
        if rows >= columns
        else "right_minimum_norm_R_equals_pinv_W0_W1"
    )
    rejection_reason = None
    if rank0 != effective_rank:
        rejection_reason = "W0 is not full rectangular rank"
    elif rank1 != effective_rank:
        rejection_reason = "W1 is not full rectangular rank"
    elif not np.isfinite(condition0) or condition0 > maximum_condition_number:
        rejection_reason = "W0 exceeds the configured condition-number threshold"
    elif not np.isfinite(condition1) or condition1 > maximum_condition_number:
        rejection_reason = "W1 exceeds the configured condition-number threshold"
    if rejection_reason is not None:
        return _empty_aligned_rectangular_transfer(
            shape=w0.shape,
            orientation=orientation,
            rank0=rank0,
            rank1=rank1,
            rank_rtol=rank_rtol,
            threshold0=threshold0,
            threshold1=threshold1,
            condition0=condition0,
            condition1=condition1,
            maximum_condition_number=maximum_condition_number,
            delta_s=ds,
            reason=rejection_reason,
        )

    tiny = np.finfo(np.float64).tiny
    inverse_singular0 = 1.0 / singular0
    if rows >= columns:
        source_basis = u0[:, :effective_rank]
        target_basis = u1[:, :effective_rank]
        orientation_overlap = target_basis.T @ source_basis
        procrustes_left, _, procrustes_right_h = np.linalg.svd(
            orientation_overlap,
            full_matrices=False,
        )
        alignment = procrustes_left @ procrustes_right_h
        aligned_target_basis = target_basis @ alignment
        core0 = source_basis.T @ w0
        core1 = aligned_target_basis.T @ w1
        core_operator = np.linalg.solve(core0.T, core1.T).T

        # L=(W1 V0 Sigma0^-1) U0^T.  U0 has orthonormal columns,
        # hence the singular values of the first thin factor are precisely
        # the nonzero singular values of L.
        supported_factor = (w1 @ vh0.T) * inverse_singular0[None, :]
        operator = (
            supported_factor @ source_basis.T
            if materialize_operator
            else None
        )
        reconstructed = supported_factor @ (source_basis.T @ w0)
        unsupported_coordinates = source_basis.T - (
            source_basis.T @ source_basis
        ) @ source_basis.T
        unsupported_squared = float(
            np.trace(
                (supported_factor.T @ supported_factor)
                @ (unsupported_coordinates @ unsupported_coordinates.T)
            )
        )
        unsupported_residual = np.sqrt(max(unsupported_squared, 0.0)) / max(
            float(np.linalg.norm(supported_factor, ord="fro")),
            tiny,
        )
        core_reconstruction = core_operator @ core0
        ambient = rows
        core_equation = "K C0=C1"
    else:
        source_basis = vh0.T[:, :effective_rank]
        target_basis = vh1.T[:, :effective_rank]
        orientation_overlap = target_basis.T @ source_basis
        procrustes_left, _, procrustes_right_h = np.linalg.svd(
            orientation_overlap,
            full_matrices=False,
        )
        alignment = procrustes_left @ procrustes_right_h
        aligned_target_basis = target_basis @ alignment
        core0 = w0 @ source_basis
        core1 = w1 @ aligned_target_basis
        core_operator = np.linalg.solve(core0, core1)

        # R=V0 (Sigma0^-1 U0^T W1).  V0 has orthonormal columns, so the
        # second thin factor carries exactly the nonzero singular values.
        supported_factor = inverse_singular0[:, None] * (u0.T @ w1)
        operator = source_basis @ supported_factor if materialize_operator else None
        reconstructed = (w0 @ source_basis) @ supported_factor
        unsupported_coordinates = source_basis - source_basis @ (
            source_basis.T @ source_basis
        )
        unsupported_squared = float(
            np.trace(
                (unsupported_coordinates.T @ unsupported_coordinates)
                @ (supported_factor @ supported_factor.T)
            )
        )
        unsupported_residual = np.sqrt(max(unsupported_squared, 0.0)) / max(
            float(np.linalg.norm(supported_factor, ord="fro")),
            tiny,
        )
        core_reconstruction = core0 @ core_operator
        ambient = columns
        core_equation = "C0 K=C1"

    overlap_singular_values = np.linalg.svd(
        orientation_overlap,
        compute_uv=False,
    )
    overlap_singular_values = np.clip(overlap_singular_values, 0.0, 1.0)
    overlap_singular_values[
        1.0 - overlap_singular_values <= 100.0 * np.finfo(np.float64).eps
    ] = 1.0
    principal_angles = np.arccos(overlap_singular_values)
    forced_zeros = max(0, 2 * effective_rank - ambient)
    forced_zeros = min(forced_zeros, effective_rank)
    principal_angles[:forced_zeros] = 0.0

    supported_singular_values = np.linalg.svd(
        supported_factor,
        compute_uv=False,
    )
    core_singular_values = np.linalg.svd(core_operator, compute_uv=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        supported_log_rates = np.log(supported_singular_values) / ds
        core_log_rates = np.log(core_singular_values) / ds
    relative_reconstruction = float(
        np.linalg.norm(reconstructed - w1, ord="fro")
        / max(float(np.linalg.norm(w1, ord="fro")), tiny)
    )
    core_residual = float(
        np.linalg.norm(core_reconstruction - core1, ord="fro")
        / max(float(np.linalg.norm(core1, ord="fro")), tiny)
    )
    alignment_residual = float(
        np.linalg.norm(aligned_target_basis - source_basis, ord="fro")
        / np.sqrt(float(effective_rank))
    )
    return AlignedRectangularTransferRecord(
        operator=operator,
        operator_materialized=bool(materialize_operator),
        core_operator=core_operator,
        available=True,
        orientation=orientation,
        matrix_shape=w0.shape,
        effective_rank=effective_rank,
        ambient_operator_dimension=ambient,
        structural_zero_count=ambient - effective_rank,
        numerical_rank0=rank0,
        numerical_rank1=rank1,
        rank_rtol=rank_rtol,
        rank_threshold0=threshold0,
        rank_threshold1=threshold1,
        condition_number0=condition0,
        condition_number1=condition1,
        maximum_condition_number=float(maximum_condition_number),
        relative_reconstruction_residual=relative_reconstruction,
        unsupported_source_action_residual=float(unsupported_residual),
        core_reconstruction_residual=core_residual,
        subspace_alignment_residual=alignment_residual,
        forced_intersection_zeros=forced_zeros,
        principal_angles=principal_angles,
        principal_angle_rates=principal_angles / abs(ds),
        supported_transfer_singular_values=supported_singular_values,
        supported_transfer_log_rates=supported_log_rates,
        supported_transfer_rate_amplitudes=np.abs(supported_log_rates),
        core_singular_values=core_singular_values,
        core_log_rates=core_log_rates,
        core_rate_amplitudes=np.abs(core_log_rates),
        nonpositive_supported_count=int(
            np.count_nonzero(supported_singular_values <= 0.0)
        ),
        nonpositive_core_count=int(np.count_nonzero(core_singular_values <= 0.0)),
        delta_s=ds,
        unavailable_reason=None,
        is_training_jacobian=False,
        operator_kind=(
            "exact_supported_rectangular_checkpoint_transfer_and_"
            "procrustes_aligned_core_not_jacobian"
        ),
        map_definition=(
            f"{orientation}; exact minimum-norm ambient interpolation with "
            f"{ambient - effective_rank} structural zero modes, plus active "
            f"subspace Procrustes quotient and {core_equation}; log singular "
            "rates are divided by delta_s. These are finite checkpoint "
            "transfers, not d beta/dW or an optimizer Jacobian."
        ),
    )


def generalized_gram_log_rates(
    W0: ArrayLike,
    W1: ArrayLike,
    delta_s: float,
    *,
    side: str = "auto",
    rcond: float | None = None,
) -> GeneralizedGramRateRecord:
    """Compute rectangular radial rates from a supported Gram pencil.

    On the positive support of ``G0``, eigenvalues of

    ``G0^{-1/2} G1 G0^{-1/2}``

    are squared radial scale ratios.  Their Gram log rates are
    ``log(lambda)/delta_s`` and singular/radial log rates are half of that.
    No pseudo-inverse transfer matrix is called a Jacobian.
    """

    w0, w1 = _matrix_pair(W0, W1)
    ds = _nonzero_delta(delta_s)
    rows, columns = w0.shape
    normalized_side = str(side).lower()
    if normalized_side == "auto":
        normalized_side = "right" if rows >= columns else "left"
    if normalized_side in {"column", "columns"}:
        normalized_side = "right"
    if normalized_side in {"row", "rows"}:
        normalized_side = "left"
    if normalized_side == "right":
        gram0 = w0.T @ w0
        gram1 = w1.T @ w1
    elif normalized_side == "left":
        gram0 = w0 @ w0.T
        gram1 = w1 @ w1.T
    else:
        raise ValueError("side must be auto, right/column, or left/row")

    eigenvalues0, eigenvectors0 = np.linalg.eigh(
        0.5 * (gram0 + gram0.T)
    )
    largest = max(float(eigenvalues0[-1]), 0.0)
    tolerance = (
        float(rcond) * max(largest, 1.0)
        if rcond is not None
        else gram0.shape[0]
        * np.finfo(np.float64).eps
        * max(largest, 1.0)
    )
    if rcond is not None and rcond < 0.0:
        raise ValueError("rcond must be non-negative")
    retained = eigenvalues0 > tolerance
    if not np.any(retained):
        raise np.linalg.LinAlgError("W0 has no numerically positive Gram support")
    basis = eigenvectors0[:, retained]
    positive0 = eigenvalues0[retained]
    inverse_sqrt = 1.0 / np.sqrt(positive0)
    reduced1 = basis.T @ gram1 @ basis
    whitened = (
        inverse_sqrt[:, None] * reduced1 * inverse_sqrt[None, :]
    )
    whitened = 0.5 * (whitened + whitened.T)
    ratios = np.linalg.eigvalsh(whitened)
    negative_tolerance = 100.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(ratios))), 1.0
    )
    if np.any(ratios < -negative_tolerance):
        raise RuntimeError("generalized Gram pencil produced a negative eigenvalue")
    ratios = np.maximum(ratios, 0.0)[::-1]
    with np.errstate(divide="ignore"):
        gram_rates = np.log(ratios) / ds
    radial_rates = 0.5 * gram_rates

    null_basis = eigenvectors0[:, ~retained]
    null_energy = (
        float(np.trace(null_basis.T @ gram1 @ null_basis))
        if null_basis.shape[1]
        else 0.0
    )
    return GeneralizedGramRateRecord(
        side=normalized_side,
        gram0_eigenvalues=eigenvalues0[::-1],
        generalized_eigenvalue_ratios=ratios,
        gram_log_rates=gram_rates,
        radial_log_rates=radial_rates,
        radial_rate_amplitudes=np.abs(radial_rates),
        retained_rank=int(np.count_nonzero(retained)),
        dropped_initial_null_directions=int(np.count_nonzero(~retained)),
        target_energy_in_initial_nullspace=null_energy,
        nonpositive_ratio_count=int(np.count_nonzero(ratios <= 0.0)),
        delta_s=ds,
        operator_kind="rectangular_generalized_gram_radial_log_rates",
        map_definition=(
            "eig(G0_support^{-1/2} G1 G0_support^{-1/2}); radial rates="
            "0.5 log(lambda)/delta_s, not a checkpoint Jacobian"
        ),
    )


def _subspace_basis(
    matrix: FloatArray,
    sector: str,
    rcond: float | None,
) -> tuple[FloatArray, int]:
    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = (
        float(rcond) * max(largest, 1.0)
        if rcond is not None
        else max(matrix.shape)
        * np.finfo(np.float64).eps
        * max(largest, 1.0)
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    if sector == "column":
        return left[:, :rank], rank
    return right_h[:rank, :].T, rank


def _grassmann_sector(
    w0: FloatArray,
    w1: FloatArray,
    ds: float,
    *,
    sector: str,
    rcond: float | None,
    angle_zero_tolerance: float,
    chart_cosine_floor: float,
) -> GrassmannSectorRateRecord:
    q0, rank0 = _subspace_basis(w0, sector, rcond)
    q1, rank1 = _subspace_basis(w1, sector, rcond)
    ambient = w0.shape[0] if sector == "column" else w0.shape[1]
    compared = min(rank0, rank1)
    unmatched = abs(rank0 - rank1)
    forced_zeros = max(0, rank0 + rank1 - ambient)
    forced_zeros = min(forced_zeros, compared)

    if compared:
        cosines = np.linalg.svd(q0.T @ q1, compute_uv=False)
        cosines = np.sort(np.clip(cosines, 0.0, 1.0))[::-1]
        angles = np.arccos(cosines)
        raw_zero_count = int(
            np.count_nonzero(1.0 - cosines <= angle_zero_tolerance)
        )
        # The dimension theorem guarantees these exact intersections even when
        # floating-point SVD reports tiny positive angles.
        angles[:forced_zeros] = 0.0
    else:
        angles = np.asarray([], dtype=np.float64)
        raw_zero_count = 0
    if unmatched:
        angles = np.concatenate(
            [angles, np.full(unmatched, np.pi / 2.0, dtype=np.float64)]
        )

    observed_zeros = max(raw_zero_count, forced_zeros)
    sine = np.sin(angles)
    cosine = np.cos(angles)
    chart = np.empty_like(angles)
    finite_chart = cosine > float(chart_cosine_floor)
    chart[finite_chart] = sine[finite_chart] / cosine[finite_chart]
    chart[~finite_chart] = np.inf
    denominator = abs(ds)
    geodesic_rates = angles / denominator
    chordal_rates = sine / denominator
    chart_rates = chart / denominator
    geodesic_distance = float(np.linalg.norm(angles))
    chordal_distance = float(np.linalg.norm(sine))
    chart_distance = (
        float(np.linalg.norm(chart)) if np.all(np.isfinite(chart)) else np.inf
    )
    return GrassmannSectorRateRecord(
        sector=sector,
        ambient_dimension=int(ambient),
        rank0=rank0,
        rank1=rank1,
        compared_dimension=compared,
        unmatched_dimensions=unmatched,
        forced_intersection_zeros=forced_zeros,
        observed_zero_angles=observed_zeros,
        additional_zero_angles=max(0, observed_zeros - forced_zeros),
        principal_angles=angles,
        geodesic_rates=geodesic_rates,
        chordal_rates=chordal_rates,
        chart_rates=chart_rates,
        chart_infinite_count=int(np.count_nonzero(~np.isfinite(chart))),
        geodesic_distance=geodesic_distance,
        chordal_distance=chordal_distance,
        chart_distance=chart_distance,
        geodesic_rate=geodesic_distance / denominator,
        chordal_rate=chordal_distance / denominator,
        chart_rate=chart_distance / denominator,
        delta_s=ds,
        operator_kind=f"{sector}_grassmann_principal_angle_rates",
        map_definition=(
            "principal angles theta with geodesic=theta, chordal=sin(theta), "
            "chart=tan(theta), each divided by |delta_s|; forced intersection "
            "zeros are counted by max(0,r0+r1-ambient)"
        ),
    )


def grassmann_flow_rates(
    W0: ArrayLike,
    W1: ArrayLike,
    delta_s: float,
    *,
    rcond: float | None = None,
    angle_zero_tolerance: float = 1e-12,
    chart_cosine_floor: float = 1e-12,
) -> GrassmannFlowRecord:
    """Return row- and column-space principal-angle flow rates."""

    w0, w1 = _matrix_pair(W0, W1)
    ds = _nonzero_delta(delta_s)
    if angle_zero_tolerance < 0.0 or chart_cosine_floor <= 0.0:
        raise ValueError("angle tolerances must be non-negative/positive")
    column = _grassmann_sector(
        w0,
        w1,
        ds,
        sector="column",
        rcond=rcond,
        angle_zero_tolerance=float(angle_zero_tolerance),
        chart_cosine_floor=float(chart_cosine_floor),
    )
    row = _grassmann_sector(
        w0,
        w1,
        ds,
        sector="row",
        rcond=rcond,
        angle_zero_tolerance=float(angle_zero_tolerance),
        chart_cosine_floor=float(chart_cosine_floor),
    )
    return GrassmannFlowRecord(
        column=column,
        row=row,
        operator_kind="row_and_column_grassmann_checkpoint_flow",
        map_definition=(
            "row/column subspace principal-angle rates; finite checkpoint "
            "geometry, not a Jacobian of training dynamics"
        ),
    )


def relative_polar_angular_flow(
    W0: ArrayLike,
    W1: ArrayLike,
    delta_s: float,
    *,
    rcond: float | None = None,
    endpoint_tolerance: float = 1e-10,
) -> RelativePolarAngularRecord:
    """Gauge-invariant finite angular flow between two full-rank matrices.

    For tall matrices ``R=P1.T P0``; for wide matrices ``R=P1 P0.T``.
    Singular values of ``R`` give tilt principal angles.  The polar factor
    ``O=P(R)`` gives twist.  Projective endpoint atoms are excluded from the
    finite ``tan^2`` samples and counted separately.
    """

    w0, w1 = _matrix_pair(W0, W1)
    ds = _nonzero_delta(delta_s)
    if endpoint_tolerance < 0.0 or endpoint_tolerance >= 0.1:
        raise ValueError("endpoint_tolerance must satisfy 0 <= tol < 0.1")
    rows, columns = w0.shape
    rank_expected = min(rows, columns)
    for name, matrix in (("W0", w0), ("W1", w1)):
        singular_values = np.linalg.svd(matrix, compute_uv=False)
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
        if int(np.count_nonzero(singular_values > tolerance)) != rank_expected:
            raise np.linalg.LinAlgError(f"{name} must have full rectangular rank")

    p0 = polar_factor(w0)
    p1 = polar_factor(w1)
    if rows >= columns:
        relative = p1.T @ p0
        orientation = "column_stiefel_relative_map_P1T_P0"
    else:
        relative = p1 @ p0.T
        orientation = "row_stiefel_relative_map_P1_P0T"

    tilt_cosines = np.sort(
        np.clip(np.linalg.svd(relative, compute_uv=False), 0.0, 1.0)
    )[::-1]
    tilt_ambient = rows if rows >= columns else columns
    forced_tilt_zeros = max(0, 2 * rank_expected - tilt_ambient)
    tilt_cosines[:forced_tilt_zeros] = 1.0
    tilt_angles = np.arccos(tilt_cosines)
    tilt_lambda = np.sin(tilt_angles) ** 2
    tilt_zero = tilt_lambda <= endpoint_tolerance
    tilt_endpoint = tilt_lambda >= 1.0 - endpoint_tolerance
    tilt_interior = ~(tilt_zero | tilt_endpoint)
    denominator = abs(ds)
    tilt_projective = (
        tilt_lambda[tilt_interior] / (1.0 - tilt_lambda[tilt_interior])
    ) / denominator

    relative_singular_values = np.linalg.svd(relative, compute_uv=False)
    relative_tolerance = (
        max(relative.shape)
        * np.finfo(np.float64).eps
        * max(float(relative_singular_values[0]), 1.0)
    )
    relative_rank = int(
        np.count_nonzero(relative_singular_values > relative_tolerance)
    )
    twist_unique = relative_rank == relative.shape[0]
    relative_rotation = polar_factor(relative)
    identity = np.eye(relative.shape[0])
    twist_operator = (
        2.0 * identity - relative_rotation - relative_rotation.T
    )
    twist_lambda = np.sort(
        np.clip(
            np.linalg.eigvalsh(0.5 * (twist_operator + twist_operator.T)),
            0.0,
            4.0,
        )
    )
    twist_angles = 2.0 * np.arcsin(np.sqrt(twist_lambda / 4.0))
    twist_scaled = twist_lambda / 4.0
    twist_zero = twist_scaled <= endpoint_tolerance
    twist_endpoint = twist_scaled >= 1.0 - endpoint_tolerance
    twist_interior = ~(twist_zero | twist_endpoint)
    twist_projective = (
        twist_scaled[twist_interior] / (1.0 - twist_scaled[twist_interior])
    ) / denominator

    return RelativePolarAngularRecord(
        relative_map=relative,
        relative_polar=relative_rotation,
        relative_rank=relative_rank,
        twist_unique=twist_unique,
        orientation=orientation,
        tilt_cosines=tilt_cosines,
        tilt_angles=tilt_angles,
        tilt_geodesic_rates=tilt_angles / denominator,
        tilt_chordal_rates=np.sin(tilt_angles) / denominator,
        tilt_projective_rates=np.sort(tilt_projective),
        tilt_forced_intersection_zeros=forced_tilt_zeros,
        tilt_zero_atoms=int(np.count_nonzero(tilt_zero)),
        tilt_endpoint_atoms=int(np.count_nonzero(tilt_endpoint)),
        twist_eigenvalues=twist_lambda,
        twist_angles=twist_angles,
        twist_geodesic_rates=twist_angles / denominator,
        twist_chordal_rates=(2.0 * np.sin(twist_angles / 2.0)) / denominator,
        twist_projective_rates=np.sort(twist_projective),
        twist_zero_atoms=int(np.count_nonzero(twist_zero)),
        twist_endpoint_atoms=int(np.count_nonzero(twist_endpoint)),
        endpoint_tolerance=float(endpoint_tolerance),
        delta_s=ds,
        operator_kind="finite_checkpoint_relative_polar_angular_flow_not_jacobian",
        map_definition=(
            "R=P1^T P0 (tall) or P1 P0^T (wide); tilt from svd(R), twist "
            "from O=P(R); geodesic/chordal/tan^2 amplitudes divided by "
            "|delta_s|; endpoint atoms excluded from projective samples. "
            "Twist is unique only when R is nonsingular."
        ),
    )


def _haar_orthogonal(dimension: int, rng: np.random.Generator) -> FloatArray:
    q, triangular = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return q * np.where(np.diag(triangular) < 0.0, -1.0, 1.0)[None, :]


def check_relative_polar_orthogonal_invariance(
    W0: ArrayLike,
    W1: ArrayLike,
    delta_s: float,
    *,
    rng: np.random.Generator | int | None = 0,
    atol: float = 1e-8,
    rtol: float = 1e-7,
    endpoint_tolerance: float = 1e-10,
) -> RelativePolarInvarianceRecord:
    """Check common ``(W0,W1)->(L W0 R^T,L W1 R^T)`` invariance."""

    w0, w1 = _matrix_pair(W0, W1)
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    left = _haar_orthogonal(w0.shape[0], generator)
    right = _haar_orthogonal(w0.shape[1], generator)
    original = relative_polar_angular_flow(
        w0,
        w1,
        delta_s,
        endpoint_tolerance=endpoint_tolerance,
    )
    transformed = relative_polar_angular_flow(
        left @ w0 @ right.T,
        left @ w1 @ right.T,
        delta_s,
        endpoint_tolerance=endpoint_tolerance,
    )
    array_pairs = (
        (original.tilt_angles, transformed.tilt_angles),
        (original.tilt_geodesic_rates, transformed.tilt_geodesic_rates),
        (original.tilt_projective_rates, transformed.tilt_projective_rates),
        (original.twist_eigenvalues, transformed.twist_eigenvalues),
        (original.twist_geodesic_rates, transformed.twist_geodesic_rates),
        (original.twist_projective_rates, transformed.twist_projective_rates),
    )
    errors = [
        float(np.max(np.abs(first - second))) if first.size else 0.0
        for first, second in array_pairs
        if first.shape == second.shape
    ]
    shapes_match = all(first.shape == second.shape for first, second in array_pairs)
    arrays_match = shapes_match and all(
        np.allclose(first, second, atol=atol, rtol=rtol)
        for first, second in array_pairs
    )
    atom_counts_original = (
        original.tilt_zero_atoms,
        original.tilt_endpoint_atoms,
        original.twist_zero_atoms,
        original.twist_endpoint_atoms,
    )
    atom_counts_transformed = (
        transformed.tilt_zero_atoms,
        transformed.tilt_endpoint_atoms,
        transformed.twist_zero_atoms,
        transformed.twist_endpoint_atoms,
    )
    counts_match = atom_counts_original == atom_counts_transformed
    return RelativePolarInvarianceRecord(
        passed=bool(arrays_match and counts_match),
        maximum_absolute_spectral_error=max(errors, default=np.inf),
        atom_counts_match=counts_match,
        operator_kind="relative_polar_common_orthogonal_invariance_audit",
        map_definition=(
            "compare angular spectra after common L/R orthogonal gauge action; "
            "R itself may conjugate but its reported spectral observables agree"
        ),
    )


def analyze_two_checkpoints(
    W0: ArrayLike,
    W1: ArrayLike,
    delta_s: float,
    *,
    maximum_condition_number: float = 1e8,
    rcond: float | None = None,
) -> TwoCheckpointAnalysisRecord:
    """Bundle the conservative two-checkpoint observables for one layer."""

    return TwoCheckpointAnalysisRecord(
        beta=finite_difference_beta(W0, W1, delta_s),
        square_transfer=conditioned_square_transfer(
            W0,
            W1,
            maximum_condition_number=maximum_condition_number,
            rcond=rcond,
        ),
        rectangular_transfer=aligned_rectangular_transfer(
            W0,
            W1,
            delta_s,
            maximum_condition_number=maximum_condition_number,
            rcond=rcond,
            materialize_operator=False,
        ),
        radial=generalized_gram_log_rates(
            W0,
            W1,
            delta_s,
            rcond=rcond,
        ),
        grassmann=grassmann_flow_rates(
            W0,
            W1,
            delta_s,
            rcond=rcond,
        ),
        relative_polar=relative_polar_angular_flow(
            W0,
            W1,
            delta_s,
            rcond=rcond,
        ),
        operator_kind="two_checkpoint_local_flow_diagnostic_bundle",
        map_definition=(
            "secant beta surrogate + conditioned square transfer + exact "
            "supported rectangular transfer/aligned core + Gram radial rates "
            "+ Grassmann rates + relative-polar tilt/twist angular flow; none "
            "is d beta/dW"
        ),
    )


# Compatibility aliases with maximally explicit scientific names.
finite_difference_beta_surrogate = finite_difference_beta
square_checkpoint_operator = conditioned_square_transfer
rectangular_checkpoint_transfer = aligned_rectangular_transfer
grassmann_rates = grassmann_flow_rates
relative_polar_flow = relative_polar_angular_flow


__all__ = [
    "AlignedRectangularTransferRecord",
    "ConditionedSquareTransferRecord",
    "FiniteDifferenceBetaRecord",
    "GeneralizedGramRateRecord",
    "GrassmannFlowRecord",
    "GrassmannSectorRateRecord",
    "RelativePolarAngularRecord",
    "RelativePolarInvarianceRecord",
    "TwoCheckpointAnalysisRecord",
    "analyze_two_checkpoints",
    "aligned_rectangular_transfer",
    "conditioned_square_transfer",
    "finite_difference_beta",
    "finite_difference_beta_surrogate",
    "generalized_gram_log_rates",
    "grassmann_flow_rates",
    "grassmann_rates",
    "check_relative_polar_orthogonal_invariance",
    "relative_polar_angular_flow",
    "relative_polar_flow",
    "rectangular_checkpoint_transfer",
    "square_checkpoint_operator",
]

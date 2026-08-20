"""Weight-only Muon quotient hypotheses for the MLP3 checkpoint suite.

The maps in this module are deliberately labelled *hypotheses*.  An unknown
sum of semiorthogonal Muon updates does not define a non-trivial exact quotient
of a final weight matrix.  These functions instead provide five auditable,
low-dimensional inverse models which can be tested on the verified final-100
checkpoint cache and then passed back through WeightWatcher as actual matrices.

Every map:

* fixes an ECS rank before applying the transformation;
* returns the rectangular-diagonal canonical representative of the
  ``O(m) x O(n)`` orbit whose non-zero Gram spectrum is the transformed ECS
  spectrum;
* records every nuisance parameter and numerical diagnostic.

The implementation never adds jitter merely to manufacture positive modes.
Exact zeros remain zeros and may make a WeightWatcher fit unavailable.  The
canonical diagonal section is also essential numerically: densely rotating a
rank-k matrix and then casting it to a float32 model parameter generically
creates tiny positive singular values in its nominal null space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class WeightQuotientUnavailable(RuntimeError):
    """A declared inverse model is inapplicable to this observed spectrum."""


@dataclass(frozen=True)
class WeightQuotientResult:
    """One canonical full-shape representative and its positive ECS spectrum."""

    weight: FloatArray
    singular_values: FloatArray
    gram_eigenvalues: FloatArray
    ecs_rank: int
    retained_rank: int
    method: str
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


def _rank(value: int, *, maximum: int) -> int:
    result = int(value)
    if float(value) != float(result) or not 1 <= result <= int(maximum):
        raise ValueError(f"ecs_rank must lie in [1, {int(maximum)}]")
    return result


def _fraction(value: float, *, name: str, include_one: bool = False) -> float:
    result = float(value)
    upper_ok = result <= 1.0 if include_one else result < 1.0
    if not np.isfinite(result) or result < 0.0 or not upper_ok:
        relation = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must lie in {relation}")
    return result


def midpoint_ecs_rank(
    pl_support_rank: int | float,
    detx_rank: int | float,
    *,
    maximum_rank: int,
) -> int:
    """Match the suite's preregistered floor midpoint rank convention."""

    maximum = int(maximum_rank)
    if maximum < 1:
        raise ValueError("maximum_rank must be positive")
    pl = int(round(float(pl_support_rank)))
    detx = int(round(float(detx_rank)))
    return int(np.clip(np.floor((max(1, pl) + detx) / 2.0), 1, maximum))


def _svd(weight: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    left, singular, right_h = np.linalg.svd(weight, full_matrices=False)
    return left, np.asarray(singular, dtype=np.float64), right_h


def _representative(
    left: FloatArray,
    right_h: FloatArray,
    singular_values: ArrayLike,
    *,
    ecs_rank: int,
    method: str,
    operator_kind: str,
    map_definition: str,
    parameters: Mapping[str, Any],
) -> WeightQuotientResult:
    values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if values.size != int(ecs_rank):
        raise ValueError("transformed singular spectrum must have ecs_rank values")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("transformed singular values must be finite and non-negative")
    tolerance = (
        max(1, values.size)
        * np.finfo(np.float64).eps
        * max(float(np.max(values, initial=0.0)), 1.0)
    )
    values = np.where(values > tolerance, values, 0.0)
    positive = np.sort(values[values > 0.0])[::-1]
    # WeightWatcher fits only the orthogonal-orbit invariant singular spectrum.
    # Use the rectangular-diagonal section rather than a dense U diag(s) V^T
    # reconstruction: entrywise float32 rounding of that dense product would
    # make a nominal rank-k matrix weakly full rank.
    matrix = np.zeros((left.shape[0], right_h.shape[1]), dtype=np.float64)
    diagonal = np.arange(values.size)
    matrix[diagonal, diagonal] = values
    observed = np.linalg.svd(matrix, compute_uv=False)
    observed = np.sort(observed[observed > tolerance])[::-1]
    if observed.size != positive.size or not np.allclose(
        observed, positive, rtol=2e-10, atol=2e-12
    ):
        raise RuntimeError("materialized representative disagrees with declared spectrum")
    return WeightQuotientResult(
        weight=np.asarray(matrix, dtype=np.float64),
        singular_values=positive,
        gram_eigenvalues=positive**2,
        ecs_rank=int(ecs_rank),
        retained_rank=int(positive.size),
        method=str(method),
        operator_kind=str(operator_kind),
        map_definition=str(map_definition),
        parameters={
            **dict(parameters),
            "orthogonal_orbit_gauge": "rectangular_diagonal_canonical_section",
        },
    )


def ecs_truncation(weight: ArrayLike, *, ecs_rank: int) -> WeightQuotientResult:
    """Frozen-midpoint ECS control with no nuisance correction."""

    matrix = _matrix(weight)
    left, singular, right_h = _svd(matrix)
    k = _rank(ecs_rank, maximum=singular.size)
    return _representative(
        left,
        right_h,
        singular[:k],
        ecs_rank=k,
        method="midpoint_ecs_control",
        operator_kind="frozen_midpoint_ecs_truncated_weight",
        map_definition=(
            "[W] under O(m)xO(n) -> rectangular_diag(s_1,...,s_k,0)"
        ),
        parameters={"ecs_rank": k, "discarded_rank": int(singular.size - k)},
    )


def uniform_singular_translation(
    weight: ArrayLike,
    *,
    ecs_rank: int,
    shift_fraction: float,
) -> WeightQuotientResult:
    """Reference polar quotient: remove one common ECS singular coordinate."""

    matrix = _matrix(weight)
    left, singular, right_h = _svd(matrix)
    k = _rank(ecs_rank, maximum=singular.size)
    fraction = _fraction(shift_fraction, name="shift_fraction")
    shift = fraction * float(singular[k - 1])
    transformed = singular[:k] - shift
    return _representative(
        left,
        right_h,
        transformed,
        ecs_rank=k,
        method="uniform_singular_translation",
        operator_kind="midpoint_ecs_uniform_singular_counterterm",
        map_definition=(
            "on [W] under O(m)xO(n), s_i -> s_i-mu on the frozen midpoint ECS"
        ),
        parameters={
            "ecs_rank": k,
            "shift_fraction": fraction,
            "mu": shift,
            "ecs_floor_singular_value": float(singular[k - 1]),
        },
    )


def gram_ridge_quotient(
    weight: ArrayLike,
    *,
    ecs_rank: int,
    tau_fraction: float,
) -> WeightQuotientResult:
    """Subtract a scalar identity counterterm from the ECS Gram spectrum."""

    matrix = _matrix(weight)
    left, singular, right_h = _svd(matrix)
    k = _rank(ecs_rank, maximum=singular.size)
    fraction = _fraction(tau_fraction, name="tau_fraction")
    floor = float(singular[k - 1] ** 2)
    tau = fraction * floor
    transformed = np.sqrt(np.maximum(singular[:k] ** 2 - tau, 0.0))
    return _representative(
        left,
        right_h,
        transformed,
        ecs_rank=k,
        method="gram_ridge",
        operator_kind="midpoint_ecs_scalar_gram_counterterm",
        map_definition=(
            "on [W] under O(m)xO(n), lambda_i=s_i^2 -> "
            "max(lambda_i-tau,0) on the ECS"
        ),
        parameters={
            "ecs_rank": k,
            "tau_fraction": fraction,
            "tau": tau,
            "ecs_floor_gram_eigenvalue": floor,
        },
    )


def _pava_nonincreasing(values: FloatArray) -> FloatArray:
    """Euclidean isotonic projection onto non-increasing sequences."""

    levels: list[float] = []
    weights: list[int] = []
    for raw in np.asarray(values, dtype=np.float64):
        levels.append(float(raw))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            total = weights[-2] + weights[-1]
            pooled = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / total
            levels[-2:] = [pooled]
            weights[-2:] = [total]
    return np.concatenate(
        [np.full(weight, level, dtype=np.float64) for level, weight in zip(levels, weights)]
    )


def blockwise_singular_quotient(
    weight: ArrayLike,
    *,
    ecs_rank: int,
    block_count: int,
    shift_fraction: float,
) -> WeightQuotientResult:
    """Remove one common singular offset inside each fixed contiguous band."""

    matrix = _matrix(weight)
    left, singular, right_h = _svd(matrix)
    k = _rank(ecs_rank, maximum=singular.size)
    blocks = int(block_count)
    if float(block_count) != float(blocks) or not 1 <= blocks <= k:
        raise ValueError("block_count must be an integer in [1, ecs_rank]")
    fraction = _fraction(shift_fraction, name="shift_fraction")
    transformed = singular[:k].copy()
    boundaries: list[tuple[int, int]] = []
    shifts: list[float] = []
    for indices in np.array_split(np.arange(k), blocks):
        start, stop = int(indices[0]), int(indices[-1]) + 1
        shift = fraction * float(np.min(singular[indices]))
        transformed[indices] -= shift
        boundaries.append((start, stop))
        shifts.append(shift)
    projected = _pava_nonincreasing(transformed)
    correction = float(np.linalg.norm(projected - transformed))
    return _representative(
        left,
        right_h,
        projected,
        ecs_rank=k,
        method="blockwise_singular",
        operator_kind="midpoint_ecs_blockwise_singular_counterterm",
        map_definition=(
            "on [W] under O(m)xO(n), partition ordered ECS modes into fixed "
            "contiguous blocks, apply s_i -> s_i-mu_b, then the declared "
            "non-increasing isotonic projection"
        ),
        parameters={
            "ecs_rank": k,
            "block_count": blocks,
            "shift_fraction": fraction,
            "block_boundaries_zero_based_half_open": tuple(boundaries),
            "block_shifts": tuple(shifts),
            "isotonic_correction_norm": correction,
        },
    )


def _smaller_gram_and_anchor_basis(
    weight: FloatArray,
    anchor: FloatArray,
) -> tuple[FloatArray, FloatArray, str]:
    if weight.shape != anchor.shape:
        raise ValueError("weight and anchor_weight must have the same shape")
    anchor_left, _, anchor_right_h = _svd(anchor)
    rows, columns = weight.shape
    if rows <= columns:
        gram = weight @ weight.T
        basis = anchor_left
        side = "left_W_WT"
    else:
        gram = weight.T @ weight
        basis = anchor_right_h.T
        side = "right_WT_W"
    return 0.5 * (gram + gram.T), basis, side


def feshbach_downfolding_quotient(
    weight: ArrayLike,
    anchor_weight: ArrayLike,
    *,
    ecs_rank: int,
    regularization_ratio: float,
    minimum_anchor_gap_ratio: float = 1.0e-6,
    maximum_condition_number: float = 1.0e12,
) -> WeightQuotientResult:
    """Downfold in a midpoint subspace frozen from an independent anchor.

    The declared reference energy is ``z=-ridge``.  Consequently
    ``H_eff=A-B(C+ridge I)^-1 B^T`` is the regularized Schur/Feshbach form.
    The linear solve is rejected when its declared conditioning bound fails.
    """

    matrix = _matrix(weight)
    anchor = _matrix(anchor_weight, name="anchor_weight")
    left, singular, right_h = _svd(matrix)
    _, anchor_singular, _ = _svd(anchor)
    k = _rank(ecs_rank, maximum=singular.size)
    ratio = float(regularization_ratio)
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("regularization_ratio must be finite and non-negative")
    bound = float(maximum_condition_number)
    if not np.isfinite(bound) or bound <= 1.0:
        raise ValueError("maximum_condition_number must exceed one")
    minimum_gap = float(minimum_anchor_gap_ratio)
    if not np.isfinite(minimum_gap) or minimum_gap < 0.0:
        raise ValueError("minimum_anchor_gap_ratio must be finite and non-negative")
    if k < anchor_singular.size:
        anchor_gap = float(anchor_singular[k - 1] - anchor_singular[k])
        anchor_gap_ratio = anchor_gap / max(
            float(anchor_singular[k - 1]), np.finfo(np.float64).tiny
        )
        if anchor_gap_ratio < minimum_gap:
            raise WeightQuotientUnavailable(
                "anchor midpoint boundary is spectrally degenerate: "
                f"relative_gap={anchor_gap_ratio:.3e} < {minimum_gap:.3e}"
            )
    else:
        anchor_gap = None
        anchor_gap_ratio = None
    gram, basis, side = _smaller_gram_and_anchor_basis(matrix, anchor)
    rotated = basis.T @ gram @ basis
    rotated = 0.5 * (rotated + rotated.T)
    first = rotated[:k, :k]
    coupling = rotated[:k, k:]
    shell = rotated[k:, k:]
    boundary_scale = float(anchor_singular[k - 1] ** 2)
    ridge = ratio * boundary_scale
    if shell.size:
        shifted = shell + ridge * np.eye(shell.shape[0])
        condition = float(np.linalg.cond(shifted))
        if not np.isfinite(condition) or condition > bound:
            raise WeightQuotientUnavailable(
                f"Feshbach shell condition number {condition:.3e} exceeds {bound:.3e}"
            )
        solved = np.linalg.solve(shifted, coupling.T)
        residual = float(
            np.linalg.norm(shifted @ solved - coupling.T)
            / max(np.linalg.norm(coupling.T), np.finfo(np.float64).tiny)
        )
        effective_raw = first - coupling @ solved
    else:
        condition = 1.0
        residual = 0.0
        effective_raw = first
    hermiticity_residual = float(
        np.linalg.norm(effective_raw - effective_raw.T)
        / max(np.linalg.norm(effective_raw), np.finfo(np.float64).tiny)
    )
    effective = 0.5 * (effective_raw + effective_raw.T)
    eigenvalues = np.linalg.eigvalsh(effective)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    tolerance = max(1, k) * np.finfo(np.float64).eps * scale
    if float(np.min(eigenvalues, initial=0.0)) < -128.0 * tolerance:
        raise WeightQuotientUnavailable(
            "Feshbach effective Gram is materially indefinite"
        )
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
    transformed = np.sqrt(eigenvalues)
    return _representative(
        left,
        right_h,
        transformed,
        ecs_rank=k,
        method="feshbach_downfolding",
        operator_kind="anchor_frozen_midpoint_feshbach_effective_gram",
        map_definition=(
            "anchor-frozen P: H_eff(z)=PHP+PHQ(z-QHQ)^-1QHP at "
            "z=-regularization; retain its spectrum and choose the rectangular-"
            "diagonal O(m)xO(n) canonical representative"
        ),
        parameters={
            "ecs_rank": k,
            "anchor_policy": "earliest_verified_tail_checkpoint_same_seed_layer",
            "smaller_gram_side": side,
            "regularization_ratio": ratio,
            "minimum_anchor_gap_ratio": minimum_gap,
            "anchor_boundary_gap": anchor_gap,
            "anchor_boundary_gap_ratio": anchor_gap_ratio,
            "anchor_boundary_gap_status": (
                "finite_shell_boundary" if k < anchor_singular.size
                else "no_shell_full_smaller_side_rank"
            ),
            "regularization": ridge,
            "reference_energy_z": -ridge,
            "shell_condition_number": condition,
            "linear_solve_relative_residual": residual,
            "coupling_frobenius_norm": float(np.linalg.norm(coupling)),
            "pre_symmetrization_relative_hermiticity_residual": (
                hermiticity_residual
            ),
        },
    )


def rectangular_d_transform_quotient(
    weight: ArrayLike,
    *,
    ecs_rank: int,
    minimum_noise_modes: int = 8,
    noise_bulk_fraction: float = 1.0,
    denominator_ridge: float = 1.0e-12,
    minimum_relative_separation: float = 1.0e-2,
) -> WeightQuotientResult:
    """Empirical rectangular D-transform deconvolution of retained spikes.

    This is the finite W-only implementation of the incoherent/free-noise
    hypothesis used by the notebooks.  It is a low-rank-spike approximation,
    not a claim to solve unrestricted full-rank rectangular free deconvolution.
    The discarded singular values supply the empirical nuisance law.
    """

    matrix = _matrix(weight)
    left, singular, right_h = _svd(matrix)
    k = _rank(ecs_rank, maximum=singular.size)
    minimum = int(minimum_noise_modes)
    if float(minimum_noise_modes) != float(minimum) or minimum < 2:
        raise ValueError("minimum_noise_modes must be an integer >= 2")
    discarded = singular[k:]
    bulk_fraction = _fraction(
        noise_bulk_fraction,
        name="noise_bulk_fraction",
        include_one=True,
    )
    if bulk_fraction <= 0.0:
        raise ValueError("noise_bulk_fraction must be positive")
    selected_count = int(np.ceil(bulk_fraction * discarded.size))
    noise = discarded[-selected_count:] if selected_count else discarded[:0]
    if noise.size < minimum:
        raise WeightQuotientUnavailable(
            f"D-transform requires at least {minimum} selected noise modes; "
            f"observed {noise.size} from {discarded.size} discarded modes"
        )
    ridge = float(denominator_ridge)
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("denominator_ridge must be finite and positive")
    separation = float(minimum_relative_separation)
    if not np.isfinite(separation) or separation < 0.0:
        raise ValueError(
            "minimum_relative_separation must be finite and non-negative"
        )
    discarded_edge = float(np.max(discarded))
    noise_edge = float(np.max(noise))
    retained_floor = float(singular[k - 1])
    relative_separation = (retained_floor - discarded_edge) / max(
        discarded_edge, np.finfo(np.float64).tiny
    )
    if relative_separation < separation:
        raise WeightQuotientUnavailable(
            "retained D-transform modes are not separated from the discarded "
            f"bulk: relative_gap={relative_separation:.3e} < {separation:.3e}"
        )
    aspect = min(matrix.shape) / max(matrix.shape)
    transformed: list[float] = []
    d_values: list[float] = []
    for observed in singular[:k]:
        denominators = observed**2 - noise**2
        scale = max(float(observed**2), np.finfo(np.float64).tiny)
        denominators = np.maximum(denominators, ridge * scale)
        phi = float(np.mean(observed / denominators))
        phi_tilde = float(aspect * phi + (1.0 - aspect) / observed)
        d_value = phi * phi_tilde
        if not np.isfinite(d_value) or d_value <= 0.0:
            raise WeightQuotientUnavailable(
                "empirical rectangular D-transform is non-positive"
            )
        latent = min(float(observed), float(1.0 / np.sqrt(d_value)))
        transformed.append(max(latent, 0.0))
        d_values.append(d_value)
    projected = _pava_nonincreasing(np.asarray(transformed, dtype=np.float64))
    return _representative(
        left,
        right_h,
        projected,
        ecs_rank=k,
        method="rectangular_d_transform",
        operator_kind="empirical_rectangular_D_transform_spike_deconvolution",
        map_definition=(
            "on [W] under O(m)xO(n), estimate the rectangular noise D-transform "
            "from a separated discarded bulk and map retained s_i to "
            "min(s_i,D(s_i)^-1/2)"
        ),
        parameters={
            "ecs_rank": k,
            "aspect_ratio_min_over_max": float(aspect),
            "discarded_mode_count": int(discarded.size),
            "noise_mode_count": int(noise.size),
            "minimum_noise_modes": minimum,
            "noise_bulk_fraction": bulk_fraction,
            "noise_bulk_window": "lowest_selected_fraction_of_discarded_modes",
            "denominator_ridge": ridge,
            "minimum_relative_separation": separation,
            "retained_to_noise_relative_gap": relative_separation,
            "noise_edge": noise_edge,
            "discarded_edge": discarded_edge,
            "d_transform_min": float(np.min(d_values)),
            "d_transform_median": float(np.median(d_values)),
            "d_transform_max": float(np.max(d_values)),
            "approximation_scope": "separated_spikes_plus_empirical_incoherent_noise",
        },
    )


def calibrated_mp_shrinker_quotient(
    weight: ArrayLike,
    *,
    ecs_rank: int,
    minimum_noise_modes: int = 8,
    noise_scale_multiplier: float = 1.0,
) -> WeightQuotientResult:
    """Analytic monotone shrinker calibrated from an MP-like noise edge.

    This supplies a small, reproducible calibrated-shrinker baseline without
    training on WeightWatcher alpha or fit quality.  It uses the optimal
    Frobenius white-noise shrinker after estimating its scale from the discarded
    spectrum.  Correlated Muon histories need a separately learned calibration.
    """

    matrix = _matrix(weight)
    left, singular, right_h = _svd(matrix)
    k = _rank(ecs_rank, maximum=singular.size)
    minimum = int(minimum_noise_modes)
    if float(minimum_noise_modes) != float(minimum) or minimum < 2:
        raise ValueError("minimum_noise_modes must be an integer >= 2")
    noise = singular[k:]
    if noise.size < minimum:
        raise WeightQuotientUnavailable(
            f"calibrated shrinker requires at least {minimum} discarded modes; "
            f"observed {noise.size}"
        )
    multiplier = float(noise_scale_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("noise_scale_multiplier must be finite and positive")
    aspect = min(matrix.shape) / max(matrix.shape)
    edge_factor = 1.0 + np.sqrt(aspect)
    noise_unit = multiplier * float(np.max(noise)) / edge_factor
    if not np.isfinite(noise_unit) or noise_unit <= 0.0:
        raise WeightQuotientUnavailable(
            "estimated white-noise singular scale is not positive"
        )
    normalized = singular[:k] / noise_unit
    radicand = (normalized**2 - aspect - 1.0) ** 2 - 4.0 * aspect
    shrunk_normalized = np.where(
        normalized > edge_factor,
        np.sqrt(np.maximum(radicand, 0.0)) / normalized,
        0.0,
    )
    transformed = noise_unit * shrunk_normalized
    transformed = _pava_nonincreasing(transformed)
    return _representative(
        left,
        right_h,
        transformed,
        ecs_rank=k,
        method="calibrated_mp_shrinker",
        operator_kind="discarded_bulk_calibrated_monotone_MP_shrinker",
        map_definition=(
            "on [W] under O(m)xO(n), estimate white-noise scale from the "
            "discarded edge and apply the analytic optimal Frobenius "
            "singular-value shrinker"
        ),
        parameters={
            "ecs_rank": k,
            "aspect_ratio_min_over_max": float(aspect),
            "noise_mode_count": int(noise.size),
            "minimum_noise_modes": minimum,
            "noise_scale_multiplier": multiplier,
            "noise_edge": float(np.max(noise)),
            "estimated_noise_unit": noise_unit,
            "normalized_bulk_edge": float(edge_factor),
            "calibration_target": "discarded_MP_like_bulk_not_WeightWatcher_fit",
        },
    )


def apply_weight_quotient(
    method: str,
    weight: ArrayLike,
    *,
    ecs_rank: int,
    anchor_weight: ArrayLike | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> WeightQuotientResult:
    """Dispatch one declared quotient hypothesis with explicit parameters."""

    options = dict(parameters or {})
    selected = str(method)
    if selected == "midpoint_ecs_control":
        return ecs_truncation(weight, ecs_rank=ecs_rank)
    if selected == "uniform_singular_translation":
        return uniform_singular_translation(weight, ecs_rank=ecs_rank, **options)
    if selected == "gram_ridge":
        return gram_ridge_quotient(weight, ecs_rank=ecs_rank, **options)
    if selected == "blockwise_singular":
        return blockwise_singular_quotient(weight, ecs_rank=ecs_rank, **options)
    if selected == "feshbach_downfolding":
        if anchor_weight is None:
            raise ValueError("feshbach_downfolding requires anchor_weight")
        return feshbach_downfolding_quotient(
            weight, anchor_weight, ecs_rank=ecs_rank, **options
        )
    if selected == "rectangular_d_transform":
        return rectangular_d_transform_quotient(weight, ecs_rank=ecs_rank, **options)
    if selected == "calibrated_mp_shrinker":
        return calibrated_mp_shrinker_quotient(weight, ecs_rank=ecs_rank, **options)
    raise ValueError(f"unknown weight quotient method: {selected!r}")


__all__ = [
    "WeightQuotientResult",
    "WeightQuotientUnavailable",
    "apply_weight_quotient",
    "blockwise_singular_quotient",
    "calibrated_mp_shrinker_quotient",
    "ecs_truncation",
    "feshbach_downfolding_quotient",
    "gram_ridge_quotient",
    "midpoint_ecs_rank",
    "rectangular_d_transform_quotient",
    "uniform_singular_translation",
]

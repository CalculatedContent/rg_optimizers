"""Adaptive trace-log geometry and one-sided RG component removal.

The discrete ECS is selected by the self-consistent bulk-effective scan in
:mod:`rg_sc_trace_log.ecs`.  During one local optimizer correction, the ECS
membership is held fixed.  This is the same local-linearization convention as
the original trace-log tracker; the difference is that the retained support
and normalization dimension now come from the adaptive ECS construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

from .ecs import (
    EffectiveRankMethod,
    SelfConsistentECS,
    SupportPolicy,
    bulk_effective_normalization_dimension,
    detx_first_below_from_dimension,
    detx_nearest_from_dimension,
    solve_self_consistent_ecs,
    working_support_rank,
)

CorrectionMode = Literal["tangent", "one_sided", "tracking"]
NormalizationResponse = Literal["frozen", "differentiated"]


@dataclass
class AdaptiveTraceLogGeometry:
    """Local adaptive trace-log coordinate and its weight-space normal."""

    residual: torch.Tensor
    gradient: torch.Tensor
    ecs: SelfConsistentECS
    working_rank: int
    pl_rank: Optional[int]
    support_policy: SupportPolicy
    transposed: bool
    positive_count: int
    smallest_retained_singular_value: float
    largest_retained_singular_value: float
    gradient_norm_sq: float
    radial_inner_product: float
    retained_gradient_norm_sq: float
    radial_gradient_norm_sq: float
    normalization_gradient_norm_sq: float
    normalization_response: NormalizationResponse
    normalization_dimension_is_cached: bool


@dataclass
class CorrectionResult:
    """Result of filtering one completed matrix displacement."""

    corrected_delta: torch.Tensor
    correction: torch.Tensor
    coefficient: float
    base_drift: float
    corrected_drift: float
    predicted_residual_after: float
    correction_ratio: float
    capped: bool
    applied: bool
    reason: str


def _orient_tall(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return a matrix with rows >= columns without changing Frobenius geometry."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(weight.shape)}.")
    if weight.shape[0] >= weight.shape[1]:
        return weight, False
    return weight.transpose(0, 1), True


def _svd_with_cpu_fallback(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin SVD with a CPU fallback for backends lacking stable SVD support."""
    try:
        return torch.linalg.svd(weight, full_matrices=False)
    except RuntimeError:
        cpu = weight.detach().to(device="cpu", dtype=torch.float32)
        u, s, vh = torch.linalg.svd(cpu, full_matrices=False)
        return (
            u.to(device=weight.device, dtype=weight.dtype),
            s.to(device=weight.device, dtype=weight.dtype),
            vh.to(device=weight.device, dtype=weight.dtype),
        )


def _effective_count_and_singular_gradient(
    singular_values: torch.Tensor,
    *,
    method: EffectiveRankMethod,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return effective count and derivative with respect to singular values.

    The derivative is evaluated on a fixed discarded-bulk subspace.  The
    discrete ECS boundary itself is not differentiated during one local step.
    """
    if singular_values.numel() == 0:
        zero = singular_values.new_zeros(())
        return zero, torch.zeros_like(singular_values)

    s = singular_values
    lam = s.square()
    total = lam.sum().clamp_min(float(eps))

    if method == "participation_ratio":
        second = lam.square().sum().clamp_min(float(eps))
        count = total.square() / second
        dcount_dlambda = (
            2.0 * total / second
            - 2.0 * total.square() * lam / second.square()
        )
        return count, 2.0 * s * dcount_dlambda

    if method == "entropy":
        lam_safe = lam.clamp_min(float(eps))
        moment = torch.sum(lam * torch.log(lam_safe))
        entropy = torch.log(total) - moment / total
        count = torch.exp(entropy)
        dentropy_dlambda = (moment / total - torch.log(lam_safe)) / total
        return count, 2.0 * s * count * dentropy_dlambda

    if method == "stable_rank":
        maximum, max_index = torch.max(lam, dim=0)
        maximum = maximum.clamp_min(float(eps))
        count = total / maximum
        derivative_lambda = torch.full_like(lam, 1.0 / maximum)
        derivative_lambda[max_index] = (maximum - total) / maximum.square()
        return count, 2.0 * s * derivative_lambda

    raise ValueError(f"Unknown effective-rank method: {method!r}")


def _fixed_ecs_result(
    eigenvalues_ascending: np.ndarray,
    *,
    rank: int,
    normalization_dimension: float,
    method: EffectiveRankMethod,
    gamma: float,
    reference_rank: Optional[int],
    positive_floor: float,
    status: str,
) -> SelfConsistentECS:
    """Build a complete ECS record for a cached discrete support and scale."""
    values = np.asarray(eigenvalues_ascending, dtype=float)
    spectral_sum = float(np.sum(values))
    retained_desc = values[::-1][:rank]
    trace_log = float(
        np.sum(np.log(retained_desc))
        + rank * np.log(float(normalization_dimension) / spectral_sum)
    )
    scale_squared = float(normalization_dimension) / spectral_sum
    nearest = detx_nearest_from_dimension(
        values,
        normalization_dimension,
        positive_floor=positive_floor,
    )
    first_below = detx_first_below_from_dimension(
        values,
        normalization_dimension,
        positive_floor=positive_floor,
    )
    norm = bulk_effective_normalization_dimension(
        values,
        rank,
        method=method,
        gamma=gamma,
        positive_floor=positive_floor,
    )
    return SelfConsistentECS(
        ecs_rank=int(rank),
        fractional_rank=float(rank),
        normalization_dimension=float(normalization_dimension),
        bulk_count=int(norm["bulk_count"]),
        bulk_effective_count=float(norm["bulk_effective_count"]),
        bulk_effective_fraction=float(norm["bulk_effective_fraction"]),
        trace_log=trace_log,
        trace_log_per_eval=trace_log / float(rank),
        lambda_cut_raw=float(retained_desc[-1]),
        lambda_cut_scaled=float(retained_desc[-1] * scale_squared),
        detx_nearest_at_selected_scale=int(nearest["m"]),
        detx_first_below_at_selected_scale=int(first_below["m"]),
        fixed_point_error_nearest=int(nearest["m"] - rank),
        fixed_point_error_first_below=int(first_below["m"] - rank),
        num_sign_change_brackets=0,
        status=status,
        method=method,
        gamma=float(gamma),
        positive_count=int(values.size),
        reference_rank=(int(reference_rank) if reference_rank is not None else None),
    )


@torch.no_grad()
def adaptive_trace_log_geometry(
    weight: torch.Tensor,
    *,
    fixed_ecs_rank: Optional[int] = None,
    fixed_normalization_dimension: Optional[float] = None,
    reference_ecs_rank: Optional[int] = None,
    pl_rank: Optional[int] = None,
    support_policy: SupportPolicy = "midpoint",
    effective_rank_method: EffectiveRankMethod = "participation_ratio",
    normalization_gamma: float = 0.0,
    normalization_response: NormalizationResponse = "frozen",
    min_ecs_size: int = 2,
    min_retained: int = 3,
    require_sign_change: bool = False,
    ridge_relative: float = 1e-6,
    positive_eigenvalue_floor: float = 0.0,
    eps: float = 1e-12,
) -> AdaptiveTraceLogGeometry:
    """Compute the adaptive trace-log coordinate and local normal.

    There are two intended operating modes.

    * **Outer-loop mode (default):** ``fixed_ecs_rank`` and
      ``fixed_normalization_dimension`` come from a WeightWatcher checkpoint.
      They are held fixed during the local correction.  This is the closest
      analogue of the original trace-log tracker, with only the ECS estimator
      replaced.
    * **Live mode:** omit ``fixed_ecs_rank`` to solve the self-consistent ECS
      from the current SVD.  When ``normalization_response='differentiated'``,
      the normal additionally contains ``grad log D(W)`` for the adaptive
      bulk-effective normalization.

    In both modes, discrete ECS membership is held fixed while taking the
    differential.  It may be refreshed before a later optimizer step.
    """
    if normalization_response not in {"frozen", "differentiated"}:
        raise ValueError(
            "normalization_response must be 'frozen' or 'differentiated'."
        )
    if not 0.0 <= float(normalization_gamma) <= 1.0:
        raise ValueError("normalization_gamma must lie in [0, 1].")
    if ridge_relative < 0.0:
        raise ValueError("ridge_relative must be non-negative.")
    if min_retained < 1:
        raise ValueError("min_retained must be positive.")
    if fixed_normalization_dimension is not None and fixed_ecs_rank is None:
        raise ValueError(
            "fixed_normalization_dimension requires fixed_ecs_rank."
        )

    original_dtype = weight.dtype
    work, was_transposed = _orient_tall(weight.detach())
    compute = work.float() if work.dtype in {torch.float16, torch.bfloat16} else work
    u, singular_values, vh = _svd_with_cpu_fallback(compute)

    if singular_values.numel() < 2:
        raise ValueError("The matrix has fewer than two singular values.")

    eigenvalues = singular_values.square()
    positive_mask = eigenvalues > float(positive_eigenvalue_floor)
    positive_count = int(torch.count_nonzero(positive_mask).item())
    if positive_count < max(2, int(min_ecs_size)):
        raise ValueError(
            f"Only {positive_count} positive eigenvalues are available."
        )

    # torch.linalg.svd returns descending singular values.
    s_positive = singular_values[:positive_count]
    eigenvalues_ascending = (
        s_positive.square().detach().cpu().double().numpy()[::-1].copy()
    )

    normalization_is_cached = fixed_normalization_dimension is not None
    if fixed_ecs_rank is None:
        ecs = solve_self_consistent_ecs(
            eigenvalues_ascending,
            method=effective_rank_method,
            gamma=float(normalization_gamma),
            min_ecs_size=int(min_ecs_size),
            reference_rank=reference_ecs_rank,
            positive_floor=float(positive_eigenvalue_floor),
            numeric_eps=float(eps),
        )
    else:
        rank = int(np.clip(int(fixed_ecs_rank), int(min_ecs_size), positive_count))
        if fixed_normalization_dimension is None:
            norm = bulk_effective_normalization_dimension(
                eigenvalues_ascending,
                rank,
                method=effective_rank_method,
                gamma=float(normalization_gamma),
                positive_floor=float(positive_eigenvalue_floor),
            )
            dimension = float(norm["normalization_dimension"])
            status = "fixed_rank_live_normalization"
        else:
            dimension = float(fixed_normalization_dimension)
            if not np.isfinite(dimension) or dimension <= 0.0:
                raise ValueError(
                    "fixed_normalization_dimension must be positive and finite."
                )
            dimension = float(np.clip(dimension, rank, positive_count))
            status = "cached_outer_loop"
        ecs = _fixed_ecs_result(
            eigenvalues_ascending,
            rank=rank,
            normalization_dimension=dimension,
            method=effective_rank_method,
            gamma=float(normalization_gamma),
            reference_rank=reference_ecs_rank,
            positive_floor=float(positive_eigenvalue_floor),
            status=status,
        )

    if require_sign_change and ecs.status not in {"sign_change", "cached_outer_loop"}:
        raise ValueError(
            "The self-consistent ECS scan did not bracket a trace-log zero."
        )

    clipped_pl_rank = (
        int(np.clip(int(pl_rank), 1, positive_count))
        if pl_rank is not None
        else None
    )
    working_rank = working_support_rank(
        ecs_rank=ecs.ecs_rank,
        pl_rank=clipped_pl_rank,
        policy=support_policy,
        minimum=int(min_retained),
        maximum=positive_count,
    )

    largest = torch.clamp(singular_values[0], min=float(eps))
    singular_floor = torch.clamp(
        largest * float(ridge_relative) ** 0.5,
        min=float(eps),
    )
    s_retained = singular_values[:working_rank]
    s_safe = torch.clamp(s_retained, min=singular_floor)
    u_retained = u[:, :working_rank]
    vh_retained = vh[:working_rank, :]

    frob_sq = torch.sum(singular_values.square()).clamp_min(float(eps))
    dimension_tensor = frob_sq.new_tensor(float(ecs.normalization_dimension))
    normalized_retained = (
        dimension_tensor * s_safe.square() / frob_sq
    ).clamp_min(float(eps))
    residual = torch.mean(torch.log(normalized_retained))

    retained_gradient = (2.0 / float(working_rank)) * (
        (u_retained * (1.0 / s_safe).unsqueeze(0)) @ vh_retained
    )
    radial_gradient = -(2.0 / frob_sq) * compute
    normalization_gradient = torch.zeros_like(compute)

    # A cached outer-loop D is a gauge convention and is intentionally frozen.
    differentiate_dimension = (
        normalization_response == "differentiated"
        and not normalization_is_cached
        and ecs.bulk_count > 0
    )
    if differentiate_dimension:
        bulk_start = int(ecs.ecs_rank)
        bulk_stop = positive_count
        s_bulk = singular_values[bulk_start:bulk_stop]
        u_bulk = u[:, bulk_start:bulk_stop]
        vh_bulk = vh[bulk_start:bulk_stop, :]
        _, dcount_ds = _effective_count_and_singular_gradient(
            s_bulk,
            method=effective_rank_method,
            eps=float(eps),
        )
        coefficient = (
            (1.0 - float(normalization_gamma))
            / max(float(ecs.normalization_dimension), float(eps))
        )
        dlog_dimension_ds = coefficient * dcount_ds
        normalization_gradient = (
            u_bulk * dlog_dimension_ds.unsqueeze(0)
        ) @ vh_bulk

    gradient_work = retained_gradient + radial_gradient + normalization_gradient
    gradient = (
        gradient_work.transpose(0, 1) if was_transposed else gradient_work
    ).to(dtype=original_dtype)

    gradient_float = gradient.float()
    gradient_norm_sq = torch.sum(gradient_float.square())
    radial_inner = torch.sum(gradient_float * weight.detach().float())

    return AdaptiveTraceLogGeometry(
        residual=residual.to(dtype=original_dtype),
        gradient=gradient,
        ecs=ecs,
        working_rank=working_rank,
        pl_rank=clipped_pl_rank,
        support_policy=support_policy,
        transposed=was_transposed,
        positive_count=positive_count,
        smallest_retained_singular_value=float(s_retained[-1].detach().cpu()),
        largest_retained_singular_value=float(s_retained[0].detach().cpu()),
        gradient_norm_sq=float(gradient_norm_sq.detach().cpu()),
        radial_inner_product=float(radial_inner.detach().cpu()),
        retained_gradient_norm_sq=float(
            torch.sum(retained_gradient.float().square()).detach().cpu()
        ),
        radial_gradient_norm_sq=float(
            torch.sum(radial_gradient.float().square()).detach().cpu()
        ),
        normalization_gradient_norm_sq=float(
            torch.sum(normalization_gradient.float().square()).detach().cpu()
        ),
        normalization_response=normalization_response,
        normalization_dimension_is_cached=normalization_is_cached,
    )


@torch.no_grad()
def correct_trace_log_component(
    delta_base: torch.Tensor,
    geometry: AdaptiveTraceLogGeometry,
    *,
    mode: CorrectionMode = "one_sided",
    tracking_gamma: float = 0.10,
    correction_scale: float = 1.0,
    max_correction_ratio: Optional[float] = 0.25,
    eps: float = 1e-12,
) -> CorrectionResult:
    """Filter the adaptive trace-log-normal component of a completed step.

    ``one_sided`` is the branch-protection mode.  It removes only negative
    first-order drift of the retained log-volume and leaves expansion intact.
    It therefore does not assume that an intermediate layer already lies on
    the trace-log target surface.
    """
    if mode not in {"tangent", "one_sided", "tracking"}:
        raise ValueError(f"Unknown correction mode: {mode!r}")
    if not 0.0 <= float(tracking_gamma) <= 1.0:
        raise ValueError("tracking_gamma must lie in [0, 1].")
    if not 0.0 <= float(correction_scale) <= 1.0:
        raise ValueError("correction_scale must lie in [0, 1].")

    grad = geometry.gradient.to(device=delta_base.device, dtype=delta_base.dtype)
    grad_norm_sq = torch.sum(grad.float().square())
    base_drift_tensor = torch.sum(grad.float() * delta_base.float())
    grad_norm_sq_value = float(grad_norm_sq.detach().cpu())
    base_drift = float(base_drift_tensor.detach().cpu())
    residual = float(geometry.residual.detach().float().cpu())

    if not torch.isfinite(grad_norm_sq) or grad_norm_sq_value <= float(eps):
        zero = torch.zeros_like(delta_base)
        return CorrectionResult(
            corrected_delta=delta_base,
            correction=zero,
            coefficient=0.0,
            base_drift=base_drift,
            corrected_drift=base_drift,
            predicted_residual_after=residual + base_drift,
            correction_ratio=0.0,
            capped=False,
            applied=False,
            reason="adaptive trace-log gradient is numerically singular",
        )

    if mode == "tangent":
        coefficient = base_drift / grad_norm_sq_value
    elif mode == "one_sided":
        coefficient = min(base_drift / grad_norm_sq_value, 0.0)
    else:
        coefficient = (
            base_drift + float(tracking_gamma) * residual
        ) / grad_norm_sq_value

    correction = -float(coefficient) * grad
    correction = float(correction_scale) * correction

    delta_norm = torch.linalg.vector_norm(delta_base.float())
    correction_norm = torch.linalg.vector_norm(correction.float())
    capped = False
    if max_correction_ratio is not None and float(max_correction_ratio) >= 0.0:
        allowed = float(max_correction_ratio) * float(delta_norm.detach().cpu())
        current = float(correction_norm.detach().cpu())
        if current > allowed and current > float(eps):
            correction = correction * (allowed / current)
            correction_norm = torch.linalg.vector_norm(correction.float())
            capped = True

    corrected = delta_base + correction
    corrected_drift_tensor = torch.sum(grad.float() * corrected.float())
    corrected_drift = float(corrected_drift_tensor.detach().cpu())
    delta_norm_value = float(delta_norm.detach().cpu())
    correction_ratio = (
        float(correction_norm.detach().cpu()) / delta_norm_value
        if delta_norm_value > float(eps)
        else 0.0
    )
    applied = bool(float(correction_norm.detach().cpu()) > float(eps))

    if applied:
        reason = "applied"
    elif mode == "one_sided":
        reason = "no contracting component"
    else:
        reason = "no removable component"

    return CorrectionResult(
        corrected_delta=corrected,
        correction=correction,
        coefficient=float(coefficient),
        base_drift=base_drift,
        corrected_drift=corrected_drift,
        predicted_residual_after=residual + corrected_drift,
        correction_ratio=correction_ratio,
        capped=capped,
        applied=applied,
        reason=reason,
    )

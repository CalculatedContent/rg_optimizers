"""Trace-log residuals, gradients, and component corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

CorrectionMode = Literal["tangent", "one_sided", "tracking"]
NormalizationMode = Literal["weightwatcher", "raw"]


@dataclass
class TraceLogGeometry:
    """Local trace-log residual and normal direction for one matrix."""

    residual: torch.Tensor
    gradient: torch.Tensor
    retained_rank: int
    transposed: bool
    smallest_retained_singular_value: float
    largest_retained_singular_value: float
    gradient_norm_sq: float
    radial_inner_product: float


@dataclass
class CorrectionResult:
    """Result of filtering one proposed matrix displacement."""

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
    """Return a view with N >= M, preserving Frobenius geometry."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(weight.shape)}")
    if weight.shape[0] >= weight.shape[1]:
        return weight, False
    return weight.transpose(0, 1), True


def _svd_with_cpu_fallback(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a thin SVD, falling back to CPU when a backend lacks support."""
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


@torch.no_grad()
def trace_log_geometry(
    weight: torch.Tensor,
    retained_rank: int,
    *,
    normalization: NormalizationMode = "weightwatcher",
    ridge_relative: float = 1e-6,
    eps: float = 1e-12,
) -> TraceLogGeometry:
    """Compute the local trace-log residual and its weight-space gradient.

    The returned residual is the trace-log *per retained eigenvalue*. Scaling
    both the residual and gradient by the same constant does not change the
    projection or tracking correction, while the per-mode convention improves
    comparability across layers.

    For ``normalization="raw"``::

        T = (1/m) sum_i log(s_i^2 / N)
        grad T = (2/m) U_R diag(1/s_i) V_R^T

    For ``normalization="weightwatcher"``::

        T = (1/m) sum_i log(M s_i^2 / ||W||_F^2)
        grad T = (2/m) U_R diag(1/s_i) V_R^T
                 - 2 W / ||W||_F^2

    The second term removes global radial scale. With an exact inverse, its
    Frobenius inner product with W cancels that of the retained term.
    """
    if normalization not in {"weightwatcher", "raw"}:
        raise ValueError(f"Unknown normalization: {normalization!r}")
    if ridge_relative < 0.0:
        raise ValueError("ridge_relative must be non-negative.")

    original_dtype = weight.dtype
    work, was_transposed = _orient_tall(weight.detach())

    compute = work.float() if work.dtype in {torch.float16, torch.bfloat16} else work
    u, singular_values, vh = _svd_with_cpu_fallback(compute)

    max_rank = int(singular_values.numel())
    m = int(max(1, min(int(retained_rank), max_rank)))

    s_all = singular_values
    s = s_all[:m]
    u_r = u[:, :m]
    vh_r = vh[:m, :]

    largest = torch.clamp(s_all[0], min=float(eps))
    singular_floor = torch.clamp(
        largest * float(ridge_relative) ** 0.5,
        min=float(eps),
    )
    s_safe = torch.clamp(s, min=singular_floor)

    n_rows = int(compute.shape[0])
    n_cols = int(compute.shape[1])
    frob_sq = torch.sum(s_all.square()).clamp_min(float(eps))

    if normalization == "raw":
        retained_eigenvalues = (s_safe.square() / float(n_rows)).clamp_min(float(eps))
    else:
        retained_eigenvalues = (
            float(n_cols) * s_safe.square() / frob_sq
        ).clamp_min(float(eps))

    residual = torch.mean(torch.log(retained_eigenvalues))

    inverse_s = 1.0 / s_safe
    retained_gradient = (2.0 / float(m)) * ((u_r * inverse_s.unsqueeze(0)) @ vh_r)

    if normalization == "weightwatcher":
        gradient_work = retained_gradient - (2.0 / frob_sq) * compute
    else:
        gradient_work = retained_gradient

    gradient = gradient_work.transpose(0, 1) if was_transposed else gradient_work
    gradient = gradient.to(dtype=original_dtype)

    gradient_norm_sq_tensor = torch.sum(gradient.float().square())
    radial_inner = torch.sum(gradient.float() * weight.detach().float())

    return TraceLogGeometry(
        residual=residual.to(dtype=original_dtype),
        gradient=gradient,
        retained_rank=m,
        transposed=was_transposed,
        smallest_retained_singular_value=float(s[-1].detach().cpu()),
        largest_retained_singular_value=float(s[0].detach().cpu()),
        gradient_norm_sq=float(gradient_norm_sq_tensor.detach().cpu()),
        radial_inner_product=float(radial_inner.detach().cpu()),
    )


@torch.no_grad()
def correct_trace_log_component(
    delta_base: torch.Tensor,
    geometry: TraceLogGeometry,
    *,
    mode: CorrectionMode,
    gamma: float = 0.10,
    correction_scale: float = 1.0,
    max_correction_ratio: Optional[float] = 0.25,
    eps: float = 1e-12,
) -> CorrectionResult:
    """Filter the trace-log-normal component of a completed optimizer step."""
    if mode not in {"tangent", "one_sided", "tracking"}:
        raise ValueError(f"Unknown correction mode: {mode!r}")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
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
            reason="trace-log gradient is numerically singular",
        )

    if mode == "tangent":
        coefficient = base_drift / grad_norm_sq_value
    elif mode == "one_sided":
        coefficient = min(base_drift / grad_norm_sq_value, 0.0)
    else:
        coefficient = (base_drift + float(gamma) * residual) / grad_norm_sq_value

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
    correction_ratio = float(
        correction_norm.detach().cpu() / delta_norm.detach().cpu()
    ) if float(delta_norm.detach().cpu()) > float(eps) else 0.0

    applied = bool(float(correction_norm.detach().cpu()) > float(eps))
    reason = "applied" if applied else "no removable component"

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

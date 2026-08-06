"""Spectral volume and shell-beta geometry for AdaptiveSpectralGuard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch


@dataclass
class SpectralGeometry:
    trace_residual: torch.Tensor
    trace_gradient: torch.Tensor
    volume_rank: int

    beta_E: torch.Tensor
    beta_gradient: torch.Tensor
    beta_reliable: bool
    shape_rank: int
    shells_used: int
    dynamic_range_decades: float

    trace_beta_inner_product_before: float
    trace_beta_inner_product_after: float

    smallest_retained_singular_value: float
    largest_retained_singular_value: float
    gradient_norm_sq_trace: float
    gradient_norm_sq_beta: float
    radial_inner_trace: float
    radial_inner_beta: float
    transposed: bool


def _orient_tall(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got {tuple(weight.shape)}")
    if weight.shape[0] >= weight.shape[1]:
        return weight, False
    return weight.transpose(0, 1), True


def _svd_with_cpu_fallback(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _cap_rank(rank: int, maximum: int, minimum: int = 1) -> int:
    return int(max(minimum, min(int(rank), int(maximum))))


def _empty_beta(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool, int, float]:
    return (
        torch.zeros((), device=weight.device, dtype=weight.dtype),
        torch.zeros_like(weight),
        False,
        0,
        math.nan,
    )


@torch.no_grad()
def spectral_geometry(
    weight: torch.Tensor,
    *,
    volume_rank: int,
    shape_rank: int,
    n_shells: int = 5,
    min_shape_retained: int = 20,
    min_shape_decades: float = 0.50,
    ridge_relative: float = 1e-6,
    eps: float = 1e-12,
) -> SpectralGeometry:
    """Compute volume and beta-E directions from one thin SVD.

    The beta direction uses frozen equal-width logarithmic shells. For shell
    energies E_k and shell centers x_k = log Lambda_k,

        beta_E = sum_k a_k log E_k,

    where a_k is the centered least-squares slope coefficient. Its gradient is
    then orthogonalized against the trace-log gradient so the shape channel is
    first-order volume preserving.
    """

    if ridge_relative < 0.0:
        raise ValueError("ridge_relative must be non-negative")
    if n_shells < 2:
        raise ValueError("n_shells must be at least two")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    original_dtype = weight.dtype
    work, transposed = _orient_tall(weight.detach())
    compute = (
        work.float()
        if work.dtype in {torch.float16, torch.bfloat16}
        else work
    )
    u, singular_values, vh = _svd_with_cpu_fallback(compute)
    if singular_values.numel() < 1:
        raise ValueError("No singular values available")

    max_rank = int(singular_values.numel())
    m_volume = _cap_rank(volume_rank, max_rank)
    m_shape = _cap_rank(shape_rank, max_rank)

    largest = singular_values[0].clamp_min(float(eps))
    singular_floor = torch.clamp(
        largest * float(ridge_relative) ** 0.5,
        min=float(eps),
    )
    s_all = singular_values
    s_volume = torch.clamp(s_all[:m_volume], min=singular_floor)
    u_volume = u[:, :m_volume]
    vh_volume = vh[:m_volume, :]

    frob_sq = torch.sum(s_all.square()).clamp_min(float(eps))
    n_cols = int(compute.shape[1])
    normalized_eigs = (
        float(n_cols) * s_volume.square() / frob_sq
    ).clamp_min(float(eps))
    trace_residual = torch.mean(torch.log(normalized_eigs))

    retained_gradient = (
        2.0
        / float(m_volume)
        * ((u_volume * (1.0 / s_volume).unsqueeze(0)) @ vh_volume)
    )
    trace_gradient_work = retained_gradient - (2.0 / frob_sq) * compute

    (
        beta_value,
        beta_gradient_work,
        beta_reliable,
        shells_used,
        dynamic_range,
    ) = _empty_beta(compute)

    if m_shape >= max(3, int(min_shape_retained)):
        s_shape = torch.clamp(s_all[:m_shape], min=singular_floor)
        eigenvalues = s_shape.square().clamp_min(float(eps))
        high = float(eigenvalues[0].detach().cpu())
        low = float(eigenvalues[-1].detach().cpu())
        if high > low > 0.0:
            dynamic_range = math.log10(high / low)
            if dynamic_range >= float(min_shape_decades):
                used = min(int(n_shells), m_shape)
                while used >= 2:
                    log_low = math.log(low)
                    log_high = math.log(high)
                    edges = torch.exp(
                        torch.linspace(
                            log_low,
                            log_high,
                            used + 1,
                            device=compute.device,
                            dtype=compute.dtype,
                        )
                    )
                    shell_ids = torch.bucketize(
                        eigenvalues,
                        edges[1:-1],
                        right=False,
                    )
                    counts = torch.bincount(
                        shell_ids,
                        minlength=used,
                    )
                    if bool(torch.all(counts > 0)):
                        break
                    used -= 1

                if used >= 3:
                    energies = torch.zeros(
                        used,
                        device=compute.device,
                        dtype=compute.dtype,
                    )
                    energies.scatter_add_(0, shell_ids, eigenvalues)
                    energies = energies.clamp_min(float(eps))

                    centers = torch.sqrt(edges[:-1] * edges[1:])
                    x = torch.log(centers)
                    x_centered = x - torch.mean(x)
                    denominator = torch.sum(x_centered.square()).clamp_min(
                        float(eps)
                    )
                    slope_coeff = x_centered / denominator
                    beta_value = torch.sum(
                        slope_coeff * torch.log(energies)
                    )

                    shell_coeff_for_value = slope_coeff[shell_ids]
                    energy_for_value = energies[shell_ids]
                    d_beta_d_s = (
                        2.0
                        * shell_coeff_for_value
                        * s_shape
                        / energy_for_value
                    )
                    u_shape = u[:, :m_shape]
                    vh_shape = vh[:m_shape, :]
                    beta_gradient_work = (
                        u_shape * d_beta_d_s.unsqueeze(0)
                    ) @ vh_shape
                    beta_reliable = True
                    shells_used = int(used)

    trace_norm_sq = torch.sum(trace_gradient_work.float().square())
    beta_before = torch.sum(
        trace_gradient_work.float() * beta_gradient_work.float()
    )
    if beta_reliable and float(trace_norm_sq.detach().cpu()) > float(eps):
        beta_gradient_work = beta_gradient_work - (
            beta_before / trace_norm_sq.to(beta_before.dtype)
        ) * trace_gradient_work
    beta_after = torch.sum(
        trace_gradient_work.float() * beta_gradient_work.float()
    )

    trace_gradient = (
        trace_gradient_work.transpose(0, 1)
        if transposed
        else trace_gradient_work
    ).to(dtype=original_dtype)
    beta_gradient = (
        beta_gradient_work.transpose(0, 1)
        if transposed
        else beta_gradient_work
    ).to(dtype=original_dtype)

    trace_norm_sq_final = torch.sum(trace_gradient.float().square())
    beta_norm_sq_final = torch.sum(beta_gradient.float().square())
    radial_trace = torch.sum(
        trace_gradient.float() * weight.detach().float()
    )
    radial_beta = torch.sum(
        beta_gradient.float() * weight.detach().float()
    )

    return SpectralGeometry(
        trace_residual=trace_residual.to(dtype=original_dtype),
        trace_gradient=trace_gradient,
        volume_rank=m_volume,
        beta_E=beta_value.to(dtype=original_dtype),
        beta_gradient=beta_gradient,
        beta_reliable=bool(beta_reliable),
        shape_rank=m_shape,
        shells_used=int(shells_used),
        dynamic_range_decades=float(dynamic_range),
        trace_beta_inner_product_before=float(beta_before.detach().cpu()),
        trace_beta_inner_product_after=float(beta_after.detach().cpu()),
        smallest_retained_singular_value=float(
            s_volume[-1].detach().cpu()
        ),
        largest_retained_singular_value=float(
            s_volume[0].detach().cpu()
        ),
        gradient_norm_sq_trace=float(trace_norm_sq_final.detach().cpu()),
        gradient_norm_sq_beta=float(beta_norm_sq_final.detach().cpu()),
        radial_inner_trace=float(radial_trace.detach().cpu()),
        radial_inner_beta=float(radial_beta.detach().cpu()),
        transposed=transposed,
    )


def relative_cap(
    correction: torch.Tensor,
    reference: torch.Tensor,
    max_ratio: Optional[float],
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float, bool]:
    """Cap ||correction|| / ||reference|| and return the final ratio."""

    correction_norm = torch.linalg.vector_norm(correction.float())
    reference_norm = torch.linalg.vector_norm(reference.float())
    reference_value = float(reference_norm.detach().cpu())
    correction_value = float(correction_norm.detach().cpu())

    if reference_value <= eps:
        return torch.zeros_like(correction), 0.0, correction_value > eps

    capped = False
    if max_ratio is not None:
        allowed = float(max_ratio) * reference_value
        if correction_value > allowed and correction_value > eps:
            correction = correction * (allowed / correction_value)
            correction_norm = torch.linalg.vector_norm(correction.float())
            correction_value = float(correction_norm.detach().cpu())
            capped = True

    return correction, correction_value / reference_value, capped


def loss_neutralize(
    correction: torch.Tensor,
    task_gradient: torch.Tensor,
    base_delta: torch.Tensor,
    *,
    allowed_conflict_ratio: float = 0.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Remove correction components that increase minibatch loss to first order.

    The allowed positive task inner product is expressed relative to the
    magnitude of the base optimizer's first-order descent.
    """

    grad = task_gradient.to(
        device=correction.device,
        dtype=correction.dtype,
    )
    grad_norm_sq = torch.sum(grad.float().square())
    base_task = torch.sum(grad.float() * base_delta.float())
    correction_task_pre = torch.sum(grad.float() * correction.float())
    denominator = abs(float(base_task.detach().cpu())) + float(eps)
    allowed = float(allowed_conflict_ratio) * denominator

    pre_value = float(correction_task_pre.detach().cpu())
    applied = False
    removed_fraction = 0.0
    before_norm = float(
        torch.linalg.vector_norm(correction.float()).detach().cpu()
    )

    if (
        float(grad_norm_sq.detach().cpu()) > float(eps)
        and pre_value > allowed
    ):
        correction = correction - (
            (correction_task_pre - allowed)
            / grad_norm_sq.to(correction_task_pre.dtype)
        ) * grad
        applied = True

    after_norm = float(
        torch.linalg.vector_norm(correction.float()).detach().cpu()
    )
    if before_norm > eps:
        removed_fraction = max(0.0, 1.0 - after_norm / before_norm)

    correction_task_post = torch.sum(grad.float() * correction.float())
    base_norm = torch.linalg.vector_norm(base_delta.float())
    correction_norm = torch.linalg.vector_norm(correction.float())
    cosine_base = float("nan")
    if (
        float(base_norm.detach().cpu()) > eps
        and float(correction_norm.detach().cpu()) > eps
    ):
        cosine_base = float(
            (
                torch.sum(base_delta.float() * correction.float())
                / (base_norm * correction_norm)
            )
            .detach()
            .cpu()
        )

    return correction, {
        "base_task_inner_product": float(base_task.detach().cpu()),
        "correction_task_inner_product_pre": pre_value,
        "correction_task_inner_product_post": float(
            correction_task_post.detach().cpu()
        ),
        "task_conflict_ratio_pre": pre_value / denominator,
        "task_conflict_ratio_post": float(
            correction_task_post.detach().cpu()
        )
        / denominator,
        "correction_base_cosine": cosine_base,
        "loss_neutral_applied": applied,
        "loss_neutral_removed_fraction": removed_fraction,
    }

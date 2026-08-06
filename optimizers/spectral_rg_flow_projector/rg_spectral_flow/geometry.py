"""Measure a completed optimizer step in spectral-shape coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .shape import (
    CollapsePotential,
    centered_log_eigenvalue_shape,
    collapse_potential_from_shape,
    effective_rank_from_shape,
    rank_alpha_proxy_from_shape,
    trivial_branch_flow_vector,
)


@dataclass
class SpectralFlowGeometry:
    working_rank: int
    collapse_potential: CollapsePotential
    shape_before: torch.Tensor
    shape_base: torch.Tensor
    spectral_displacement: torch.Tensor
    trivial_flow_vector: torch.Tensor
    flow_vector_norm_sq: float
    base_flow_component: float
    base_projection_coefficient: float
    base_alignment_cosine: float
    collapse_potential_before: float
    collapse_potential_base: float
    effective_rank_before: float
    effective_rank_base: float
    rank_alpha_proxy_before: float
    rank_alpha_proxy_base: float
    transposed: bool


def orient_tall(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(weight.shape)}.")
    if weight.shape[0] >= weight.shape[1]:
        return weight, False
    return weight.transpose(0, 1), True


def svd_with_cpu_fallback(
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


def _clip_rank(value: int, lower: int, upper: int) -> int:
    return max(int(lower), min(int(value), int(upper)))


@torch.no_grad()
def spectral_flow_geometry(
    weight_before: torch.Tensor,
    weight_base: torch.Tensor,
    retained_rank: int,
    *,
    potential: CollapsePotential = "participation_ratio",
    eps: float = 1e-12,
) -> tuple[
    SpectralFlowGeometry,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return shape-flow geometry and the base proposal SVD ``(U, s, Vh)``."""
    if weight_before.shape != weight_base.shape:
        raise ValueError("weight_before and weight_base must have the same shape.")

    before_tall, transposed_before = orient_tall(weight_before.detach())
    base_tall, transposed_base = orient_tall(weight_base.detach())
    if transposed_before != transposed_base:
        raise RuntimeError("Matrix orientation changed unexpectedly.")

    compute_before = (
        before_tall.float()
        if before_tall.dtype in {torch.float16, torch.bfloat16}
        else before_tall
    )
    compute_base = (
        base_tall.float()
        if base_tall.dtype in {torch.float16, torch.bfloat16}
        else base_tall
    )
    _, s_before, _ = svd_with_cpu_fallback(compute_before)
    u_base, s_base, vh_base = svd_with_cpu_fallback(compute_base)

    maximum_rank = min(s_before.numel(), s_base.numel())
    if maximum_rank < 2:
        raise ValueError("Need at least two singular values for spectral flow.")
    rank = _clip_rank(retained_rank, 2, maximum_rank)
    shape_before = centered_log_eigenvalue_shape(s_before, rank, eps=eps)
    shape_base = centered_log_eigenvalue_shape(s_base, rank, eps=eps)
    displacement = shape_base - shape_before
    flow_vector = trivial_branch_flow_vector(
        shape_before,
        potential=potential,
        eps=eps,
    )
    flow_norm_sq_tensor = torch.sum(flow_vector.square())
    flow_norm_sq = float(flow_norm_sq_tensor.detach().cpu())
    component_tensor = torch.sum(displacement * flow_vector)
    component = float(component_tensor.detach().cpu())
    coefficient = component / flow_norm_sq if flow_norm_sq > float(eps) else 0.0

    displacement_norm = torch.linalg.vector_norm(displacement)
    flow_norm = torch.sqrt(flow_norm_sq_tensor.clamp_min(float(eps)))
    denominator = float((displacement_norm * flow_norm).detach().cpu())
    cosine = component / denominator if denominator > float(eps) else 0.0

    potential_before = collapse_potential_from_shape(
        shape_before,
        potential=potential,
        eps=eps,
    )
    potential_base = collapse_potential_from_shape(
        shape_base,
        potential=potential,
        eps=eps,
    )
    rank_before = effective_rank_from_shape(shape_before, eps=eps)
    rank_base = effective_rank_from_shape(shape_base, eps=eps)

    geometry = SpectralFlowGeometry(
        working_rank=rank,
        collapse_potential=potential,
        shape_before=shape_before,
        shape_base=shape_base,
        spectral_displacement=displacement,
        trivial_flow_vector=flow_vector,
        flow_vector_norm_sq=flow_norm_sq,
        base_flow_component=component,
        base_projection_coefficient=coefficient,
        base_alignment_cosine=cosine,
        collapse_potential_before=float(potential_before.detach().cpu()),
        collapse_potential_base=float(potential_base.detach().cpu()),
        effective_rank_before=float(rank_before.detach().cpu()),
        effective_rank_base=float(rank_base.detach().cpu()),
        rank_alpha_proxy_before=rank_alpha_proxy_from_shape(shape_before, eps=eps),
        rank_alpha_proxy_base=rank_alpha_proxy_from_shape(shape_base, eps=eps),
        transposed=transposed_base,
    )
    return geometry, u_base, s_base, vh_base

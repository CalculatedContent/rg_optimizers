"""Finite one-sided projection away from the spectral collapse surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .geometry import (
    SpectralFlowGeometry,
    orient_tall,
    spectral_flow_geometry,
    svd_with_cpu_fallback,
)
from .shape import (
    CollapsePotential,
    centered_log_eigenvalue_shape,
    collapse_potential_from_shape,
    effective_rank_from_shape,
    rank_alpha_proxy_from_shape,
)


@dataclass
class SpectralFlowCorrection:
    corrected_weight: torch.Tensor
    correction: torch.Tensor
    corrected_shape: torch.Tensor
    corrected_spectral_displacement: torch.Tensor
    corrected_flow_component: float
    collapse_potential_corrected: float
    effective_rank_corrected: float
    rank_alpha_proxy_corrected: float
    projection_coefficient: float
    spectral_component_removed_norm: float
    correction_ratio: float
    capped: bool
    applied: bool
    reason: str
    geometry: SpectralFlowGeometry


def _unchanged(
    weight_base: torch.Tensor,
    geometry: SpectralFlowGeometry,
    reason: str,
) -> SpectralFlowCorrection:
    return SpectralFlowCorrection(
        corrected_weight=weight_base,
        correction=torch.zeros_like(weight_base),
        corrected_shape=geometry.shape_base,
        corrected_spectral_displacement=geometry.spectral_displacement,
        corrected_flow_component=geometry.base_flow_component,
        collapse_potential_corrected=geometry.collapse_potential_base,
        effective_rank_corrected=geometry.effective_rank_base,
        rank_alpha_proxy_corrected=geometry.rank_alpha_proxy_base,
        projection_coefficient=0.0,
        spectral_component_removed_norm=0.0,
        correction_ratio=0.0,
        capped=False,
        applied=False,
        reason=reason,
        geometry=geometry,
    )


@torch.no_grad()
def remove_trivial_branch_component(
    weight_before: torch.Tensor,
    weight_base: torch.Tensor,
    retained_rank: int,
    *,
    potential: CollapsePotential = "participation_ratio",
    projection_strength: float = 1.0,
    min_alignment_cosine: float = 0.0,
    max_abs_log_eigenvalue_correction: Optional[float] = 0.25,
    max_correction_ratio: Optional[float] = 0.25,
    preserve_frobenius_norm: bool = True,
    eps: float = 1e-12,
) -> SpectralFlowCorrection:
    """Remove only positive shape-flow projection toward the F0 surrogate."""
    if not 0.0 <= float(projection_strength) <= 1.0:
        raise ValueError("projection_strength must lie in [0, 1].")
    if not -1.0 <= float(min_alignment_cosine) <= 1.0:
        raise ValueError("min_alignment_cosine must lie in [-1, 1].")
    if max_abs_log_eigenvalue_correction is not None and (
        max_abs_log_eigenvalue_correction < 0.0
    ):
        raise ValueError(
            "max_abs_log_eigenvalue_correction must be non-negative or None."
        )
    if max_correction_ratio is not None and max_correction_ratio < 0.0:
        raise ValueError("max_correction_ratio must be non-negative or None.")

    geometry, u_base, s_base, vh_base = spectral_flow_geometry(
        weight_before,
        weight_base,
        retained_rank,
        potential=potential,
        eps=eps,
    )
    if geometry.flow_vector_norm_sq <= float(eps):
        return _unchanged(weight_base, geometry, "trivial-branch flow vector is numerically zero")
    if geometry.base_projection_coefficient <= 0.0:
        return _unchanged(weight_base, geometry, "base step does not flow toward the trivial branch")
    if geometry.base_alignment_cosine < float(min_alignment_cosine):
        return _unchanged(weight_base, geometry, "trivial-flow alignment is below the configured threshold")

    coefficient = float(projection_strength) * geometry.base_projection_coefficient
    delta_shape = -coefficient * geometry.trivial_flow_vector
    if max_abs_log_eigenvalue_correction is not None:
        limit = float(max_abs_log_eigenvalue_correction)
        delta_shape = torch.clamp(delta_shape, min=-limit, max=limit)
        delta_shape = delta_shape - torch.mean(delta_shape)

    rank = geometry.working_rank
    s_candidate = s_base.clone()
    s_candidate[:rank] = s_candidate[:rank] * torch.exp(0.5 * delta_shape)
    if preserve_frobenius_norm:
        original_norm = torch.linalg.vector_norm(s_base)
        candidate_norm = torch.linalg.vector_norm(s_candidate).clamp_min(float(eps))
        s_candidate = s_candidate * (original_norm / candidate_norm)

    _, was_transposed = orient_tall(weight_base.detach())
    corrected_tall = (u_base * s_candidate.unsqueeze(0)) @ vh_base
    corrected_weight = corrected_tall.transpose(0, 1) if was_transposed else corrected_tall
    corrected_weight = corrected_weight.to(dtype=weight_base.dtype)
    correction = corrected_weight - weight_base

    base_delta_norm = torch.linalg.vector_norm((weight_base - weight_before).float())
    correction_norm = torch.linalg.vector_norm(correction.float())
    capped = False
    if max_correction_ratio is not None:
        allowed = float(max_correction_ratio) * float(base_delta_norm.detach().cpu())
        current = float(correction_norm.detach().cpu())
        if current > allowed and current > float(eps):
            correction = correction * (allowed / current)
            corrected_weight = weight_base + correction
            correction_norm = torch.linalg.vector_norm(correction.float())
            capped = True

    correction_ratio = (
        float(correction_norm.detach().cpu()) / float(base_delta_norm.detach().cpu())
        if float(base_delta_norm.detach().cpu()) > float(eps)
        else 0.0
    )

    _, s_corrected, _ = svd_with_cpu_fallback(orient_tall(corrected_weight.detach())[0])
    corrected_shape = centered_log_eigenvalue_shape(s_corrected, rank, eps=eps)
    corrected_displacement = corrected_shape - geometry.shape_before
    corrected_component = float(
        torch.sum(corrected_displacement * geometry.trivial_flow_vector).detach().cpu()
    )
    potential_corrected = collapse_potential_from_shape(
        corrected_shape,
        potential=potential,
        eps=eps,
    )
    effective_rank_corrected = effective_rank_from_shape(corrected_shape, eps=eps)
    removed_norm = float(
        torch.linalg.vector_norm(
            geometry.spectral_displacement - corrected_displacement
        ).detach().cpu()
    )
    applied = bool(float(correction_norm.detach().cpu()) > float(eps))
    return SpectralFlowCorrection(
        corrected_weight=corrected_weight,
        correction=correction,
        corrected_shape=corrected_shape,
        corrected_spectral_displacement=corrected_displacement,
        corrected_flow_component=corrected_component,
        collapse_potential_corrected=float(potential_corrected.detach().cpu()),
        effective_rank_corrected=float(effective_rank_corrected.detach().cpu()),
        rank_alpha_proxy_corrected=rank_alpha_proxy_from_shape(corrected_shape, eps=eps),
        projection_coefficient=coefficient,
        spectral_component_removed_norm=removed_norm,
        correction_ratio=correction_ratio,
        capped=capped,
        applied=applied,
        reason="applied" if applied else "projection produced no finite correction",
        geometry=geometry,
    )

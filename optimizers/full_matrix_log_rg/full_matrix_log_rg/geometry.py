"""Full matrix-log geometry plus radial and legacy projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .support import MatrixLogSupport, orient_tall

CorrectionMode = Literal["radial", "modewise"]


@dataclass
class MatrixLogGeometry:
    potential: torch.Tensor
    log_eigenvalues: torch.Tensor
    gradient: torch.Tensor
    retained_rank: int
    normalization: str
    normalization_dimension: float
    gradient_norm_sq: float
    min_retained_singular_value: float
    max_retained_singular_value: float
    transposed: bool
    reference_work: torch.Tensor
    mode_left_vectors: torch.Tensor
    mode_right_vectors: torch.Tensor
    singular_values: torch.Tensor
    frobenius_norm_sq: torch.Tensor


@dataclass
class ProjectionResult:
    corrected_delta: torch.Tensor
    correction: torch.Tensor
    mode: str
    base_drift: float
    corrected_drift: float
    coefficient: float
    correction_ratio: float
    applied: bool
    capped: bool
    inward_mode_count: int
    base_inward_mode_norm: float
    corrected_inward_mode_norm: float


def _svd(matrix: torch.Tensor):
    try:
        return torch.linalg.svd(matrix, full_matrices=False)
    except (RuntimeError, NotImplementedError):
        cpu = matrix.detach().cpu().float()
        u, s, vh = torch.linalg.svd(cpu, full_matrices=False)
        return u.to(matrix), s.to(matrix), vh.to(matrix)


def _eigh(matrix: torch.Tensor):
    try:
        return torch.linalg.eigh(matrix)
    except (RuntimeError, NotImplementedError):
        values, vectors = torch.linalg.eigh(matrix.detach().cpu().double())
        return values.to(matrix), vectors.to(matrix)


def _basis_from_rank(work: torch.Tensor, retained_rank: int) -> torch.Tensor:
    _, _, vh = _svd(work)
    rank = int(max(1, min(int(retained_rank), int(vh.shape[0]))))
    return vh[:rank].transpose(0, 1).contiguous()


@torch.no_grad()
def full_matrix_log_geometry(
    weight: torch.Tensor,
    retained_rank: int | None = None,
    *,
    support: MatrixLogSupport | None = None,
    right_basis: torch.Tensor | None = None,
    normalization: str = "full_m",
    normalization_dimension: float | None = None,
    ridge_relative: float = 1e-6,
    eps: float = 1e-12,
) -> MatrixLogGeometry:
    if normalization not in {"full_m", "self_consistent"}:
        raise ValueError(f"Unknown normalization: {normalization!r}")
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got {tuple(weight.shape)}")
    original_dtype = weight.dtype
    work, transposed = orient_tall(weight.detach())
    compute = work.float() if work.dtype in {torch.float16, torch.bfloat16} else work

    if support is not None:
        if bool(support.transposed) != bool(transposed):
            raise ValueError("Cached support orientation does not match parameter")
        basis = support.right_basis
        retained_rank = int(support.retained_rank)
        if normalization_dimension is None:
            normalization_dimension = support.dimension(normalization)
    elif right_basis is not None:
        basis = right_basis
        retained_rank = int(retained_rank or right_basis.shape[1])
    else:
        if retained_rank is None:
            raise ValueError ("Provide support, right_basis, or retained_rank")
        basis = _basis_from_rank(compute, retained_rank)

    basis = torch.as_tensor(basis).detach().to(compute)
    if basis.ndim != 2 or int(basis.shape[0]) != int(compute.shape[1]):
        raise ValueError("Right basis has incompatible shape")
    rank = int(max(1, min(int(retained_rank), int(basis.shape[1]))))
    basis = basis[:, :rank]
    d = float(
        normalization_dimension
        if normalization_dimension is not None
        else compute.shape[1]
    )
    if d <= 0.0:
        raise ValueError("normalization dimension must be positive")

    projected = compute @ basis
    covariance = projected.transpose(0, 1) @ projected
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    eigenvalues, eigenvectors = _eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    largest = torch.clamp(eigenvalues[0], min=float(eps))
    floor = torch.clamp(largest * float(ridge_relative), min=float(eps))
    safe = torch.clamp(eigenvalues, min=floor)
    singular_values = torch.sqrt(safe)
    right = basis @ eigenvectors
    left = (projected @ eigenvectors) / singular_values.unsqueeze(0)

    frob_sq = torch.sum(compute.square()).clamp_min(float(eps))
    normalized = (d * safe / frob_sq).clamp_min(float(eps))
    logs = torch.log(normalized)
    potential = 0.5 * torch.mean(logs.square())
    retained_gradient = (2.0 / float(rank)) * (
        (left * (logs / singular_values).unsqueeze(0)) @ right.transpose(0, 1)
    )
    radial_gradient = (2.0 * torch.mean(logs) / frob_sq) * compute
    gradient_work = retained_gradient - radial_gradient
    gradient = gradient_work.transpose(0, 1) if transposed else gradient_work
    gradient = gradient.to(dtype=original_dtype)

    return MatrixLogGeometry(
        potential=potential.to(dtype=original_dtype),
        log_eigenvalues=logs.to(dtype=original_dtype),
        gradient=gradient,
        retained_rank=rank,
        normalization=normalization,
        normalization_dimension=d,
        gradient_norm_sq=float(torch.sum(gradient.float().square()).cpu()),
        min_retained_singular_value=float(singular_values[-1].cpu()),
        max_retained_singular_value=float(singular_values[0].cpu()),
        transposed=transposed,
        reference_work=compute,
        mode_left_vectors=left,
        mode_right_vectors=right,
        singular_values=singular_values,
        frobenius_norm_sq=frob_sq,
    )


def mode_drifts(delta: torch.Tensor, geometry: MatrixLogGeometry) -> torch.Tensor:
    work = delta.transpose(0, 1) if geometry.transposed else delta
    work = work.to(geometry.reference_work)
    mode_delta = torch.sum(
        (work @ geometry.mode_right_vectors) * geometry.mode_left_vectors,
        dim=0,
    )
    radial_delta = (
        2.0
        * torch.sum(geometry.reference_work * work)
        / geometry.frobenius_norm_sq
    )
    return 2.0 * mode_delta / geometry.singular_values - radial_delta


def mode_gram(geometry: MatrixLogGeometry) -> torch.Tensor:
    rank = int(geometry.retained_rank)
    s = geometry.singular_values
    return torch.diag(4.0 / s.square()) - (
        4.0 / geometry.frobenius_norm_sq
    ) * torch.ones((rank, rank), device=s.device, dtype=s.dtype)


def correction_from_coefficients(
    coefficients: torch.Tensor,
    geometry: MatrixLogGeometry,
    *,
    output_like: torch.Tensor,
) -> torch.Tensor:
    work = 2.0 * (
        (
            geometry.mode_left_vectors
            * (coefficients / geometry.singular_values).unsqueeze(0)
        )
        @ geometry.mode_right_vectors.transpose(0, 1)
   )
    work = work - (
        2.0 * torch.sum(coefficients) / geometry.frobenius_norm_sq
    ) * geometry.reference_work
    result = work.transpose(0, 1) if geometry.transposed else work
    return result.to(output_like)


def _cap(correction, delta, max_ratio, eps):
    base = torch.linalg.vector_norm(delta.float())
    corr = torch.linalg.vector_norm(correction.float())
    capped = False
    if max_ratio is not None:
        allowed = float(max_ratio) * float(base.cpu())
        current = float(corr.cpu())
        if current > allowed and current > eps:
            correction = correction * (allowed / current)
            corr = torch.linalg.vector_norm(correction.float())
            capped = True
    denom = float(base.cpu())
    ratio = float(corr.cpu()) / denom if denom > eps else 0.0
    return correction, ratio, capped


@torch.no_grad()
def remove_inward_matrix_log_flow(
    delta_base: torch.Tensor,
    geometry: MatrixLogGeometry,
    *,
    mode: CorrectionMode = "radial",
    projection_strength: float = 1.0,
    max_correction_ratio: float | None = 0.10,
    gram_ridge_relative: float = 1e-6,
    eps: float = 1e-12,
) -> ProjectionResult:
    """Radial reference and original modewise equality-system ablation."""
    if mode not in {"radial", "modewise"}:
        raise ValueError(f"Unknown legacy mode: {mode!r}")
    drifts = mode_drifts(delta_base, geometry)
    logs = geometry.log_eigenvalues.to(drifts)
    inward = logs * drifts < 0.0
    inward_count = int(torch.count_nonzero(inward))
    base_inward = (
        float(torch.linalg.vector_norm(drifts[inward]).cpu())
        if inward_count
        else 0.0
    )
    gradient = geometry.gradient.to(delta_base)
    base_drift = float(torch.sum(gradient.float() * delta_base.float()).cpu())

    if mode == "radial":
        norm_sq = float(torch.sum(gradient.float().square()).cpu())
        if norm_sq <= eps or base_drift >= 0.0:
            zero = torch.zeros_like(delta_base)
            return ProjectionResult(
                delta_base, zero, mode, base_drift, base_drift, 0.0, 0.0,
                False, False, inward_count, base_inward, base_inward,
            )
        coefficient = base_drift / norm_sq
        correction = -float(projection_strength) * coefficient * gradient
    else:
        if inward_count == 0:
            zero = torch.zeros_like(delta_base)
            return ProjectionResult(
                delta_base, zero, mode, base_drift, base_drift, 0.0, 0.0,
                False, False, 0, 0.0, 0.0,
            )
        gram = mode_gram(geometry)
        target = torch.zeros_like(drifts)
        target[inward] = -float(projection_strength) * drifts[inward]
        scale = torch.max(torch.diagonal(gram)).clamp_min(float(eps))
        system = gram + float(gram_ridge_relative) * scale * torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )
        try:
            coefficients = torch.linalg.solve(system, target)
        except (RuntimeError, NotImplementedError):
            coefficients = torch.linalg.lstsq(
                system.cpu().double(), target.cpu().double().unsqueeze(-1)
            ).solution.squeeze(-1).to(system)
        correction = correction_from_coefficients(
            coefficients, geometry, output_like=delta_base
        )
        coefficient = float(torch.linalg.vector_norm(coefficients).cpu())

    correction, ratio, capped = _cap(
        correction, delta_base, max_correction_ratio, eps
    )
    corrected = delta_base + correction
    corrected_drift = float(
        torch.sum(gradient.float() * corrected.float()).cpu()
    )
    corrected_modes = mode_drifts(corrected, geometry)
    corrected_inward = (
        float(torch.linalg.vector_norm(corrected_modes[inward]).cpu())
        if inward_count
        else 0.0
    )
    return ProjectionResult(
        corrected_delta=corrected,
        correction=correction,
        mode=mode,
        base_drift=base_drift,
        corrected_drift=corrected_drift,
        coefficient=float(coefficient),
        correction_ratio=ratio,
        applied=bool(float(torch.linalg.vector_norm(correction.float()).cpu()) > eps),
        capped=capped,
        inward_mode_count=inward_count,
        base_inward_mode_norm=base_inward,
        corrected_inward_mode_norm=corrected_inward,
    )


__all__ = [
    "MatrixLogGeometry",
    "ProjectionResult",
    "correction_from_coefficients",
    "full_matrix_log_geometry",
    "mode_drifts",
    "mode_gram",
    "remove_inward_matrix_log_flow",
]

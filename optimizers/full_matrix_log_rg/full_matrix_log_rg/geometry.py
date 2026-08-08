"""Full matrix-log geometry for removing flow toward the isotropic trivial manifold."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class MatrixLogGeometry:
    potential: torch.Tensor
    log_eigenvalues: torch.Tensor
    gradient: torch.Tensor
    retained_rank: int
    normalization_dimension: float
    gradient_norm_sq: float
    min_retained_singular_value: float
    max_retained_singular_value: float
    transposed: bool


def _orient_tall(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if weight.ndim != 2:
        raise ValueError(f'Expected matrix parameter, got {tuple(weight.shape)}')
    if weight.shape[0] >= weight.shape[1]:
        return weight, False
    return weight.transpose(0, 1), True


def _svd(weight: torch.Tensor):
    try:
        return torch.linalg.svd(weight, full_matrices=False)
    except RuntimeError:
        cpu = weight.detach().to('cpu', dtype=torch.float32)
        u, s, vh = torch.linalg.svd(cpu, full_matrices=False)
        return u.to(weight.device), s.to(weight.device), vh.to(weight.device)


@torch.no_grad()
def full_matrix_log_geometry(
    weight: torch.Tensor,
    retained_rank: int,
    *,
    normalization_dimension: float | None = None,
    ridge_relative: float = 1e-6,
    eps: float = 1e-12,
) -> MatrixLogGeometry:
    """Return Phi=(1/2m)||log X_tilde_R||_F^2 and its frozen-support gradient."""
    original_dtype = weight.dtype
    work, transposed = _orient_tall(weight.detach())
    compute = work.float() if work.dtype in {torch.float16, torch.bfloat16} else work
    u, s_all, vh = _svd(compute)
    max_rank = int(s_all.numel())
    m = int(max(1, min(int(retained_rank), max_rank)))
    d = float(normalization_dimension) if normalization_dimension is not None else float(compute.shape[1])

    frob_sq = torch.sum(s_all.square()).clamp_min(float(eps))
    largest = torch.clamp(s_all[0], min=float(eps))
    floor = torch.clamp(largest * float(ridge_relative) ** 0.5, min=float(eps))
    s = torch.clamp(s_all[:m], min=floor)
    u_r = u[:, :m]
    vh_r = vh[:m, :]

    lam_tilde = (d * s.square() / frob_sq).clamp_min(float(eps))
    ell = torch.log(lam_tilde)
    potential = 0.5 * torch.mean(ell.square())

    retained = (2.0 / float(m)) * ((u_r * (ell / s).unsqueeze(0)) @ vh_r)
    radial = (2.0 * torch.mean(ell) / frob_sq) * compute
    grad_work = retained - radial
    grad = grad_work.transpose(0, 1) if transposed else grad_work
    grad = grad.to(dtype=original_dtype)
    grad_norm_sq = torch.sum(grad.float().square())

    return MatrixLogGeometry(
        potential=potential.to(dtype=original_dtype),
        log_eigenvalues=ell.to(dtype=original_dtype),
        gradient=grad,
        retained_rank=m,
        normalization_dimension=d,
        gradient_norm_sq=float(grad_norm_sq.detach().cpu()),
        min_retained_singular_value=float(s[-1].detach().cpu()),
        max_retained_singular_value=float(s[0].detach().cpu()),
        transposed=transposed,
    )


@dataclass
class ProjectionResult:
    corrected_delta: torch.Tensor
    correction: torch.Tensor
    base_drift: float
    corrected_drift: float
    coefficient: float
    correction_ratio: float
    applied: bool
    capped: bool


@torch.no_grad()
def remove_inward_matrix_log_flow(
    delta_base: torch.Tensor,
    geometry: MatrixLogGeometry,
    *,
    projection_strength: float = 1.0,
    max_correction_ratio: float | None = 0.10,
    eps: float = 1e-12,
) -> ProjectionResult:
    """Remove only first-order flow that decreases Phi, i.e. toward X_tilde_R = I."""
    grad = geometry.gradient.to(delta_base.device, delta_base.dtype)
    grad_norm_sq = torch.sum(grad.float().square())
    drift = torch.sum(grad.float() * delta_base.float())
    g2 = float(grad_norm_sq.detach().cpu())
    d = float(drift.detach().cpu())

    if (not torch.isfinite(grad_norm_sq)) or g2 <= float(eps) or d >= 0.0:
        z = torch.zeros_like(delta_base)
        return ProjectionResult(delta_base, z, d, d, 0.0, 0.0, False, False)

    coefficient = d / g2
    correction = -float(projection_strength) * coefficient * grad
    base_norm = torch.linalg.vector_norm(delta_base.float())
    corr_norm = torch.linalg.vector_norm(correction.float())
    capped = False
    if max_correction_ratio is not None:
        allowed = float(max_correction_ratio) * float(base_norm.detach().cpu())
        current = float(corr_norm.detach().cpu())
        if current > allowed and current > float(eps):
            correction = correction * (allowed / current)
            corr_norm = torch.linalg.vector_norm(correction.float())
            capped = True

    corrected = delta_base + correction
    corrected_drift = float(torch.sum(grad.float() * corrected.float()).detach().cpu())
    denom = float(base_norm.detach().cpu())
    ratio = float(corr_norm.detach().cpu()) / denom if denom > float(eps) else 0.0
    return ProjectionResult(
        corrected, correction, d, corrected_drift, coefficient, ratio,
        bool(float(corr_norm.detach().cpu()) > float(eps)), capped,
    )

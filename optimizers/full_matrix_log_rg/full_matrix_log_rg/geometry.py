"""Full matrix-log geometry and legacy one-sided flow projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import CorrectionMode
from .support import MatrixLogSupport, orient_tall


@dataclass
class MatrixLogGeometry:
    """Local full matrix-log geometry on one frozen retained subspace."""

    potential: torch.Tensor
    log_eigenvalues: torch.Tensor
    gradient: torch.Tensor
    retained_rank: int
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
    cone_active_set_size: int = 0
    cone_iterations: int = 0
    cone_converged: bool = True
    max_signed_violation_before: float = 0.0
    max_signed_violation_after: float = 0.0


def _svd_with_cpu_fallback(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.svd(matrix, full_matrices=False)
    except (RuntimeError, NotImplementedError):
        cpu = matrix.detach().to(device="cpu", dtype=torch.float32)
        u, s, vh = torch.linalg.svd(cpu, full_matrices=False)
        return (
            u.to(device=matrix.device, dtype=matrix.dtype),
            s.to(device=matrix.device, dtype=matrix.dtype),
            vh.to(device=matrix.device, dtype=matrix.dtype),
        )


def _eigh_with_cpu_fallback(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.eigh(matrix)
    except (RuntimeError, NotImplementedError):
        cpu = matrix.detach().to(device="cpu", dtype=torch.float32)
        values, vectors = torch.linalg.eigh(cpu)
        return (
            values.to(device=matrix.device, dtype=matrix.dtype),
            vectors.to(device=matrix.device, dtype=matrix.dtype),
        )


def solve_with_cpu_fallback(
    matrix: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    try:
        return torch.linalg.solve(matrix, vector)
    except (RuntimeError, NotImplementedError):
        cpu_matrix = matrix.detach().to(device="cpu", dtype=torch.float64)
        cpu_vector = vector.detach().to(device="cpu", dtype=torch.float64)
        solution = torch.linalg.solve(cpu_matrix, cpu_vector)
        return solution.to(device=matrix.device, dtype=matrix.dtype)


def _basis_from_rank(
    work: torch.Tensor,
    retained_rank: int,
) -> torch.Tensor:
    _, _, vh = _svd_with_cpu_fallback(work)
    rank = int(max(1, min(int(retained_rank), int(vh.shape[0]))))
    return vh[:rank, :].transpose(0, 1).contiguous()


@torch.no_grad()
def full_matrix_log_geometry(
    weight: torch.Tensor,
    retained_rank: int | None = None,
    *,
    support: MatrixLogSupport | None = None,
    right_basis: torch.Tensor | None = None,
    normalization_dimension: float | None = None,
    ridge_relative: float = 1e-6,
    eps: float = 1e-12,
) -> MatrixLogGeometry:
    """Compute ``Phi=(1/2m)||log X_tilde_R||_F^2`` and its gradient."""

    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(weight.shape)}")
    if ridge_relative < 0.0:
        raise ValueError("ridge_relative must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    original_dtype = weight.dtype
    work, transposed = orient_tall(weight.detach())
    compute = work.float() if work.dtype in {torch.float16, torch.bfloat16} else work

    if support is not None:
        if bool(support.transposed) != bool(transposed):
            raise ValueError("Cached support orientation does not match the parameter")
        basis = support.right_basis
        retained_rank = int(support.retained_rank)
        if normalization_dimension is None:
            normalization_dimension = float(support.normalization_dimension)
    elif right_basis is not None:
        basis = right_basis
        if retained_rank is None:
            retained_rank = int(right_basis.shape[1])
    else:
        if retained_rank is None:
            raise ValueError("Provide support, right_basis, or retained_rank")
        basis = _basis_from_rank(compute, int(retained_rank))

    basis = torch.as_tensor(basis).detach().to(
        device=compute.device,
        dtype=compute.dtype,
    )
    if basis.ndim != 2 or int(basis.shape[0]) != int(compute.shape[1]):
        raise ValueError(
            f"Right basis has shape {tuple(basis.shape)}, expected ({compute.shape[1]}, m)"
        )

    rank = int(max(1, min(int(retained_rank), int(basis.shape[1]))))
    basis = basis[:, :rank]
    dimension = (
        float(normalization_dimension)
        if normalization_dimension is not None
        else float(compute.shape[1])
    )
    if dimension <= 0.0:
        raise ValueError("normalization_dimension must be positive")

    projected = compute @ basis
    covariance = projected.transpose(0, 1) @ projected
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    eigenvalues, eigenvectors = _eigh_with_cpu_fallback(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    largest = torch.clamp(eigenvalues[0], min=float(eps))
    floor = torch.clamp(largest * float(ridge_relative), min=float(eps))
    safe_eigenvalues = torch.clamp(eigenvalues, min=floor)
    singular_values = torch.sqrt(safe_eigenvalues)

    effective_right = basis @ eigenvectors
    left = (projected @ eigenvectors) / singular_values.unsqueeze(0)

    frob_sq = torch.sum(compute.square()).clamp_min(float(eps))
    normalized = (dimension * safe_eigenvalues / frob_sq).clamp_min(float(eps))
    log_eigenvalues = torch.log(normalized)
    potential = 0.5 * torch.mean(log_eigenvalues.square())

    retained_gradient = (2.0 / float(rank)) * (
        (left * (log_eigenvalues / singular_values).unsqueeze(0))
        @ effective_right.transpose(0, 1)
    )
    radial_gradient = (
        2.0 * torch.mean(log_eigenvalues) / frob_sq
    ) * compute
    gradient_work = retained_gradient - radial_gradient
    gradient = gradient_work.transpose(0, 1) if transposed else gradient_work
    gradient = gradient.to(dtype=original_dtype)
    gradient_norm_sq = torch.sum(gradient.float().square())

    return MatrixLogGeometry(
        potential=potential.to(dtype=original_dtype),
        log_eigenvalues=log_eigenvalues.to(dtype=original_dtype),
        gradient=gradient,
        retained_rank=rank,
        normalization_dimension=dimension,
        gradient_norm_sq=float(gradient_norm_sq.detach().cpu()),
        min_retained_singular_value=float(singular_values[-1].detach().cpu()),
        max_retained_singular_value=float(singular_values[0].detach().cpu()),
        transposed=transposed,
        reference_work=compute,
        mode_left_vectors=left,
        mode_right_vectors=effective_right,
        singular_values=singular_values,
        frobenius_norm_sq=frob_sq,
    )


def _orient_like_geometry(
    matrix: torch.Tensor,
    geometry: MatrixLogGeometry,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(matrix.shape)}")
    return matrix.transpose(0, 1) if geometry.transposed else matrix


def matrix_log_mode_drifts(
    delta_base: torch.Tensor,
    geometry: MatrixLogGeometry,
) -> torch.Tensor:
    """Return first-order drifts of every retained normalized log eigenvalue."""

    delta_work = _orient_like_geometry(delta_base, geometry).to(
        device=geometry.reference_work.device,
        dtype=geometry.reference_work.dtype,
    )
    mode_delta = torch.sum(
        (delta_work @ geometry.mode_right_vectors)
        * geometry.mode_left_vectors,
        dim=0,
    )
    radial_delta = (
        2.0
        * torch.sum(geometry.reference_work * delta_work)
        / geometry.frobenius_norm_sq
    )
    return 2.0 * mode_delta / geometry.singular_values - radial_delta


def matrix_log_mode_gram(geometry: MatrixLogGeometry) -> torch.Tensor:
    """Gram matrix of the individual log-eigenvalue gradients."""

    rank = int(geometry.retained_rank)
    diagonal = torch.diag(4.0 / geometry.singular_values.square())
    radial = (
        4.0 / geometry.frobenius_norm_sq
    ) * torch.ones(
        (rank, rank),
        device=diagonal.device,
        dtype=diagonal.dtype,
    )
    return diagonal - radial


def matrix_log_correction_from_coefficients(
    coefficients: torch.Tensor,
    geometry: MatrixLogGeometry,
    *,
    output_like: torch.Tensor,
) -> torch.Tensor:
    """Construct ``sum_i coefficients_i * grad ell_i`` in parameter coordinates."""

    coefficients = torch.as_tensor(coefficients).to(
        device=geometry.reference_work.device,
        dtype=geometry.reference_work.dtype,
    )
    correction_work = 2.0 * (
        (
            geometry.mode_left_vectors
            * (coefficients / geometry.singular_values).unsqueeze(0)
        )
        @ geometry.mode_right_vectors.transpose(0, 1)
    )
    correction_work = correction_work - (
        2.0 * torch.sum(coefficients) / geometry.frobenius_norm_sq
    ) * geometry.reference_work
    correction = (
        correction_work.transpose(0, 1)
        if geometry.transposed
        else correction_work
    )
    return correction.to(device=output_like.device, dtype=output_like.dtype)


def apply_correction_cap(
    correction: torch.Tensor,
    delta_base: torch.Tensor,
    *,
    max_correction_ratio: float | None,
    eps: float,
) -> tuple[torch.Tensor, float, bool]:
    base_norm = torch.linalg.vector_norm(delta_base.float())
    correction_norm = torch.linalg.vector_norm(correction.float())
    capped = False
    if max_correction_ratio is not None:
        allowed = float(max_correction_ratio) * float(base_norm.detach().cpu())
        current = float(correction_norm.detach().cpu())
        if current > allowed and current > float(eps):
            correction = correction * (allowed / current)
            correction_norm = torch.linalg.vector_norm(correction.float())
            capped = True
    denominator = float(base_norm.detach().cpu())
    ratio = (
        float(correction_norm.detach().cpu()) / denominator
        if denominator > float(eps)
        else 0.0
    )
    return correction, ratio, capped


def _empty_projection(
    delta_base: torch.Tensor,
    *,
    mode: str,
    base_drift: float,
    inward: torch.Tensor,
    base_mode_drifts: torch.Tensor,
) -> ProjectionResult:
    inward_count = int(torch.count_nonzero(inward).detach().cpu())
    inward_norm = (
        float(torch.linalg.vector_norm(base_mode_drifts[inward]).detach().cpu())
        if inward_count
        else 0.0
    )
    zero = torch.zeros_like(delta_base)
    return ProjectionResult(
        corrected_delta=delta_base,
        correction=zero,
        mode=mode,
        base_drift=base_drift,
        corrected_drift=base_drift,
        coefficient=0.0,
        correction_ratio=0.0,
        applied=False,
        capped=False,
        inward_mode_count=inward_count,
        base_inward_mode_norm=inward_norm,
        corrected_inward_mode_norm=inward_norm,
    )


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
    """Legacy radial or exactly-targeted modewise projection.

    ``modewise`` is retained as a historical ablation. The active-set cone
    projector is the corrected primary method.
    """

    if mode not in {"radial", "modewise"}:
        raise ValueError(
            "remove_inward_matrix_log_flow supports only radial/modewise; "
            "use project_matrix_log_cone for mode='cone'"
        )
    if not 0.0 <= float(projection_strength) <= 1.0:
        raise ValueError("projection_strength must lie in [0, 1]")

    base_mode_drifts = matrix_log_mode_drifts(delta_base, geometry)
    log_values = geometry.log_eigenvalues.to(
        device=base_mode_drifts.device,
        dtype=base_mode_drifts.dtype,
    )
    inward = (log_values * base_mode_drifts) < 0.0
    inward_count = int(torch.count_nonzero(inward).detach().cpu())
    base_inward_norm = (
        float(torch.linalg.vector_norm(base_mode_drifts[inward]).detach().cpu())
        if inward_count
        else 0.0
    )

    grad = geometry.gradient.to(delta_base.device, delta_base.dtype)
    base_drift = float(
        torch.sum(grad.float() * delta_base.float()).detach().cpu()
    )

    if mode == "radial":
        grad_norm_sq = torch.sum(grad.float().square())
        g2 = float(grad_norm_sq.detach().cpu())
        if (
            not bool(torch.isfinite(grad_norm_sq).item())
            or g2 <= float(eps)
            or base_drift >= 0.0
        ):
            return _empty_projection(
                delta_base,
                mode=mode,
                base_drift=base_drift,
                inward=inward,
                base_mode_drifts=base_mode_drifts,
            )
        coefficient = base_drift / g2
        correction = -float(projection_strength) * coefficient * grad
    else:
        if inward_count == 0:
            return _empty_projection(
                delta_base,
                mode=mode,
                base_drift=base_drift,
                inward=inward,
                base_mode_drifts=base_mode_drifts,
            )
        gram = matrix_log_mode_gram(geometry)
        target = torch.zeros_like(base_mode_drifts)
        target[inward] = -float(projection_strength) * base_mode_drifts[inward]
        scale = torch.max(torch.diagonal(gram)).clamp_min(float(eps))
        ridge = float(gram_ridge_relative) * scale
        coefficients = solve_with_cpu_fallback(
            gram
            + ridge
            * torch.eye(
                geometry.retained_rank,
                device=gram.device,
                dtype=gram.dtype,
            ),
            target,
        )
        correction = matrix_log_correction_from_coefficients(
            coefficients,
            geometry,
            output_like=delta_base,
        )
        coefficient = float(torch.linalg.vector_norm(coefficients).detach().cpu())

    correction, ratio, capped = apply_correction_cap(
        correction,
        delta_base,
        max_correction_ratio=max_correction_ratio,
        eps=eps,
    )
    corrected = delta_base + correction
    corrected_drift = float(
        torch.sum(grad.float() * corrected.float()).detach().cpu()
    )
    corrected_mode_drifts = matrix_log_mode_drifts(corrected, geometry)
    corrected_inward_norm = (
        float(torch.linalg.vector_norm(corrected_mode_drifts[inward]).detach().cpu())
        if inward_count
        else 0.0
    )
    applied = bool(
        float(torch.linalg.vector_norm(correction.float()).detach().cpu()) > float(eps)
    )

    return ProjectionResult(
        corrected_delta=corrected,
        correction=correction,
        mode=mode,
        base_drift=base_drift,
        corrected_drift=corrected_drift,
        coefficient=float(coefficient),
        correction_ratio=ratio,
        applied=applied,
        capped=capped,
        inward_mode_count=inward_count,
        base_inward_mode_norm=base_inward_norm,
        corrected_inward_mode_norm=corrected_inward_norm,
    )

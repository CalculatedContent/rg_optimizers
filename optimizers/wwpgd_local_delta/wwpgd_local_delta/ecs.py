"""Self-consistent ECS scan and local-delta projection geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ECSScanResult:
    """Finite self-consistent ECS scan result."""

    rank: int
    normalization_dimension: float
    bulk_effective_count: float
    trace_log_per_eval: float
    status: str
    candidate_rank_count: int
    spectral_count: int
    total_energy: float


@dataclass(frozen=True)
class LocalECSGeometry:
    """Retained ECS basis in the oriented tall layer coordinates."""

    basis: torch.Tensor
    scan: ECSScanResult
    transposed: bool
    projection_side: str
    oriented_rows: int
    oriented_columns: int


@dataclass(frozen=True)
class LocalDeltaCorrectionResult:
    """Diagnostics from damping the epoch displacement outside the ECS."""

    corrected_delta: torch.Tensor
    ecs_rank: int
    ecs_fraction: float
    normalization_dimension: float
    bulk_effective_count: float
    trace_log_per_eval: float
    status: str
    transposed: bool
    projection_side: str
    base_delta_norm: float
    ecs_delta_norm: float
    orthogonal_delta_norm: float
    post_orthogonal_delta_norm: float
    removed_delta_norm: float
    orthogonal_fraction: float
    post_orthogonal_fraction: float
    removed_fraction_of_base: float
    observed_orthogonal_damping: float
    expected_orthogonal_damping: float
    damping_error: float
    pythagorean_error: float
    correction_identity_error: float
    correction_fraction: float


def _as_matrix(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim < 2:
        raise ValueError("weight must be at least two-dimensional")
    if weight.ndim == 2:
        return weight
    return weight.reshape(weight.shape[0], -1)


def _orient_tall(matrix: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Orient a layer as N x M with N >= M, matching TraceLogRG/WeightWatcher."""
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got {tuple(matrix.shape)}")
    if matrix.shape[0] >= matrix.shape[1]:
        return matrix, False
    return matrix.transpose(0, 1), True


def _svd_with_cpu_fallback(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compute = matrix.float() if matrix.dtype in {torch.float16, torch.bfloat16} else matrix
    try:
        return torch.linalg.svd(compute, full_matrices=False)
    except RuntimeError:
        cpu = compute.detach().to(device="cpu", dtype=torch.float32)
        u, s, vh = torch.linalg.svd(cpu, full_matrices=False)
        return (
            u.to(device=matrix.device, dtype=compute.dtype),
            s.to(device=matrix.device, dtype=compute.dtype),
            vh.to(device=matrix.device, dtype=compute.dtype),
        )


def _participation_count(values: torch.Tensor, eps: float) -> float:
    if values.numel() == 0:
        return 0.0
    s1 = torch.sum(values)
    s2 = torch.sum(values * values)
    if float(s1.detach().cpu()) <= eps or float(s2.detach().cpu()) <= eps:
        return 0.0
    return float(((s1 * s1) / (s2 + eps)).detach().cpu())


def select_self_consistent_ecs(
    singular_values: torch.Tensor,
    *,
    min_retained: int = 3,
    max_retained: Optional[int] = None,
    normalization_gamma: float = 0.0,
    eps: float = 1e-12,
) -> ECSScanResult:
    """Select an ECS rank using the bulk-effective trace-log scan.

    For candidate retained rank ``m`` the adaptive normalization dimension is

        D(m) = m + r_bulk(m) + gamma * ((M-m) - r_bulk(m)),

    and the selected rank is adjacent to a zero crossing of

        F(m) = mean_top_m log(D(m) * lambda_i / sum_j lambda_j),

    or the minimum-absolute-residual candidate if a finite spectrum has no
    crossing.
    """

    if not 0.0 <= float(normalization_gamma) <= 1.0:
        raise ValueError("normalization_gamma must lie in [0, 1]")
    s = singular_values.detach().float().abs()
    lambdas = s * s
    lambdas = lambdas[torch.isfinite(lambdas) & (lambdas > float(eps))]
    lambdas = lambdas.clamp_min(eps)
    if lambdas.numel() == 0:
        raise ValueError("no positive singular values available for ECS scan")
    lambdas, _ = torch.sort(lambdas, descending=True)
    spectral_count = int(lambdas.numel())
    lo = max(1, int(min_retained))
    hi = spectral_count if max_retained is None else min(int(max_retained), spectral_count)
    if hi < lo:
        raise ValueError("not enough singular values for requested retained rank")

    total = torch.sum(lambdas).clamp_min(eps)
    gamma = float(normalization_gamma)
    candidates: list[tuple[int, float, float, float]] = []
    for rank in range(lo, hi + 1):
        retained = lambdas[:rank]
        bulk = lambdas[rank:]
        r_bulk = _participation_count(bulk, eps)
        bulk_count = float(max(0, spectral_count - rank))
        dimension = float(rank) + r_bulk + gamma * (bulk_count - r_bulk)
        dimension = max(float(rank), min(float(spectral_count), dimension))
        trace_log = torch.mean(torch.log((dimension * retained / total).clamp_min(eps)))
        candidates.append((rank, dimension, r_bulk, float(trace_log.detach().cpu())))

    exact = [item for item in candidates if abs(item[3]) <= eps]
    if exact:
        selected = min(exact, key=lambda item: item[0])
        status = "exact_zero"
    else:
        bracket_endpoints: list[tuple[int, float, float, float]] = []
        for left, right in zip(candidates[:-1], candidates[1:]):
            if left[3] * right[3] < 0.0:
                bracket_endpoints.extend((left, right))
        if bracket_endpoints:
            selected = min(bracket_endpoints, key=lambda item: (abs(item[3]), item[0]))
            status = "zero_crossing"
        else:
            selected = min(candidates, key=lambda item: (abs(item[3]), item[0]))
            status = "min_abs_residual"

    rank, dimension, r_bulk, trace_log = selected
    return ECSScanResult(
        rank=int(rank),
        normalization_dimension=float(dimension),
        bulk_effective_count=float(r_bulk),
        trace_log_per_eval=float(trace_log),
        status=status,
        candidate_rank_count=len(candidates),
        spectral_count=spectral_count,
        total_energy=float(total.detach().cpu()),
    )


def local_ecs_geometry(
    weight: torch.Tensor,
    *,
    min_retained: int = 3,
    max_retained: Optional[int] = None,
    normalization_gamma: float = 0.0,
    eps: float = 1e-12,
) -> LocalECSGeometry:
    """Return the ECS basis with the same tall orientation as TraceLogRG.

    For an original tall/square matrix the ECS acts on the right and the
    retained update is ``Delta @ P_R``.  For an original wide matrix, the layer
    is transposed before forming ``X = W^T W/N``; after mapping back, the ECS
    acts on the left and the retained update is ``P_R @ Delta``.
    """

    matrix = _as_matrix(weight).detach()
    if matrix.numel() == 0:
        raise ValueError("empty weight matrix")
    oriented, transposed = _orient_tall(matrix)
    _, singular_values, vh = _svd_with_cpu_fallback(oriented)
    scan = select_self_consistent_ecs(
        singular_values,
        min_retained=min_retained,
        max_retained=max_retained,
        normalization_gamma=normalization_gamma,
        eps=eps,
    )
    basis = vh[: scan.rank].transpose(0, 1).to(device=weight.device, dtype=weight.dtype)
    return LocalECSGeometry(
        basis=basis,
        scan=scan,
        transposed=transposed,
        projection_side="left" if transposed else "right",
        oriented_rows=int(oriented.shape[0]),
        oriented_columns=int(oriented.shape[1]),
    )


def split_delta_by_ecs(
    delta: torch.Tensor,
    geometry: LocalECSGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an update into retained and orthogonal ECS components."""

    original_shape = delta.shape
    matrix = _as_matrix(delta)
    oriented = matrix.transpose(0, 1) if geometry.transposed else matrix
    basis = geometry.basis.to(device=oriented.device, dtype=oriented.dtype)
    retained_oriented = (oriented @ basis) @ basis.transpose(0, 1)
    orthogonal_oriented = oriented - retained_oriented
    if geometry.transposed:
        retained = retained_oriented.transpose(0, 1)
        orthogonal = orthogonal_oriented.transpose(0, 1)
    else:
        retained = retained_oriented
        orthogonal = orthogonal_oriented
    return retained.reshape(original_shape), orthogonal.reshape(original_shape)


def damp_delta_outside_ecs(
    delta: torch.Tensor,
    reference_weight: torch.Tensor,
    *,
    correction_fraction: float,
    min_retained: int = 3,
    max_retained: Optional[int] = None,
    normalization_gamma: float = 0.0,
    eps: float = 1e-12,
) -> LocalDeltaCorrectionResult:
    """Fractionally damp the completed displacement outside the local ECS."""

    frac = float(correction_fraction)
    if not 0.0 <= frac <= 1.0:
        raise ValueError("correction_fraction must lie in [0, 1]")
    delta_matrix = _as_matrix(delta)
    ref_matrix = _as_matrix(reference_weight)
    if delta_matrix.shape != ref_matrix.shape:
        raise ValueError("delta and reference_weight must have compatible matrix shapes")

    geometry = local_ecs_geometry(
        ref_matrix,
        min_retained=min_retained,
        max_retained=max_retained,
        normalization_gamma=normalization_gamma,
        eps=eps,
    )
    delta_ecs, delta_orth = split_delta_by_ecs(delta, geometry)
    corrected = delta_ecs + (1.0 - frac) * delta_orth
    _, post_orth = split_delta_by_ecs(corrected, geometry)

    base_norm_t = torch.linalg.vector_norm(delta.float())
    ecs_norm_t = torch.linalg.vector_norm(delta_ecs.float())
    orth_norm_t = torch.linalg.vector_norm(delta_orth.float())
    post_orth_norm_t = torch.linalg.vector_norm(post_orth.float())
    removed = delta - corrected
    removed_norm_t = torch.linalg.vector_norm(removed.float())

    base_norm = float(base_norm_t.detach().cpu())
    ecs_norm = float(ecs_norm_t.detach().cpu())
    orth_norm = float(orth_norm_t.detach().cpu())
    post_orth_norm = float(post_orth_norm_t.detach().cpu())
    removed_norm = float(removed_norm_t.detach().cpu())
    denom = max(base_norm, float(eps))
    expected_damping = 1.0 - frac
    orth_threshold = max(float(eps), 1e-6 * denom)
    if orth_norm > orth_threshold:
        observed_damping = post_orth_norm / orth_norm
        damping_error = abs(observed_damping - expected_damping)
    else:
        observed_damping = expected_damping
        damping_error = 0.0

    pythagorean_error = abs(base_norm**2 - ecs_norm**2 - orth_norm**2) / max(base_norm**2, float(eps))
    identity_target = delta - frac * delta_orth
    identity_error = float(torch.linalg.vector_norm((corrected - identity_target).float()).detach().cpu()) / denom

    scan = geometry.scan
    return LocalDeltaCorrectionResult(
        corrected_delta=corrected,
        ecs_rank=scan.rank,
        ecs_fraction=float(scan.rank / max(scan.spectral_count, 1)),
        normalization_dimension=scan.normalization_dimension,
        bulk_effective_count=scan.bulk_effective_count,
        trace_log_per_eval=scan.trace_log_per_eval,
        status=scan.status,
        transposed=geometry.transposed,
        projection_side=geometry.projection_side,
        base_delta_norm=base_norm,
        ecs_delta_norm=ecs_norm,
        orthogonal_delta_norm=orth_norm,
        post_orthogonal_delta_norm=post_orth_norm,
        removed_delta_norm=removed_norm,
        orthogonal_fraction=orth_norm / denom,
        post_orthogonal_fraction=post_orth_norm / denom,
        removed_fraction_of_base=removed_norm / denom,
        observed_orthogonal_damping=observed_damping,
        expected_orthogonal_damping=expected_damping,
        damping_error=damping_error,
        pythagorean_error=pythagorean_error,
        correction_identity_error=identity_error,
        correction_fraction=frac,
    )

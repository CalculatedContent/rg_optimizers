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
class LocalDeltaCorrectionResult:
    """Diagnostics from damping the epoch displacement outside the ECS."""

    corrected_delta: torch.Tensor
    ecs_rank: int
    normalization_dimension: float
    bulk_effective_count: float
    trace_log_per_eval: float
    status: str
    base_delta_norm: float
    ecs_delta_norm: float
    orthogonal_delta_norm: float
    removed_delta_norm: float
    orthogonal_fraction: float
    removed_fraction_of_base: float
    correction_fraction: float


def _as_matrix(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim < 2:
        raise ValueError("weight must be at least two-dimensional")
    if weight.ndim == 2:
        return weight
    return weight.reshape(weight.shape[0], -1)


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
    """Select an ECS rank using a finite self-consistent trace-log scan.

    This mirrors the self-consistent scan used in the trace-log experiments but
    intentionally restricts the normalizing dimension to the positive finite
    spectrum observed by the SVD. For local update projection we only need the
    retained right singular subspace; null directions are treated as outside the
    measured finite spectrum.
    """

    s = singular_values.detach().float().abs()
    lambdas = (s * s).clamp_min(eps)
    lambdas = lambdas[torch.isfinite(lambdas)]
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
        tail_count = float(max(0, spectral_count - rank))
        dimension = float(rank) + r_bulk + gamma * (tail_count - r_bulk)
        dimension = max(float(rank), min(float(spectral_count), dimension))
        trace_log = torch.mean(torch.log((dimension * retained / total).clamp_min(eps)))
        candidates.append((rank, dimension, r_bulk, float(trace_log.detach().cpu())))

    selected = min(candidates, key=lambda item: abs(item[3]))
    status = "min_abs_residual"
    for left, right in zip(candidates[:-1], candidates[1:]):
        f_left = left[3]
        f_right = right[3]
        if f_left == 0.0:
            selected = left
            status = "exact_zero"
            break
        if f_left * f_right <= 0.0:
            selected = left if abs(f_left) <= abs(f_right) else right
            status = "zero_crossing"
            break

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


def right_ecs_basis(
    weight: torch.Tensor,
    *,
    min_retained: int = 3,
    max_retained: Optional[int] = None,
    normalization_gamma: float = 0.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, ECSScanResult]:
    """Return the retained right-singular ECS basis for a layer matrix."""

    matrix = _as_matrix(weight).detach()
    if matrix.numel() == 0:
        raise ValueError("empty weight matrix")
    _, singular_values, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    scan = select_self_consistent_ecs(
        singular_values,
        min_retained=min_retained,
        max_retained=max_retained,
        normalization_gamma=normalization_gamma,
        eps=eps,
    )
    basis = vh[: scan.rank].transpose(0, 1).to(device=weight.device, dtype=weight.dtype)
    return basis, scan


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
    """Dampen the component of a completed displacement outside the ECS.

    The ECS is a right-singular-vector subspace. For a layer matrix with shape
    [out, in], the retained component is Delta @ P_R.
    """

    frac = float(correction_fraction)
    if not 0.0 <= frac <= 1.0:
        raise ValueError("correction_fraction must lie in [0, 1]")
    original_shape = delta.shape
    delta_matrix = _as_matrix(delta)
    ref_matrix = _as_matrix(reference_weight)
    if delta_matrix.shape != ref_matrix.shape:
        raise ValueError("delta and reference_weight must have compatible matrix shapes")

    basis, scan = right_ecs_basis(
        ref_matrix,
        min_retained=min_retained,
        max_retained=max_retained,
        normalization_gamma=normalization_gamma,
        eps=eps,
    )
    delta_ecs = (delta_matrix @ basis) @ basis.transpose(0, 1)
    delta_orth = delta_matrix - delta_ecs
    corrected = delta_ecs + (1.0 - frac) * delta_orth

    base_norm = float(torch.linalg.vector_norm(delta_matrix).detach().cpu())
    ecs_norm = float(torch.linalg.vector_norm(delta_ecs).detach().cpu())
    orth_norm = float(torch.linalg.vector_norm(delta_orth).detach().cpu())
    removed_norm = float(torch.linalg.vector_norm(frac * delta_orth).detach().cpu())
    denom = max(base_norm, float(eps))

    return LocalDeltaCorrectionResult(
        corrected_delta=corrected.reshape(original_shape),
        ecs_rank=scan.rank,
        normalization_dimension=scan.normalization_dimension,
        bulk_effective_count=scan.bulk_effective_count,
        trace_log_per_eval=scan.trace_log_per_eval,
        status=scan.status,
        base_delta_norm=base_norm,
        ecs_delta_norm=ecs_norm,
        orthogonal_delta_norm=orth_norm,
        removed_delta_norm=removed_norm,
        orthogonal_fraction=orth_norm / denom,
        removed_fraction_of_base=removed_norm / denom,
        correction_fraction=frac,
    )

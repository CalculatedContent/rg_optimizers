"""Self-consistent ECS selection and SVD-space gradient projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np
import torch

ProjectionMode = Literal["core", "rank_m_tangent"]


@dataclass(frozen=True)
class ECSSelection:
    rank: int
    fractional_rank: float
    normalization_dimension: float
    bulk_count: int
    bulk_effective_count: float
    bulk_effective_fraction: float
    trace_log: float
    trace_log_per_eval: float
    lambda_cut: float
    positive_count: int
    number_of_sign_change_brackets: int
    status: str
    reference_rank: Optional[int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ECSSVDState:
    selection: ECSSelection
    full_weight: torch.Tensor
    truncated_weight: torch.Tensor
    left_vectors: torch.Tensor
    singular_values: torch.Tensor
    right_vectors_t: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.selection.rank)

    @property
    def left_ecs(self) -> torch.Tensor:
        return self.left_vectors[:, : self.rank]

    @property
    def right_ecs(self) -> torch.Tensor:
        return self.right_vectors_t[: self.rank, :].transpose(0, 1)


def _clean_positive_eigenvalues(eigenvalues: object) -> np.ndarray:
    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    values = np.sort(values)
    if values.size < 1:
        raise ValueError("at least one positive finite eigenvalue is required")
    return values


def participation_ratio(values: object) -> float:
    x = _clean_positive_eigenvalues(values)
    total = float(np.sum(x))
    second = float(np.sum(x * x))
    return float(total * total / second) if second > 0.0 else 0.0


def _fractional_crossing(
    ranks: np.ndarray,
    residuals: np.ndarray,
    selected_index: int,
) -> float:
    if np.isclose(residuals[selected_index], 0.0, atol=1e-14, rtol=1e-12):
        return float(ranks[selected_index])
    candidate_pairs: list[tuple[int, int]] = []
    if selected_index > 0:
        candidate_pairs.append((selected_index - 1, selected_index))
    if selected_index + 1 < len(ranks):
        candidate_pairs.append((selected_index, selected_index + 1))
    for left, right in candidate_pairs:
        f0 = float(residuals[left])
        f1 = float(residuals[right])
        if f0 == 0.0:
            return float(ranks[left])
        if f1 == 0.0:
            return float(ranks[right])
        if np.signbit(f0) != np.signbit(f1):
            denominator = abs(f0) + abs(f1)
            if denominator > 0.0:
                fraction = abs(f0) / denominator
                return float(ranks[left] + fraction * (ranks[right] - ranks[left]))
    return float(ranks[selected_index])


def select_self_consistent_ecs(
    eigenvalues: object,
    *,
    min_rank: int = 2,
    normalization_gamma: float = 0.0,
    reference_rank: Optional[int] = None,
    numeric_epsilon: float = 1e-12,
) -> ECSSelection:
    """Select the bulk-effective self-consistent trace-log ECS.

    Every candidate retained rank is evaluated.  All exact zeros and both
    endpoints of every sign-change bracket are eligible.  The chosen integer
    minimizes the absolute trace-log residual; ``reference_rank`` is used only
    as a continuity tie-breaker.
    """

    if not 0.0 <= float(normalization_gamma) <= 1.0:
        raise ValueError("normalization_gamma must lie in [0, 1]")
    if min_rank < 1:
        raise ValueError("min_rank must be positive")

    values = _clean_positive_eigenvalues(eigenvalues)
    count = int(values.size)
    minimum = int(np.clip(min_rank, 1, count))
    ranks = np.arange(minimum, count + 1, dtype=np.int64)
    bulk_counts = count - ranks

    prefix_sum = np.concatenate(([0.0], np.cumsum(values)))
    prefix_second = np.concatenate(([0.0], np.cumsum(values * values)))
    bulk_sum = prefix_sum[bulk_counts]
    bulk_second = prefix_second[bulk_counts]
    bulk_effective = np.zeros_like(bulk_sum)
    valid_bulk = bulk_second > 0.0
    bulk_effective[valid_bulk] = (
        bulk_sum[valid_bulk] ** 2 / bulk_second[valid_bulk]
    )

    dimensions = (
        ranks.astype(np.float64)
        + bulk_effective
        + float(normalization_gamma)
        * (bulk_counts.astype(np.float64) - bulk_effective)
    )
    dimensions = np.clip(dimensions, ranks.astype(np.float64), float(count))

    spectral_sum = float(np.sum(values))
    descending = values[::-1]
    cumulative_log = np.cumsum(np.log(descending))
    residuals = cumulative_log[ranks - 1] + ranks * np.log(
        dimensions / spectral_sum
    )

    exact = np.flatnonzero(
        np.isclose(residuals, 0.0, atol=numeric_epsilon, rtol=1e-10)
    )
    sign_changes = np.flatnonzero(
        np.signbit(residuals[:-1]) != np.signbit(residuals[1:])
    )
    bracket_candidates: set[int] = set(int(i) for i in exact)
    for index in sign_changes:
        bracket_candidates.add(int(index))
        bracket_candidates.add(int(index + 1))

    if bracket_candidates:
        candidates = np.array(sorted(bracket_candidates), dtype=np.int64)
        status = "trace_log_crossing"
    else:
        candidates = np.arange(len(ranks), dtype=np.int64)
        status = "nearest_residual"

    absolute = np.abs(residuals[candidates])
    best_absolute = float(np.min(absolute))
    tie_tolerance = max(numeric_epsilon, 1e-10 * max(1.0, best_absolute))
    tied = candidates[np.abs(absolute - best_absolute) <= tie_tolerance]
    if reference_rank is not None and tied.size > 1:
        distance = np.abs(ranks[tied] - int(reference_rank))
        tied = tied[distance == np.min(distance)]
    selected_index = int(tied[np.argmin(ranks[tied])])

    rank = int(ranks[selected_index])
    bulk_count = int(bulk_counts[selected_index])
    effective = float(bulk_effective[selected_index])
    fraction = float(effective / bulk_count) if bulk_count > 0 else float("nan")
    trace_log = float(residuals[selected_index])
    return ECSSelection(
        rank=rank,
        fractional_rank=_fractional_crossing(ranks, residuals, selected_index),
        normalization_dimension=float(dimensions[selected_index]),
        bulk_count=bulk_count,
        bulk_effective_count=effective,
        bulk_effective_fraction=fraction,
        trace_log=trace_log,
        trace_log_per_eval=float(trace_log / rank),
        lambda_cut=float(descending[rank - 1]),
        positive_count=count,
        number_of_sign_change_brackets=int(sign_changes.size),
        status=status,
        reference_rank=None if reference_rank is None else int(reference_rank),
    )


def compute_ecs_svd(
    weight: torch.Tensor,
    *,
    min_rank: int = 2,
    normalization_gamma: float = 0.0,
    reference_rank: Optional[int] = None,
    svd_device: Literal["cpu", "model"] = "cpu",
    numeric_epsilon: float = 1e-12,
) -> ECSSVDState:
    if weight.ndim != 2:
        raise ValueError("ECS SVD requires a matrix parameter")
    source = weight.detach()
    compute_device = source.device if svd_device == "model" else torch.device("cpu")
    # Float32 is materially faster than float64 for the MNIST matrices.  The
    # discrete ECS scan itself is still performed in NumPy float64.
    svd_input = source.to(device=compute_device, dtype=torch.float32)
    left, singular, right_t = torch.linalg.svd(svd_input, full_matrices=False)
    selection = select_self_consistent_ecs(
        singular.detach().cpu().double().numpy() ** 2,
        min_rank=min_rank,
        normalization_gamma=normalization_gamma,
        reference_rank=reference_rank,
        numeric_epsilon=numeric_epsilon,
    )
    rank = int(selection.rank)
    truncated = (left[:, :rank] * singular[:rank]) @ right_t[:rank, :]
    target_kwargs = {"device": source.device, "dtype": source.dtype}
    return ECSSVDState(
        selection=selection,
        full_weight=source.clone(),
        truncated_weight=truncated.to(**target_kwargs),
        left_vectors=left.to(**target_kwargs),
        singular_values=singular.to(**target_kwargs),
        right_vectors_t=right_t.to(**target_kwargs),
    )


def project_gradient_to_ecs(
    gradient: torch.Tensor,
    state: ECSSVDState,
    *,
    mode: ProjectionMode = "core",
) -> torch.Tensor:
    """Project a matrix gradient into the current rank-``m`` ECS geometry.

    ``core`` keeps both row and column directions inside the retained singular
    subspaces.  ``rank_m_tangent`` uses the orthogonal tangent projection for
    the rank-``m`` matrix manifold and therefore also permits infinitesimal
    rotations of those subspaces.
    """

    if gradient.shape != state.full_weight.shape:
        raise ValueError("gradient and weight shapes do not match")
    left = state.left_ecs
    right = state.right_ecs
    left_projector = left @ left.transpose(0, 1)
    right_projector = right @ right.transpose(0, 1)
    core = left_projector @ gradient @ right_projector
    if mode == "core":
        return core
    if mode == "rank_m_tangent":
        return left_projector @ gradient + gradient @ right_projector - core
    raise ValueError(f"unknown projection mode {mode!r}")

"""Bulk-effective self-consistent Effective Correlation Space selection.

This is the adaptive ECS construction used by the self-consistent-gap
experiment.  It is copied into this optimizer folder so the experiment remains
self-contained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Optional

import numpy as np

EffectiveRankMethod = Literal["participation_ratio", "entropy", "stable_rank"]
SupportPolicy = Literal["ecs", "midpoint", "power_law"]


@dataclass(frozen=True)
class SelfConsistentECS:
    ecs_rank: int
    fractional_rank: float
    normalization_dimension: float
    bulk_count: int
    bulk_effective_count: float
    bulk_effective_fraction: float
    trace_log: float
    trace_log_per_eval: float
    status: str
    method: EffectiveRankMethod
    gamma: float
    positive_count: int
    reference_rank: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveSupportState:
    """Outer-loop state installed in the optimizer for one matrix parameter."""

    ecs_rank: int
    normalization_dimension: float
    bulk_effective_count: float
    trace_log_per_eval: float
    status: str
    method: EffectiveRankMethod = "participation_ratio"
    normalization_gamma: float = 0.0
    pl_rank: Optional[int] = None
    working_rank: Optional[int] = None
    alpha: float = float("nan")
    erg_gap_sc: float = float("nan")
    source_epoch: Optional[int] = None
    source_global_step: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "AdaptiveSupportState":
        if isinstance(value, cls):
            return cls(**value.to_dict())
        if isinstance(value, Mapping):
            return cls(**dict(value))
        if isinstance(value, (int, np.integer)):
            rank = int(value)
            return cls(
                ecs_rank=rank,
                normalization_dimension=float(rank),
                bulk_effective_count=0.0,
                trace_log_per_eval=float("nan"),
                status="fixed_rank",
                working_rank=rank,
            )
        raise TypeError(
            "Support values must be AdaptiveSupportState, mappings, or integer ranks."
        )


def clean_positive_eigenvalues(
    eigenvalues: Any,
    *,
    positive_floor: float = 0.0,
) -> np.ndarray:
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    values = values[values > float(positive_floor)]
    values = np.sort(values)
    if values.size < 2:
        raise ValueError("Need at least two positive finite eigenvalues.")
    return values


def effective_contributor_count(
    values: Any,
    *,
    method: EffectiveRankMethod = "participation_ratio",
) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size == 0:
        return 0.0
    total = float(np.sum(x))
    if total <= 0.0:
        return 0.0

    if method == "participation_ratio":
        second = float(np.sum(x * x))
        return float(total * total / second) if second > 0.0 else 0.0
    if method == "entropy":
        p = x / total
        return float(np.exp(-np.sum(p * np.log(p))))
    if method == "stable_rank":
        maximum = float(np.max(x))
        return float(total / maximum) if maximum > 0.0 else 0.0
    raise ValueError(f"Unknown effective-rank method: {method!r}")


def _bulk_effective_prefixes(
    values_ascending: np.ndarray,
    *,
    method: EffectiveRankMethod,
) -> np.ndarray:
    """Effective count for prefixes of lengths 0,...,M-1."""
    values = values_ascending
    total_count = values.size
    prefix_sum = np.concatenate(([0.0], np.cumsum(values)))

    if method == "participation_ratio":
        prefix_second = np.concatenate(([0.0], np.cumsum(values * values)))
        counts = np.arange(total_count, dtype=int)
        sums = prefix_sum[counts]
        second = prefix_second[counts]
        out = np.zeros(total_count, dtype=float)
        valid = second > 0.0
        out[valid] = sums[valid] ** 2 / second[valid]
        return out

    if method == "entropy":
        xlogx = values * np.log(values)
        prefix_xlogx = np.concatenate(([0.0], np.cumsum(xlogx)))
        counts = np.arange(total_count, dtype=int)
        sums = prefix_sum[counts]
        moments = prefix_xlogx[counts]
        out = np.zeros(total_count, dtype=float)
        valid = sums > 0.0
        entropy = np.zeros(total_count, dtype=float)
        entropy[valid] = np.log(sums[valid]) - moments[valid] / sums[valid]
        out[valid] = np.exp(entropy[valid])
        return out

    if method == "stable_rank":
        out = np.zeros(total_count, dtype=float)
        for count in range(1, total_count):
            maximum = values[count - 1]
            out[count] = prefix_sum[count] / maximum if maximum > 0.0 else 0.0
        return out

    raise ValueError(f"Unknown effective-rank method: {method!r}")


def candidate_arrays(
    eigenvalues_ascending: Any,
    *,
    method: EffectiveRankMethod = "participation_ratio",
    gamma: float = 0.0,
    min_ecs_size: int = 2,
    positive_floor: float = 0.0,
) -> dict[str, np.ndarray]:
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    values = clean_positive_eigenvalues(
        eigenvalues_ascending,
        positive_floor=positive_floor,
    )
    total_count = int(values.size)
    minimum = int(np.clip(int(min_ecs_size), 1, total_count))
    ranks = np.arange(minimum, total_count + 1, dtype=int)
    bulk_counts = total_count - ranks
    bulk_effective_all = _bulk_effective_prefixes(values, method=method)
    bulk_effective = bulk_effective_all[bulk_counts]

    dimensions = (
        ranks.astype(float)
        + bulk_effective
        + float(gamma) * (bulk_counts.astype(float) - bulk_effective)
    )
    dimensions = np.clip(dimensions, ranks.astype(float), float(total_count))

    spectral_sum = float(np.sum(values))
    raw_log_desc_cumulative = np.cumsum(np.log(values[::-1]))
    retained_log_sum = raw_log_desc_cumulative[ranks - 1]
    trace_log = retained_log_sum + ranks * np.log(dimensions / spectral_sum)
    trace_log_per_eval = trace_log / ranks

    bulk_fraction = np.full(ranks.size, np.nan, dtype=float)
    nonempty = bulk_counts > 0
    bulk_fraction[nonempty] = bulk_effective[nonempty] / bulk_counts[nonempty]

    return {
        "values": values,
        "candidate_m": ranks,
        "normalization_dimension": dimensions,
        "bulk_count": bulk_counts,
        "bulk_effective_count": bulk_effective,
        "bulk_effective_fraction": bulk_fraction,
        "trace_log": trace_log,
        "trace_log_per_eval": trace_log_per_eval,
    }


def _select_candidate(
    ranks: np.ndarray,
    residuals: np.ndarray,
    *,
    reference_rank: Optional[int],
    numeric_eps: float,
) -> tuple[int, float, int, str]:
    exact = np.flatnonzero(np.isclose(residuals, 0.0, atol=numeric_eps, rtol=0.0))
    brackets = np.flatnonzero(residuals[:-1] * residuals[1:] < 0.0)
    candidates: list[int] = []
    roots: list[float] = []

    for index in exact:
        candidates.append(int(index))
        roots.append(float(ranks[index]))
    for index in brackets:
        left = int(index)
        right = left + 1
        candidates.extend([left, right])
        denominator = residuals[right] - residuals[left]
        if abs(float(denominator)) > numeric_eps:
            root = ranks[left] - residuals[left] * (
                ranks[right] - ranks[left]
            ) / denominator
        else:
            root = 0.5 * (ranks[left] + ranks[right])
        roots.append(float(root))

    candidates = sorted(set(candidates))
    if candidates:
        def score(index: int) -> tuple[float, float, float]:
            distance = (
                abs(float(ranks[index]) - float(reference_rank))
                if reference_rank is not None
                else 0.0
            )
            return abs(float(residuals[index])), distance, -float(ranks[index])

        chosen = min(candidates, key=score)
        fractional = min(roots, key=lambda root: abs(root - float(ranks[chosen])))
        status = "sign_change"
    else:
        chosen = int(np.nanargmin(np.abs(residuals)))
        fractional = float(ranks[chosen])
        status = "nearest_no_sign_change"

    return chosen, float(fractional), int(brackets.size), status


def solve_self_consistent_ecs(
    eigenvalues_ascending: Any,
    *,
    method: EffectiveRankMethod = "participation_ratio",
    gamma: float = 0.0,
    min_ecs_size: int = 2,
    reference_rank: Optional[int] = None,
    positive_floor: float = 0.0,
    numeric_eps: float = 1e-12,
) -> SelfConsistentECS:
    arrays = candidate_arrays(
        eigenvalues_ascending,
        method=method,
        gamma=gamma,
        min_ecs_size=min_ecs_size,
        positive_floor=positive_floor,
    )
    ranks = arrays["candidate_m"]
    residuals = arrays["trace_log_per_eval"]
    chosen, fractional, _, status = _select_candidate(
        ranks,
        residuals,
        reference_rank=reference_rank,
        numeric_eps=numeric_eps,
    )
    rank = int(ranks[chosen])
    return SelfConsistentECS(
        ecs_rank=rank,
        fractional_rank=fractional,
        normalization_dimension=float(arrays["normalization_dimension"][chosen]),
        bulk_count=int(arrays["bulk_count"][chosen]),
        bulk_effective_count=float(arrays["bulk_effective_count"][chosen]),
        bulk_effective_fraction=float(arrays["bulk_effective_fraction"][chosen]),
        trace_log=float(arrays["trace_log"][chosen]),
        trace_log_per_eval=float(residuals[chosen]),
        status=status,
        method=method,
        gamma=float(gamma),
        positive_count=int(arrays["values"].size),
        reference_rank=(int(reference_rank) if reference_rank is not None else None),
    )


def working_support_rank(
    *,
    ecs_rank: int,
    pl_rank: Optional[int],
    policy: SupportPolicy = "midpoint",
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    ecs = int(ecs_rank)
    pl = int(pl_rank) if pl_rank is not None else None
    if policy == "ecs":
        rank = ecs
    elif policy == "midpoint":
        rank = int(np.floor((ecs + pl) / 2.0)) if pl is not None else ecs
    elif policy == "power_law":
        rank = pl if pl is not None else ecs
    else:
        raise ValueError(f"Unknown support policy: {policy!r}")

    upper = int(maximum) if maximum is not None else max(rank, int(minimum))
    return int(np.clip(rank, int(minimum), upper))

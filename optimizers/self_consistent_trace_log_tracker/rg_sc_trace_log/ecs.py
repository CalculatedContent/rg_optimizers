"""Self-consistent Effective Correlation Space (ECS) selection.

This module implements the bulk-effective normalization introduced in the
self-consistent ECS notebook.  For a candidate retained rank ``m`` and a
positive ESD ``lambda_1 <= ... <= lambda_M``, the normalization dimension is

    D(m) = m + r_bulk(m) + gamma * ((M - m) - r_bulk(m)),

where ``r_bulk`` is an effective contributor count for the discarded bulk.
The default contributor count is the participation ratio.  The selected ECS
is an integer adjacent to a zero of

    F(m) = (1/m) sum_{i in top-m} log(D(m) * lambda_i / sum_j lambda_j).

The exhaustive integer scan is authoritative.  It is inexpensive for the
matrix sizes used in the MNIST experiments and avoids unstable fixed-point
iteration of the discrete retained rank.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Optional

import numpy as np
import pandas as pd

EffectiveRankMethod = Literal["participation_ratio", "entropy", "stable_rank"]
SupportPolicy = Literal["ecs", "midpoint", "power_law"]


@dataclass(frozen=True)
class SelfConsistentECS:
    """Selected self-consistent ECS and its adaptive normalization."""

    ecs_rank: int
    fractional_rank: float
    normalization_dimension: float
    bulk_count: int
    bulk_effective_count: float
    bulk_effective_fraction: float
    trace_log: float
    trace_log_per_eval: float
    lambda_cut_raw: float
    lambda_cut_scaled: float
    detx_nearest_at_selected_scale: int
    detx_first_below_at_selected_scale: int
    fixed_point_error_nearest: int
    fixed_point_error_first_below: int
    num_sign_change_brackets: int
    status: str
    method: EffectiveRankMethod
    gamma: float
    positive_count: int
    reference_rank: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveSupportState:
    """Optimizer-facing support state for one matrix parameter."""

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
            )
        raise TypeError(
            "Support values must be AdaptiveSupportState, mappings, or integer ranks."
        )


def clean_positive_eigenvalues(
    eigenvalues: Any,
    *,
    positive_floor: float = 0.0,
) -> np.ndarray:
    """Return finite positive eigenvalues in ascending order."""
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    values = values[values > float(positive_floor)]
    values = np.sort(values)
    if values.size < 2:
        raise ValueError("Need at least two positive finite eigenvalues.")
    return values


def rescale_eigenvalues_with_dimension(
    eigenvalues_ascending: Any,
    normalization_dimension: float,
    *,
    positive_floor: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Scale an ESD by ``D / sum(evals)``.

    Returns the rescaled ESD and the corresponding multiplicative scale on
    singular values/weights.
    """
    values = clean_positive_eigenvalues(
        eigenvalues_ascending,
        positive_floor=positive_floor,
    )
    dimension = float(normalization_dimension)
    if not np.isfinite(dimension) or dimension <= 0.0:
        raise ValueError("normalization_dimension must be positive and finite.")
    spectral_sum = float(np.sum(values))
    scale_squared = dimension / spectral_sum
    return values * scale_squared, float(np.sqrt(scale_squared))


def effective_contributor_count(
    values: Any,
    *,
    method: EffectiveRankMethod = "participation_ratio",
) -> float:
    """Scale-invariant effective number of positive contributors."""
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
        entropy = -float(np.sum(p * np.log(p)))
        return float(np.exp(entropy))

    if method == "stable_rank":
        maximum = float(np.max(x))
        return float(total / maximum) if maximum > 0.0 else 0.0

    raise ValueError(f"Unknown effective-rank method: {method!r}")


def bulk_effective_normalization_dimension(
    eigenvalues_ascending: Any,
    retained_count: int,
    *,
    method: EffectiveRankMethod = "participation_ratio",
    gamma: float = 0.0,
    positive_floor: float = 0.0,
) -> dict[str, float]:
    """Count retained ECS modes fully and discarded bulk modes effectively."""
    values = clean_positive_eigenvalues(
        eigenvalues_ascending,
        positive_floor=positive_floor,
    )
    total_count = int(values.size)
    retained = int(retained_count)
    if retained < 1 or retained > total_count:
        raise ValueError(f"retained_count must lie in [1, {total_count}].")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")

    bulk_count = total_count - retained
    bulk = values[:bulk_count]
    bulk_effective = effective_contributor_count(bulk, method=method)
    dimension = (
        retained
        + bulk_effective
        + float(gamma) * (bulk_count - bulk_effective)
    )
    dimension = float(np.clip(dimension, retained, total_count))
    return {
        "normalization_dimension": dimension,
        "bulk_count": float(bulk_count),
        "bulk_effective_count": float(bulk_effective),
        "bulk_effective_fraction": (
            float(bulk_effective / bulk_count) if bulk_count > 0 else float("nan")
        ),
    }


def tail_trace_log_curve(
    eigenvalues_ascending: Any,
    normalization_dimension: float,
    *,
    positive_floor: float = 0.0,
) -> pd.DataFrame:
    """Cumulative trace-log for the largest ``m=1,...,M`` eigenvalues."""
    values = clean_positive_eigenvalues(
        eigenvalues_ascending,
        positive_floor=positive_floor,
    )
    scaled, weight_scale = rescale_eigenvalues_with_dimension(
        values,
        normalization_dimension,
        positive_floor=positive_floor,
    )
    descending = scaled[::-1]
    cumulative = np.cumsum(np.log(descending))
    ranks = np.arange(1, values.size + 1, dtype=int)
    return pd.DataFrame(
        {
            "m": ranks,
            "trace_log": cumulative,
            "trace_log_per_eval": cumulative / ranks,
            "lambda_cut_scaled": descending,
            "weight_scale": weight_scale,
            "normalization_dimension": float(normalization_dimension),
        }
    )


def detx_nearest_from_dimension(
    eigenvalues_ascending: Any,
    normalization_dimension: float,
    *,
    positive_floor: float = 0.0,
) -> dict[str, float]:
    """Retained rank whose cumulative trace-log is nearest zero."""
    values = clean_positive_eigenvalues(
        eigenvalues_ascending,
        positive_floor=positive_floor,
    )
    curve = tail_trace_log_curve(
        values,
        normalization_dimension,
        positive_floor=positive_floor,
    )
    index = int(np.nanargmin(np.abs(curve["trace_log"].to_numpy(dtype=float))))
    rank = int(curve.iloc[index]["m"])
    return {
        "m": rank,
        "trace_log": float(curve.iloc[index]["trace_log"]),
        "trace_log_per_eval": float(curve.iloc[index]["trace_log_per_eval"]),
        "lambda_cut_raw": float(values[-rank]),
        "lambda_cut_scaled": float(curve.iloc[index]["lambda_cut_scaled"]),
        "normalization_dimension": float(normalization_dimension),
    }


def detx_first_below_from_dimension(
    eigenvalues_ascending: Any,
    normalization_dimension: float,
    *,
    positive_floor: float = 0.0,
) -> dict[str, float]:
    """Log-domain version of WeightWatcher's first-product-below-one rule."""
    values = clean_positive_eigenvalues(
        eigenvalues_ascending,
        positive_floor=positive_floor,
    )
    curve = tail_trace_log_curve(
        values,
        normalization_dimension,
        positive_floor=positive_floor,
    )
    below = np.flatnonzero(curve["trace_log"].to_numpy(dtype=float) < 0.0)
    index = int(below[0]) if below.size else len(curve) - 1
    rank = int(curve.iloc[index]["m"])
    return {
        "m": rank,
        "trace_log": float(curve.iloc[index]["trace_log"]),
        "trace_log_per_eval": float(curve.iloc[index]["trace_log_per_eval"]),
        "lambda_cut_raw": float(values[-rank]),
        "lambda_cut_scaled": float(curve.iloc[index]["lambda_cut_scaled"]),
        "normalization_dimension": float(normalization_dimension),
    }


def _effective_bulk_arrays(
    values: np.ndarray,
    *,
    method: EffectiveRankMethod,
) -> np.ndarray:
    """Effective bulk count for every prefix length ``0,...,M-1``."""
    total_count = values.size
    prefix_sum = np.concatenate(([0.0], np.cumsum(values)))

    if method == "participation_ratio":
        prefix_second = np.concatenate(([0.0], np.cumsum(values * values)))
        out = np.zeros(total_count, dtype=float)
        bulk_counts = np.arange(total_count, dtype=int)
        sums = prefix_sum[bulk_counts]
        seconds = prefix_second[bulk_counts]
        valid = seconds > 0.0
        out[valid] = sums[valid] ** 2 / seconds[valid]
        return out

    if method == "entropy":
        xlogx = np.zeros_like(values)
        positive = values > 0.0
        xlogx[positive] = values[positive] * np.log(values[positive])
        prefix_xlogx = np.concatenate(([0.0], np.cumsum(xlogx)))
        out = np.zeros(total_count, dtype=float)
        bulk_counts = np.arange(total_count, dtype=int)
        sums = prefix_sum[bulk_counts]
        moments = prefix_xlogx[bulk_counts]
        valid = sums > 0.0
        entropy = np.zeros(total_count, dtype=float)
        entropy[valid] = np.log(sums[valid]) - moments[valid] / sums[valid]
        out[valid] = np.exp(entropy[valid])
        return out

    if method == "stable_rank":
        out = np.zeros(total_count, dtype=float)
        for bulk_count in range(1, total_count):
            maximum = values[bulk_count - 1]
            out[bulk_count] = (
                prefix_sum[bulk_count] / maximum if maximum > 0.0 else 0.0
            )
        return out

    raise ValueError(f"Unknown effective-rank method: {method!r}")


def _candidate_arrays(
    eigenvalues_ascending: Any,
    *,
    method: EffectiveRankMethod,
    gamma: float,
    min_ecs_size: int,
    positive_floor: float,
) -> dict[str, np.ndarray]:
    values = clean_positive_eigenvalues(
        eigenvalues_ascending,
        positive_floor=positive_floor,
    )
    total_count = values.size
    minimum = int(max(1, min(int(min_ecs_size), total_count)))
    ranks = np.arange(minimum, total_count + 1, dtype=int)
    bulk_counts = total_count - ranks

    bulk_effective_all = _effective_bulk_arrays(values, method=method)
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
        "bulk_count": bulk_counts.astype(int),
        "bulk_effective_count": bulk_effective,
        "bulk_effective_fraction": bulk_fraction,
        "trace_log": trace_log,
        "trace_log_per_eval": trace_log_per_eval,
    }


def self_consistent_candidate_scan(
    eigenvalues_ascending: Any,
    *,
    method: EffectiveRankMethod = "participation_ratio",
    gamma: float = 0.0,
    min_ecs_size: int = 2,
    positive_floor: float = 0.0,
) -> pd.DataFrame:
    """Evaluate the adaptive trace-log residual for every candidate ECS rank."""
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    arrays = _candidate_arrays(
        eigenvalues_ascending,
        method=method,
        gamma=float(gamma),
        min_ecs_size=int(min_ecs_size),
        positive_floor=float(positive_floor),
    )
    return pd.DataFrame(
        {
            "candidate_m": arrays["candidate_m"],
            "normalization_dimension": arrays["normalization_dimension"],
            "bulk_count": arrays["bulk_count"],
            "bulk_effective_count": arrays["bulk_effective_count"],
            "bulk_effective_fraction": arrays["bulk_effective_fraction"],
            "trace_log": arrays["trace_log"],
            "trace_log_per_eval": arrays["trace_log_per_eval"],
            "method": method,
            "gamma": float(gamma),
        }
    )


def _select_index(
    ranks: np.ndarray,
    residuals: np.ndarray,
    *,
    reference_rank: Optional[int],
    numeric_eps: float,
) -> tuple[int, float, int, str]:
    exact = np.flatnonzero(np.isclose(residuals, 0.0, atol=numeric_eps, rtol=0.0))
    brackets = np.flatnonzero(residuals[:-1] * residuals[1:] < 0.0)

    candidates: list[int] = []
    fractional_roots: list[float] = []
    for index in exact:
        candidates.append(int(index))
        fractional_roots.append(float(ranks[index]))
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
        fractional_roots.append(float(root))

    candidates = sorted(set(candidates))
    if candidates:
        def score(index: int) -> tuple[float, float, float]:
            distance = (
                abs(float(ranks[index]) - float(reference_rank))
                if reference_rank is not None
                else 0.0
            )
            return (
                abs(float(residuals[index])),
                distance,
                -float(ranks[index]),
            )

        chosen = min(candidates, key=score)
        fractional = min(
            fractional_roots,
            key=lambda root: abs(root - float(ranks[chosen])),
        )
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
    """Solve the adaptive ECS by exhaustive integer scan."""
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")

    arrays = _candidate_arrays(
        eigenvalues_ascending,
        method=method,
        gamma=float(gamma),
        min_ecs_size=int(min_ecs_size),
        positive_floor=float(positive_floor),
    )
    ranks = arrays["candidate_m"]
    residuals = arrays["trace_log_per_eval"]
    chosen, fractional, bracket_count, status = _select_index(
        ranks,
        residuals,
        reference_rank=reference_rank,
        numeric_eps=float(numeric_eps),
    )

    values = arrays["values"]
    rank = int(ranks[chosen])
    dimension = float(arrays["normalization_dimension"][chosen])
    nearest = detx_nearest_from_dimension(
        values,
        dimension,
        positive_floor=positive_floor,
    )
    first_below = detx_first_below_from_dimension(
        values,
        dimension,
        positive_floor=positive_floor,
    )
    scale_squared = dimension / float(np.sum(values))

    return SelfConsistentECS(
        ecs_rank=rank,
        fractional_rank=fractional,
        normalization_dimension=dimension,
        bulk_count=int(arrays["bulk_count"][chosen]),
        bulk_effective_count=float(arrays["bulk_effective_count"][chosen]),
        bulk_effective_fraction=float(
            arrays["bulk_effective_fraction"][chosen]
        ),
        trace_log=float(arrays["trace_log"][chosen]),
        trace_log_per_eval=float(residuals[chosen]),
        lambda_cut_raw=float(values[-rank]),
        lambda_cut_scaled=float(values[-rank] * scale_squared),
        detx_nearest_at_selected_scale=int(nearest["m"]),
        detx_first_below_at_selected_scale=int(first_below["m"]),
        fixed_point_error_nearest=int(nearest["m"] - rank),
        fixed_point_error_first_below=int(first_below["m"] - rank),
        num_sign_change_brackets=bracket_count,
        status=status,
        method=method,
        gamma=float(gamma),
        positive_count=int(values.size),
        reference_rank=(int(reference_rank) if reference_rank is not None else None),
    )


def working_support_rank(
    *,
    ecs_rank: int,
    pl_rank: Optional[int],
    policy: SupportPolicy = "ecs",
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    """Choose the optimizer's working support from ECS and PL ranks."""
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

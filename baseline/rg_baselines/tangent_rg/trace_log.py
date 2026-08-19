"""Trace-log diagnostics with explicit support and normalization provenance."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .powerlaw_fit import positive_values


def normalize_esd(eigenvalues: Any, *, dimension: float) -> np.ndarray:
    """Scale positive eigenvalues so that their sum equals ``dimension``."""

    values = positive_values(eigenvalues)
    dimension = float(dimension)
    if not np.isfinite(dimension) or dimension <= 0.0:
        raise ValueError("dimension must be positive and finite")
    return values * (dimension / float(np.sum(values)))


def tail_trace_log_curve(
    eigenvalues: Any,
    *,
    normalization_dimension: float,
) -> pd.DataFrame:
    """Cumulative trace-log of the largest ``m=1,...,M`` normalized modes."""

    normalized = normalize_esd(
        eigenvalues,
        dimension=float(normalization_dimension),
    )
    descending = normalized[::-1]
    trace = np.cumsum(np.log(descending))
    ranks = np.arange(1, normalized.size + 1, dtype=int)
    return pd.DataFrame(
        {
            "m": ranks,
            "trace_log_total": trace,
            "trace_log_per_eval": trace / ranks,
            "lambda_cut_scaled": descending,
            "normalization_dimension": float(normalization_dimension),
        }
    )


def trace_log_at_rank(
    eigenvalues: Any,
    *,
    rank: int,
    normalization_dimension: float,
    rank_source: str,
) -> dict[str, Any]:
    """Evaluate trace-log at an independently supplied support rank."""

    curve = tail_trace_log_curve(
        eigenvalues,
        normalization_dimension=normalization_dimension,
    )
    rank = int(rank)
    if not 1 <= rank <= len(curve):
        raise ValueError(f"rank must lie in [1, {len(curve)}]")
    selected = curve.iloc[rank - 1]
    return {
        "support_rank": rank,
        "support_rank_source": str(rank_source),
        "support_selected_from_same_trace_log": False,
        "normalization_dimension": float(normalization_dimension),
        "trace_log_total": float(selected["trace_log_total"]),
        "trace_log_per_eval": float(selected["trace_log_per_eval"]),
        "lambda_cut_scaled": float(selected["lambda_cut_scaled"]),
    }


def nearest_trace_log_zero(
    eigenvalues: Any,
    *,
    normalization_dimension: float,
    minimum_rank: int = 1,
) -> dict[str, Any]:
    """Same-curve nearest-zero diagnostic; never a fixed-point certificate."""

    curve = tail_trace_log_curve(
        eigenvalues,
        normalization_dimension=normalization_dimension,
    )
    subset = curve[curve["m"] >= int(minimum_rank)]
    if subset.empty:
        raise ValueError("minimum_rank exceeds the positive spectral rank")
    position = int(
        np.nanargmin(np.abs(subset["trace_log_total"].to_numpy(dtype=float)))
    )
    selected = subset.iloc[position]
    trace_values = subset["trace_log_total"].to_numpy(dtype=float)
    brackets = int(np.count_nonzero(trace_values[:-1] * trace_values[1:] < 0.0))
    return {
        "support_rank": int(selected["m"]),
        "support_rank_source": "same_curve_nearest_zero_diagnostic",
        "support_selected_from_same_trace_log": True,
        "normalization_dimension": float(normalization_dimension),
        "trace_log_total": float(selected["trace_log_total"]),
        "trace_log_per_eval": float(selected["trace_log_per_eval"]),
        "lambda_cut_scaled": float(selected["lambda_cut_scaled"]),
        "num_sign_change_brackets": brackets,
    }


def compare_supports(
    eigenvalues: Any,
    *,
    normalization_dimension: float,
    supports: Iterable[tuple[str, int]],
    include_nearest_zero_audit: bool = True,
) -> pd.DataFrame:
    """Evaluate preregistered WW/PL/ECS supports side by side."""

    rows = [
        trace_log_at_rank(
            eigenvalues,
            rank=rank,
            normalization_dimension=normalization_dimension,
            rank_source=source,
        )
        for source, rank in supports
    ]
    if include_nearest_zero_audit:
        rows.append(
            nearest_trace_log_zero(
                eigenvalues,
                normalization_dimension=normalization_dimension,
            )
        )
    return pd.DataFrame(rows)


def trace_log_passes(
    row: dict[str, Any] | pd.Series,
    *,
    tolerance_per_eval: float,
) -> bool:
    """Declared fixed-point condition for an independently chosen support."""

    if bool(row.get("support_selected_from_same_trace_log", False)):
        return False
    value = float(row["trace_log_per_eval"])
    return bool(np.isfinite(value) and abs(value) <= float(tolerance_per_eval))

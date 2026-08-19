"""Fixed-point qualification and seed-level uncertainty tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from rg_baselines.statistics import summarize_numeric_metrics


def qualify_fixed_point(
    observations: pd.DataFrame,
    *,
    alpha_target: float = 2.0,
    alpha_half_width: float = 0.25,
    max_ks_D: float = 0.15,
    minimum_tail: int = 8,
    trace_log_tolerance: float = 0.10,
    persistence_measurements: int = 5,
    required_fraction: float = 0.80,
    group_columns: Sequence[str] = ("optimizer", "seed", "layer"),
) -> pd.DataFrame:
    """Certify persistent convergence, never a single crossing of alpha two.

    ``observations`` must already select the preregistered fit variant and the
    independently chosen trace-log support.  The classifier's rank-ten ESD is
    not silently exempted; its limited statistical power remains visible.
    """

    required = {
        *group_columns,
        "step",
        "alpha",
        "ks_D",
        "n_tail",
        "trace_log_per_eval",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"fixed-point table is missing columns: {sorted(missing)}")
    if "support_selected_from_same_trace_log" in observations.columns:
        if observations["support_selected_from_same_trace_log"].fillna(False).any():
            raise ValueError(
                "same-curve nearest-zero/detX supports cannot certify trace-log convergence"
            )

    rows: list[dict[str, Any]] = []
    for keys, group in observations.groupby(list(group_columns), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, keys))
        tail = group.sort_values("step").tail(int(persistence_measurements)).copy()
        alpha = pd.to_numeric(tail["alpha"], errors="coerce")
        distance = pd.to_numeric(tail["ks_D"], errors="coerce")
        n_tail = pd.to_numeric(tail["n_tail"], errors="coerce")
        trace = pd.to_numeric(tail["trace_log_per_eval"], errors="coerce")
        alpha_ok = (alpha - float(alpha_target)).abs() <= float(alpha_half_width)
        fit_ok = (distance <= float(max_ks_D)) & (n_tail >= int(minimum_tail))
        trace_ok = trace.abs() <= float(trace_log_tolerance)
        joint = alpha_ok & fit_ok & trace_ok
        count = int(len(tail))
        passed = int(joint.sum())
        fraction = float(passed / count) if count else 0.0
        rows.append(
            {
                **identity,
                "measurements_available": count,
                "measurements_required": int(persistence_measurements),
                "alpha_target": float(alpha_target),
                "alpha_half_width": float(alpha_half_width),
                "max_ks_D": float(max_ks_D),
                "minimum_tail": int(minimum_tail),
                "trace_log_tolerance": float(trace_log_tolerance),
                "alpha_pass_count": int(alpha_ok.sum()),
                "fit_quality_pass_count": int(fit_ok.sum()),
                "trace_log_pass_count": int(trace_ok.sum()),
                "joint_pass_count": passed,
                "joint_pass_fraction": fraction,
                "fixed_point_qualified": bool(
                    count >= int(persistence_measurements)
                    and fraction >= float(required_fraction)
                ),
                "low_rank_warning": str(identity.get("layer", "")) == "fc3.weight",
            }
        )
    return pd.DataFrame(rows)


def seed_confidence_intervals(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metrics: Sequence[str],
    expected_seeds: Sequence[int] = (1337, 2027, 31415),
) -> pd.DataFrame:
    """Two-sided 95% Student-t intervals across complete independent runs."""

    if "seed" not in frame.columns:
        raise ValueError("seed is the unit of replication and is required")
    expected = {int(seed) for seed in expected_seeds}
    present = {int(seed) for seed in frame["seed"].dropna().unique()}
    missing = expected - present
    if missing:
        raise RuntimeError(f"incomplete replicate set; missing seeds {sorted(missing)}")
    groups = tuple(column for column in group_columns if column != "seed")
    summary = summarize_numeric_metrics(
        frame[frame["seed"].isin(expected)],
        group_columns=groups,
        metrics=metrics,
    )
    incomplete = summary[pd.to_numeric(summary["n"], errors="coerce") != len(expected)]
    if not incomplete.empty:
        identity = [
            column
            for column in (*groups, "metric", "n")
            if column in incomplete.columns
        ]
        raise RuntimeError(
            "confidence intervals require every independent seed in every row:\n"
            + incomplete[identity].to_string(index=False)
        )
    return summary


def merge_fit_and_trace(
    fits: pd.DataFrame,
    traces: pd.DataFrame,
    *,
    keys: Sequence[str] = ("optimizer", "seed", "step", "layer", "fit_variant"),
    trace_support_source: str = "powerlaw_tail_count",
) -> pd.DataFrame:
    """Build the long-form qualification input without ambiguous joins."""

    selected = traces[traces["support_rank_source"] == trace_support_source].copy()
    available = [key for key in keys if key in fits.columns and key in selected.columns]
    if not available:
        raise ValueError("fit and trace frames have no declared join keys")
    return fits.merge(selected, on=available, how="inner", suffixes=("", "_trace"))


def manifest_scientific_claims() -> Mapping[str, str]:
    """Canonical labels persisted in reports and notebook captions."""

    return {
        "weight_esd": "observed weight spectrum",
        "finite_flow": "two-checkpoint beta/finite-flow surrogate, not a Jacobian",
        "polar_projection_jacobian": "Jacobian of W -> polar(W), not optimizer flow",
        "normalized_gram_jacobian": "Jacobian of W -> Gram(W)/trace(Gram(W))",
        "calibrated_training_map": (
            "local derivative conditional on batch, loss, optimizer and scheduler state"
        ),
        "weights_only_identifiability": (
            "optimizer beta and its Jacobian are not identifiable from W alone"
        ),
    }

"""Replicate aggregation and 95% Student-t confidence intervals.

The baseline notebooks use independent random seeds as the experimental
replicates.  Error bars are two-sided 95% confidence intervals for the mean,

    mean +/- t_{0.975, n-1} * sample_std / sqrt(n).

This is deliberately different from plotting batch-to-batch variation or
standard deviation.  The unit of replication is a complete training run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


# Two-sided 95% Student-t critical values t_{0.975, df}.  The baseline
# notebooks use n=3 (df=2), but the table supports common larger replicate
# counts without adding a SciPy dependency.
_T_CRITICAL_975: dict[int, float] = {
    1: 12.7062047364,
    2: 4.3026527297,
    3: 3.1824463053,
    4: 2.7764451052,
    5: 2.5705818356,
    6: 2.4469118511,
    7: 2.3646242510,
    8: 2.3060041350,
    9: 2.2621571629,
    10: 2.2281388520,
    11: 2.2009851601,
    12: 2.1788128297,
    13: 2.1603686565,
    14: 2.1447866879,
    15: 2.1314495456,
    16: 2.1199052992,
    17: 2.1098155778,
    18: 2.1009220402,
    19: 2.0930240544,
    20: 2.0859634473,
    21: 2.0796138447,
    22: 2.0738730679,
    23: 2.0686576104,
    24: 2.0638985616,
    25: 2.0595385528,
    26: 2.0555294386,
    27: 2.0518305165,
    28: 2.0484071418,
    29: 2.0452296421,
    30: 2.0422724563,
}


def student_t_critical_95(sample_count: int) -> float:
    """Return the two-sided 95% Student-t critical value for ``sample_count``.

    For more than 31 replicates, a smooth large-df approximation is sufficient
    for plotting.  The exact values that matter for the default three-seed
    protocol are tabulated above.
    """

    n = int(sample_count)
    if n < 2:
        return float("nan")
    df = n - 1
    if df in _T_CRITICAL_975:
        return _T_CRITICAL_975[df]
    # Cornish-Fisher expansion around the 0.975 standard-normal quantile.
    z = 1.959963984540054
    inv_df = 1.0 / float(df)
    return float(
        z
        + (z**3 + z) * inv_df / 4.0
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) * inv_df**2 / 96.0
    )


def summarize_numeric_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metrics: Sequence[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate numeric metrics into long-form mean/error-bar rows.

    Parameters
    ----------
    frame:
        One row per independent replicate and measurement unit.
    group_columns:
        Columns that identify a plotted point, for example
        ``("run", "optimizer", "epoch")`` or
        ``("run", "optimizer", "layer", "epoch")``.
    metrics:
        Numeric columns to summarize.
    confidence:
        Currently only 0.95 is supported so the reported interval has one
        unambiguous interpretation throughout the notebooks.

    Returns
    -------
    pandas.DataFrame
        Long-form rows with ``metric``, ``n``, ``mean``, sample ``std``,
        ``sem``, Student-t critical value, interval half-width, bounds, minimum,
        and maximum.
    """

    if not math.isclose(float(confidence), 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Only two-sided 95% Student-t confidence intervals are supported.")

    groups = tuple(str(column) for column in group_columns)
    metric_names = tuple(str(metric) for metric in metrics)
    missing = [column for column in (*groups, *metric_names) if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot summarize missing columns: {missing}")

    rows: list[dict[str, object]] = []
    grouped = frame.groupby(list(groups), dropna=False, sort=True)
    for keys, group in grouped:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(groups, key_tuple))
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            n = int(values.size)
            if n == 0:
                continue
            mean = float(np.mean(values))
            if n >= 2:
                std = float(np.std(values, ddof=1))
                sem = float(std / math.sqrt(n))
                critical = student_t_critical_95(n)
                half_width = float(critical * sem)
                ci_low = float(mean - half_width)
                ci_high = float(mean + half_width)
            else:
                std = sem = critical = half_width = ci_low = ci_high = float("nan")

            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "confidence": 0.95,
                    "critical_value": critical,
                    "ci_half_width": half_width,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )

    return pd.DataFrame(rows)


def require_complete_summary(
    summary: pd.DataFrame,
    *,
    expected_replicates: int,
    required_metrics: Sequence[str],
) -> None:
    """Reject missing or statistically incomplete required summary rows."""

    if summary.empty:
        raise RuntimeError("replicate summary is empty")
    missing_metrics = set(required_metrics) - set(summary["metric"].astype(str))
    if missing_metrics:
        raise RuntimeError(f"replicate summary is missing metrics: {sorted(missing_metrics)}")

    required = summary[summary["metric"].isin(required_metrics)].copy()
    if (required["n"].astype(int) != int(expected_replicates)).any():
        bad = required.loc[
            required["n"].astype(int) != int(expected_replicates),
            [column for column in required.columns if column in {"metric", "epoch", "layer", "n"}],
        ]
        raise RuntimeError(
            "required error bars do not contain every seed:\n"
            + bad.to_string(index=False)
        )
    if int(expected_replicates) >= 2:
        numeric = required[["mean", "sem", "ci_half_width", "ci_low", "ci_high"]]
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise RuntimeError("required confidence-interval statistics contain non-finite values")

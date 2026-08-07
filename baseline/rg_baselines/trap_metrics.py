"""Correlation-trap extraction from WeightWatcher randomized analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import SpectralCheckpoint


def _integer_count(value: Any, *, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WeightWatcher returned no valid {name}") from exc
    if not np.isfinite(number) or number < 0.0 or not np.isclose(number, round(number)):
        raise ValueError(f"WeightWatcher returned invalid {name}={value!r}")
    return int(round(number))


def correlation_trap_count_from_row(row: pd.Series) -> int:
    """Return the randomized MP outlier count reported by WeightWatcher.

    WeightWatcher exposes the canonical result as ``num_traps``. The internal
    randomized-MP alias ``rand_num_spikes`` is accepted only for API
    compatibility, and both values must agree when both columns are present.
    """

    values: dict[str, int] = {}
    for column in ("num_traps", "rand_num_spikes"):
        if column in row.index and not pd.isna(row[column]):
            values[column] = _integer_count(row[column], name=column)
    if not values:
        raise ValueError(
            "WeightWatcher analyze(randomize=True) did not return num_traps"
        )
    if len(set(values.values())) != 1:
        raise ValueError(f"WeightWatcher trap-count columns disagree: {values}")
    return next(iter(values.values()))


def attach_correlation_traps(checkpoint: SpectralCheckpoint) -> SpectralCheckpoint:
    """Attach strict per-layer ``num_traps`` values to a spectral checkpoint."""

    details = checkpoint.details.copy()
    metrics = checkpoint.metrics.copy()
    if "layer_id" not in details or "layer_id" not in metrics:
        raise RuntimeError("WeightWatcher trap merge requires layer_id")

    valid_metrics = metrics[metrics["status"].astype(str).eq("ok")].copy()
    required_layer_ids = set(valid_metrics["layer_id"].astype(int))
    relevant_details = details[
        details["layer_id"].astype(int).isin(required_layer_ids)
    ].copy()

    trap_rows: list[dict[str, Any]] = []
    for _, row in relevant_details.iterrows():
        trap_rows.append(
            {
                "layer_id": int(row["layer_id"]),
                "num_traps": correlation_trap_count_from_row(row),
            }
        )
    traps = pd.DataFrame(trap_rows, columns=["layer_id", "num_traps"])
    if traps["layer_id"].duplicated().any():
        raise RuntimeError("WeightWatcher returned duplicate layer_id values")

    # Preserve the canonical count in the full details table when it is present,
    # but do not make a skipped/non-matrix row invalidate otherwise complete FC metrics.
    details["num_traps"] = details.apply(
        lambda row: (
            correlation_trap_count_from_row(row)
            if int(row["layer_id"]) in required_layer_ids
            else np.nan
        ),
        axis=1,
    )
    metrics = metrics.drop(columns=["num_traps", "num_traps_source"], errors="ignore")
    metrics = metrics.merge(traps, on="layer_id", how="left", validate="many_to_one")
    metrics["num_traps_source"] = (
        "WeightWatcher analyze(randomize=True) randomized MP fit"
    )

    valid = metrics[metrics["status"].astype(str).eq("ok")]
    if valid.empty or valid["num_traps"].isna().any():
        raise RuntimeError("A valid WeightWatcher layer is missing num_traps")
    counts = valid["num_traps"].to_numpy(dtype=float)
    if (counts < 0.0).any() or not np.allclose(counts, np.rint(counts)):
        raise RuntimeError("WeightWatcher num_traps must be non-negative integers")

    checkpoint.details = details
    checkpoint.metrics = metrics
    return checkpoint

"""Static spectral diagnostics used by the RG experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def normalize_esd_like_weightwatcher(eigenvalues: Any) -> dict[str, Any]:
    """Clean a complete ESD and rescale it to mean eigenvalue one."""
    raw = np.asarray(eigenvalues, dtype=float).reshape(-1)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raise ValueError("WeightWatcher returned an empty ESD.")

    scale = max(float(np.max(np.abs(raw))), 1.0)
    raw = np.where(np.abs(raw) <= 1e-12 * scale, 0.0, raw)
    if np.any(raw < 0.0):
        raise ValueError("The ESD contains negative eigenvalues after cleanup.")

    trace = float(np.sum(raw))
    if trace <= 0.0:
        raise ValueError("The ESD has non-positive trace.")

    full_dimension = int(raw.size)
    raw_positive_desc = np.sort(raw[raw > 0.0])[::-1]
    if raw_positive_desc.size < 3:
        raise ValueError("Fewer than three positive eigenvalues are available.")

    normalized_desc = full_dimension * raw_positive_desc / trace
    return {
        "raw_complete": raw,
        "raw_positive_desc": raw_positive_desc,
        "normalized_desc": normalized_desc,
        "full_dimension": full_dimension,
        "positive_count": int(raw_positive_desc.size),
        "zero_count": int(full_dimension - raw_positive_desc.size),
        "spectral_trace": trace,
    }


def trace_log_rank(normalized_desc: np.ndarray) -> tuple[int, float]:
    """Return the retained rank whose cumulative trace-log is nearest zero."""
    values = np.asarray(normalized_desc, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("No positive normalized eigenvalues.")
    cumulative = np.cumsum(np.log(values))
    index = int(np.argmin(np.abs(cumulative)))
    return index + 1, float(cumulative[index])


def shell_balance_metrics(
    values_desc: Any,
    *,
    n_shells: int = 5,
    min_count_per_shell: int = 1,
    min_retained_for_reliable: int = 20,
    min_shells_for_reliable: int = 3,
    min_decades_for_reliable: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Measure energy balance across equal-width logarithmic shells.

    For shell energy G_b = sum_{i in b} lambda_i, fit
    log G_b = intercept + beta_E log Lambda_b. Scale balance is beta_E=0.
    """
    values = np.asarray(values_desc, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    n = int(values.size)

    defaults = {
        "beta_E_midpoint": np.nan,
        "alpha_from_beta_midpoint": np.nan,
        "shell_energy_rms_midpoint": np.nan,
        "shell_roughness_rms_midpoint": np.nan,
        "rg_shell_shift_rms_midpoint": np.nan,
        "shells_used_midpoint": 0,
        "dynamic_range_decades_midpoint": np.nan,
        "scale_balance_reliable": False,
    }
    empty = pd.DataFrame(columns=[
        "shell", "lambda_left", "lambda_right", "lambda_center", "count",
        "energy", "relative_energy", "log_energy", "fitted_log_energy",
        "log_balance_residual", "log_roughness_residual",
    ])
    if n < 3 or int(n_shells) < 2:
        return defaults, empty

    low = float(np.min(values))
    high = float(np.max(values))
    if not high > low:
        return defaults, empty

    dynamic_range = float(np.log10(high / low))
    used = min(int(n_shells), max(2, n // max(int(min_count_per_shell), 1)))
    while used >= 2:
        edges = np.geomspace(low, np.nextafter(high, np.inf), used + 1)
        counts, _ = np.histogram(values, bins=edges)
        energies, _ = np.histogram(values, bins=edges, weights=values)
        if np.all(counts >= int(min_count_per_shell)) and np.all(energies > 0.0):
            break
        used -= 1
    if used < 2:
        return defaults, empty

    centers = np.sqrt(edges[:-1] * edges[1:])
    log_centers = np.log(centers)
    log_energies = np.log(energies)
    beta, intercept = np.polyfit(log_centers, log_energies, deg=1)
    fitted = intercept + beta * log_centers

    flat_residual = log_energies - np.mean(log_energies)
    roughness_residual = log_energies - fitted
    log_steps = np.diff(log_energies)
    geometric_mean_energy = float(np.exp(np.mean(log_energies)))

    reliable = bool(
        n >= int(min_retained_for_reliable)
        and used >= int(min_shells_for_reliable)
        and dynamic_range >= float(min_decades_for_reliable)
    )
    metrics = {
        "beta_E_midpoint": float(beta),
        "alpha_from_beta_midpoint": float(2.0 - beta),
        "shell_energy_rms_midpoint": float(np.sqrt(np.mean(flat_residual**2))),
        "shell_roughness_rms_midpoint": float(np.sqrt(np.mean(roughness_residual**2))),
        "rg_shell_shift_rms_midpoint": (
            float(np.sqrt(np.mean(log_steps**2))) if log_steps.size else np.nan
        ),
        "shells_used_midpoint": int(used),
        "dynamic_range_decades_midpoint": dynamic_range,
        "scale_balance_reliable": reliable,
    }
    shell_table = pd.DataFrame({
        "shell": np.arange(1, used + 1),
        "lambda_left": edges[:-1],
        "lambda_right": edges[1:],
        "lambda_center": centers,
        "count": counts.astype(int),
        "energy": energies,
        "relative_energy": energies / geometric_mean_energy,
        "log_energy": log_energies,
        "fitted_log_energy": fitted,
        "log_balance_residual": flat_residual,
        "log_roughness_residual": roughness_residual,
    })
    return metrics, shell_table

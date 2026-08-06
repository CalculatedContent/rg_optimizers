"""WeightWatcher checkpoint adapter for AdaptiveSpectralGuard."""

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class WeightWatcherCheckpoint:
    details: pd.DataFrame
    metrics: pd.DataFrame


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_value(row: pd.Series, names: list[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row.index:
            value = row[name]
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            return value
    return default


def _layer_name(row: pd.Series) -> str:
    for column in ("longname", "name"):
        if column in row.index and pd.notna(row[column]):
            return str(row[column])
    return ""


def _match_parameter_name(
    layer_name: str,
    parameter_names: list[str],
) -> Optional[str]:
    candidates = [layer_name]
    if layer_name and not layer_name.endswith(".weight"):
        candidates.append(f"{layer_name}.weight")
    for candidate in candidates:
        if candidate in parameter_names:
            return candidate
    matches = [
        parameter
        for parameter in parameter_names
        if any(
            parameter.endswith(candidate)
            for candidate in candidates
            if candidate
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _analyze_compat(
    watcher: Any,
    *,
    min_evals: int,
    max_evals: Optional[int],
) -> pd.DataFrame:
    try:
        parameters = inspect.signature(watcher.analyze).parameters
    except (TypeError, ValueError):
        parameters = {}

    kwargs: dict[str, Any] = {
        "plot": False,
        "randomize": False,
        "min_evals": int(min_evals),
        "savefig": False,
    }
    if not parameters or "start_ids" in parameters:
        kwargs["start_ids"] = 0
    if not parameters or "ERG" in parameters:
        kwargs["ERG"] = True
    elif "detX" in parameters:
        kwargs["detX"] = True
    else:
        raise RuntimeError(
            "This WeightWatcher version exposes neither ERG nor detX"
        )
    if max_evals is not None:
        kwargs["max_evals"] = int(max_evals)
    return watcher.analyze(**kwargs)


def normalize_esd(eigenvalues: Any) -> np.ndarray:
    raw = np.asarray(eigenvalues, dtype=float).reshape(-1)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raise ValueError("WeightWatcher returned an empty ESD")
    scale = max(float(np.max(np.abs(raw))), 1.0)
    raw = np.where(np.abs(raw) <= 1e-12 * scale, 0.0, raw)
    if np.any(raw < 0.0):
        raise ValueError("Negative eigenvalues remain after cleanup")
    trace = float(np.sum(raw))
    if trace <= 0.0:
        raise ValueError("The ESD has non-positive trace")
    positive = np.sort(raw[raw > 0.0])[::-1]
    if positive.size < 3:
        raise ValueError("Fewer than three positive eigenvalues")
    return raw.size * positive / trace


def shell_beta_metrics(
    values_desc: Any,
    *,
    n_shells: int = 5,
    min_retained: int = 20,
    min_shells: int = 3,
    min_decades: float = 0.50,
) -> dict[str, Any]:
    values = np.asarray(values_desc, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    defaults = {
        "beta_E_midpoint": np.nan,
        "alpha_from_beta_midpoint": np.nan,
        "shell_energy_rms_midpoint": np.nan,
        "shells_used_midpoint": 0,
        "dynamic_range_decades_midpoint": np.nan,
        "scale_balance_reliable": False,
    }
    if values.size < 3:
        return defaults
    low, high = float(values.min()), float(values.max())
    if not high > low > 0.0:
        return defaults

    dynamic_range = math.log10(high / low)
    used = min(int(n_shells), int(values.size))
    while used >= 2:
        edges = np.geomspace(low, np.nextafter(high, np.inf), used + 1)
        counts, _ = np.histogram(values, bins=edges)
        energies, _ = np.histogram(values, bins=edges, weights=values)
        if np.all(counts > 0) and np.all(energies > 0):
            break
        used -= 1
    if used < 2:
        return defaults

    centers = np.sqrt(edges[:-1] * edges[1:])
    x = np.log(centers)
    y = np.log(energies)
    beta, _ = np.polyfit(x, y, 1)
    flat = y - np.mean(y)
    reliable = bool(
        values.size >= int(min_retained)
        and used >= int(min_shells)
        and dynamic_range >= float(min_decades)
    )
    return {
        "beta_E_midpoint": float(beta),
        "alpha_from_beta_midpoint": float(2.0 - beta),
        "shell_energy_rms_midpoint": float(
            np.sqrt(np.mean(flat**2))
        ),
        "shells_used_midpoint": int(used),
        "dynamic_range_decades_midpoint": float(dynamic_range),
        "scale_balance_reliable": reliable,
    }


def analyze_weightwatcher_checkpoint(
    model: torch.nn.Module,
    *,
    run_label: str,
    epoch: int,
    global_step: int,
    min_evals: int = 10,
    max_evals: Optional[int] = None,
    n_shells: int = 5,
    min_beta_retained: int = 20,
    min_beta_decades: float = 0.50,
) -> WeightWatcherCheckpoint:
    """Return direct WeightWatcher alpha/ERG metrics plus beta diagnostics."""

    try:
        import weightwatcher as ww
    except ImportError as exc:
        raise ImportError(
            "Install WeightWatcher with `pip install weightwatcher`"
        ) from exc

    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.eval()
    watcher = ww.WeightWatcher(model=model_cpu)
    details = _analyze_compat(
        watcher,
        min_evals=min_evals,
        max_evals=max_evals,
    )
    if not isinstance(details, pd.DataFrame) or details.empty:
        return WeightWatcherCheckpoint(pd.DataFrame(), pd.DataFrame())

    parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2
    ]
    rows: list[dict[str, Any]] = []

    for _, row in details.iterrows():
        layer_id = _safe_int(_row_value(row, ["layer_id"]))
        if layer_id is None:
            continue
        layer_name = _layer_name(row)
        parameter_name = _match_parameter_name(
            layer_name,
            parameter_names,
        )

        try:
            alpha = _safe_float(_row_value(row, ["alpha"]))
            if not np.isfinite(alpha):
                raise ValueError(
                    "WeightWatcher did not return a usable alpha"
                )
            erg_gap = _safe_int(_row_value(row, ["ERG_gap"]))
            if erg_gap is None:
                raise ValueError(
                    "WeightWatcher analyze(..., ERG=True) did not return ERG_gap"
                )
            detx = _safe_int(
                _row_value(row, ["detX_num", "num_ERG_spikes"])
            )
            mpl = _safe_int(
                _row_value(
                    row,
                    ["num_pl_spikes", "num_evals_in_tail", "tail_size"],
                )
            )
            if detx is None or detx <= 0:
                raise ValueError(
                    "WeightWatcher did not return a usable detX_num"
                )
            if mpl is None or mpl <= 0:
                raise ValueError(
                    "WeightWatcher did not return a usable num_pl_spikes"
                )

            esd = watcher.get_ESD(layer=int(layer_id))
            normalized = normalize_esd(esd)
            count = int(normalized.size)
            detx = int(np.clip(detx, 3, count))
            mpl = int(np.clip(mpl, 3, count))
            midpoint = int(
                np.clip(math.floor((detx + mpl) / 2.0), 3, count)
            )
            retained = normalized[:midpoint]
            trace_log = float(np.mean(np.log(retained)))
            beta = shell_beta_metrics(
                retained,
                n_shells=n_shells,
                min_retained=min_beta_retained,
                min_decades=min_beta_decades,
            )

            rows.append(
                {
                    "run": str(run_label),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "layer_id": int(layer_id),
                    "layer_name": layer_name,
                    "parameter_name": parameter_name,
                    "alpha": float(alpha),
                    "alpha_source": "WeightWatcher",
                    "ERG_gap": int(erg_gap),
                    "ERG_gap_source": "WeightWatcher",
                    "detX_num": int(detx),
                    "num_pl_spikes": int(mpl),
                    "m_midpoint": int(midpoint),
                    "boundary_overlap_ratio": float(
                        min(detx, mpl) / max(detx, mpl)
                    ),
                    "trace_log_midpoint_per_eval": trace_log,
                    "geometric_mean_midpoint": float(np.exp(trace_log)),
                    "xmin": _safe_float(_row_value(row, ["xmin"])),
                    **beta,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "run": str(run_label),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "layer_id": int(layer_id),
                    "layer_name": layer_name,
                    "parameter_name": parameter_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return WeightWatcherCheckpoint(
        details=details.copy(),
        metrics=pd.DataFrame(rows),
    )

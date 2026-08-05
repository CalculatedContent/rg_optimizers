"""WeightWatcher checkpoint analysis and midpoint ECS selection."""

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from .spectral import (
    normalize_esd_like_weightwatcher,
    shell_balance_metrics,
    trace_log_rank,
)


@dataclass
class WeightWatcherCheckpoint:
    """WeightWatcher results and midpoint supports for one checkpoint."""

    details: pd.DataFrame
    metrics: pd.DataFrame
    supports: dict[str, int]


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


def _match_parameter_name(layer_name: str, parameter_names: list[str]) -> Optional[str]:
    candidates = [layer_name]
    if layer_name and not layer_name.endswith(".weight"):
        candidates.append(f"{layer_name}.weight")
    for candidate in candidates:
        if candidate in parameter_names:
            return candidate

    suffix_matches = [
        parameter_name
        for parameter_name in parameter_names
        if any(parameter_name.endswith(candidate) for candidate in candidates if candidate)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    short = layer_name.split(".")[-1] if layer_name else ""
    short_matches = [
        parameter_name
        for parameter_name in parameter_names
        if parameter_name.endswith(f"{short}.weight")
    ]
    return short_matches[0] if len(short_matches) == 1 else None


def _analyze_compat(
    watcher: Any,
    *,
    min_evals: int,
    max_evals: Optional[int],
) -> pd.DataFrame:
    """Run WeightWatcher without passing max_evals=None."""
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
        raise RuntimeError("This WeightWatcher version exposes neither ERG nor detX.")

    if max_evals is not None:
        max_evals_int = int(max_evals)
        kwargs["max_evals"] = max_evals_int
        if max_evals_int > 0 and (not parameters or "max_N" in parameters):
            default_max_n = 50000
            if parameters and parameters["max_N"].default not in {
                inspect.Parameter.empty,
                None,
            }:
                try:
                    default_max_n = int(parameters["max_N"].default)
                except (TypeError, ValueError):
                    pass
            kwargs["max_N"] = max(default_max_n, max_evals_int)
    return watcher.analyze(**kwargs)


def analyze_weightwatcher_checkpoint(
    model: torch.nn.Module,
    *,
    run_label: str,
    epoch: int,
    global_step: int = 0,
    min_evals: int = 10,
    max_evals: Optional[int] = None,
    n_shells: int = 5,
    min_count_per_shell: int = 1,
    min_retained_for_reliable: int = 20,
    min_shells_for_reliable: int = 3,
    min_decades_for_reliable: float = 0.5,
) -> WeightWatcherCheckpoint:
    """Collect alpha, ERG gap, midpoint trace-log, and beta_E for a model."""
    try:
        import weightwatcher as ww
    except ImportError as exc:
        raise ImportError(
            "WeightWatcher is required. Install it with `pip install weightwatcher`."
        ) from exc

    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.eval()
    watcher = ww.WeightWatcher(model=model_cpu)
    details = _analyze_compat(
        watcher,
        min_evals=int(min_evals),
        max_evals=max_evals,
    )
    if not isinstance(details, pd.DataFrame) or details.empty:
        return WeightWatcherCheckpoint(pd.DataFrame(), pd.DataFrame(), {})

    parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.ndim == 2
    ]
    rows: list[dict[str, Any]] = []
    supports: dict[str, int] = {}

    for _, row in details.iterrows():
        layer_id = _safe_int(_row_value(row, ["layer_id"]), default=None)
        if layer_id is None:
            continue
        layer_name = _layer_name(row)
        parameter_name = _match_parameter_name(layer_name, parameter_names)

        try:
            esd = watcher.get_ESD(layer=int(layer_id))
            spectrum = normalize_esd_like_weightwatcher(esd)
            raw_desc = spectrum["raw_positive_desc"]
            normalized_desc = spectrum["normalized_desc"]
            n_positive = spectrum["positive_count"]

            m_trace = _safe_int(
                _row_value(row, ["detX_num", "num_ERG_spikes"]),
                default=None,
            )
            trace_source = "WeightWatcher"
            if m_trace is None or m_trace <= 0:
                m_trace, _ = trace_log_rank(normalized_desc)
                trace_source = "log-domain fallback"

            m_pl = _safe_int(
                _row_value(row, ["num_pl_spikes", "num_evals_in_tail", "tail_size"]),
                default=None,
            )
            xmin = _safe_float(_row_value(row, ["xmin"]))
            if (m_pl is None or m_pl <= 0) and np.isfinite(xmin) and xmin > 0.0:
                m_pl = int(np.count_nonzero(raw_desc >= xmin))
            if m_pl is None or m_pl <= 0:
                raise ValueError("WeightWatcher did not provide a usable PL boundary.")

            m_trace = int(np.clip(m_trace, 3, n_positive))
            m_pl = int(np.clip(m_pl, 3, n_positive))
            m_midpoint = int(np.clip(math.floor((m_trace + m_pl) / 2.0), 3, n_positive))

            midpoint = normalized_desc[:m_midpoint]
            trace_log_per_eval = float(np.mean(np.log(midpoint)))
            shell_metrics, _ = shell_balance_metrics(
                midpoint,
                n_shells=int(n_shells),
                min_count_per_shell=int(min_count_per_shell),
                min_retained_for_reliable=int(min_retained_for_reliable),
                min_shells_for_reliable=int(min_shells_for_reliable),
                min_decades_for_reliable=float(min_decades_for_reliable),
            )
            alpha = _safe_float(_row_value(row, ["alpha"]))
            erg_gap = int(m_trace - m_pl)

            rows.append({
                "run": str(run_label),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "layer_id": int(layer_id),
                "layer_name": layer_name,
                "parameter_name": parameter_name,
                "alpha": alpha,
                "xmin": xmin,
                "detX_num": int(m_trace),
                "num_pl_spikes": int(m_pl),
                "ERG_gap": erg_gap,
                "m_midpoint": int(m_midpoint),
                "boundary_overlap_ratio": float(min(m_trace, m_pl) / max(m_trace, m_pl)),
                "trace_boundary_source": trace_source,
                "trace_log_midpoint_per_eval": trace_log_per_eval,
                "geometric_mean_midpoint": float(np.exp(trace_log_per_eval)),
                **shell_metrics,
                "status": "ok",
                "error": "",
            })
            if parameter_name is not None:
                supports[parameter_name] = int(m_midpoint)

        except Exception as exc:
            rows.append({
                "run": str(run_label),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "layer_id": int(layer_id),
                "layer_name": layer_name,
                "parameter_name": parameter_name,
                "status": "failed",
                "error": str(exc),
            })

    return WeightWatcherCheckpoint(
        details=details.copy(),
        metrics=pd.DataFrame(rows),
        supports=supports,
    )

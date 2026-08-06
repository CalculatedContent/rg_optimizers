"""WeightWatcher outer-loop analysis for the spectral RG-flow projector."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from .ecs import (
    AdaptiveSupportState,
    EffectiveRankMethod,
    SupportPolicy,
    clean_positive_eigenvalues,
    solve_self_consistent_ecs,
    working_support_rank,
)


@dataclass
class SpectralFlowCheckpoint:
    details: pd.DataFrame
    metrics: pd.DataFrame
    supports: dict[str, AdaptiveSupportState]


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
        return int(round(float(value)))
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
            value = str(row[column])
            if value:
                return value
    return ""


def _match_parameter_name(
    layer_name: str,
    short_name: str,
    parameter_names: list[str],
) -> Optional[str]:
    candidates: list[str] = []
    for candidate in (layer_name, short_name):
        if not candidate:
            continue
        candidates.append(candidate)
        if not candidate.endswith(".weight"):
            candidates.append(f"{candidate}.weight")

    for candidate in candidates:
        if candidate in parameter_names:
            return candidate
    suffixes = [
        name
        for name in parameter_names
        if any(name.endswith(candidate) for candidate in candidates)
    ]
    return suffixes[0] if len(suffixes) == 1 else None


def _analyze_compat(
    watcher: Any,
    *,
    min_evals: int,
    max_evals: Optional[int],
    svd_method: str,
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
    if not parameters or "vectors" in parameters:
        kwargs["vectors"] = False
    if not parameters or "start_ids" in parameters:
        kwargs["start_ids"] = 0
    if not parameters or "ERG" in parameters:
        kwargs["ERG"] = True
    elif "detX" in parameters:
        kwargs["detX"] = True
    if not parameters or "svd_method" in parameters:
        kwargs["svd_method"] = str(svd_method)
    if max_evals is not None:
        kwargs["max_evals"] = int(max_evals)
        if not parameters or "max_N" in parameters:
            kwargs["max_N"] = max(50000, int(max_evals))
    return watcher.analyze(**kwargs)


def _make_get_esd_params(*, min_evals: int, svd_method: str) -> Optional[dict[str, Any]]:
    try:
        from weightwatcher.constants import (
            DEFAULT_PARAMS,
            MIN_EVALS,
            PLOT,
            RANDOMIZE,
            SAVEFIG,
            SVD_METHOD,
            VECTORS,
        )
    except (ImportError, AttributeError):
        return None
    params = DEFAULT_PARAMS.copy()
    params[SVD_METHOD] = str(svd_method)
    params[MIN_EVALS] = int(min_evals)
    params[PLOT] = False
    params[SAVEFIG] = False
    params[RANDOMIZE] = False
    params[VECTORS] = False
    return params


def analyze_weightwatcher_checkpoint(
    model: torch.nn.Module,
    *,
    run_label: str,
    epoch: int,
    global_step: int = 0,
    min_evals: int = 8,
    max_evals: Optional[int] = None,
    svd_method: str = "accurate",
    effective_rank_method: EffectiveRankMethod = "participation_ratio",
    normalization_gamma: float = 0.0,
    support_policy: SupportPolicy = "midpoint",
    min_ecs_size: int = 2,
    min_retained: int = 20,
    positive_eigenvalue_floor: float = 0.0,
    previous_supports: Optional[
        Mapping[str, AdaptiveSupportState | Mapping[str, Any] | int]
    ] = None,
) -> SpectralFlowCheckpoint:
    """Use WeightWatcher for ESD/alpha/PL and recompute the adaptive ECS."""
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
        min_evals=min_evals,
        max_evals=max_evals,
        svd_method=svd_method,
    )
    if not isinstance(details, pd.DataFrame) or details.empty:
        return SpectralFlowCheckpoint(pd.DataFrame(), pd.DataFrame(), {})

    get_esd_params = _make_get_esd_params(
        min_evals=min_evals,
        svd_method=svd_method,
    )
    parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.ndim == 2
    ]
    previous: dict[str, AdaptiveSupportState] = {}
    for name, value in (previous_supports or {}).items():
        try:
            previous[str(name)] = AdaptiveSupportState.from_value(value)
        except (TypeError, ValueError):
            pass

    rows: list[dict[str, Any]] = []
    supports: dict[str, AdaptiveSupportState] = {}

    for _, row in details.iterrows():
        layer_id = _safe_int(_row_value(row, ["layer_id"]), default=None)
        if layer_id is None:
            continue
        short_name = str(_row_value(row, ["name"], default=""))
        layer_name = _layer_name(row)
        parameter_name = _match_parameter_name(
            layer_name,
            short_name,
            parameter_names,
        )
        try:
            kwargs: dict[str, Any] = {
                "model": model_cpu,
                "layer": int(layer_id),
                "random": False,
            }
            if get_esd_params is not None:
                kwargs["params"] = get_esd_params
            raw_esd = watcher.get_ESD(**kwargs)
            values = clean_positive_eigenvalues(
                raw_esd,
                positive_floor=positive_eigenvalue_floor,
            )
            positive_count = int(values.size)

            alpha = _safe_float(_row_value(row, ["alpha"]))
            if not np.isfinite(alpha) or alpha <= 0.0:
                raise ValueError("WeightWatcher did not return a usable alpha.")
            m_pl = _safe_int(
                _row_value(row, ["num_pl_spikes", "num_evals_in_tail", "tail_size"]),
                default=None,
            )
            if m_pl is None or m_pl <= 0:
                raise ValueError("WeightWatcher did not return num_pl_spikes.")
            m_pl = int(np.clip(m_pl, 1, positive_count))

            m_ww = _safe_int(
                _row_value(row, ["detX_num", "num_ERG_spikes"]),
                default=None,
            )
            gap_ww = _safe_float(_row_value(row, ["ERG_gap"]))

            prior = None
            for key in (parameter_name, layer_name, short_name):
                if key and key in previous:
                    prior = previous[key]
                    break
            reference_rank = prior.ecs_rank if prior is not None else m_ww

            sc = solve_self_consistent_ecs(
                values,
                method=effective_rank_method,
                gamma=normalization_gamma,
                min_ecs_size=min_ecs_size,
                reference_rank=reference_rank,
                positive_floor=positive_eigenvalue_floor,
            )
            m_sc = int(sc.ecs_rank)
            m_working = working_support_rank(
                ecs_rank=m_sc,
                pl_rank=m_pl,
                policy=support_policy,
                minimum=min_retained,
                maximum=positive_count,
            )
            erg_gap_sc = int(m_sc - m_pl)
            midpoint = 0.5 * (m_sc + m_pl)
            relative_gap = erg_gap_sc / midpoint if midpoint > 0.0 else np.nan

            state = AdaptiveSupportState(
                ecs_rank=m_sc,
                normalization_dimension=float(sc.normalization_dimension),
                bulk_effective_count=float(sc.bulk_effective_count),
                trace_log_per_eval=float(sc.trace_log_per_eval),
                status=sc.status,
                method=effective_rank_method,
                normalization_gamma=normalization_gamma,
                pl_rank=m_pl,
                working_rank=m_working,
                alpha=alpha,
                erg_gap_sc=float(erg_gap_sc),
                source_epoch=int(epoch),
                source_global_step=int(global_step),
            )
            if parameter_name is not None and m_working >= min_retained:
                supports[parameter_name] = state

            rows.append(
                {
                    "run": str(run_label),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "layer_id": int(layer_id),
                    "layer_name": layer_name,
                    "parameter_name": parameter_name,
                    "alpha": alpha,
                    "num_pl_spikes": m_pl,
                    "detX_num_WW": m_ww,
                    "ERG_gap_WW": gap_ww,
                    "detX_num_SC": m_sc,
                    "detX_num_SC_fractional": float(sc.fractional_rank),
                    "ERG_gap_SC": erg_gap_sc,
                    "ERG_gap_SC_relative": relative_gap,
                    "m_working": m_working,
                    "M_normalization_SC": float(sc.normalization_dimension),
                    "bulk_effective_count_SC": float(sc.bulk_effective_count),
                    "trace_log_SC_per_eval": float(sc.trace_log_per_eval),
                    "SC_status": sc.status,
                    "support_policy": support_policy,
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

    return SpectralFlowCheckpoint(
        details=details.copy(),
        metrics=pd.DataFrame(rows),
        supports=supports,
    )

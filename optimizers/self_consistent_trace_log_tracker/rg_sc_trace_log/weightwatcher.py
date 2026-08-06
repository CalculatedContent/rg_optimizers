"""WeightWatcher checkpoint analysis with a self-consistent adaptive ECS."""

from __future__ import annotations

import copy
import inspect
import math
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
class SelfConsistentWeightWatcherCheckpoint:
    """WeightWatcher results plus adaptive ECS states for one checkpoint."""

    details: pd.DataFrame
    metrics: pd.DataFrame
    supports: dict[str, AdaptiveSupportState]
    candidate_scans: pd.DataFrame


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
        result = int(round(float(value)))
        return result
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
    raw_candidates = [layer_name, short_name]
    candidates: list[str] = []
    for candidate in raw_candidates:
        if not candidate:
            continue
        candidates.append(candidate)
        if not candidate.endswith(".weight"):
            candidates.append(f"{candidate}.weight")

    for candidate in candidates:
        if candidate in parameter_names:
            return candidate

    suffix_matches = [
        parameter_name
        for parameter_name in parameter_names
        if any(parameter_name.endswith(candidate) for candidate in candidates)
    ]
    return suffix_matches[0] if len(suffix_matches) == 1 else None


def _analyze_compat(
    watcher: Any,
    *,
    min_evals: int,
    max_evals: Optional[int],
    svd_method: str,
) -> pd.DataFrame:
    """Run WeightWatcher while tolerating minor API differences."""
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
    else:
        raise RuntimeError("This WeightWatcher version exposes neither ERG nor detX.")
    if not parameters or "svd_method" in parameters:
        kwargs["svd_method"] = str(svd_method)

    if max_evals is not None:
        max_evals_int = int(max_evals)
        kwargs["max_evals"] = max_evals_int
        if max_evals_int > 0 and (not parameters or "max_N" in parameters):
            kwargs["max_N"] = max(50000, max_evals_int)

    return watcher.analyze(**kwargs)


def _make_get_esd_params(
    *,
    min_evals: int,
    svd_method: str,
) -> Optional[dict[str, Any]]:
    """Build complete WeightWatcher params when its constants are available."""
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
    min_retained: int = 3,
    positive_eigenvalue_floor: float = 0.0,
    previous_supports: Optional[
        Mapping[str, AdaptiveSupportState | Mapping[str, Any] | int]
    ] = None,
    save_candidate_scans: bool = False,
) -> SelfConsistentWeightWatcherCheckpoint:
    """Measure alpha/PL with WeightWatcher and recompute the ECS ourselves.

    WeightWatcher remains the sole source of the ESD, alpha, and PL boundary.
    Its full-``M`` ``detX_num``/``ERG_gap`` are retained only as audits.  The
    optimizer support is selected from the notebook's bulk-effective adaptive
    normalization.
    """
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
        svd_method=str(svd_method),
    )
    if not isinstance(details, pd.DataFrame) or details.empty:
        return SelfConsistentWeightWatcherCheckpoint(
            pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()
        )

    get_esd_params = _make_get_esd_params(
        min_evals=int(min_evals),
        svd_method=str(svd_method),
    )
    parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.ndim == 2
    ]

    previous_by_name: dict[str, AdaptiveSupportState] = {}
    for supplied_name, value in (previous_supports or {}).items():
        try:
            state = AdaptiveSupportState.from_value(value)
        except (TypeError, ValueError):
            continue
        previous_by_name[str(supplied_name)] = state

    rows: list[dict[str, Any]] = []
    scan_frames: list[pd.DataFrame] = []
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
            get_kwargs: dict[str, Any] = {
                "model": model_cpu,
                "layer": int(layer_id),
                "random": False,
            }
            if get_esd_params is not None:
                get_kwargs["params"] = get_esd_params
            raw_esd = watcher.get_ESD(**get_kwargs)
            values = clean_positive_eigenvalues(
                raw_esd,
                positive_floor=float(positive_eigenvalue_floor),
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
                raise ValueError(
                    "WeightWatcher did not return a usable num_pl_spikes value."
                )
            m_pl = int(np.clip(m_pl, 1, positive_count))

            m_ww = _safe_int(
                _row_value(row, ["detX_num", "num_ERG_spikes"]),
                default=None,
            )
            if m_ww is not None:
                m_ww = int(np.clip(m_ww, 1, positive_count))
            gap_ww = _safe_float(_row_value(row, ["ERG_gap"]))

            prior = None
            for key in (parameter_name, layer_name, short_name):
                if key and key in previous_by_name:
                    prior = previous_by_name[key]
                    break
            reference_rank = prior.ecs_rank if prior is not None else m_ww

            sc = solve_self_consistent_ecs(
                values,
                method=effective_rank_method,
                gamma=float(normalization_gamma),
                min_ecs_size=int(min_ecs_size),
                reference_rank=reference_rank,
                positive_floor=float(positive_eigenvalue_floor),
            )
            m_sc = int(sc.ecs_rank)
            m_working = working_support_rank(
                ecs_rank=m_sc,
                pl_rank=m_pl,
                policy=support_policy,
                minimum=int(min_retained),
                maximum=positive_count,
            )
            erg_gap_sc = int(m_sc - m_pl)
            midpoint = 0.5 * (m_sc + m_pl)
            erg_gap_sc_relative = (
                float(erg_gap_sc / midpoint) if midpoint > 0.0 else np.nan
            )

            state = AdaptiveSupportState(
                ecs_rank=m_sc,
                normalization_dimension=float(sc.normalization_dimension),
                bulk_effective_count=float(sc.bulk_effective_count),
                trace_log_per_eval=float(sc.trace_log_per_eval),
                status=str(sc.status),
                method=effective_rank_method,
                normalization_gamma=float(normalization_gamma),
                pl_rank=m_pl,
                working_rank=m_working,
                alpha=alpha,
                erg_gap_sc=float(erg_gap_sc),
                source_epoch=int(epoch),
                source_global_step=int(global_step),
            )
            if parameter_name is not None:
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
                    "alpha_source": "WeightWatcher",
                    "xmin": _safe_float(_row_value(row, ["xmin"])),
                    "num_pl_spikes": m_pl,
                    "detX_num_WW": m_ww,
                    "ERG_gap_WW": gap_ww,
                    "detX_num_SC": m_sc,
                    "detX_num_SC_fractional": float(sc.fractional_rank),
                    "ERG_gap_SC": erg_gap_sc,
                    "ERG_gap_SC_relative": erg_gap_sc_relative,
                    "m_working": m_working,
                    "support_policy": support_policy,
                    "M_normalization_SC": float(sc.normalization_dimension),
                    "bulk_count_SC": int(sc.bulk_count),
                    "bulk_effective_count_SC": float(sc.bulk_effective_count),
                    "bulk_effective_fraction_SC": float(sc.bulk_effective_fraction),
                    "trace_log_SC": float(sc.trace_log),
                    "trace_log_SC_per_eval": float(sc.trace_log_per_eval),
                    "lambda_cut_SC_raw": float(sc.lambda_cut_raw),
                    "lambda_cut_SC_scaled": float(sc.lambda_cut_scaled),
                    "SC_status": str(sc.status),
                    "SC_num_sign_change_brackets": int(sc.num_sign_change_brackets),
                    "SC_fixed_point_error_nearest": int(sc.fixed_point_error_nearest),
                    "SC_fixed_point_error_first_below": int(
                        sc.fixed_point_error_first_below
                    ),
                    "SC_effective_rank_method": effective_rank_method,
                    "SC_gamma": float(normalization_gamma),
                    "num_positive_eigenvalues": positive_count,
                    "status": "ok",
                    "error": "",
                }
            )

            if save_candidate_scans:
                # Import lazily to avoid an unnecessary dataframe in normal runs.
                from .ecs import self_consistent_candidate_scan

                scan = self_consistent_candidate_scan(
                    values,
                    method=effective_rank_method,
                    gamma=float(normalization_gamma),
                    min_ecs_size=int(min_ecs_size),
                    positive_floor=float(positive_eigenvalue_floor),
                )
                scan.insert(0, "run", str(run_label))
                scan.insert(1, "epoch", int(epoch))
                scan.insert(2, "global_step", int(global_step))
                scan.insert(3, "layer_id", int(layer_id))
                scan.insert(4, "layer_name", layer_name)
                scan["selected_m"] = m_sc
                scan_frames.append(scan)

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

    scans = pd.concat(scan_frames, ignore_index=True) if scan_frames else pd.DataFrame()
    return SelfConsistentWeightWatcherCheckpoint(
        details=details.copy(),
        metrics=pd.DataFrame(rows),
        supports=supports,
        candidate_scans=scans,
    )

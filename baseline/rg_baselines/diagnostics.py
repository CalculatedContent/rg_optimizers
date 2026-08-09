"""WeightWatcher and exact epoch-level spectral diagnostics for the baselines."""

from __future__ import annotations

import copy
import inspect
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class SpectralCheckpoint:
    """One complete epoch checkpoint."""

    details: pd.DataFrame
    metrics: pd.DataFrame
    esd_arrays: dict[str, np.ndarray]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _row_value(row: pd.Series, names: tuple[str, ...], default: Any = np.nan) -> Any:
    for name in names:
        if name not in row.index:
            continue
        value = row[name]
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return default


def _sanitize_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def clean_positive_eigenvalues(
    values: Any,
    *,
    expected_dimension: Optional[int] = None,
) -> np.ndarray:
    """Return positive eigenvalues in ascending order.

    When ``expected_dimension`` is supplied, fail closed if the ESD is
    incomplete, non-finite, or rank deficient. This preserves
    WeightWatcher's full-M normalization instead of silently renormalizing a
    filtered positive-rank spectrum.
    """

    evals = np.asarray(values, dtype=float).reshape(-1)
    if expected_dimension is not None:
        expected = int(expected_dimension)
        if expected < 2:
            raise ValueError("expected spectral dimension must be at least two")
        if evals.size != expected:
            raise ValueError(
                "ESD dimension mismatch: "
                f"expected {expected} eigenvalues, received {evals.size}"
            )
        if not np.all(np.isfinite(evals)):
            raise ValueError("full ESD contains non-finite eigenvalues")
        if np.any(evals <= 0.0):
            positive = int(np.count_nonzero(evals > 0.0))
            raise ValueError(
                "rank-deficient ESD: "
                f"expected {expected} positive eigenvalues, found {positive}"
            )
    else:
        evals = evals[np.isfinite(evals) & (evals > 0.0)]

    evals = np.sort(evals)
    if evals.size < 2:
        raise ValueError("fewer than two finite positive eigenvalues")
    return evals
def _entropy_effective_rank(evals: np.ndarray) -> float:
    total = float(np.sum(evals))
    if total <= 0.0:
        return np.nan
    p = evals / total
    return float(np.exp(-np.sum(p * np.log(p))))


def spectral_metrics_from_esd(
    raw_evals_ascending: Any,
    normalized_evals_ascending: Any,
    *,
    detx_num: int,
    num_pl_spikes: int,
    erg_gap: int,
    expected_dimension: Optional[int] = None,
) -> dict[str, float | int]:
    """Compute transparent metrics from one WeightWatcher ESD.

    ``normalized_evals_ascending`` must be produced by WeightWatcher's own
    ``RMT_Util.rescale_eigenvalues``. The trace-log boundary and gap are not
    recomputed here: the supplied ``detx_num``, ``num_pl_spikes``, and
    ``erg_gap`` must come from ``watcher.analyze(ERG=True)``.

    ``expected_dimension`` is the full spectral dimension
    ``min(weight.shape)``. Strict baseline measurements require all of those
    eigenvalues to be finite and positive so WeightWatcher's normalization is
    not silently changed by positive-eigenvalue filtering.
    """

    raw = clean_positive_eigenvalues(
        raw_evals_ascending,
        expected_dimension=expected_dimension,
    )
    normalized = clean_positive_eigenvalues(
        normalized_evals_ascending,
        expected_dimension=expected_dimension,
    )
    if raw.size != normalized.size:
        raise ValueError("raw and normalized ESDs have different sizes")

    count = int(raw.size)
    normalized_sum = float(np.sum(normalized))
    if not np.isclose(
        normalized_sum,
        float(count),
        rtol=1e-10,
        atol=1e-10 * max(count, 1),
    ):
        raise ValueError(
            "WeightWatcher normalization audit failed: "
            f"sum={normalized_sum:.17g}, expected={count}"
        )

    m_detx = int(detx_num)
    m_pl = int(num_pl_spikes)
    if not 1 <= m_detx <= count:
        raise ValueError(
            f"detX_num must lie in [1, {count}], received {m_detx}"
        )
    if not 1 <= m_pl <= count:
        raise ValueError(
            f"num_pl_spikes must lie in [1, {count}], received {m_pl}"
        )

    expected_gap = m_detx - m_pl
    if int(erg_gap) != expected_gap:
        raise ValueError(
            f"WeightWatcher ERG_gap audit failed: {erg_gap} != {m_detx} - {m_pl}"
        )
    m_midpoint = int(math.floor((m_detx + m_pl) / 2.0))

    raw_desc = raw[::-1]
    normalized_desc = normalized[::-1]
    midpoint = normalized_desc[:m_midpoint]
    midpoint_logs = np.log(midpoint)
    trace_log_total = float(np.sum(midpoint_logs))
    trace_log_per_eval = float(np.mean(midpoint_logs))

    spectral_sum = float(np.sum(raw))
    max_eval = float(raw[-1])
    min_eval = float(raw[0])
    probabilities = raw / spectral_sum
    participation_ratio = float(1.0 / np.sum(probabilities**2))

    def energy_fraction(m: int) -> float:
        return float(np.sum(raw_desc[: int(m)]) / spectral_sum)

    return {
        "num_positive_eigenvalues": count,
        "detX_num": m_detx,
        "num_pl_spikes": m_pl,
        "ERG_gap": int(erg_gap),
        "ERG_gap_audit": expected_gap,
        "m_midpoint": m_midpoint,
        "boundary_overlap_ratio": float(min(m_detx, m_pl) / max(m_detx, m_pl)),
        "trace_log_midpoint_total": trace_log_total,
        "trace_log_midpoint_per_eval": trace_log_per_eval,
        "geometric_mean_midpoint": float(np.exp(trace_log_per_eval)),
        "midpoint_span_decades": float(
            np.log10(normalized_desc[0] / normalized_desc[m_midpoint - 1])
        ),
        "spectral_sum": spectral_sum,
        "frobenius_norm": float(np.sqrt(spectral_sum)),
        "spectral_norm": float(np.sqrt(max_eval)),
        "stable_rank": float(spectral_sum / max_eval),
        "participation_ratio": participation_ratio,
        "entropy_effective_rank": _entropy_effective_rank(raw),
        "largest_eigenvalue": max_eval,
        "smallest_positive_eigenvalue": min_eval,
        "eigenvalue_condition_number": float(max_eval / min_eval),
        "top1_energy_fraction": float(max_eval / spectral_sum),
        "pl_energy_fraction": energy_fraction(m_pl),
        "detx_energy_fraction": energy_fraction(m_detx),
        "midpoint_energy_fraction": energy_fraction(m_midpoint),
        "rescaled_eigenvalue_sum": normalized_sum,
        "rescale_sum_minus_num_eigenvalues": float(normalized_sum - count),
        "normalized_lambda_max": float(normalized_desc[0]),
        "normalized_lambda_midpoint_cut": float(normalized_desc[m_midpoint - 1]),
    }


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
    randomize: bool,
) -> pd.DataFrame:
    """Run WeightWatcher with ERG enabled across minor API variants."""

    try:
        parameters = inspect.signature(watcher.analyze).parameters
    except (TypeError, ValueError):
        parameters = {}

    kwargs: dict[str, Any] = {
        "plot": False,
        "randomize": bool(randomize),
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
        raise RuntimeError("WeightWatcher exposes neither ERG nor detX")
    if not parameters or "svd_method" in parameters:
        kwargs["svd_method"] = str(svd_method)
    if max_evals is not None:
        kwargs["max_evals"] = int(max_evals)
        if not parameters or "max_N" in parameters:
            kwargs["max_N"] = max(50000, int(max_evals))
    return watcher.analyze(**kwargs)


def _get_esd_params(*, min_evals: int, svd_method: str) -> Optional[dict[str, Any]]:
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


def _get_esd_compat(
    watcher: Any,
    *,
    model: torch.nn.Module,
    layer_id: int,
    params: Optional[dict[str, Any]],
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "layer": int(layer_id),
        "random": False,
    }
    if params is not None:
        kwargs["params"] = params
    try:
        return watcher.get_ESD(**kwargs)
    except TypeError:
        return watcher.get_ESD(layer=int(layer_id))


def _rescale_with_weightwatcher(evals: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        from weightwatcher.RMT_Util import rescale_eigenvalues
    except ImportError:
        try:
            from weightwatcher import RMT_Util
        except ImportError as exc:
            raise ImportError(
                "Could not import WeightWatcher's RMT_Util.rescale_eigenvalues"
            ) from exc
        rescale_eigenvalues = RMT_Util.rescale_eigenvalues
    scaled, weight_scale = rescale_eigenvalues(np.asarray(evals, dtype=float).copy())
    return np.asarray(scaled, dtype=float), float(weight_scale)


def measure_weightwatcher_checkpoint(
    model: torch.nn.Module,
    *,
    run_label: str,
    epoch: int,
    global_step: int,
    min_evals: int = 8,
    max_evals: Optional[int] = None,
    svd_method: str = "accurate",
    randomize: bool = False,
) -> SpectralCheckpoint:
    """Measure all requested original full-M metrics at one epoch."""

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
        randomize=randomize,
    )
    if not isinstance(details, pd.DataFrame) or details.empty:
        raise RuntimeError("WeightWatcher returned no layer details")

    details_out = details.copy()
    details_out.insert(0, "run", str(run_label))
    details_out.insert(1, "epoch", int(epoch))
    details_out.insert(2, "global_step", int(global_step))

    parameter_map = {
        name: parameter for name, parameter in model.named_parameters() if parameter.ndim == 2
    }
    parameter_names = list(parameter_map)
    get_esd_params = _get_esd_params(min_evals=min_evals, svd_method=svd_method)

    rows: list[dict[str, Any]] = []
    esd_arrays: dict[str, np.ndarray] = {}
    for _, row in details.iterrows():
        layer_id = _safe_int(_row_value(row, ("layer_id",)), default=None)
        if layer_id is None:
            continue
        short_name = str(_row_value(row, ("name",), default=""))
        long_name = _layer_name(row)
        parameter_name = _match_parameter_name(long_name, short_name, parameter_names)
        layer_short = (
            parameter_name.removesuffix(".weight").split(".")[-1]
            if parameter_name
            else short_name.split(".")[-1]
        )

        base_record: dict[str, Any] = {
            "run": str(run_label),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "layer_id": int(layer_id),
            "layer_name": long_name,
            "layer": layer_short,
            "parameter_name": parameter_name,
            "status": "failed",
            "error": "",
        }
        try:
            alpha = _safe_float(_row_value(row, ("alpha",)))
            if not np.isfinite(alpha):
                raise ValueError("WeightWatcher did not return alpha")
            detx_num = _safe_int(
                _row_value(row, ("detX_num", "num_ERG_spikes")), default=None
            )
            num_pl_spikes = _safe_int(
                _row_value(row, ("num_pl_spikes", "num_evals_in_tail", "tail_size")),
                default=None,
            )
            erg_gap_value = _safe_float(_row_value(row, ("ERG_gap",)))
            if detx_num is None or detx_num <= 0:
                raise ValueError("WeightWatcher did not return detX_num")
            if num_pl_spikes is None or num_pl_spikes <= 0:
                raise ValueError("WeightWatcher did not return num_pl_spikes")
            if not np.isfinite(erg_gap_value):
                raise ValueError("WeightWatcher did not return ERG_gap")
            erg_gap = int(round(erg_gap_value))

            parameter = parameter_map.get(parameter_name) if parameter_name else None
            if parameter is None:
                raise ValueError(
                    "WeightWatcher layer could not be matched to a model matrix"
                )
            expected_dimension = int(min(parameter.shape))
            raw_esd = clean_positive_eigenvalues(
                _get_esd_compat(
                    watcher,
                    model=model_cpu,
                    layer_id=int(layer_id),
                    params=get_esd_params,
                ),
                expected_dimension=expected_dimension,
            )
            normalized_esd, weight_scale = _rescale_with_weightwatcher(raw_esd)
            computed = spectral_metrics_from_esd(
                raw_esd,
                normalized_esd,
                detx_num=int(detx_num),
                num_pl_spikes=int(num_pl_spikes),
                erg_gap=erg_gap,
                expected_dimension=expected_dimension,
            )

            record = {
                **base_record,
                "status": "ok",
                "alpha": alpha,
                "alpha_minus_2": float(alpha - 2.0),
                "abs_alpha_minus_2": float(abs(alpha - 2.0)),
                "alpha_source": "WeightWatcher",
                "ERG_gap_source": "WeightWatcher analyze(ERG=True)",
                "detX_source": "WeightWatcher analyze(ERG=True)",
                "num_pl_spikes_source": "WeightWatcher power-law fit",
                "normalization_source": "WeightWatcher RMT_Util.rescale_eigenvalues",
                "weight_scale": weight_scale,
                "xmin": _safe_float(_row_value(row, ("xmin",))),
                "xmax": _safe_float(_row_value(row, ("xmax",))),
                "layer_rows": int(parameter.shape[0]),
                "layer_cols": int(parameter.shape[1]),
                "layer_parameter_count": int(parameter.numel()),
                **computed,
            }
            for column in (
                "D",
                "sigma",
                "warning",
                "num_evals",
                "rank_loss",
                "alpha_weighted",
                "log_alpha_norm",
                "log_norm",
                "log_spectral_norm",
                "norm",
            ):
                if column in row.index:
                    record[f"ww_{column}"] = row[column]
            rows.append(record)

            prefix = f"epoch_{int(epoch):03d}__{_sanitize_key(layer_short)}"
            esd_arrays[f"{prefix}__raw_ascending"] = raw_esd
            esd_arrays[f"{prefix}__weightwatcher_rescaled_ascending"] = normalized_esd
        except Exception as exc:
            base_record["error"] = str(exc)
            rows.append(base_record)

    return SpectralCheckpoint(
        details=details_out,
        metrics=pd.DataFrame(rows),
        esd_arrays=esd_arrays,
    )

"""WeightWatcher and local spectral diagnostics for local-delta experiments."""

from __future__ import annotations

import copy
import inspect
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .ecs import local_ecs_geometry


def _layer_matrix(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().reshape(weight.shape[0], -1).cpu()


def rank_slope_alpha_proxy(weight: torch.Tensor, eps: float = 1e-12) -> tuple[float, float]:
    """Return ``(q, alpha_proxy)`` from the full rank-ordered spectrum."""
    matrix = _layer_matrix(weight)
    if min(matrix.shape) < 3:
        return float("nan"), float("nan")
    singular_values = torch.linalg.svdvals(matrix).clamp_min(eps)
    lambdas = torch.sort(singular_values * singular_values, descending=True).values
    n = int(lambdas.numel())
    x = torch.log(torch.arange(1, n + 1, dtype=torch.float32))
    x = x - x.mean()
    y = torch.log(lambdas) - torch.log(lambdas).mean()
    denom = torch.sum(x * x).clamp_min(eps)
    q = -float((torch.sum(x * y) / denom).detach().cpu())
    alpha = 1.0 + (1.0 / q) if q > eps else float("inf")
    return q, alpha


def fallback_spectral_diagnostics(
    model: nn.Module,
    *,
    epoch: int,
    run_label: str,
    seed: int,
    optimizer_kind: str,
    arm: str,
    min_retained: int = 3,
    normalization_gamma: float = 0.0,
    diagnostic_error: str = "",
) -> pd.DataFrame:
    """Direct-SVD diagnostics used as an explicit fallback/audit."""
    rows: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or not name.endswith("weight"):
            continue
        matrix = _layer_matrix(parameter)
        if min(matrix.shape) < min_retained:
            continue
        geometry = local_ecs_geometry(
            matrix,
            min_retained=min_retained,
            normalization_gamma=normalization_gamma,
        )
        q, alpha_proxy = rank_slope_alpha_proxy(parameter)
        singular_values = torch.linalg.svdvals(matrix)
        lambdas = singular_values.square()
        spectral_norm = float(lambdas.max().detach().cpu())
        stable_rank = float(
            (lambdas.sum() / lambdas.max().clamp_min(1e-12)).detach().cpu()
        )
        rows.append(
            {
                "run_label": run_label,
                "seed": seed,
                "optimizer_kind": optimizer_kind,
                "arm": arm,
                "epoch": epoch,
                "layer_name": name,
                "name": name,
                "alpha": alpha_proxy,
                "alpha_proxy": alpha_proxy,
                "rank_slope_q": q,
                "detX_num": np.nan,
                "num_pl_spikes": np.nan,
                "ERG_gap": np.nan,
                "ecs_rank_local": geometry.scan.rank,
                "ecs_fraction_local": geometry.scan.rank
                / max(geometry.scan.spectral_count, 1),
                "trace_log_per_eval_local": geometry.scan.trace_log_per_eval,
                "bulk_effective_count_local": geometry.scan.bulk_effective_count,
                "projection_side_local": geometry.projection_side,
                "spectral_norm": spectral_norm,
                "stable_rank": stable_rank,
                "diagnostic_source": "fallback_svd",
                "diagnostic_error": diagnostic_error,
            }
        )
    return pd.DataFrame(rows)


def _analyze_compat(watcher: Any, *, min_evals: int, svd_method: str) -> pd.DataFrame:
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
        raise RuntimeError("WeightWatcher exposes neither ERG nor detX analysis.")
    if not parameters or "svd_method" in parameters:
        kwargs["svd_method"] = str(svd_method)
    return watcher.analyze(**kwargs)


def _match_local_layer(layer_name: str, local_names: list[str]) -> str | None:
    candidates = [layer_name]
    if not layer_name.endswith(".weight"):
        candidates.append(f"{layer_name}.weight")
    for candidate in candidates:
        if candidate in local_names:
            return candidate
    matches = [
        name
        for name in local_names
        if any(name.endswith(candidate) for candidate in candidates)
    ]
    return matches[0] if len(matches) == 1 else None


def analyze_weightwatcher_or_fallback(
    model: nn.Module,
    *,
    epoch: int,
    run_label: str,
    seed: int,
    optimizer_kind: str,
    arm: str,
    ww_enabled: bool = True,
    ww_required: bool = False,
    ww_min_evals: int = 8,
    ww_svd_method: str = "accurate",
    min_retained: int = 3,
    normalization_gamma: float = 0.0,
) -> pd.DataFrame:
    """Run WeightWatcher on a CPU copy and attach direct-SVD audit metrics."""
    fallback = fallback_spectral_diagnostics(
        model,
        epoch=epoch,
        run_label=run_label,
        seed=seed,
        optimizer_kind=optimizer_kind,
        arm=arm,
        min_retained=min_retained,
        normalization_gamma=normalization_gamma,
    )
    if not ww_enabled:
        return fallback

    try:
        import weightwatcher as ww  # type: ignore

        model_cpu = copy.deepcopy(model).to("cpu")
        model_cpu.eval()
        watcher = ww.WeightWatcher(model=model_cpu)
        details = _analyze_compat(
            watcher,
            min_evals=int(ww_min_evals),
            svd_method=str(ww_svd_method),
        )
        if not isinstance(details, pd.DataFrame) or details.empty:
            raise RuntimeError("WeightWatcher returned no layer rows.")

        details = details.copy()
        key = "longname" if "longname" in details.columns else "name"
        details["layer_name"] = details[key].astype(str)
        details["run_label"] = run_label
        details["seed"] = seed
        details["optimizer_kind"] = optimizer_kind
        details["arm"] = arm
        details["epoch"] = epoch
        details["diagnostic_source"] = "weightwatcher"
        details["diagnostic_error"] = ""
        if "ERG_gap" not in details.columns:
            if "detX_num" in details.columns and "num_pl_spikes" in details.columns:
                details["ERG_gap"] = details["detX_num"] - details["num_pl_spikes"]
            else:
                details["ERG_gap"] = np.nan

        local = fallback.set_index("layer_name", drop=False)
        local_names = list(local.index.astype(str))
        audit_columns = [
            "alpha_proxy",
            "rank_slope_q",
            "ecs_rank_local",
            "ecs_fraction_local",
            "trace_log_per_eval_local",
            "bulk_effective_count_local",
            "projection_side_local",
        ]
        for column in audit_columns:
            details[column] = np.nan if column != "projection_side_local" else ""
        for idx, row in details.iterrows():
            match = _match_local_layer(str(row["layer_name"]), local_names)
            if match is None:
                continue
            local_row = local.loc[match]
            for column in audit_columns:
                details.at[idx, column] = local_row[column]
        return details
    except Exception as exc:  # pragma: no cover - optional package/backend
        if ww_required:
            raise RuntimeError(
                f"WeightWatcher analysis failed at epoch {epoch}: {exc}"
            ) from exc
        return fallback_spectral_diagnostics(
            model,
            epoch=epoch,
            run_label=run_label,
            seed=seed,
            optimizer_kind=optimizer_kind,
            arm=arm,
            min_retained=min_retained,
            normalization_gamma=normalization_gamma,
            diagnostic_error=repr(exc),
        )

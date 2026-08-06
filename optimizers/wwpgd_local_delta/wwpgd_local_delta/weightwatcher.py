"""WeightWatcher and fallback spectral diagnostics for local-delta experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .ecs import select_self_consistent_ecs


def _layer_matrix(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().reshape(weight.shape[0], -1).cpu()


def rank_slope_alpha_proxy(weight: torch.Tensor, eps: float = 1e-12) -> tuple[float, float]:
    """Return (q, alpha_proxy) from rank-ordered singular-value slope."""
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
    q = -float(torch.sum(x * y) / denom)
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
) -> pd.DataFrame:
    """Cheap diagnostics used when WeightWatcher is unavailable or fails."""
    rows: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or not name.endswith("weight"):
            continue
        matrix = _layer_matrix(parameter)
        if min(matrix.shape) < min_retained:
            continue
        singular_values = torch.linalg.svdvals(matrix)
        scan = select_self_consistent_ecs(
            singular_values,
            min_retained=min_retained,
            normalization_gamma=normalization_gamma,
        )
        q, alpha_proxy = rank_slope_alpha_proxy(parameter)
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
                "detX_num": scan.rank,
                "num_pl_spikes": np.nan,
                "ERG_gap": np.nan,
                "ecs_rank_local": scan.rank,
                "trace_log_per_eval_local": scan.trace_log_per_eval,
                "bulk_effective_count_local": scan.bulk_effective_count,
                "diagnostic_source": "fallback_svd",
            }
        )
    return pd.DataFrame(rows)


def analyze_weightwatcher_or_fallback(
    model: nn.Module,
    *,
    epoch: int,
    run_label: str,
    seed: int,
    optimizer_kind: str,
    arm: str,
    ww_enabled: bool = True,
    min_retained: int = 3,
    normalization_gamma: float = 0.0,
) -> pd.DataFrame:
    """Run WeightWatcher when available; otherwise return fallback SVD metrics."""
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

        watcher = ww.WeightWatcher(model=model)
        try:
            details = watcher.analyze(ERG=True, randomize=False, plot=False)
        except TypeError:
            details = watcher.analyze(detX=True, randomize=False, plot=False)
        details = details.copy()
        key = "longname" if "longname" in details.columns else "name"
        details["layer_name"] = details[key].astype(str)
        details["run_label"] = run_label
        details["seed"] = seed
        details["optimizer_kind"] = optimizer_kind
        details["arm"] = arm
        details["epoch"] = epoch
        details["diagnostic_source"] = "weightwatcher"
        if "ERG_gap" not in details.columns:
            if "detX_num" in details.columns and "num_pl_spikes" in details.columns:
                details["ERG_gap"] = details["detX_num"] - details["num_pl_spikes"]
            else:
                details["ERG_gap"] = np.nan
        local = fallback.set_index("layer_name")
        for col in [
            "alpha_proxy",
            "rank_slope_q",
            "ecs_rank_local",
            "trace_log_per_eval_local",
            "bulk_effective_count_local",
        ]:
            details[col] = np.nan
        for idx, row in details.iterrows():
            lname = str(row["layer_name"])
            matches = [
                name
                for name in local.index
                if name.endswith(lname) or lname.endswith(name.replace(".weight", ""))
            ]
            if matches:
                local_row = local.loc[matches[0]]
                for col in [
                    "alpha_proxy",
                    "rank_slope_q",
                    "ecs_rank_local",
                    "trace_log_per_eval_local",
                    "bulk_effective_count_local",
                ]:
                    details.at[idx, col] = local_row[col]
        return details
    except Exception as exc:  # pragma: no cover - depends on optional package
        fallback = fallback.copy()
        fallback["diagnostic_error"] = repr(exc)
        return fallback

"""Retained bases and full-M/self-consistent SETOL normalizations."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np
import pandas as pd
import torch

NormalizationMode = Literal["full_m", "self_consistent"]

@dataclass
class MatrixLogSupport:
    retained_rank: int
    normalization_dimension: float
    right_basis: torch.Tensor
    transposed: bool
    checkpoint_epoch: int = 0
    full_dimension: float | None = None
    self_consistent_dimension: float | None = None
    bulk_effective_rank: float = 0.0

    def dimension(self, mode: NormalizationMode) -> float:
        if mode not in {"full_m", "self_consistent"}:
            raise ValueError(f"unknown normalization: {mode!r}")
        value = self.full_dimension if mode == "full_m" else self.self_consistent_dimension
        value = self.normalization_dimension if value is None else value
        if float(value) <= 0:
            raise ValueError("normalization dimension must be positive")
        return float(value)

    def state_dict(self) -> dict:
        return {
            "retained_rank": int(self.retained_rank),
            "normalization_dimension": float(self.normalization_dimension),
            "full_dimension": self.full_dimension,
            "self_consistent_dimension": self.self_consistent_dimension,
            "bulk_effective_rank": float(self.bulk_effective_rank),
            "right_basis": self.right_basis.detach().cpu().float(),
            "transposed": bool(self.transposed),
            "checkpoint_epoch": int(self.checkpoint_epoch),
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "MatrixLogSupport":
        legacy = float(payload["normalization_dimension"])
        return cls(
            int(payload["retained_rank"]), legacy,
            torch.as_tensor(payload["right_basis"]).detach().cpu().float(),
            bool(payload.get("transposed", False)), int(payload.get("checkpoint_epoch", 0)),
            float(payload.get("full_dimension") or legacy),
            float(payload.get("self_consistent_dimension") or legacy),
            float(payload.get("bulk_effective_rank", 0.0)),
        )

@dataclass
class SupportCheckpoint:
    details: pd.DataFrame
    metrics: pd.DataFrame
    supports: dict[str, MatrixLogSupport]
    esd_arrays: dict[str, object]

def orient_tall(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if weight.ndim != 2:
        raise ValueError(f"expected matrix, got {tuple(weight.shape)}")
    return (weight, False) if weight.shape[0] >= weight.shape[1] else (weight.T, True)

def self_consistent_dimension_from_eigenvalues(
    eigenvalues_descending: torch.Tensor | np.ndarray,
    retained_rank: int, *, gamma: float = 0.0, eps: float = 1e-12,
) -> tuple[float, float]:
    """D_R=m+r_bulk+gamma[(M-m)-r_bulk], with PR bulk rank."""
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0,1]")
    values = torch.as_tensor(eigenvalues_descending).detach().cpu().double().reshape(-1)
    values = values[torch.isfinite(values) & (values > 0)]
    if not values.numel():
        raise ValueError("no positive finite eigenvalues")
    values = torch.sort(values, descending=True).values
    total = int(values.numel()); m = max(1, min(int(retained_rank), total)); bulk = values[m:]
    r_bulk = 0.0 if not bulk.numel() else float(bulk.sum()) ** 2 / max(float(bulk.square().sum()), eps)
    d = m + r_bulk + float(gamma) * ((total - m) - r_bulk)
    return min(float(total), max(float(m), float(d))), float(r_bulk)

def build_support(weight: torch.Tensor, retained_rank: int, *, epoch: int = 0) -> MatrixLogSupport:
    work, transposed = orient_tall(weight.detach())
    _, singular_values, vh = torch.linalg.svd(work.cpu().float(), full_matrices=False)
    rank = max(1, min(int(retained_rank), int(vh.shape[0])))
    full = float(work.shape[1]); d_sc, r_bulk = self_consistent_dimension_from_eigenvalues(singular_values.square(), rank)
    return MatrixLogSupport(rank, full, vh[:rank].T.contiguous(), transposed, int(epoch), full, d_sc, r_bulk)

def analyze_supports(
    model: torch.nn.Module, *, run_label: str = "FullMatrixLogRG", epoch: int = 0,
    global_step: int = 0, min_evals: int = 8, max_evals: int | None = None,
    svd_method: str = "accurate", randomize: bool = True,
    parameter_names: Iterable[str] | None = None, build_bases: bool = True,
) -> SupportCheckpoint:
    try:
        from rg_baselines import measure_weightwatcher_checkpoint
        from rg_baselines.trap_metrics import attach_correlation_traps
    except ImportError as exc:
        raise ImportError("install './baseline[experiment]' first") from exc
    checkpoint = attach_correlation_traps(measure_weightwatcher_checkpoint(
        model, run_label=run_label, epoch=int(epoch), global_step=int(global_step),
        min_evals=int(min_evals), max_evals=max_evals, svd_method=str(svd_method),
        randomize=bool(randomize),
    ))
    allowed = set(parameter_names) if parameter_names is not None else None
    params = {n:p for n,p in model.named_parameters() if p.ndim == 2 and (allowed is None or n in allowed)}
    supports: dict[str, MatrixLogSupport] = {}; metrics = checkpoint.metrics.copy()
    if build_bases and not metrics.empty:
        for _, row in metrics[metrics["status"].eq("ok")].iterrows():
            name = row.get("parameter_name")
            if isinstance(name, str) and name in params:
                support = build_support(params[name], int(row["m_midpoint"]), epoch=int(epoch)); supports[name] = support
                mask = metrics["parameter_name"].eq(name)
                metrics.loc[mask, "D_full_M"] = support.dimension("full_m")
                metrics.loc[mask, "D_self_consistent"] = support.dimension("self_consistent")
                metrics.loc[mask, "bulk_effective_rank"] = support.bulk_effective_rank
    if build_bases and allowed is not None:
        missing = sorted(allowed.difference(supports))
        if missing:
            raise RuntimeError("WeightWatcher produced no support for: " + ", ".join(missing))
    return SupportCheckpoint(checkpoint.details, metrics, supports, checkpoint.esd_arrays)

__all__ = ["MatrixLogSupport", "SupportCheckpoint", "analyze_supports", "build_support", "orient_tall", "self_consistent_dimension_from_eigenvalues"]

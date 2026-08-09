"""WeightWatcher midpoint support selection and cached retained bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import torch

from .config import EffectiveRankMethod, NormalizationMode


def _effective_contributor_count(
    values: torch.Tensor,
    *,
    method: EffectiveRankMethod,
    eps: float = 1e-30,
) -> float:
    x = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    x = x[torch.isfinite(x) & (x > 0.0)]
    if x.numel() == 0:
        return 0.0
    total = torch.sum(x)
    if float(total) <= eps:
        return 0.0
    if method == "participation_ratio":
        second = torch.sum(x.square())
        return float((total.square() / second.clamp_min(eps)).item())
    if method == "entropy":
        p = x / total
        return float(torch.exp(-torch.sum(p * torch.log(p))).item())
    if method == "stable_rank":
        return float((total / torch.max(x).clamp_min(eps)).item())
    raise ValueError(f"Unknown effective-rank method: {method!r}")


@dataclass
class MatrixLogSupport:
    """Frozen retained subspace used between slower WeightWatcher checkpoints.

    The full singular spectrum is cached on CPU so the optimizer can evaluate
    both full-``M`` and bulk-effective self-consistent normalizations without
    recomputing a layer SVD.
    """

    retained_rank: int
    normalization_dimension: float
    right_basis: torch.Tensor
    transposed: bool
    checkpoint_epoch: int = 0
    eigenvalues_ascending: torch.Tensor | None = None

    def matrix_dimension(self) -> int:
        if self.eigenvalues_ascending is not None:
            return int(torch.as_tensor(self.eigenvalues_ascending).numel())
        return int(round(float(self.normalization_dimension)))

    def normalization_dimension_for(
        self,
        mode: NormalizationMode,
        *,
        method: EffectiveRankMethod = "participation_ratio",
        gamma: float = 0.0,
    ) -> float:
        if mode == "full_m":
            return float(self.matrix_dimension())
        if mode != "self_consistent":
            raise ValueError(f"Unknown normalization mode: {mode!r}")
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if self.eigenvalues_ascending is None:
            return float(self.normalization_dimension)

        values = torch.as_tensor(
            self.eigenvalues_ascending,
            dtype=torch.float64,
            device="cpu",
        ).reshape(-1)
        total_count = int(values.numel())
        retained = int(max(1, min(int(self.retained_rank), total_count)))
        bulk_count = total_count - retained
        bulk = values[:bulk_count]
        effective = _effective_contributor_count(bulk, method=method)
        dimension = (
            retained
            + effective
            + float(gamma) * (float(bulk_count) - effective)
        )
        return float(min(float(total_count), max(float(retained), dimension)))

    def bulk_effective_count(
        self,
        *,
        method: EffectiveRankMethod = "participation_ratio",
    ) -> float:
        if self.eigenvalues_ascending is None:
            return float("nan")
        values = torch.as_tensor(self.eigenvalues_ascending, dtype=torch.float64)
        retained = int(max(1, min(int(self.retained_rank), values.numel())))
        return _effective_contributor_count(
            values[: values.numel() - retained], method=method
        )

    def state_dict(self) -> dict:
        return {
            "retained_rank": int(self.retained_rank),
            "normalization_dimension": float(self.normalization_dimension),
            "right_basis": self.right_basis.detach().to(
                device="cpu", dtype=torch.float32
            ),
            "transposed": bool(self.transposed),
            "checkpoint_epoch": int(self.checkpoint_epoch),
            "eigenvalues_ascending": (
                None
                if self.eigenvalues_ascending is None
                else torch.as_tensor(self.eigenvalues_ascending)
                .detach()
                .to(device="cpu", dtype=torch.float32)
            ),
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "MatrixLogSupport":
        eigenvalues = payload.get("eigenvalues_ascending")
        return cls(
            retained_rank=int(payload["retained_rank"]),
            normalization_dimension=float(payload["normalization_dimension"]),
            right_basis=torch.as_tensor(payload["right_basis"]).detach().to(
                device="cpu", dtype=torch.float32
            ),
            transposed=bool(payload.get("transposed", False)),
            checkpoint_epoch=int(payload.get("checkpoint_epoch", 0)),
            eigenvalues_ascending=(
                None
                if eigenvalues is None
                else torch.as_tensor(eigenvalues).detach().to(
                    device="cpu", dtype=torch.float32
                )
            ),
        )


@dataclass
class SupportCheckpoint:
    details: pd.DataFrame
    metrics: pd.DataFrame
    supports: dict[str, MatrixLogSupport]
    esd_arrays: dict[str, object]


def orient_tall(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(weight.shape)}")
    if weight.shape[0] >= weight.shape[1]:
        return weight, False
    return weight.transpose(0, 1), True


def _build_support(
    weight: torch.Tensor,
    retained_rank: int,
    *,
    epoch: int,
) -> MatrixLogSupport:
    work, transposed = orient_tall(weight.detach())
    cpu = work.to(device="cpu", dtype=torch.float32)
    _, singular_values, vh = torch.linalg.svd(cpu, full_matrices=False)
    rank = int(max(1, min(int(retained_rank), int(vh.shape[0]))))
    basis = vh[:rank, :].transpose(0, 1).contiguous()
    eigenvalues = singular_values.square().flip(0).contiguous()
    return MatrixLogSupport(
        retained_rank=rank,
        normalization_dimension=float(cpu.shape[1]),
        right_basis=basis,
        transposed=transposed,
        checkpoint_epoch=int(epoch),
        eigenvalues_ascending=eigenvalues,
    )


def analyze_supports(
    model: torch.nn.Module,
    *,
    run_label: str = "FullMatrixLogRG",
    epoch: int = 0,
    global_step: int = 0,
    min_evals: int = 8,
    max_evals: int | None = None,
    svd_method: str = "accurate",
    randomize: bool = True,
    parameter_names: Iterable[str] | None = None,
    build_bases: bool = True,
) -> SupportCheckpoint:
    """Use the baseline's strict WeightWatcher path, then cache midpoint bases."""

    try:
        from rg_baselines import measure_weightwatcher_checkpoint
        from rg_baselines.trap_metrics import attach_correlation_traps
    except ImportError as exc:
        raise ImportError(
            "analyze_supports requires the sibling baseline package. Install from "
            "the repository root with `python -m pip install -e './baseline[experiment]'`."
        ) from exc

    checkpoint = measure_weightwatcher_checkpoint(
        model,
        run_label=run_label,
        epoch=int(epoch),
        global_step=int(global_step),
        min_evals=int(min_evals),
        max_evals=max_evals,
        svd_method=str(svd_method),
        randomize=bool(randomize),
    )
    checkpoint = attach_correlation_traps(checkpoint)

    allowed = set(parameter_names) if parameter_names is not None else None
    parameter_map = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2 and (allowed is None or name in allowed)
    }
    supports: dict[str, MatrixLogSupport] = {}
    if build_bases and not checkpoint.metrics.empty:
        successful = checkpoint.metrics[checkpoint.metrics["status"].eq("ok")]
        for _, row in successful.iterrows():
            parameter_name = row.get("parameter_name")
            if not isinstance(parameter_name, str) or parameter_name not in parameter_map:
                continue
            rank = int(row["m_midpoint"])
            supports[parameter_name] = _build_support(
                parameter_map[parameter_name], rank, epoch=int(epoch)
            )

    if build_bases and allowed is not None:
        missing = sorted(allowed.difference(supports))
        if missing:
            raise RuntimeError(
                "WeightWatcher did not produce usable midpoint supports for: "
                + ", ".join(missing)
            )

    return SupportCheckpoint(
        details=checkpoint.details,
        metrics=checkpoint.metrics,
        supports=supports,
        esd_arrays=checkpoint.esd_arrays,
    )

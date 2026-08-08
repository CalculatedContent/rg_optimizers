"""WeightWatcher midpoint support selection and cached retained bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import torch


@dataclass
class MatrixLogSupport:
    """Frozen retained subspace used between slower WeightWatcher checkpoints."""

    retained_rank: int
    normalization_dimension: float
    right_basis: torch.Tensor
    transposed: bool
    checkpoint_epoch: int = 0

    def state_dict(self) -> dict:
        return {
            "retained_rank": int(self.retained_rank),
            "normalization_dimension": float(self.normalization_dimension),
            "right_basis": self.right_basis.detach().to(
                device="cpu", dtype=torch.float32
            ),
            "transposed": bool(self.transposed),
            "checkpoint_epoch": int(self.checkpoint_epoch),
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "MatrixLogSupport":
        return cls(
            retained_rank=int(payload["retained_rank"]),
            normalization_dimension=float(payload["normalization_dimension"]),
            right_basis=torch.as_tensor(payload["right_basis"]).detach().to(
                device="cpu", dtype=torch.float32
            ),
            transposed=bool(payload.get("transposed", False)),
            checkpoint_epoch=int(payload.get("checkpoint_epoch", 0)),
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
    _, _, vh = torch.linalg.svd(cpu, full_matrices=False)
    rank = int(max(1, min(int(retained_rank), int(vh.shape[0]))))
    basis = vh[:rank, :].transpose(0, 1).contiguous()
    return MatrixLogSupport(
        retained_rank=rank,
        normalization_dimension=float(cpu.shape[1]),
        right_basis=basis,
        transposed=transposed,
        checkpoint_epoch=int(epoch),
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
    """Use the baseline's strict WeightWatcher path, then cache midpoint bases.

    The baseline package is intentionally used here so alpha, detX_num,
    num_pl_spikes, ERG_gap, correlation traps, and rescaling follow the exact
    reference protocol.
    """

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

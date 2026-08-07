"""Self-consistent ECS and optional WeightWatcher epoch diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn

from .config import ExperimentConfig
from .ecs import compute_ecs_svd, participation_ratio


def direct_spectral_rows(
    model: nn.Module,
    *,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    config: ExperimentConfig,
    previous_ranks: dict[str, int],
) -> list[dict[str, Any]]:
    named_parameters = dict(model.named_parameters())
    rows: list[dict[str, Any]] = []
    for parameter_name in config.trace_wall.parameter_names:
        parameter = named_parameters[parameter_name]
        state = compute_ecs_svd(
            parameter,
            min_rank=config.trace_wall.min_ecs_rank,
            normalization_gamma=config.trace_wall.normalization_gamma,
            reference_rank=previous_ranks.get(parameter_name),
            svd_device=config.trace_wall.svd_device,
            numeric_epsilon=config.trace_wall.numerical_epsilon,
        )
        previous_ranks[parameter_name] = int(state.selection.rank)
        singular = state.singular_values.detach().cpu().double().numpy()
        eigenvalues = singular**2
        spectral_sum = float(np.sum(eigenvalues))
        rank = int(state.selection.rank)
        retained_energy = float(np.sum(eigenvalues[:rank]) / spectral_sum)
        rows.append(
            {
                "arm": arm,
                "seed": int(seed),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "layer": parameter_name.removesuffix(".weight").split(".")[-1],
                "parameter_name": parameter_name,
                "ecs_rank": rank,
                "ecs_fractional_rank": state.selection.fractional_rank,
                "ecs_positive_count": state.selection.positive_count,
                "ecs_rank_fraction": rank / state.selection.positive_count,
                "ecs_normalization_dimension": (
                    state.selection.normalization_dimension
                ),
                "ecs_bulk_effective_count": state.selection.bulk_effective_count,
                "ecs_bulk_effective_fraction": (
                    state.selection.bulk_effective_fraction
                ),
                "ecs_trace_log": state.selection.trace_log,
                "ecs_trace_log_per_eval": state.selection.trace_log_per_eval,
                "ecs_lambda_cut": state.selection.lambda_cut,
                "ecs_sign_change_brackets": (
                    state.selection.number_of_sign_change_brackets
                ),
                "ecs_status": state.selection.status,
                "retained_energy_fraction": retained_energy,
                "weight_frobenius_norm": float(np.sqrt(spectral_sum)),
                "spectral_norm": float(singular[0]),
                "stable_rank": float(spectral_sum / eigenvalues[0]),
                "participation_ratio": participation_ratio(eigenvalues),
                "alpha": float("nan"),
                "ERG_gap": float("nan"),
                "detX_num": float("nan"),
                "num_pl_spikes": float("nan"),
                "weightwatcher_status": "not_requested",
            }
        )
    return rows


def merge_weightwatcher(
    rows: list[dict[str, Any]],
    model: nn.Module,
    *,
    arm: str,
    epoch: int,
    global_step: int,
    config: ExperimentConfig,
) -> None:
    if not config.measure_weightwatcher:
        return
    try:
        from rg_baselines.diagnostics import measure_weightwatcher_checkpoint
    except ImportError as exc:
        if config.require_weightwatcher:
            raise ImportError(
                "WeightWatcher diagnostics require the repository's baseline/ "
                "folder on sys.path and the weightwatcher package installed"
            ) from exc
        for row in rows:
            row["weightwatcher_status"] = "unavailable"
        return

    try:
        checkpoint = measure_weightwatcher_checkpoint(
            model,
            run_label=f"{arm}",
            epoch=epoch,
            global_step=global_step,
            min_evals=config.weightwatcher_min_evals,
            max_evals=None,
            svd_method=config.weightwatcher_svd_method,
            randomize=False,
        )
    except Exception:
        if config.require_weightwatcher:
            raise
        for row in rows:
            row["weightwatcher_status"] = "failed"
        return

    metrics = checkpoint.metrics
    by_parameter = {
        str(record["parameter_name"]): record
        for record in metrics.to_dict(orient="records")
        if record.get("status") == "ok" and record.get("parameter_name")
    }
    for row in rows:
        record = by_parameter.get(str(row["parameter_name"]))
        if record is None:
            if config.require_weightwatcher:
                raise RuntimeError(
                    f"WeightWatcher did not return {row['parameter_name']}"
                )
            row["weightwatcher_status"] = "missing_layer"
            continue
        row["weightwatcher_status"] = "ok"
        for key in (
            "alpha",
            "ERG_gap",
            "detX_num",
            "num_pl_spikes",
            "m_midpoint",
            "trace_log_midpoint_total",
            "trace_log_midpoint_per_eval",
            "boundary_overlap_ratio",
        ):
            if key in record:
                row[key] = record[key]


def measure_spectral(
    model: nn.Module,
    *,
    arm: str,
    seed: int,
    epoch: int,
    global_step: int,
    config: ExperimentConfig,
    previous_ranks: dict[str, int],
) -> list[dict[str, Any]]:
    rows = direct_spectral_rows(
        model,
        arm=arm,
        seed=seed,
        epoch=epoch,
        global_step=global_step,
        config=config,
        previous_ranks=previous_ranks,
    )
    merge_weightwatcher(
        rows,
        model,
        arm=arm,
        epoch=epoch,
        global_step=global_step,
        config=config,
    )
    return rows

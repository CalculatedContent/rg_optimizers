"""Construction, scheduling, and audit rows for the MNIST baselines."""

from __future__ import annotations

import math
from typing import Any

import torch

from .config import BaselineConfig
from .muon import (
    MuonWithAuxAdamW,
    SGDMomentumMuon,
    zeropower_via_newton_schulz_5,
)


def _decay_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: torch.nn.Module,
    config: BaselineConfig,
) -> torch.optim.Optimizer:
    config.validate()
    if config.optimizer == "sgd_momentum":
        return torch.optim.SGD(
            _decay_groups(model, config.sgd_weight_decay),
            lr=config.sgd_learning_rate,
            momentum=config.sgd_momentum,
            dampening=config.sgd_dampening,
            nesterov=config.sgd_nesterov,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            _decay_groups(model, config.adamw_weight_decay),
            lr=config.adamw_learning_rate,
            betas=(config.adamw_beta1, config.adamw_beta2),
            eps=config.adamw_eps,
            amsgrad=config.adamw_amsgrad,
        )
    return MuonWithAuxAdamW(
        model.named_parameters(),
        muon_parameter_names=config.muon_parameter_names,
        muon_lr=config.muon_learning_rate,
        muon_momentum=config.muon_momentum,
        muon_nesterov=config.muon_nesterov,
        muon_weight_decay=config.muon_weight_decay,
        newton_schulz_steps=config.muon_newton_schulz_steps,
        muon_eps=config.muon_eps,
        auxiliary_lr=config.muon_aux_learning_rate,
        auxiliary_betas=(config.muon_aux_beta1, config.muon_aux_beta2),
        auxiliary_eps=config.muon_aux_eps,
        auxiliary_weight_decay=config.muon_aux_weight_decay,
    )


def warmup_cosine_learning_rate(
    update_index: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Linear warm-up followed by cosine decay to a non-zero floor.

    ``update_index`` is zero-based and denotes the optimizer update about to be
    taken. A one-epoch warm-up therefore genuinely ramps over every minibatch in
    the first epoch instead of jumping directly to the peak at epoch one.
    """

    if total_steps < 2:
        raise ValueError("total_steps must be at least two")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup < total")
    if update_index < 0:
        raise ValueError("update_index must be non-negative")
    if peak_lr <= 0.0 or min_lr < 0.0 or min_lr > peak_lr:
        raise ValueError("peak_lr/min_lr are inconsistent")

    index = min(int(update_index), int(total_steps) - 1)
    if warmup_steps and index < warmup_steps:
        return float(peak_lr) * (index + 1) / float(warmup_steps)

    progress = (index - warmup_steps) / max(
        1,
        total_steps - warmup_steps - 1,
    )
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + cosine * (float(peak_lr) - float(min_lr))


def _profile_values(
    config: BaselineConfig,
) -> tuple[int, tuple[float, float], tuple[float, float] | None]:
    if config.optimizer == "sgd_momentum":
        return (
            config.sgd_warmup_epochs,
            (config.sgd_learning_rate, config.sgd_min_learning_rate),
            None,
        )
    if config.optimizer == "adamw":
        return (
            config.adamw_warmup_epochs,
            (config.adamw_learning_rate, config.adamw_min_learning_rate),
            None,
        )
    return (
        config.muon_warmup_epochs,
        (config.muon_learning_rate, config.muon_min_learning_rate),
        (config.muon_aux_learning_rate, config.muon_aux_min_learning_rate),
    )


def scheduled_learning_rates(
    config: BaselineConfig,
    *,
    update_index: int,
    total_steps: int,
    steps_per_epoch: int,
) -> dict[str, float]:
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    warmup_epochs, primary, auxiliary = _profile_values(config)
    warmup_steps = int(warmup_epochs) * int(steps_per_epoch)
    values = {
        "primary": warmup_cosine_learning_rate(
            update_index,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=primary[0],
            min_lr=primary[1],
        )
    }
    if auxiliary is not None:
        values["auxiliary"] = warmup_cosine_learning_rate(
            update_index,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=auxiliary[0],
            min_lr=auxiliary[1],
        )
    return values


def set_scheduled_learning_rates(
    optimizer: torch.optim.Optimizer,
    config: BaselineConfig,
    *,
    update_index: int,
    total_steps: int,
    steps_per_epoch: int,
) -> dict[str, float]:
    values = scheduled_learning_rates(
        config,
        update_index=update_index,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
    )
    if config.optimizer != "sgd_momentum_muon":
        for group in optimizer.param_groups:
            group["lr"] = values["primary"]
        return values

    for group in optimizer.param_groups:
        kind = str(group.get("kind", ""))
        group["lr"] = (
            values["primary"] if kind == "muon" else values["auxiliary"]
        )
    return values


def optimizer_group_rows(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    optimizer_label: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, group in enumerate(optimizer.param_groups):
        names = list(group.get("names", []))
        betas = group.get("betas", (float("nan"), float("nan")))
        rows.append(
            {
                "run": optimizer_label,
                "epoch": int(epoch),
                "group_index": index,
                "kind": str(
                    group.get("kind", optimizer.__class__.__name__)
                ),
                "names": "|".join(names),
                "parameter_tensors": len(group["params"]),
                "parameter_count": sum(
                    int(parameter.numel()) for parameter in group["params"]
                ),
                "learning_rate": float(group.get("lr", float("nan"))),
                "momentum": float(group.get("momentum", float("nan"))),
                "beta1": float(betas[0]),
                "beta2": float(betas[1]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "nesterov": bool(group.get("nesterov", False)),
            }
        )
    return rows


__all__ = [
    "MuonWithAuxAdamW",
    "SGDMomentumMuon",
    "zeropower_via_newton_schulz_5",
    "build_optimizer",
    "warmup_cosine_learning_rate",
    "scheduled_learning_rates",
    "set_scheduled_learning_rates",
    "optimizer_group_rows",
]

"""Construction, scheduling, and audit rows for the three MNIST baselines."""

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


def build_optimizer(
    model: torch.nn.Module,
    config: BaselineConfig,
) -> torch.optim.Optimizer:
    config.validate()
    if config.optimizer == "sgd_momentum":
        decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
        no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
        return torch.optim.SGD(
            [
                {"params": decay, "weight_decay": config.sgd_weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.sgd_learning_rate,
            momentum=config.sgd_momentum,
            dampening=config.sgd_dampening,
            nesterov=config.sgd_nesterov,
        )
    if config.optimizer == "adamw":
        decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
        no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": config.adamw_weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
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
    epoch_index: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Linear warm-up followed by cosine decay to a non-zero floor."""

    if total_epochs < 2:
        raise ValueError("total_epochs must be at least two")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must satisfy 0 <= warmup < total")
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    if warmup_epochs and epoch_index < warmup_epochs:
        return float(peak_lr) * (epoch_index + 1) / warmup_epochs
    progress = (epoch_index - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + cosine * (float(peak_lr) - float(min_lr))


def scheduled_learning_rates(
    config: BaselineConfig,
    *,
    epoch_index: int,
) -> dict[str, float]:
    if config.optimizer == "sgd_momentum":
        return {
            "primary": warmup_cosine_learning_rate(
                epoch_index,
                total_epochs=config.epochs,
                warmup_epochs=config.sgd_warmup_epochs,
                peak_lr=config.sgd_learning_rate,
                min_lr=config.sgd_min_learning_rate,
            )
        }
    if config.optimizer == "adamw":
        return {
            "primary": warmup_cosine_learning_rate(
                epoch_index,
                total_epochs=config.epochs,
                warmup_epochs=config.adamw_warmup_epochs,
                peak_lr=config.adamw_learning_rate,
                min_lr=config.adamw_min_learning_rate,
            )
        }
    return {
        "primary": warmup_cosine_learning_rate(
            epoch_index,
            total_epochs=config.epochs,
            warmup_epochs=config.muon_warmup_epochs,
            peak_lr=config.muon_learning_rate,
            min_lr=config.muon_min_learning_rate,
        ),
        "auxiliary": warmup_cosine_learning_rate(
            epoch_index,
            total_epochs=config.epochs,
            warmup_epochs=config.muon_warmup_epochs,
            peak_lr=config.muon_aux_learning_rate,
            min_lr=config.muon_aux_min_learning_rate,
        ),
    }


def set_scheduled_learning_rates(
    optimizer: torch.optim.Optimizer,
    config: BaselineConfig,
    *,
    epoch_index: int,
) -> dict[str, float]:
    values = scheduled_learning_rates(config, epoch_index=epoch_index)
    if config.optimizer != "sgd_momentum_muon":
        for group in optimizer.param_groups:
            group["lr"] = values["primary"]
        return values

    for group in optimizer.param_groups:
        kind = str(group.get("kind", ""))
        group["lr"] = values["primary"] if kind == "muon" else values["auxiliary"]
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
                "kind": str(group.get("kind", optimizer.__class__.__name__)),
                "names": "|".join(names),
                "parameter_tensors": len(group["params"]),
                "parameter_count": sum(int(p.numel()) for p in group["params"]),
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

"""Construction and audit rows for the three clean baselines."""
from __future__ import annotations
from typing import Any
import torch
from .config import BaselineConfig
from .muon import SGDMomentumMuon, zeropower_via_newton_schulz_5


def build_optimizer(model: torch.nn.Module, config: BaselineConfig) -> torch.optim.Optimizer:
    config.validate()
    if config.optimizer=="sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=config.sgd_learning_rate,
            momentum=config.sgd_momentum, dampening=config.sgd_dampening,
            nesterov=config.sgd_nesterov, weight_decay=config.sgd_weight_decay)
    if config.optimizer=="adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.adamw_learning_rate,
            betas=(config.adamw_beta1,config.adamw_beta2), eps=config.adamw_eps,
            weight_decay=config.adamw_weight_decay, amsgrad=config.adamw_amsgrad)
    return SGDMomentumMuon(model.named_parameters(),
        muon_parameter_names=config.muon_parameter_names,
        muon_lr=config.muon_learning_rate, muon_momentum=config.muon_momentum,
        muon_nesterov=config.muon_nesterov, muon_weight_decay=config.muon_weight_decay,
        newton_schulz_steps=config.muon_newton_schulz_steps, muon_eps=config.muon_eps,
        auxiliary_lr=config.muon_aux_learning_rate,
        auxiliary_momentum=config.muon_aux_momentum,
        auxiliary_dampening=config.muon_aux_dampening,
        auxiliary_nesterov=config.muon_aux_nesterov,
        auxiliary_weight_decay=config.muon_aux_weight_decay)


def optimizer_group_rows(optimizer: torch.optim.Optimizer, *, epoch: int,
                         optimizer_label: str) -> list[dict[str,Any]]:
    rows=[]
    for index, group in enumerate(optimizer.param_groups):
        names=list(group.get("names",[]))
        rows.append(dict(run=optimizer_label, epoch=int(epoch), group_index=index,
            kind=str(group.get("kind",optimizer.__class__.__name__)),
            names="|".join(names), parameter_tensors=len(group["params"]),
            parameter_count=sum(int(p.numel()) for p in group["params"]),
            learning_rate=float(group.get("lr",float("nan"))),
            momentum=float(group.get("momentum",float("nan"))),
            weight_decay=float(group.get("weight_decay",0.0)),
            nesterov=bool(group.get("nesterov",False))))
    return rows

__all__=["SGDMomentumMuon","zeropower_via_newton_schulz_5","build_optimizer","optimizer_group_rows"]

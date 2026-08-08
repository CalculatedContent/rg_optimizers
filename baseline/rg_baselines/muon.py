"""Muon with auxiliary AdamW for the MNIST MLP baseline."""

from __future__ import annotations

import math
from typing import Iterable

import torch


@torch.no_grad()
def zeropower_via_newton_schulz_5(
    update: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate a 2-D update's polar factor with Muon's quintic map."""

    if update.ndim != 2:
        raise ValueError(f"Muon requires a matrix, got {tuple(update.shape)}")
    if steps < 1 or eps <= 0:
        raise ValueError("steps and eps must be positive")
    dtype = update.dtype
    transposed = update.shape[0] > update.shape[1]
    x = update.T if transposed else update
    x = x.to(torch.bfloat16 if x.device.type == "cuda" else torch.float32)
    x = x / torch.linalg.vector_norm(x.float()).clamp_min(float(eps)).to(x.dtype)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(dtype)


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Muon on named hidden matrices and AdamW on all auxiliary parameters.

    The parameter partition follows the reference Muon recipe: hidden 2-D
    weights receive Muon, while the classifier and all biases/gains receive
    AdamW. Auxiliary matrix parameters receive decoupled weight decay; 1-D
    parameters do not.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        muon_parameter_names: tuple[str, ...],
        muon_lr: float,
        muon_momentum: float,
        muon_nesterov: bool,
        muon_weight_decay: float,
        newton_schulz_steps: int,
        muon_eps: float,
        auxiliary_lr: float,
        auxiliary_betas: tuple[float, float],
        auxiliary_eps: float,
        auxiliary_weight_decay: float,
    ) -> None:
        named = [
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad
        ]
        requested = set(muon_parameter_names)
        found = {name for name, _ in named if name in requested}
        if requested - found:
            raise ValueError(f"Muon parameters not found: {sorted(requested - found)}")
        if any(parameter.ndim != 2 for name, parameter in named if name in requested):
            raise ValueError("Muon parameters must be matrices")

        muon_named = [(name, parameter) for name, parameter in named if name in requested]
        auxiliary_named = [(name, parameter) for name, parameter in named if name not in requested]
        auxiliary_decay = [
            (name, parameter) for name, parameter in auxiliary_named if parameter.ndim >= 2
        ]
        auxiliary_no_decay = [
            (name, parameter) for name, parameter in auxiliary_named if parameter.ndim < 2
        ]

        groups: list[dict] = []
        if muon_named:
            groups.append(
                {
                    "params": [parameter for _, parameter in muon_named],
                    "kind": "muon",
                    "names": [name for name, _ in muon_named],
                    "lr": float(muon_lr),
                    "momentum": float(muon_momentum),
                    "nesterov": bool(muon_nesterov),
                    "weight_decay": float(muon_weight_decay),
                    "newton_schulz_steps": int(newton_schulz_steps),
                    "eps": float(muon_eps),
                }
            )
        for group_name, entries, weight_decay in (
            ("adamw_decay", auxiliary_decay, auxiliary_weight_decay),
            ("adamw_no_decay", auxiliary_no_decay, 0.0),
        ):
            if entries:
                groups.append(
                    {
                        "params": [parameter for _, parameter in entries],
                        "kind": group_name,
                        "names": [name for name, _ in entries],
                        "lr": float(auxiliary_lr),
                        "betas": tuple(float(value) for value in auxiliary_betas),
                        "eps": float(auxiliary_eps),
                        "weight_decay": float(weight_decay),
                    }
                )

        super().__init__(groups, defaults={})
        self.assignment = {
            name: ("muon" if name in requested else "adamw") for name, _ in named
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            kind = str(group["kind"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    raise RuntimeError("sparse gradients are unsupported")
                state = self.state[parameter]

                if kind == "muon":
                    momentum_value = float(group["momentum"])
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        buffer = torch.zeros_like(gradient)
                        state["momentum_buffer"] = buffer
                    buffer.lerp_(gradient, 1.0 - momentum_value)
                    update_source = (
                        gradient.lerp(buffer, momentum_value)
                        if bool(group["nesterov"])
                        else buffer
                    )
                    update = zeropower_via_newton_schulz_5(
                        update_source,
                        steps=int(group["newton_schulz_steps"]),
                        eps=float(group["eps"]),
                    )
                    decay = float(group.get("weight_decay", 0.0))
                    if decay:
                        parameter.mul_(max(0.0, 1.0 - lr * decay))
                    shape_scale = math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
                    parameter.add_(update, alpha=-lr * shape_scale)
                    continue

                beta1, beta2 = (float(value) for value in group["betas"])
                step = int(state.get("step", 0)) + 1
                state["step"] = step
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None:
                    exp_avg = torch.zeros_like(gradient)
                    exp_avg_sq = torch.zeros_like(gradient)
                    state["exp_avg"] = exp_avg
                    state["exp_avg_sq"] = exp_avg_sq

                decay = float(group.get("weight_decay", 0.0))
                if decay:
                    parameter.mul_(max(0.0, 1.0 - lr * decay))
                exp_avg.lerp_(gradient, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denominator = (
                    exp_avg_sq.sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(float(group["eps"]))
                )
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)
        return loss


# Historical import alias retained for old notebooks and result readers.
SGDMomentumMuon = MuonWithAuxAdamW

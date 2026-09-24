from __future__ import annotations

import math
from typing import Iterable

import torch

from .base_optimizers import (
    Muon,
    OptimizerHandle,
    cosine_learning_rate,
    load_optimizer_state_dict,
    optimizer_state_dict,
    set_learning_rates,
    zero_grad,
    zeropower_via_newton_schulz_5,
)


class MuonHyperBall(torch.optim.Optimizer):
    """Muon followed by a relative Frobenius trust-region projection.

    Ordinary Muon, including its multiplicative matrix weight decay, first
    proposes a complete displacement ``delta``. HyperBall applies

        delta * min(1, rho * ||W||_F / (||delta||_F + eps)).

    The map is radial, so it changes only displacement magnitude.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        newton_schulz_steps: int = 5,
        eps: float = 1e-7,
        relative_radius: float = 0.01,
        hyperball_eps: float = 1e-12,
    ) -> None:
        params = list(params)
        if not params:
            raise ValueError("Muon-HyperBall requires at least one parameter")
        if any(parameter.ndim != 2 for parameter in params):
            raise ValueError("Muon-HyperBall accepts only 2-D parameters")
        if relative_radius <= 0:
            raise ValueError("relative_radius must be positive")
        if hyperball_eps <= 0:
            raise ValueError("hyperball_eps must be positive")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "weight_decay": float(weight_decay),
            "newton_schulz_steps": int(newton_schulz_steps),
            "eps": float(eps),
            "relative_radius": float(relative_radius),
            "hyperball_eps": float(hyperball_eps),
        }
        super().__init__(params, defaults)
        self._last_hyperball_summary: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scales: list[torch.Tensor] = []
        active: list[torch.Tensor] = []
        radii: list[torch.Tensor] = []
        proposed_uwr: list[torch.Tensor] = []
        applied_uwr: list[torch.Tensor] = []
        proposed_norms: list[torch.Tensor] = []
        applied_norms: list[torch.Tensor] = []

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum_value = float(group["momentum"])
            relative_radius = float(group["relative_radius"])
            hyperball_eps = float(group["hyperball_eps"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    raise RuntimeError(
                        "Muon-HyperBall does not support sparse gradients"
                    )

                state = self.state[parameter]
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
                update.mul_(
                    math.sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
                )

                # Complete reference-Muon displacement, including matrix decay.
                delta = update.mul(-lr)
                decay = float(group["weight_decay"])
                decay_factor = max(0.0, 1.0 - lr * decay) if decay else 1.0
                if decay_factor != 1.0:
                    delta.add_(parameter, alpha=decay_factor - 1.0)

                weight_norm = torch.linalg.vector_norm(parameter)
                proposed_norm = torch.linalg.vector_norm(delta)
                eps_tensor = proposed_norm.new_tensor(hyperball_eps)
                radius = weight_norm * relative_radius
                raw_scale = radius / (proposed_norm + eps_tensor)
                scale = torch.where(
                    proposed_norm == 0,
                    proposed_norm.new_ones(()),
                    raw_scale.clamp(max=1.0),
                )

                applied_norm = proposed_norm * scale
                denominator = weight_norm.clamp_min(eps_tensor)
                proposed_ratio = proposed_norm / denominator
                applied_ratio = applied_norm / denominator

                delta.mul_(scale)
                parameter.add_(delta)

                scales.append(scale.detach())
                active.append((scale < (1.0 - 1e-7)).to(scale.dtype).detach())
                radii.append(radius.detach())
                proposed_uwr.append(proposed_ratio.detach())
                applied_uwr.append(applied_ratio.detach())
                proposed_norms.append(proposed_norm.detach())
                applied_norms.append(applied_norm.detach())

        if scales:
            scale_tensor = torch.stack(scales)
            active_tensor = torch.stack(active)
            radius_tensor = torch.stack(radii)
            proposed_uwr_tensor = torch.stack(proposed_uwr)
            applied_uwr_tensor = torch.stack(applied_uwr)
            proposed_norm_tensor = torch.stack(proposed_norms)
            applied_norm_tensor = torch.stack(applied_norms)
            self._last_hyperball_summary = {
                "matrix_updates": scale_tensor.new_tensor(
                    float(scale_tensor.numel())
                ),
                "active_updates": active_tensor.sum(),
                "scale_sum": scale_tensor.sum(),
                "scale_min": scale_tensor.min(),
                "radius_sum": radius_tensor.sum(),
                "proposed_uwr_max": proposed_uwr_tensor.max(),
                "applied_uwr_max": applied_uwr_tensor.max(),
                "proposed_update_norm_max": proposed_norm_tensor.max(),
                "applied_update_norm_max": applied_norm_tensor.max(),
            }
        else:
            self._last_hyperball_summary = None
        return loss

    def last_hyperball_summary(self) -> dict[str, torch.Tensor] | None:
        if self._last_hyperball_summary is None:
            return None
        return {
            key: value.detach().clone()
            for key, value in self._last_hyperball_summary.items()
        }


def _named_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _decay_groups(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    weight_decay: float,
) -> list[dict]:
    decay = [parameter for _, parameter in named_parameters if parameter.ndim >= 2]
    no_decay = [parameter for _, parameter in named_parameters if parameter.ndim < 2]
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer_handles(model, profile: dict) -> list[OptimizerHandle]:
    named = _named_parameters(model)
    family = str(profile["family"])
    if family not in {"muon", "muon_hyperball"}:
        raise ValueError(f"unsupported optimizer family: {family}")

    hidden = [
        parameter
        for name, parameter in named
        if name.startswith("blocks.") and parameter.ndim == 2
    ]
    hidden_ids = {id(parameter) for parameter in hidden}
    auxiliary_named = [
        (name, parameter)
        for name, parameter in named
        if id(parameter) not in hidden_ids
    ]
    if not hidden or not auxiliary_named:
        raise ValueError(
            "Muon partition must contain hidden matrices and auxiliary parameters"
        )

    common = dict(
        lr=float(profile["matrix_learning_rate"]),
        momentum=float(profile["momentum"]),
        nesterov=bool(profile["nesterov"]),
        weight_decay=float(profile["matrix_weight_decay"]),
        newton_schulz_steps=int(profile["newton_schulz_steps"]),
        eps=float(profile.get("muon_epsilon", 1e-7)),
    )
    if family == "muon":
        primary = Muon(hidden, **common)
    else:
        primary = MuonHyperBall(
            hidden,
            **common,
            relative_radius=float(profile["hyperball_relative_radius"]),
            hyperball_eps=float(profile["hyperball_epsilon"]),
        )

    auxiliary = torch.optim.AdamW(
        _decay_groups(auxiliary_named, float(profile["aux_weight_decay"])),
        lr=float(profile["aux_learning_rate"]),
        betas=(float(profile["beta1"]), float(profile["beta2"])),
        eps=float(profile["epsilon"]),
    )
    return [
        OptimizerHandle(
            role="primary",
            optimizer=primary,
            peak_lr=float(profile["matrix_learning_rate"]),
            min_lr=float(profile["matrix_min_learning_rate"]),
        ),
        OptimizerHandle(
            role="auxiliary",
            optimizer=auxiliary,
            peak_lr=float(profile["aux_learning_rate"]),
            min_lr=float(profile["aux_min_learning_rate"]),
        ),
    ]


def optimizer_step(
    handles: list[OptimizerHandle],
) -> dict[str, torch.Tensor] | None:
    summary: dict[str, torch.Tensor] | None = None
    for handle in handles:
        handle.optimizer.step()
        if isinstance(handle.optimizer, MuonHyperBall):
            summary = handle.optimizer.last_hyperball_summary()
    return summary


__all__ = [
    "Muon",
    "MuonHyperBall",
    "OptimizerHandle",
    "cosine_learning_rate",
    "load_optimizer_state_dict",
    "make_optimizer_handles",
    "optimizer_state_dict",
    "optimizer_step",
    "set_learning_rates",
    "zero_grad",
    "zeropower_via_newton_schulz_5",
]

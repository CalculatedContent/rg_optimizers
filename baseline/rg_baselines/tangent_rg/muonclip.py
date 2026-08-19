"""MLP-specific MuonClip-RMS optimizer used by the tangent-RG protocol.

The original MuonClip name covers both a Muon matrix update and transformer
query/key logit clipping.  MLP3 has no query/key logits, so QK clipping is
explicitly **not applicable** here.  ``MuonClipRMSWithAuxAdamW`` applies the
repository's canonical Newton--Schulz polar map to the momentum source and
rescales every nonzero matrix direction to exactly ``rms_scale`` RMS.  All
classifier and bias parameters use the same in-repository auxiliary AdamW
implementation as ordinary Muon.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from ..muon import zeropower_via_newton_schulz_5


QK_CLIPPING_APPLICABLE = False


@torch.no_grad()
def matrix_update_components(
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor | None,
    *,
    momentum: float,
    nesterov: bool,
    newton_schulz_steps: int,
    epsilon: float,
    rms_scale: float,
) -> dict[str, torch.Tensor | float]:
    """Return the Muon source, polar factor, and RMS-matched direction.

    The returned ``momentum_buffer`` is a new tensor and the supplied buffer is
    not mutated, which lets the capture path preview the exact optimizer step.
    A mathematically degenerate all-zero polar factor remains zero rather than
    inventing a direction; ``effective_rms`` records that exceptional case.
    """

    if gradient.ndim != 2:
        raise ValueError(f"MuonClip-RMS requires a matrix, got {tuple(gradient.shape)}")
    if not 0.0 <= float(momentum) < 1.0:
        raise ValueError("momentum must lie in [0, 1)")
    if int(newton_schulz_steps) < 1 or float(epsilon) <= 0.0:
        raise ValueError("Newton--Schulz steps/epsilon are invalid")
    if float(rms_scale) <= 0.0:
        raise ValueError("rms_scale must be positive")

    previous = torch.zeros_like(gradient) if momentum_buffer is None else momentum_buffer
    next_buffer = previous.lerp(gradient, 1.0 - float(momentum))
    source = (
        gradient.lerp(next_buffer, float(momentum))
        if bool(nesterov)
        else next_buffer
    )
    polar = zeropower_via_newton_schulz_5(
        source,
        steps=int(newton_schulz_steps),
        eps=float(epsilon),
    )
    polar_rms_tensor = polar.float().square().mean().sqrt()
    polar_rms = float(polar_rms_tensor.detach().cpu())
    if polar_rms == 0.0:
        direction = torch.zeros_like(polar)
        effective_rms = 0.0
    else:
        direction = polar * (float(rms_scale) / polar_rms_tensor.to(polar.dtype))
        effective_rms = float(direction.float().square().mean().sqrt().detach().cpu())
    return {
        "momentum_buffer": next_buffer,
        "source": source,
        "polar": polar,
        "direction": direction,
        "polar_rms": polar_rms,
        "effective_rms": effective_rms,
    }


class MuonClipRMSWithAuxAdamW(torch.optim.Optimizer):
    """Muon polar directions with exact RMS matching plus auxiliary AdamW."""

    qk_clipping_applicable = QK_CLIPPING_APPLICABLE

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
        rms_scale: float,
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
        if not requested or len(requested) != len(muon_parameter_names):
            raise ValueError("MuonClip-RMS parameter names must be non-empty and unique")
        found = {name for name, _ in named if name in requested}
        if requested - found:
            raise ValueError(f"MuonClip-RMS parameters not found: {sorted(requested - found)}")
        if any(parameter.ndim != 2 for name, parameter in named if name in requested):
            raise ValueError("MuonClip-RMS parameters must be matrices")
        if not 0.0 <= float(muon_momentum) < 1.0:
            raise ValueError("MuonClip-RMS momentum must lie in [0, 1)")
        if float(rms_scale) <= 0.0:
            raise ValueError("MuonClip-RMS rms_scale must be positive")
        if float(muon_lr) <= 0.0 or float(auxiliary_lr) <= 0.0:
            raise ValueError("MuonClip-RMS learning rates must be positive")
        if int(newton_schulz_steps) < 1 or min(muon_eps, auxiliary_eps) <= 0.0:
            raise ValueError("MuonClip-RMS optimizer epsilon/NS settings are invalid")
        if min(muon_weight_decay, auxiliary_weight_decay) < 0.0:
            raise ValueError("MuonClip-RMS weight decay must be non-negative")
        if any(not 0.0 <= float(beta) < 1.0 for beta in auxiliary_betas):
            raise ValueError("MuonClip-RMS auxiliary beta values must lie in [0, 1)")

        matrix_named = [(name, parameter) for name, parameter in named if name in requested]
        auxiliary_named = [(name, parameter) for name, parameter in named if name not in requested]
        auxiliary_decay = [
            (name, parameter) for name, parameter in auxiliary_named if parameter.ndim >= 2
        ]
        auxiliary_no_decay = [
            (name, parameter) for name, parameter in auxiliary_named if parameter.ndim < 2
        ]

        groups: list[dict[str, Any]] = []
        if matrix_named:
            groups.append(
                {
                    "params": [parameter for _, parameter in matrix_named],
                    "kind": "muonclip_rms",
                    "names": [name for name, _ in matrix_named],
                    "lr": float(muon_lr),
                    "momentum": float(muon_momentum),
                    "nesterov": bool(muon_nesterov),
                    "weight_decay": float(muon_weight_decay),
                    "newton_schulz_steps": int(newton_schulz_steps),
                    "eps": float(muon_eps),
                    "rms_scale": float(rms_scale),
                    "qk_clipping_applicable": False,
                }
            )
        for kind, entries, decay in (
            ("adamw_decay", auxiliary_decay, auxiliary_weight_decay),
            ("adamw_no_decay", auxiliary_no_decay, 0.0),
        ):
            if entries:
                groups.append(
                    {
                        "params": [parameter for _, parameter in entries],
                        "kind": kind,
                        "names": [name for name, _ in entries],
                        "lr": float(auxiliary_lr),
                        "betas": tuple(float(value) for value in auxiliary_betas),
                        "eps": float(auxiliary_eps),
                        "weight_decay": float(decay),
                    }
                )

        super().__init__(groups, defaults={})
        self.assignment = {
            name: ("muonclip_rms" if name in requested else "adamw")
            for name, _ in named
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

                if kind == "muonclip_rms":
                    parts = matrix_update_components(
                        gradient,
                        state.get("momentum_buffer"),
                        momentum=float(group["momentum"]),
                        nesterov=bool(group["nesterov"]),
                        newton_schulz_steps=int(group["newton_schulz_steps"]),
                        epsilon=float(group["eps"]),
                        rms_scale=float(group["rms_scale"]),
                    )
                    state["momentum_buffer"] = parts["momentum_buffer"]
                    decay = float(group.get("weight_decay", 0.0))
                    if decay:
                        parameter.mul_(max(0.0, 1.0 - lr * decay))
                    parameter.add_(parts["direction"], alpha=-lr)
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
                denominator = (
                    exp_avg_sq.sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(float(group["eps"]))
                )
                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-lr / bias_correction1,
                )
        return loss


__all__ = [
    "MuonClipRMSWithAuxAdamW",
    "QK_CLIPPING_APPLICABLE",
    "matrix_update_components",
]

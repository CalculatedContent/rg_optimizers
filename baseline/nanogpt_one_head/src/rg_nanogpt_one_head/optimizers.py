from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch


@dataclass
class OptimizerHandle:
    role: str
    optimizer: torch.optim.Optimizer
    peak_lr: float
    min_lr: float

    def set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


@torch.no_grad()
def zeropower_via_newton_schulz_5(
    update: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the polar factor of a matrix update using Muon's quintic map.

    The calculation is deliberately performed in float32 for Apple-MPS
    reliability and matched cross-accelerator numerics, then converted back to
    the parameter dtype.
    """
    if update.ndim != 2:
        raise ValueError(
            f"Muon requires a matrix update, got shape={tuple(update.shape)}"
        )
    if steps < 1 or eps <= 0:
        raise ValueError("steps and eps must be positive")
    original_dtype = update.dtype
    transposed = update.shape[0] > update.shape[1]
    x = update.T if transposed else update
    x = x.float()
    x = x / torch.linalg.vector_norm(x).clamp_min(float(eps))
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(original_dtype)


class Muon(torch.optim.Optimizer):
    """Single-device Muon for hidden 2-D transformer matrices."""

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
    ) -> None:
        params = list(params)
        if not params:
            raise ValueError("Muon requires at least one parameter")
        if any(parameter.ndim != 2 for parameter in params):
            raise ValueError("Muon accepts only 2-D parameters")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "weight_decay": float(weight_decay),
            "newton_schulz_steps": int(newton_schulz_steps),
            "eps": float(eps),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum_value = float(group["momentum"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
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
                    math.sqrt(
                        max(
                            1.0,
                            parameter.shape[0] / parameter.shape[1],
                        )
                    )
                )
                decay = float(group["weight_decay"])
                if decay:
                    parameter.mul_(max(0.0, 1.0 - lr * decay))
                parameter.add_(update, alpha=-lr)
        return loss


def cosine_learning_rate(
    update_index: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Linear warm-up followed by cosine decay to a nonzero floor."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if update_index < 0:
        raise ValueError("update_index must be nonnegative")
    if warmup_steps and update_index < warmup_steps:
        return float(peak_lr) * (update_index + 1) / warmup_steps
    progress = (update_index - warmup_steps) / max(
        1,
        total_steps - warmup_steps - 1,
    )
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + cosine * (float(peak_lr) - float(min_lr))


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
    decay = [
        parameter
        for _, parameter in named_parameters
        if parameter.ndim >= 2
    ]
    no_decay = [
        parameter
        for _, parameter in named_parameters
        if parameter.ndim < 2
    ]
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer_handles(
    model,
    profile: dict,
) -> list[OptimizerHandle]:
    named = _named_parameters(model)
    family = str(profile["family"])

    if family == "sgd":
        optimizer = torch.optim.SGD(
            _decay_groups(named, float(profile["weight_decay"])),
            lr=float(profile["learning_rate"]),
            momentum=float(profile["momentum"]),
            dampening=float(profile.get("dampening", 0.0)),
            nesterov=bool(profile.get("nesterov", True)),
        )
        return [
            OptimizerHandle(
                role="primary",
                optimizer=optimizer,
                peak_lr=float(profile["learning_rate"]),
                min_lr=float(profile["min_learning_rate"]),
            )
        ]

    if family in {"adam", "adamw"}:
        optimizer_class = (
            torch.optim.Adam if family == "adam" else torch.optim.AdamW
        )
        optimizer = optimizer_class(
            _decay_groups(named, float(profile["weight_decay"])),
            lr=float(profile["learning_rate"]),
            betas=(
                float(profile["beta1"]),
                float(profile["beta2"]),
            ),
            eps=float(profile["epsilon"]),
        )
        return [
            OptimizerHandle(
                role="primary",
                optimizer=optimizer,
                peak_lr=float(profile["learning_rate"]),
                min_lr=float(profile["min_learning_rate"]),
            )
        ]

    if family != "muon":
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
            "Muon partition must contain both hidden matrices and "
            "auxiliary parameters"
        )

    muon = Muon(
        hidden,
        lr=float(profile["matrix_learning_rate"]),
        momentum=float(profile["momentum"]),
        nesterov=bool(profile["nesterov"]),
        weight_decay=float(profile["matrix_weight_decay"]),
        newton_schulz_steps=int(profile["newton_schulz_steps"]),
        eps=float(profile.get("muon_epsilon", 1e-7)),
    )
    auxiliary = torch.optim.AdamW(
        _decay_groups(
            auxiliary_named,
            float(profile["aux_weight_decay"]),
        ),
        lr=float(profile["aux_learning_rate"]),
        betas=(
            float(profile["beta1"]),
            float(profile["beta2"]),
        ),
        eps=float(profile["epsilon"]),
    )
    return [
        OptimizerHandle(
            role="primary",
            optimizer=muon,
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


def set_learning_rates(
    handles: list[OptimizerHandle],
    *,
    update_index: int,
    total_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for handle in handles:
        value = cosine_learning_rate(
            update_index,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=handle.peak_lr,
            min_lr=handle.min_lr,
        )
        handle.set_lr(value)
        values[handle.role] = value
    return values


def zero_grad(handles: list[OptimizerHandle]) -> None:
    for handle in handles:
        handle.optimizer.zero_grad(set_to_none=True)


def optimizer_step(handles: list[OptimizerHandle]) -> None:
    for handle in handles:
        handle.optimizer.step()


def optimizer_state_dict(handles: list[OptimizerHandle]) -> list[dict]:
    return [handle.optimizer.state_dict() for handle in handles]


def load_optimizer_state_dict(
    handles: list[OptimizerHandle],
    states: list[dict],
) -> None:
    if len(handles) != len(states):
        raise RuntimeError("optimizer-handle count changed across resume")
    for handle, state in zip(handles, states, strict=True):
        # Optimizer.load_state_dict owns device/dtype restoration semantics.
        # In particular, built-in AdamW may intentionally keep its scalar
        # step state on CPU while moving moment tensors to the parameter
        # device. Do not recursively force every tensor onto XLA.
        handle.optimizer.load_state_dict(state)

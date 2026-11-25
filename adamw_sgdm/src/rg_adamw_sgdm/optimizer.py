"""AdamW direction with a same-history, per-tensor SGDM step magnitude.

Derived from ``short_test_adamw_graft/graft.py``. The mathematical update is
unchanged; it is performed directly rather than running AdamW and rewriting
its result. All update arithmetic stays on the parameter's device.
"""

import math
from numbers import Real
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


def _real(value, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a Python real scalar")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}")


def _validate_group(group: dict) -> None:
    for name in ("lr", "weight_decay"):
        _real(group[name], name)
    for name in ("eps", "reference_lr_ratio", "norm_epsilon"):
        _real(group[name], name, positive=True)
    betas = group["betas"]
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("betas must contain two real scalars")
    for beta in betas:
        _real(beta, "beta")
        if beta >= 1:
            raise ValueError("betas must be in [0, 1)")
    if group["graft"] is not None and type(group["graft"]) is not bool:
        raise TypeError("graft must be True, False, or None")
    for name in ("amsgrad", "maximize"):
        if type(group[name]) is not bool:
            raise TypeError(f"{name} must be a bool")
    if group["norm_dtype"] not in (None, torch.float32, torch.float64):
        raise ValueError("norm_dtype must be None, torch.float32, or torch.float64")
    fixed = group.get("_fixed_betas", betas)
    if tuple(betas) != tuple(fixed):
        raise ValueError(
            "AdamWSGDM requires fixed betas for the same-history SGDM reference. "
            "Use LR-only scheduling (e.g. OneCycleLR with cycle_momentum=False)."
        )


def _validate_precision(parameter: Tensor, group: dict) -> None:
    for name, dtype in (
        ("eps", parameter.dtype),
        ("norm_epsilon", group["norm_dtype"] or parameter.dtype),
    ):
        limits = torch.finfo(dtype)
        if not limits.tiny <= group[name] <= limits.max:
            raise ValueError(f"{name} must be a positive normal value representable in {dtype}")


class AdamWSGDM(Optimizer):
    r"""AdamW with the RG experiments' SGD-momentum magnitude graft.

    For each selected parameter tensor, form AdamW's adaptive displacement
    A and replace it by ``A * ||lr * ratio * m / (1-beta1)|| / ||A||``.
    The target is the zero-dampening, non-Nesterov SGDM buffer on this run's
    own gradients. AdamW's decoupled decay remains separate from that norm.

    Args:
        params: Parameters or ordinary PyTorch parameter-group dictionaries.
        lr: Adam learning rate. Also scales the reference LR via the ratio.
        betas: Fixed first/second moment coefficients.
        eps: Positive Adam denominator epsilon.
        weight_decay: Standard decoupled AdamW weight decay.
        reference_lr_ratio: SGDM reference LR divided by current Adam LR.
        graft: True grafts every tensor in a group, False uses native AdamW
            arithmetic, None grafts only tensors with ndim >= 2.
        amsgrad: Use the running maximum of the second moment.
        maximize: Perform gradient ascent.
        norm_epsilon: Positive floor for the native Adam displacement norm.
        norm_dtype: Optional norm accumulation dtype; None uses param dtype.

    Parameters must be FP32 or FP64. Keep FP32 parameter storage and use
    autocast/GradScaler for mixed precision. This avoids low-precision moment
    loss and matches the original experiments. Dense gradients, LR schedulers,
    closures and optimizer checkpoints are supported. Sparse gradients, tensor
    learning rates, differentiable steps, CUDA graph capture and sharded/FSDP
    parameter norms are outside this version's contract.

    The default bare-parameter rule includes embeddings because a parameter
    iterable carries no module information. Use ``parameter_groups(model)``
    to exclude embeddings, or select grafted groups explicitly.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        reference_lr_ratio: float = 100.0,
        graft: bool | None = None,
        amsgrad: bool = False,
        maximize: bool = False,
        norm_epsilon: float = 1e-30,
        norm_dtype: torch.dtype | None = None,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            reference_lr_ratio=reference_lr_ratio, graft=graft,
            amsgrad=amsgrad, maximize=maximize, norm_epsilon=norm_epsilon,
            norm_dtype=norm_dtype,
        )
        _validate_group(defaults)
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict) -> None:
        group = dict(param_group)
        parameters = group["params"]
        if isinstance(parameters, Tensor):
            parameters = [parameters]
        elif isinstance(parameters, set):
            raise TypeError("optimizer parameters must have a deterministic order")
        else:
            parameters = list(parameters)
        group["params"] = parameters
        merged = {**self.defaults, **group}
        _validate_group(merged)
        for parameter in parameters:
            self._validate_parameter(parameter)
            _validate_precision(parameter, merged)
        group["_fixed_betas"] = tuple(merged["betas"])
        super().add_param_group(group)

    @staticmethod
    def _validate_parameter(parameter: Tensor) -> None:
        if not isinstance(parameter, Tensor):
            raise TypeError("optimizer parameters must be tensors")
        if parameter.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "AdamWSGDM requires FP32 or FP64 parameter storage; keep the "
                "model in FP32 and use autocast for BF16/FP16 compute."
            )
        if parameter.layout != torch.strided or parameter.is_meta:
            raise TypeError("AdamWSGDM requires materialized, dense parameters")

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        """Perform one update without reading device scalars on the host."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Validate before mutating any parameter/state, including newly thawed
        # groups and sparse gradients. No tensor-value checks or host syncs.
        for group in self.param_groups:
            _validate_group(group)
            for parameter in group["params"]:
                self._validate_parameter(parameter)
                _validate_precision(parameter, group)
                if parameter.grad is not None and parameter.grad.layout != torch.strided:
                    raise RuntimeError("AdamWSGDM does not support sparse gradients")

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = -parameter.grad if group["maximize"] else parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.int64, device=parameter.device)
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                if group["amsgrad"] and "max_exp_avg_sq" not in state:
                    state["max_exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"].add_(1)
                moment, second = state["exp_avg"], state["exp_avg_sq"]
                moment.lerp_(grad, 1.0 - beta1)
                second.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                variance = second
                if group["amsgrad"]:
                    variance = state["max_exp_avg_sq"]
                    torch.maximum(variance, second, out=variance)

                # A device step tensor prevents per-step host extraction and
                # changing Python bias-correction constants in XLA graphs.
                step = state["step"].to(dtype=parameter.dtype)
                # expm1 avoids cancellation and beta rounding to 1 in FP32
                # when a valid coefficient is extremely close to one.
                bias1 = -torch.expm1(step * math.log(beta1)) if beta1 else torch.ones_like(step)
                bias2 = -torch.expm1(step * math.log(beta2)) if beta2 else torch.ones_like(step)
                delta = -lr * (moment / bias1) / ((variance / bias2).sqrt() + group["eps"])

                do_graft = group["graft"]
                if do_graft is None:
                    do_graft = parameter.ndim >= 2
                if do_graft:
                    norm_dtype = group["norm_dtype"] or parameter.dtype
                    target = (
                        lr * group["reference_lr_ratio"] / (1.0 - beta1)
                        * torch.linalg.vector_norm(moment.to(dtype=norm_dtype))
                    )
                    native_norm = torch.linalg.vector_norm(delta.to(dtype=norm_dtype))
                    gain = target / native_norm.clamp_min(group["norm_epsilon"])
                    delta.mul_(gain.to(dtype=parameter.dtype))

                parameter.mul_(1.0 - lr * group["weight_decay"])
                parameter.add_(delta)
        return loss

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore this optimizer's checkpoints on the current parameter device.

        Load the model and construct its optimizer on the destination device
        first. Native AdamW and the old MagnitudeGraft wrapper have different
        checkpoint contracts; their states must not be loaded silently here.
        """
        if "optimizer" in state_dict:
            raise ValueError("Legacy MagnitudeGraft wrapper checkpoints need explicit conversion")
        if "param_groups" not in state_dict or "state" not in state_dict:
            raise ValueError("Expected an AdamWSGDM optimizer state_dict")
        for group in state_dict["param_groups"]:
            if "reference_lr_ratio" not in group or "_fixed_betas" not in group:
                raise ValueError("Checkpoint was not created by AdamWSGDM")
            _validate_group(group)
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for parameter in group["params"]:
                self._validate_parameter(parameter)
                _validate_precision(parameter, group)
                state = self.state.get(parameter)
                if not state:
                    continue
                state["step"] = state["step"].to(device=parameter.device, dtype=torch.int64)
                for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if name in state:
                        state[name] = state[name].to(device=parameter.device, dtype=parameter.dtype)

"""The established AdamW-direction / own-history SGDM-magnitude graft.

Pinned numerical reference: short_test_adamw_graft/graft.py at repository commit
98891bed61a816277b7a92f296e44173704f5910, SHA256
e6c9fcf0e750e0efda8b716410623eff5e3f590751d5a345482e97a4b10c45a3.

This port keeps all arithmetic involving scheduled learning rates and Adam step
counters on the parameter device. No per-step scalar extraction, duplicate
native update, or copy of the previous matrices is required. Adam's moments and
both bias corrections remain; only each transformer matrix's update norm is
grafted. Decoupled weight decay is outside that norm.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

import torch


OPTIMIZER_ID = "adamw_sgdm_graft_v1"
GRAFT_SCOPE = "transformer_physical_matrices"
REFERENCE_COMMIT = "98891bed61a816277b7a92f296e44173704f5910"
REFERENCE_SHA256 = "e6c9fcf0e750e0efda8b716410623eff5e3f590751d5a345482e97a4b10c45a3"
_BLOCK_WEIGHT = re.compile(
    r"transformer\.h\.(\d+)\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))\.weight"
)


def _number(name, value, *, minimum=0.0, strict=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum or (strict and value == minimum)):
        raise ValueError(f"{name} must be a finite {'positive' if strict else 'nonnegative'} number")
    return float(value)


class TensorLrAdamWSGDMGraft(torch.optim.AdamW):
    """Exact magnitude-graft equations with native Adam state slot names.

    For a fixed beta1, b=m/(1-beta1) is the uncorrected, zero-dampening,
    non-Nesterov SGDM buffer on this run's own gradients. Each physical block
    matrix keeps Adam's direction and receives norm reference_lr*||b||_F.
    Packed QKV is one matrix. Other parameters follow ordinary AdamW.

    The caller clips gradients before step and sets each group's LR to a device
    scalar on XLA. Adam counters are independent scalar tensors on the parameter
    device, and advance only for parameters with a gradient.
    """

    optimizer_id = OPTIMIZER_ID

    def __init__(self, model, *, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0.0, reference_lr_ratio=400.0, norm_epsilon=1e-30,
                 weight_decay_scope="matrix_only"):
        lr = _number("lr", lr)
        if len(betas) != 2:
            raise ValueError("betas must contain beta1 and beta2")
        beta1, beta2 = (_number("beta", value) for value in betas)
        if beta1 >= 1 or beta2 >= 1:
            raise ValueError("beta1 and beta2 must be less than one")
        eps = _number("eps", eps, strict=True)
        weight_decay = _number("weight_decay", weight_decay)
        self.reference_lr_ratio = _number("reference_lr_ratio", reference_lr_ratio, strict=True)
        self.norm_epsilon = _number("norm_epsilon", norm_epsilon, strict=True)
        if weight_decay_scope != "matrix_only":
            raise ValueError("weight_decay_scope must be matrix_only (ndim >= 2)")
        self.weight_decay_scope = weight_decay_scope
        self._fixed_betas = (beta1, beta2)
        self._fixed_eps = eps
        named = tuple(model.named_parameters())  # Deduplicates the tied output head.
        selected = tuple((name, p) for name, p in named if _BLOCK_WEIGHT.fullmatch(name) and p.ndim == 2)
        if not selected:
            raise ValueError("The model must contain the four physical matrices in each GPT block")
        blocks = {int(_BLOCK_WEIGHT.fullmatch(name).group(1)) for name, _ in selected}
        expected = {f"transformer.h.{index}.{projection}.weight"
                    for index in range(max(blocks) + 1)
                    for projection in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")}
        if {name for name, _ in selected} != expected:
            raise ValueError("Graft scope must contain all four physical matrices in every contiguous GPT block")
        self.grafted_parameter_names = tuple(name for name, _ in selected)
        self.grafted_matrix_count = len(selected)
        self.matrix_names = self.grafted_parameter_names  # Canonical wrapper's terminology.
        self._grafted_parameters = frozenset(p for _, p in selected)
        self._parameter_inventory = [{"name": name, "shape": list(p.shape)} for name, p in named]
        groups = [
            {"params": [p for _, p in named if p.ndim >= 2],
             "parameter_names": [name for name, p in named if p.ndim >= 2],
             "weight_decay": weight_decay, "role": "matrices_and_embeddings"},
            {"params": [p for _, p in named if p.ndim < 2],
             "parameter_names": [name for name, p in named if p.ndim < 2],
             "weight_decay": 0.0, "role": "bias_and_layernorm"},
        ]
        super().__init__(groups, lr=lr, betas=self._fixed_betas, eps=eps,
                         weight_decay=0.0, amsgrad=False, foreach=False,
                         maximize=False, capturable=False, differentiable=False, fused=False)
        self._group_contracts = [self._group_contract(group) for group in self.param_groups]
        self._saved_contract = self.graft_contract()

    @staticmethod
    def _group_contract(group):
        return {key: deepcopy(group.get(key)) for key in
                ("betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach",
                 "capturable", "differentiable", "fused", "role", "parameter_names")}

    def graft_contract(self):
        """JSON-compatible identity recorded alongside native optimizer slots."""
        return {"schema_version": 1, "optimizer": OPTIMIZER_ID, "graft_scope": GRAFT_SCOPE,
                "reference_commit": REFERENCE_COMMIT, "reference_sha256": REFERENCE_SHA256,
                "reference_lr_ratio": self.reference_lr_ratio, "norm_epsilon": self.norm_epsilon,
                "betas": list(self._fixed_betas), "eps": self._fixed_eps,
                "weight_decay_scope": self.weight_decay_scope,
                "grafted_parameter_names": list(self.grafted_parameter_names),
                "grafted_matrix_count": self.grafted_matrix_count,
                "parameter_inventory": deepcopy(self._parameter_inventory)}

    def _check_contract(self):
        if self.graft_contract() != self._saved_contract:
            raise ValueError("The graft contract changed; start a new immutable run")
        if [self._group_contract(group) for group in self.param_groups] != self._group_contracts:
            raise ValueError("Fixed Adam betas/epsilon/decay or parameter-group scope changed")

    @torch.no_grad()
    def step(self, closure=None):
        self._check_contract()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or parameter.is_complex():
                    raise RuntimeError("TensorLrAdamWSGDMGraft requires dense real gradients/parameters")
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                counter, first, second = state["step"], state["exp_avg"], state["exp_avg_sq"]
                counter.add_(1)
                # Match native Adam's moment recurrences, without rewriting them
                # after changing the applied matrix displacement.
                first.lerp_(gradient, 1.0 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                # A tensor exponent avoids step.item() and changing graph
                # constants. Float64 parameters use float64 bias arithmetic;
                # production parameters/counters remain float32.
                exponent = counter.to(dtype=parameter.dtype)
                correction1 = 1.0 - beta1 ** exponent
                correction2 = 1.0 - beta2 ** exponent
                if parameter in self._grafted_parameters:
                    first_hat = first / correction1
                    second_hat = second / correction2
                    delta = (-lr) * first_hat / (second_hat.sqrt() + group["eps"])
                    reference_lr = lr * self.reference_lr_ratio
                    target = (reference_lr / (1.0 - beta1)) * torch.linalg.vector_norm(first)
                    gain = target / torch.linalg.vector_norm(delta).clamp_min(self.norm_epsilon)
                    delta = delta * gain
                else:
                    # Native AdamW equations for embeddings, tied output,
                    # normalizations and biases, including both corrections.
                    denominator = second.sqrt() / correction2.sqrt() + group["eps"]
                    delta = (first / denominator) * ((-lr) / correction1)
                if group["weight_decay"] != 0.0:
                    parameter.mul_(1.0 - lr * group["weight_decay"])
                parameter.add_(delta)
        return loss

    def state_dict(self):
        self._check_contract()
        result = super().state_dict()
        result["graft_contract"] = self.graft_contract()
        return result

    def load_state_dict(self, state_dict):
        if state_dict.get("graft_contract") != self._saved_contract:
            raise ValueError("Checkpoint graft settings or parameter inventory differ from this run")
        if [self._group_contract(group) for group in state_dict.get("param_groups", [])] != self._group_contracts:
            raise ValueError("Checkpoint Adam betas/epsilon/decay or parameter-group scope differ")
        native = {key: value for key, value in state_dict.items() if key != "graft_contract"}
        super().load_state_dict(native)
        # AdamW's default loader keeps uncapturable step tensors on CPU.
        # This port always performs tensor step arithmetic on the parameter
        # device; do not depend on the trainer to repair this after loading.
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state:
                    continue
                if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("Checkpoint has an unexpected Adam state layout")
                counter = state["step"]
                if not isinstance(counter, torch.Tensor) or counter.numel() != 1:
                    raise ValueError("Checkpoint Adam step must be a scalar tensor")
                state["step"] = counter.to(device=parameter.device, dtype=torch.float32).reshape(())
                for key in ("exp_avg", "exp_avg_sq"):
                    if state[key].shape != parameter.shape:
                        raise ValueError("Checkpoint Adam moment shape differs from its parameter")
                    state[key] = state[key].to(device=parameter.device, dtype=parameter.dtype)
            if isinstance(group["lr"], torch.Tensor):
                if group["lr"].numel() != 1:
                    raise ValueError("Checkpoint learning rate must be a scalar tensor")
                if group["params"]:
                    group["lr"] = group["lr"].to(device=group["params"][0].device).reshape(())
        self._check_contract()


def make_optimizer(model, config):
    """Build the established graft; decay includes 2D embeddings when enabled."""
    if config.optimizer != OPTIMIZER_ID or config.graft_scope != GRAFT_SCOPE:
        raise ValueError("Expected the adamw_sgdm_graft_v1 transformer-physical-matrix contract")
    return TensorLrAdamWSGDMGraft(
        model, lr=config.peak_lr, betas=(config.beta1, config.beta2), eps=config.eps,
        weight_decay=config.weight_decay, reference_lr_ratio=config.reference_lr_ratio,
        norm_epsilon=config.norm_epsilon, weight_decay_scope=config.weight_decay_scope,
    )

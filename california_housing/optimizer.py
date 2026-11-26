"""Frozen Adam-direction / momentum-magnitude graft, with explicit MLP targets.

Arithmetic follows short_test_adamw_graft/graft.py. Only the matrix selector
changes: all five MLP weights are grafted; biases retain native AdamW updates.
"""

import math
import torch

MATRIX_NAMES = tuple(f"fc{i}.weight" for i in range(1, 6))


class MagnitudeGraft:
    def __init__(self, model, optimizer, *, reference_lr_ratio):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("MagnitudeGraft requires torch.optim.AdamW")
        if not math.isfinite(reference_lr_ratio) or reference_lr_ratio <= 0:
            raise ValueError("reference_lr_ratio must be finite and positive")
        self.optimizer = optimizer
        self.reference_lr_ratio = float(reference_lr_ratio)
        named = dict(model.named_parameters())
        if any(name not in named or named[name].ndim != 2 for name in MATRIX_NAMES):
            raise ValueError("Expected exactly the five named MLP weight matrices")
        self.matrix_names = MATRIX_NAMES
        self._weights = {named[name] for name in self.matrix_names}
        optimizer_weights = {p for group in optimizer.param_groups for p in group["params"]}
        if not self._weights.issubset(optimizer_weights):
            raise ValueError("Optimizer does not contain all grafted matrices")

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        before = {p: p.clone() for p in self._weights if p.grad is not None}
        self.optimizer.step()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p not in before:
                    continue
                state = self.optimizer.state[p]
                step = state["step"].item()
                m = state["exp_avg"]
                v = state["max_exp_avg_sq"] if group["amsgrad"] else state["exp_avg_sq"]
                m_hat = m / (1.0 - beta1**step)
                v_hat = v / (1.0 - beta2**step)
                adam_delta = -lr * m_hat / (v_hat.sqrt() + group["eps"])
                reference_lr = lr * self.reference_lr_ratio
                target = (reference_lr / (1.0 - beta1)) * torch.linalg.vector_norm(m)
                scale = target / torch.linalg.vector_norm(adam_delta).clamp_min(1e-30)
                p.copy_(before[p] * (1.0 - lr * group["weight_decay"]) + adam_delta * scale)

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(),
                "reference_lr_ratio": self.reference_lr_ratio, "matrix_names": self.matrix_names}

    def load_state_dict(self, state):
        if (state["reference_lr_ratio"] != self.reference_lr_ratio
                or tuple(state["matrix_names"]) != self.matrix_names):
            raise ValueError("Checkpoint graft settings differ from this run")
        self.optimizer.load_state_dict(state["optimizer"])


def make_optimizer(model, config):
    name = config["optimizer"]
    lr = float(config["peak_lr"])
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("peak_lr must be finite and positive")
    if name == "sgdm":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               dampening=0, nesterov=False, weight_decay=0, foreach=False)
    if name not in ("adamw", "modified_adamw"):
        raise ValueError(f"Unknown optimizer {name!r}")
    native = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99),
                              eps=1e-8, weight_decay=0, foreach=False, fused=False)
    if name == "adamw":
        return native
    return MagnitudeGraft(model, native,
                          reference_lr_ratio=float(config["reference_lr_ratio"]))

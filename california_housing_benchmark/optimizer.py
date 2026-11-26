"""Native optimizers and the established Adam-direction/SGDM-magnitude graft.

The graft's step arithmetic matches short_test_adamw_graft/graft.py. Only the
matrix selector is generalized to this published MLP. Decoupled weight decay
is retained separately from the rescaled Adam update; biases remain native.
"""
import math

import torch


class MagnitudeGraft:
    def __init__(self, model, optimizer, *, reference_lr_ratio):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("MagnitudeGraft requires torch.optim.AdamW")
        if not math.isfinite(reference_lr_ratio) or reference_lr_ratio <= 0:
            raise ValueError("reference_lr_ratio must be finite and positive")
        self.optimizer = optimizer
        self.reference_lr_ratio = float(reference_lr_ratio)
        named = dict(model.named_parameters())
        self.matrix_names = tuple(name for name, p in named.items() if p.ndim == 2)
        if not self.matrix_names:
            raise ValueError("The model has no graftable matrices")
        self._weights = {named[name] for name in self.matrix_names}
        optimized = {p for group in optimizer.param_groups for p in group["params"]}
        if not self._weights.issubset(optimized):
            raise ValueError("Optimizer omits a grafted matrix")

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
                "reference_lr_ratio": self.reference_lr_ratio,
                "matrix_names": self.matrix_names}

    def load_state_dict(self, state):
        if (state["reference_lr_ratio"] != self.reference_lr_ratio
                or tuple(state["matrix_names"]) != self.matrix_names):
            raise ValueError("Checkpoint graft settings differ")
        self.optimizer.load_state_dict(state["optimizer"])


def make_optimizer(model, config):
    name = config["optimizer"]
    lr = float(config.get("lr", config.get("peak_lr", 0.0)))
    wd = float(config.get("weight_decay", 0.0))
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(wd) or wd < 0:
        raise ValueError("Invalid learning rate or weight decay")
    # The 2026 reference excludes biases from weight decay. Its remaining
    # numerical embeddings and normalization modules do not occur in this MLP.
    groups = [{"params": [p for n, p in model.named_parameters() if n.endswith("bias")],
               "weight_decay": 0.0},
              {"params": [p for n, p in model.named_parameters() if not n.endswith("bias")],
               "weight_decay": wd}]
    if name == "sgdm":
        return torch.optim.SGD(groups, lr=lr, momentum=float(config.get("momentum", 0.9)),
                               dampening=float(config.get("dampening", 0.0)),
                               nesterov=False, foreach=False)
    if name not in ("adamw", "modified_adamw"):
        raise ValueError(f"Unknown optimizer {name!r}")
    native = torch.optim.AdamW(groups, lr=lr, betas=tuple(config.get("betas", [0.9, 0.999])),
                              eps=float(config.get("eps", 1e-8)), foreach=False, fused=False)
    if name == "adamw":
        return native
    ratio = config.get("reference_lr_ratio")
    if ratio is None:
        ratio = float(config["reference_lr"]) / lr
    return MagnitudeGraft(model, native, reference_lr_ratio=float(ratio))

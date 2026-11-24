"""The MNIST paper's per-matrix magnitude graft, applied to a character GPT."""

import math
import re

import torch


_BLOCK_WEIGHT = re.compile(
    r"transformer\.h\.\d+\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))\.weight"
)


class MagnitudeGraft:
    """Retain the AdamW direction and match a same-history momentum-SGD norm.

    The SGDM reference has zero dampening, no Nesterov correction, and fixed
    momentum equal to Adam's fixed beta1 (0.9 for the frozen GPT recipe).
    Gradients are clipped by the caller before step(). Native AdamW handles
    embeddings, tied weights, normalization parameters, biases, and moments.
    """

    def __init__(self, model, optimizer, *, reference_lr_ratio=100.0):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("MagnitudeGraft requires a torch.optim.AdamW optimizer")
        if not math.isfinite(reference_lr_ratio) or reference_lr_ratio <= 0:
            raise ValueError("reference_lr_ratio must be finite and positive")
        self.optimizer = optimizer
        self.reference_lr_ratio = float(reference_lr_ratio)
        blocks = {
            name: p for name, p in model.named_parameters()
            if _BLOCK_WEIGHT.fullmatch(name) and p.ndim == 2
        }
        self.matrix_names = tuple(blocks)
        self._weights = set(blocks.values())
        optimizer_weights = {p for group in optimizer.param_groups for p in group["params"]}
        if not blocks or not self._weights.issubset(optimizer_weights):
            raise ValueError("The AdamW optimizer must contain the GPT block matrices")

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

                # With zero initial states, m/(1-beta1) equals the uncorrected
                # SGDM buffer for these exact gradients, by its recurrence.
                reference_lr = lr * self.reference_lr_ratio
                target = (reference_lr / (1.0 - beta1)) * torch.linalg.vector_norm(m)
                scale = target / torch.linalg.vector_norm(adam_delta).clamp_min(1e-30)
                # AdamW's decoupled decay is separate from the update norm.
                p.copy_(before[p] * (1.0 - lr * group["weight_decay"]) + adam_delta * scale)

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "reference_lr_ratio": self.reference_lr_ratio,
            "matrix_names": self.matrix_names,
        }

    def load_state_dict(self, state):
        if (state["reference_lr_ratio"] != self.reference_lr_ratio
                or tuple(state["matrix_names"]) != self.matrix_names):
            raise ValueError("Checkpoint graft settings differ from this run")
        self.optimizer.load_state_dict(state["optimizer"])

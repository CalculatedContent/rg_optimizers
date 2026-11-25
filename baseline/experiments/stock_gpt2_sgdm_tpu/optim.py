"""Conventional momentum SGD with a genuinely device-side learning rate.

Torch 2.6's scalar SGD implementation passes ``-lr`` to ``Tensor.add_``'s
scalar ``alpha`` argument.  A tensor learning rate is consequently extracted
back to the host.  Here the final update multiplies tensors instead, so XLA can
reuse one graph while the learning rate changes.  The momentum recurrence,
coupled decay, parameter groups, and momentum-buffer checkpoint keys retain
the native SGD convention.  This is not a normalized-momentum optimizer.
"""
from __future__ import annotations

import torch


class TensorLrSGD(torch.optim.SGD):
    """Dense SGD, dampening=0 and Nesterov=False, with native SGD state dicts.

    A Python-number learning rate works for CPU execution.  For XLA, callers
    must assign a scalar tensor on the parameter device to each group's ``lr``
    before stepping.  No tensor learning rate is converted to a host scalar in
    ``step``.  Unsupported SGD variants fail explicitly, including after a
    checkpoint load or a parameter-group edit.
    """

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, *, maximize=False,
                 foreach=False, differentiable=False, fused=None):
        self._check_options({"dampening": dampening, "nesterov": nesterov,
                             "maximize": maximize, "foreach": foreach,
                             "differentiable": differentiable, "fused": fused})
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov,
                         maximize=maximize, foreach=foreach,
                         differentiable=differentiable, fused=fused)
        for group in self.param_groups:
            self._check_options(group)

    @staticmethod
    def _check_options(group):
        if group.get("dampening", 0.0) != 0.0 or group.get("nesterov", False):
            raise ValueError("TensorLrSGD requires dampening=0 and nesterov=False")
        if any(group.get(name, False) for name in
               ("maximize", "foreach", "differentiable", "fused")):
            raise ValueError("TensorLrSGD requires maximize/foreach/differentiable/fused disabled")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._check_options(group)
            lr = group["lr"]
            momentum, weight_decay = group["momentum"], group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                direction = parameter.grad
                if direction.is_sparse:
                    raise RuntimeError("TensorLrSGD supports dense gradients only")
                if weight_decay != 0.0:
                    direction = direction.add(parameter, alpha=weight_decay)
                if momentum != 0.0:
                    state = self.state[parameter]
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        # Match native SGD: no (1-momentum) factor, including
                        # on the first update, and decay is inside momentum.
                        buffer = state["momentum_buffer"] = direction.clone().detach()
                    else:
                        buffer.mul_(momentum).add_(direction)
                    direction = buffer
                # Do not pass a tensor LR as add_'s scalar alpha argument.
                parameter.add_(direction * (-lr))
        return loss


def make_optimizer(model, config):
    """Use coupled matrix/embedding decay, with no bias or LayerNorm decay."""
    groups = [
        {"params": [p for p in model.parameters() if p.ndim >= 2],
         "weight_decay": config.weight_decay, "role": "matrices_and_embeddings"},
        {"params": [p for p in model.parameters() if p.ndim < 2],
         "weight_decay": 0.0, "role": "bias_and_layernorm"},
    ]
    return TensorLrSGD(groups, lr=config.peak_lr, momentum=config.momentum,
                       dampening=0.0, nesterov=False, foreach=False)

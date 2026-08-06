"""Muon update used by the SGD-momentum-plus-Muon baseline."""
from __future__ import annotations
from typing import Iterable
import torch

@torch.no_grad()
def zeropower_via_newton_schulz_5(update: torch.Tensor, *, steps: int=5, eps: float=1e-7) -> torch.Tensor:
    """Approximate a 2-D update's polar factor with Muon's quintic map."""
    if update.ndim != 2: raise ValueError(f"Muon requires a matrix, got {update.shape}")
    if steps < 1 or eps <= 0: raise ValueError("steps and eps must be positive")
    dtype=update.dtype
    transposed=update.shape[0] > update.shape[1]
    x=update.T if transposed else update
    x=x.to(torch.bfloat16 if x.device.type=="cuda" else torch.float32)
    x=x/torch.linalg.vector_norm(x.float()).clamp_min(float(eps)).to(x.dtype)
    a,b,c=3.4445,-4.7750,2.0315
    for _ in range(int(steps)):
        gram=x@x.T
        x=a*x+(b*gram+c*(gram@gram))@x
    if transposed: x=x.T
    return x.to(dtype)

class SGDMomentumMuon(torch.optim.Optimizer):
    """Muon on named 2-D weights; classical SGD-momentum on all other parameters."""
    def __init__(self, named_parameters: Iterable[tuple[str,torch.nn.Parameter]], *,
                 muon_parameter_names: tuple[str,...], muon_lr: float,
                 muon_momentum: float, muon_nesterov: bool, muon_weight_decay: float,
                 newton_schulz_steps: int, muon_eps: float, auxiliary_lr: float,
                 auxiliary_momentum: float, auxiliary_dampening: float,
                 auxiliary_nesterov: bool, auxiliary_weight_decay: float) -> None:
        named=[(n,p) for n,p in named_parameters if p.requires_grad]
        requested=set(muon_parameter_names)
        found={n for n,p in named if n in requested}
        if requested-found: raise ValueError(f"Muon parameters not found: {sorted(requested-found)}")
        if any(p.ndim!=2 for n,p in named if n in requested):
            raise ValueError("Muon parameters must be matrices")
        groups=[]
        mu=[(n,p) for n,p in named if n in requested]
        aux=[(n,p) for n,p in named if n not in requested]
        if mu:
            groups.append(dict(params=[p for _,p in mu], kind="muon", names=[n for n,_ in mu],
                               lr=float(muon_lr), momentum=float(muon_momentum),
                               nesterov=bool(muon_nesterov), weight_decay=float(muon_weight_decay),
                               newton_schulz_steps=int(newton_schulz_steps), eps=float(muon_eps)))
        if aux:
            groups.append(dict(params=[p for _,p in aux], kind="sgd", names=[n for n,_ in aux],
                               lr=float(auxiliary_lr), momentum=float(auxiliary_momentum),
                               dampening=float(auxiliary_dampening), nesterov=bool(auxiliary_nesterov),
                               weight_decay=float(auxiliary_weight_decay)))
        super().__init__(groups, defaults={})
        self.assignment={n:("muon" if n in requested else "sgd") for n,_ in named}

    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for group in self.param_groups:
            lr=float(group["lr"]); momentum=float(group["momentum"])
            for p in group["params"]:
                if p.grad is None: continue
                grad=p.grad.detach()
                if grad.is_sparse: raise RuntimeError("sparse gradients are unsupported")
                decay=float(group.get("weight_decay",0.0))
                if decay: p.mul_(max(0.0,1.0-lr*decay))
                state=self.state[p]
                buf=state.get("momentum_buffer")
                if group["kind"]=="muon":
                    if buf is None:
                        buf=torch.zeros_like(grad); state["momentum_buffer"]=buf
                    buf.lerp_(grad,1.0-momentum)
                    update=grad.lerp(buf,momentum) if group["nesterov"] else buf
                    update=zeropower_via_newton_schulz_5(
                        update, steps=int(group["newton_schulz_steps"]), eps=float(group["eps"]))
                    p.add_(update,alpha=-lr*max(1.0,p.shape[0]/p.shape[1])**0.5)
                else:
                    damp=float(group.get("dampening",0.0))
                    if momentum:
                        if buf is None:
                            buf=grad.clone(); state["momentum_buffer"]=buf
                        else: buf.mul_(momentum).add_(grad,alpha=1.0-damp)
                        update=grad.add(buf,alpha=momentum) if group["nesterov"] else buf
                    else: update=grad
                    p.add_(update,alpha=-lr)
        return loss

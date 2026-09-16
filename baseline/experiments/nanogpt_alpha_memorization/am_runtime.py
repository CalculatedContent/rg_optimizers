"""Pinned baseline imports, explicit mathematical attention, safe persistence."""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import types
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def baseline(cfg):
    import yaml
    for rel, expected in cfg['source_blobs'].items():
        data=(REPO/rel).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=expected:
            raise ValueError(f'Pinned source changed: {rel}. Review and version the protocol.')
    return yaml.safe_load((REPO/cfg['source_recipe']).read_text())


def imports():
    path=str(REPO/'baseline/nanogpt_one_head/src')
    if path not in sys.path: sys.path.insert(0,path)
    from rg_nanogpt_one_head.model import GPT,GPTConfig
    from rg_nanogpt_one_head.optimizers import make_optimizer_handles,set_learning_rates,zero_grad,optimizer_step
    return GPT,GPTConfig,make_optimizer_handles,set_learning_rates,zero_grad,optimizer_step


def math_attention(self,x):
    b,t,c=x.shape; d=c//self.n_head
    q=self.q_proj(x).view(b,t,self.n_head,d).transpose(1,2)
    k=self.k_proj(x).view(b,t,self.n_head,d).transpose(1,2)
    v=self.v_proj(x).view(b,t,self.n_head,d).transpose(1,2)
    # Scale before matmul; avoid reliance on fused accelerator SDPA in this
    # NEW protocol. This is not proof of the cause of the old MPS crash.
    scores=(q/math.sqrt(d))@k.transpose(-2,-1)
    scores=scores.masked_fill(~self.causal_mask[:,:,:t,:t],float('-inf'))
    probabilities=scores.softmax(-1)
    if self.training and self.dropout:
        probabilities=torch.nn.functional.dropout(probabilities,p=self.dropout)
    y=(probabilities@v).transpose(1,2).contiguous().view(b,t,c)
    return self.resid_dropout(self.out_proj(y))


def make_model(source,cfg,seed,device):
    GPT,GPTConfig,*_=imports()
    torch.manual_seed(seed)
    spec=dict(source['model']); spec.update(cfg['model_overrides'])
    model=GPT(GPTConfig(**spec))
    for block in model.blocks:
        block.attn.forward=types.MethodType(math_attention,block.attn)
    return model.to(device)


def environment(device):
    if device=='mps' and not torch.backends.mps.is_available(): raise ValueError('MPS is not available.')
    if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is not available.')
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','weightwatcher')}
    return {'device':device,'platform':platform.platform(),'python':platform.python_version(),
            'versions':versions,'attention':'explicit_scaled_matmul_softmax','precision':'float32',
            'mps_fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','unset')}


def code_hash():
    names=['am_data.py','am_metrics.py','am_runtime.py','am_spectral.py','am_train.py']
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}


def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def model_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def atomic_save(path,value):
    path=Path(path); temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        torch.save(value,f); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def safe_clip(model,maximum):
    """Never mutate invalid gradients. Resolve finite-entry reduction overflow."""
    try:
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),maximum,error_if_nonfinite=True)
        return float(norm),False
    except RuntimeError as exc:
        if 'non-finite' not in str(exc): raise
        values=[p.grad.detach().cpu().double() for p in model.parameters() if p.grad is not None]
        if not values or not all(bool(torch.isfinite(v).all()) for v in values):
            raise FloatingPointError('Nonfinite gradient entries; preserved before clipping.')
        norms=[]
        for v in values:
            scale=float(v.abs().max())
            norms.append(scale*float(torch.linalg.vector_norm(v/scale)) if scale else 0.)
        norm=math.hypot(*norms)
        if not math.isfinite(norm): raise FloatingPointError('Global gradient norm is not representable.')
        factor=min(1.,maximum/(norm+1e-6))
        for p in model.parameters():
            if p.grad is not None: p.grad.mul_(factor)
        return norm,True


def matrices(model):
    # named_parameters removes aliases, so tied embedding/output is counted once.
    return [(n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim==2]

"""Device, evaluation, and one-epoch training utilities."""
from __future__ import annotations
import random, time
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def choose_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def parameter_l2_norm(model:torch.nn.Module)->float:
    return float(sum(float(torch.sum(p.detach().float()**2).cpu()) for p in model.parameters())**0.5)

@torch.no_grad()
def evaluate(model:torch.nn.Module, loader:DataLoader, *, device:torch.device,
             max_batches:Optional[int]=None)->dict[str,float|int]:
    model.eval(); loss_sum=0.0; correct=seen=0
    for i,(x,y) in enumerate(loader,start=1):
        if max_batches is not None and i>int(max_batches): break
        x,y=x.to(device),y.to(device); logits=model(x); loss=F.cross_entropy(logits,y)
        loss_sum+=float(loss.item())*y.numel(); correct+=int((logits.argmax(1)==y).sum()); seen+=y.numel()
    return {"loss":loss_sum/max(seen,1),"accuracy":correct/max(seen,1),"examples":seen}

def train_one_epoch(model:torch.nn.Module, optimizer:torch.optim.Optimizer, loader:DataLoader,
                    *, device:torch.device, grad_clip_norm:float)->dict[str,float|int]:
    model.train(); loss_sum=0.0; correct=seen=0; norms=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
        logits=model(x); loss=F.cross_entropy(logits,y); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(grad_clip_norm)); optimizer.step()
        n=y.numel(); loss_sum+=float(loss.item())*n; correct+=int((logits.argmax(1)==y).sum()); seen+=n
        norms.append(float(torch.as_tensor(norm).detach().cpu()))
    a=np.asarray(norms,float)
    return {"online_train_loss":loss_sum/max(seen,1),"online_train_accuracy":correct/max(seen,1),
            "mean_gradient_norm_before_clip":float(a.mean()),"median_gradient_norm_before_clip":float(np.median(a)),
            "max_gradient_norm_before_clip":float(a.max()),"batches":len(norms)}

def performance_row(*, config, epoch:int, global_step:int, train_eval:dict, test_eval:dict,
                    online:Optional[dict], parameter_norm:float, train_time:float,
                    evaluation_time:float, ww_time:float, device:torch.device)->dict:
    o=online or {}
    return {"run":config.optimizer_label,"optimizer":config.optimizer,"epoch":epoch,"global_step":global_step,
        "train_loss":float(train_eval["loss"]),"train_accuracy":float(train_eval["accuracy"]),
        "train_examples_evaluated":int(train_eval["examples"]),"test_loss":float(test_eval["loss"]),
        "test_accuracy":float(test_eval["accuracy"]),"test_examples_evaluated":int(test_eval["examples"]),
        "online_train_loss":float(o.get("online_train_loss",np.nan)),
        "online_train_accuracy":float(o.get("online_train_accuracy",np.nan)),
        "mean_gradient_norm_before_clip":float(o.get("mean_gradient_norm_before_clip",np.nan)),
        "median_gradient_norm_before_clip":float(o.get("median_gradient_norm_before_clip",np.nan)),
        "max_gradient_norm_before_clip":float(o.get("max_gradient_norm_before_clip",np.nan)),
        "batches":int(o.get("batches",0)),"parameter_l2_norm":parameter_norm,
        "train_time_sec":train_time,"evaluation_time_sec":evaluation_time,
        "weightwatcher_time_sec":ww_time,"epoch_total_time_sec":train_time+evaluation_time+ww_time,
        "device":str(device),"seed":int(config.seed)}

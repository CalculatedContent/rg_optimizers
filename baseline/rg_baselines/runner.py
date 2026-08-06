"""Run one clean MLP3/MNIST optimizer baseline."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from .config import BaselineConfig
from .diagnostics import measure_weightwatcher_checkpoint
from .engine import choose_device,evaluate,parameter_l2_norm,performance_row,set_seed,train_one_epoch
from .model import MLP3
from .optimizers import build_optimizer,optimizer_group_rows
from .results import BaselineResult,validate_result

def run_baseline(config:BaselineConfig, *, data_dir:str|Path="./data",
                 device:Optional[torch.device]=None, output_dir:Optional[str|Path]=None,
                 progress:bool=True)->BaselineResult:
    config.validate(); set_seed(config.seed); device=device or choose_device()
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    train=datasets.MNIST(str(data_dir),train=True,download=True,transform=tf)
    test=datasets.MNIST(str(data_dir),train=False,download=True,transform=tf)
    gen=torch.Generator().manual_seed(config.seed)
    train_loader=DataLoader(train,batch_size=config.batch_size,shuffle=True,generator=gen,
                            num_workers=config.num_workers)
    train_eval=DataLoader(train,batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers)
    test_loader=DataLoader(test,batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers)
    model=MLP3().to(device); optimizer=build_optimizer(model,config); step=0
    performance=[]; spectral=[]; details=[]; groups=[]; esds={}
    def measure(epoch:int,online:Optional[dict],train_time:float)->None:
        t=time.perf_counter(); tr=evaluate(model,train_eval,device=device,max_batches=config.train_eval_max_batches)
        te=evaluate(model,test_loader,device=device); eval_time=time.perf_counter()-t
        t=time.perf_counter(); ckpt=measure_weightwatcher_checkpoint(model,run_label=config.optimizer_label,
            epoch=epoch,global_step=step,min_evals=config.ww_min_evals,max_evals=config.ww_max_evals,
            svd_method=config.ww_svd_method,randomize=config.ww_randomize); ww_time=time.perf_counter()-t
        performance.append(performance_row(config=config,epoch=epoch,global_step=step,train_eval=tr,test_eval=te,
            online=online,parameter_norm=parameter_l2_norm(model),train_time=train_time,
            evaluation_time=eval_time,ww_time=ww_time,device=device))
        spectral.append(ckpt.metrics); details.append(ckpt.details); esds.update(ckpt.esd_arrays)
        groups.extend(optimizer_group_rows(optimizer,epoch=epoch,optimizer_label=config.optimizer_label))
    measure(0,None,0.0)
    if progress:
        r=performance[-1]; print(f"epoch=000 | {config.optimizer_label} | train loss={r['train_loss']:.4f} acc={r['train_accuracy']:.4f} | test loss={r['test_loss']:.4f} acc={r['test_accuracy']:.4f}")
    checkpoint_dir=Path(output_dir)/"checkpoints" if output_dir and config.save_epoch_checkpoints else None
    if checkpoint_dir: checkpoint_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,config.epochs+1):
        t=time.perf_counter(); online=train_one_epoch(model,optimizer,train_loader,device=device,
                                                      grad_clip_norm=config.grad_clip_norm)
        train_time=time.perf_counter()-t; step+=len(train_loader); measure(epoch,online,train_time)
        if checkpoint_dir: torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict()},checkpoint_dir/f"epoch_{epoch:03d}.pt")
        if progress:
            r=performance[-1]; print(f"epoch={epoch:03d} | {config.optimizer_label} | train loss={r['train_loss']:.4f} acc={r['train_accuracy']:.4f} | test loss={r['test_loss']:.4f} acc={r['test_accuracy']:.4f}")
    p=pd.DataFrame(performance); s=pd.concat(spectral,ignore_index=True); d=pd.concat(details,ignore_index=True)
    g=pd.DataFrame(groups); combined=s.merge(p,on=["run","epoch","global_step"],how="left",validate="many_to_one")
    result=BaselineResult(config,p,s,d,g,combined,esds,model,optimizer)
    if config.strict_metrics: validate_result(result)
    if output_dir: result.save(output_dir)
    return result

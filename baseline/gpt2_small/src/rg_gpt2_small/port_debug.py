"""Small, device-reduced numerical reports for diagnosing the TPU port.

Only scalar summaries cross to the CPU. Full gradients are never copied here.
Instrumentation changes execution boundaries, so record it as a diagnostic run.
"""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import torch
from rg_nanogpt_one_head import runtime as rt


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def scalar(value):
    return value if math.isfinite(value) else str(value)


def environment(output):
    versions={}
    for package in ('torch','torch-xla','libtpu','libtpu-nightly','numpy','PyYAML'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    variables=('PJRT_DEVICE','TPU_ACCELERATOR_TYPE','XLA_USE_BF16','XLA_DOWNCAST_BF16',
               'XLA_FLAGS','XLA_MATMUL_PRECISION','PT_XLA_DEBUG_LEVEL','OMP_NUM_THREADS')
    write(Path(output)/'diagnostics/environment.json',{
        'python':platform.python_version(),'platform':platform.platform(),'versions':versions,
        'source_commit':os.environ.get('RG_GPT2_SOURCE_COMMIT'),
        'environment':{key:os.environ.get(key) for key in variables},
        'mode':'diagnostic; additional synchronization and per-tensor reductions enabled',
        'attribution':'unconfirmed: model/training code, port, runtime, and hardware not yet isolated'})


def stage(output,name,step):
    row={'stage':name,'update':step,'unix_time':time.time()}
    write(Path(output)/'diagnostics/current_stage.json',row)
    print(json.dumps(row),flush=True)


def xla_metrics(output,label,device):
    if torch.device(device).type!='xla': return
    from torch_xla.debug import metrics
    path=Path(output)/'diagnostics'/f'xla-{label}.txt'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(metrics.metrics_report())


def check_tensors(named,output,label,step,device):
    named=list(named)
    if not named: return
    stage(output,label+'_started',step)
    rows=[]
    for name,tensor in named:
        value=tensor.detach().float()
        rows.append(torch.stack((torch.isfinite(value).all().float(),value.amin(),value.amax())))
    table=torch.stack(rows)
    rt.synchronize(torch.device(device))
    # A single small transfer, following an explicit execution barrier.
    values=table.cpu().tolist()
    records=[]
    for (name,tensor),(finite,minimum,maximum) in zip(named,values):
        negative_variance=name.endswith('/exp_avg_sq') and minimum<0
        records.append({'tensor':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype),
                        'all_finite':bool(finite),'min':scalar(minimum),'max':scalar(maximum),
                        'negative_second_moment':negative_variance})
    bad=[r for r in records if not r['all_finite'] or r['negative_second_moment']]
    report={'stage':label,'update':step,'records':records,'invalid_tensors':bad}
    write(Path(output)/'diagnostics'/f'{step:06d}-{label}.json',report)
    xla_metrics(output,f'{step:06d}-{label}',device)
    if bad:
        write(Path(output)/'diagnostics/first_invalid_tensors.json',report)
        print(json.dumps({'invalid_stage':label,'update':step,'invalid_tensors':bad}),flush=True)
        raise RuntimeError(f'Invalid tensor at {label}, update {step}; see diagnostics/first_invalid_tensors.json')
    stage(output,label+'_passed',step)


def optimizer_tensors(model,handles):
    names={id(p):name for name,p in model.named_parameters()}
    yield from (('weight/'+name,p) for name,p in model.named_parameters())
    for handle in handles:
        for parameter,state in handle.optimizer.state.items():
            for key,value in state.items():
                # Adam's CPU step counter is metadata, not a TPU moment tensor.
                if torch.is_tensor(value) and value.device==parameter.device and value.numel()>1:
                    yield f'{handle.role}/{names[id(parameter)]}/{key}',value


def failure(output,error):
    current=Path(output)/'diagnostics/current_stage.json'
    write(Path(output)/'TPU_PORT_FAILURE.json',{
        'status':'failed','attribution':'unconfirmed; not yet demonstrated to be an upstream TPU/XLA bug',
        'source_commit':os.environ.get('RG_GPT2_SOURCE_COMMIT'),
        'last_stage':json.loads(current.read_text()) if current.exists() else None,
        'exception_type':type(error).__name__,'exception':str(error),
        'traceback':traceback.format_exc(),
        'evidence':['manifest.json','diagnostics/environment.json','diagnostics/'],
        'next_comparison':'Replay the failing batch/state on CPU and TPU before upstream attribution.'})

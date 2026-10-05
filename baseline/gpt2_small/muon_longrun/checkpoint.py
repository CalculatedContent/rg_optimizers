"""Atomic full-state saves, bounded local retention, explicit deterministic restore."""
from dataclasses import asdict
import os
import random
import numpy as np
import torch
from common import BATCH_TOKENS, PERMANENT, atomic_json, sync_dir


def cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return value


def state(model,muon,adam,stream,rt,step,schedule,identity):
    rt.step(wait=True)
    rng={'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
    if rt.tpu: rng['xla']=int(rt.xm.get_rng_state(device=rt.device))
    return dict(schema=2,step=step,tokens_seen=step*BATCH_TOKENS,config=asdict(model.config),
                model=cpu(model.state_dict()),muon=cpu(muon.state_dict()),adam=cpu(adam.state_dict()),
                data_cursor=stream.state_dict(),rng=rng,scheduler=asdict(schedule),
                next_update=schedule.values(step),identity=identity)


def save(root,payload):
    step=payload['step']; folder=root/'checkpoints'; folder.mkdir(exist_ok=True)
    path=folder/f'step_{step:07d}.pt'; temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        torch.save(payload,f); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)
    sync_dir(folder)
    link=root/'checkpoint_latest.pt.tmp'; link.unlink(missing_ok=True)
    os.link(path,link); link.replace(root/'checkpoint_latest.pt')
    atomic_json(root/'checkpoint_latest.json',{'file':str(path.relative_to(root)),'step':step,
                'tokens_seen':payload['tokens_seen'],'next_update':payload['next_update'],
                'scheduler':payload['scheduler'],'schema':2})
    rolling=sorted(p for p in folder.glob('step_*.pt') if int(p.stem.split('_')[1]) not in PERMANENT)
    for old in rolling[:-2]: old.unlink()
    atomic_json(root/'checkpoint_inventory.json',{'files':[str(p.relative_to(root)) for p in sorted(folder.glob('*.pt'))],
                'permanent_steps':sorted(PERMANENT),'rolling_keep':2})
    return path


def restore(payload,model,muon,adam,stream,rt,schedule,identity):
    if payload.get('schema')!=2 or payload['scheduler']!=asdict(schedule) or payload['identity']!=identity:
        raise RuntimeError('Resume source, scheduler, corpus or numerical settings differ')
    if payload['config']!=asdict(model.config): raise RuntimeError('Model configuration mismatch')
    model.load_state_dict(payload['model'],strict=True)
    for p in (*model.parameters(),*model.buffers()): rt.replicate(p)
    muon.load_state_dict(payload['muon']); adam.load_state_dict(payload['adam'])
    for group in muon.groups: rt.shard_matrices(group['buffer'])
    # Adam counters/moments must follow the live parameter device, including capturable steps.
    for p,values in adam.state.items():
        for key,value in values.items():
            if isinstance(value,torch.Tensor):
                values[key]=value.to(rt.device); rt.replicate(values[key])
    stream.load_state_dict(payload['data_cursor'])
    torch.set_rng_state(payload['rng']['torch']); np.random.set_state(payload['rng']['numpy'])
    random.setstate(payload['rng']['python'])
    rt.step(wait=True)
    # Restore the seed AFTER materializing restored state; mark_step may advance it.
    if rt.tpu: rt.xm.set_rng_state(payload['rng']['xla'],device=rt.device)
    return int(payload['step'])


def assert_same(a,b,path='state'):
    """Exact CPU comparison of full replay state; used only by the startup gate."""
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b): raise RuntimeError('TPU replay differs at '+path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b): raise RuntimeError('TPU replay differs at '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise RuntimeError('TPU replay keys differ at '+path)
        for k in a: assert_same(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b): raise RuntimeError('TPU replay lengths differ at '+path)
        for i,(x,y) in enumerate(zip(a,b)): assert_same(x,y,path+'.'+str(i))
    elif a!=b: raise RuntimeError('TPU replay differs at '+path)

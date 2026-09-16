"""All-matrix WeightWatcher measurements; no alpha values are edited."""
from __future__ import annotations
import importlib.metadata
import math
import random
import numpy as np
import torch
from am_runtime import matrices,model_hash


def measure(model,cfg,seed,step):
    import weightwatcher as ww
    if importlib.metadata.version('weightwatcher')!=cfg['version']:
        raise ValueError('Install the pinned weightwatcher version '+cfg['version'])
    if cfg.get('fix_fingers')!='clip_xmax': raise ValueError('clip_xmax is required.')
    state=model_hash(model); np_state=np.random.get_state(); py_state=random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+104729)
            np.random.seed((seed+104729)%(2**32)); random.seed(seed+104729)
            holder=torch.nn.ModuleDict(); mapping={}
            for i,(name,w) in enumerate(matrices(model)):
                key=f'p{i:03d}'
                holder[key]=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device='cpu')
                with torch.no_grad(): holder[key].weight.copy_(w.detach().float().cpu())
                mapping[key]=name
            table=ww.WeightWatcher(model=holder).analyze(**{k:v for k,v in cfg.items() if k!='version'})
            if not {'alpha','raw_alpha','D'}.issubset(table.columns) or len(table)!=len(mapping):
                raise RuntimeError('WeightWatcher schema/matrix coverage mismatch; no guard decision made.')
            rows=[]
            for _,r in table.iterrows():
                text=str(r.get('longname',''))+' '+str(r.get('name',''))
                keys=[k for k in mapping if k in text]
                if len(keys)!=1: raise RuntimeError('Unbound WeightWatcher matrix: '+text)
                row={k:(v.item() if isinstance(v,np.generic) else v) for k,v in r.items()}
                row.update(matrix=mapping[keys[0]],step=step,model_sha256=state,
                           alpha_clip_xmax=float(r['alpha']),alpha_raw=float(r['raw_alpha']))
                row['fit_supported']=bool(np.isfinite(r['alpha']) and np.isfinite(r['D'])
                    and float(r.get('num_pl_spikes',0))>=20 and float(r['D'])<=0.2)
                row['correlation_test']='not_calibrated; inspect randomized ESD and rand_distance'
                rows.append(row)
            if len({r['matrix'] for r in rows})!=len(mapping): raise RuntimeError('Duplicate WW matrix.')
            return rows
    finally:
        np.random.set_state(np_state); random.setstate(py_state)
        if model_hash(model)!=state: raise RuntimeError('Spectral monitoring mutated weights.')


def offenders(rows,target):
    # A numerical controller criterion, not certification of a physical PL phase.
    return [r['matrix'] for r in rows if not all(
        isinstance(r.get(k),(int,float)) and math.isfinite(r[k]) and r[k]>=target
        for k in ('alpha_raw','alpha_clip_xmax'))]


def save_rows(path,rows):
    import pandas as pd
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); pd.DataFrame(rows).to_csv(tmp,index=False); tmp.replace(path)

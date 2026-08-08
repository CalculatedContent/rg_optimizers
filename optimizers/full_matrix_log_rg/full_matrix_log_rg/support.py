from __future__ import annotations
import math, copy
from dataclasses import dataclass
import pandas as pd
import torch

@dataclass
class SupportCheckpoint:
    details: pd.DataFrame
    metrics: pd.DataFrame
    supports: dict[str, int]

def _match(name, params):
    c=[name, name+'.weight' if name and not name.endswith('.weight') else name]
    for x in c:
        if x in params: return x
    m=[p for p in params if any(x and p.endswith(x) for x in c)]
    return m[0] if len(m)==1 else None

def analyze_supports(model: torch.nn.Module, *, epoch: int=0, min_evals: int=10) -> SupportCheckpoint:
    import weightwatcher as ww
    m=copy.deepcopy(model).to('cpu').eval(); w=ww.WeightWatcher(model=m)
    details=w.analyze(plot=False, randomize=False, ERG=True, min_evals=min_evals, savefig=False)
    names=[n for n,p in model.named_parameters() if p.ndim==2]; rows=[]; supports={}
    for _,r in details.iterrows():
        lname=str(r.get('longname', r.get('name',''))); pname=_match(lname,names)
        try:
            mt=int(r['detX_num']); mp=int(r['num_pl_spikes']); mm=max(3, math.floor((mt+mp)/2))
            if pname is not None:
                mm=min(mm,min(dict(model.named_parameters())[pname].shape)); supports[pname]=mm
            rows.append({'epoch':epoch,'layer_name':lname,'parameter_name':pname,'alpha':float(r.get('alpha',float('nan'))),'detX_num':mt,'num_pl_spikes':mp,'ERG_gap':int(r.get('ERG_gap',mt-mp)),'m_midpoint':mm})
        except Exception: pass
    return SupportCheckpoint(details, pd.DataFrame(rows), supports)

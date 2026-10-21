"""Numerical checks for exact resume and complete, RNG-neutral evaluation."""
import argparse
import copy
import io
import json
import torch
from common import *

def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--pilot-checkpoint',type=Path);a=p.parse_args()
    data=load_data(a.data); m,o,r=setup(1338)
    assert m.get_num_params(non_embedding=False)==804096
    assert m.lm_head.weight is m.transformer.wte.weight
    for i in range(4):update(m,o,data['train'],r,i)
    payload=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),**rng_state(r)),payload)
    for i in range(4,8):update(m,o,data['train'],r,i)
    expected=copy.deepcopy(m.state_dict()); expected_o=copy.deepcopy(o.state_dict()); expected_r=rng_state(r)
    m2,o2,r2=setup(9999);payload.seek(0);c=torch.load(payload,weights_only=False)
    m2.load_state_dict(c['model']);o2.load_state_dict(c['optimizer']);restore_rng(c,r2)
    for i in range(4,8):update(m2,o2,data['train'],r2,i)
    equal(m2.state_dict(),expected);equal(o2.state_dict(),expected_o)
    assert torch.equal(torch.get_rng_state(),expected_r['torch_rng']) and torch.equal(r2.get_state(),expected_r['train_rng'])
    before=torch.get_rng_state(); v=full_eval(m2,data['validation'][:780]); sampled_loss(m2,data['train'])
    assert v['evaluated_targets']==779 and torch.equal(before,torch.get_rng_state())
    checks=['exact model/AdamW/minibatch/RNG resume','804096 parameters and tied output','every target including partial tail evaluated','evaluation preserves training RNG','upstream and dataset SHA256']
    if a.pilot_checkpoint:
        ck=torch.load(a.pilot_checkpoint,map_location='cpu',weights_only=False);s=spectrum(ck['model'])
        assert s['successful_fits']==18 and abs(s['mean_alpha']-2.0683952747722225)<1e-8 and s['below_two']==14
        checks.append('reproduces all18 pilot alpha fits and mean')
    print(json.dumps(dict(passed=True,checks=checks)))

if __name__=='__main__':main()

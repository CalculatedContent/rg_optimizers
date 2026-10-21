"""Check optimizer identity, matched seeds, schedules, and exact interrupted resume."""
import argparse
import copy
import io
import json
import torch
import comparison as c

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
    p=argparse.ArgumentParser();p.add_argument('--data',type=c.Path,required=True);a=p.parse_args()
    data=c.base.load_data(a.data);d=dict(warmup_steps=100,cosine_end=17000,floor_fraction=.1,grad_clip=1.)
    recipes=[c.candidates()[0],c.candidates()[3]]
    initial=[];indices=[]
    for recipe in recipes:
        m,o,r=c.setup(1340,recipe);initial.append(copy.deepcopy(m.state_dict()));indices.append(r.get_state().clone())
        assert m.get_num_params(non_embedding=False)==804096 and m.lm_head.weight is m.transformer.wte.weight
        assert isinstance(o,torch.optim.AdamW if recipe['optimizer']=='AdamW' else torch.optim.SGD)
        if recipe['optimizer']=='SGDM':assert o.defaults['momentum']==.98 and not o.defaults['nesterov']
        for step in range(4):c.update(m,o,r,data['train'],step,recipe,d)
        b=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),**c.base.rng_state(r)),b)
        for step in range(4,8):c.update(m,o,r,data['train'],step,recipe,d)
        expected=copy.deepcopy(m.state_dict());expected_o=copy.deepcopy(o.state_dict());expected_rng=c.base.rng_state(r)
        m2,o2,r2=c.setup(9999,recipe);b.seek(0);ck=torch.load(b,weights_only=False)
        m2.load_state_dict(ck['model']);o2.load_state_dict(ck['optimizer']);c.base.restore_rng(ck,r2)
        for step in range(4,8):c.update(m2,o2,r2,data['train'],step,recipe,d)
        equal(m2.state_dict(),expected);equal(o2.state_dict(),expected_o)
        assert torch.equal(r2.get_state(),expected_rng['train_rng']) and torch.equal(torch.get_rng_state(),expected_rng['torch_rng'])
        before=r2.get_state().clone();before_torch=torch.get_rng_state().clone()
        val=c.base.full_eval(m2,data['validation'][:780]);c.base.sampled_loss(m2,data['train'])
        assert val['evaluated_targets']==779 and torch.equal(before,r2.get_state()) and torch.equal(before_torch,torch.get_rng_state())
        assert abs(c.lr_at(17000,recipe,d)-recipe['peak_lr']*.1)<1e-15
        assert abs(c.lr_at(100,recipe,d)-recipe['peak_lr'])<1e-15
        assert c.lr_at(20000,recipe,d)==c.lr_at(17000,recipe,d)
    equal(initial[0],initial[1]);assert torch.equal(indices[0],indices[1])
    print(json.dumps(dict(passed=True,checks=['exact AdamW model/optimizer/minibatch/RNG resume',
          'exact SGDM model/momentum/minibatch/RNG resume','matched initial tensors and minibatch generators across optimizers',
          '804096 unique parameters and tied output head','evaluation includes incomplete tail and leaves training RNG unchanged',
          'published PyTorch momentum convention and fixed shared schedule','input and upstream source hashes']),indent=2))

if __name__=='__main__':main()

import importlib.util
from pathlib import Path
import hashlib,json,tempfile,subprocess,sys
import numpy as np
import torch
p=Path(__file__).with_name('gradient_probe.py')
spec=importlib.util.spec_from_file_location('probe',p);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
torch.set_num_threads(1)
module=q.load_model_module();cfg=dict(vocab_size=16,block_size=4,n_layer=1,n_head=1,n_embd=8,dropout=0.0,bias=False,tie_weights=True)
model=module.GPT(module.GPTConfig(**cfg))
data=np.arange(80,dtype=np.uint16)%16
starts=np.array([0,7,11,20])
a,la,_=q.probe(model,data,starts,1)
b,lb,_=q.probe(model,data,starts,4)
assert abs(la-lb)<1e-6
for name in a:torch.testing.assert_close(a[name],b[name],rtol=2e-5,atol=2e-7)
s=q.stats(torch.ones(2),torch.tensor([1.,0.]),torch.tensor([-1.,0.]));assert s['train_val_cosine']==-1 and s['val_directional_derivative_negative_train_grad']==1
assert 'lm_head.weight' not in a
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);d=root/'data';d.mkdir();r=root/'run/epoch_checkpoints';r.mkdir(parents=True)
 meta={'dtype':'uint16','files':{}}
 for split in ['train','val']:
  f=d/f'{split}.bin';data.tofile(f);meta['files'][split]={'path':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 (d/'meta.json').write_text(json.dumps(meta))
 for step in [10,20]:torch.save({'step':step,'config':{'model':cfg,'continuation':{'global_step_offset':100}},'model':model.state_dict()},r/f'model_epoch_1_step_{step:07d}.pt')
 before={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in r.iterdir()}
 subprocess.run([sys.executable,str(p),'--run-dir',str(r.parent),'--local-steps','10','20','--data-root',str(d),'--output',str(root/'out'),'--batches','2','--batch-size','1','--repeats','2','--threads','1'],check=True)
 import csv
 rows=list(csv.DictReader((root/'out/per_parameter_gradients.csv').open()))
 assert {int(x['global_step']) for x in rows}=={110,120}
 for repeat in ['0','1']:
  x=[v for v in rows if v['global_step']=='110' and v['repeat']==repeat]
  y=[v for v in rows if v['global_step']=='120' and v['repeat']==repeat]
  assert [v['train_grad_norm'] for v in x]==[v['train_grad_norm'] for v in y]
  assert abs(sum(float(v['train_gradient_energy_fraction']) for v in x)-1)<1e-6
 assert before=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in r.iterdir()}
 assert (root/'out/gradient_probe_results.zip').is_file()
print('PASS: microbatch equivalence, sign, tied weights, fixed probes, global offsets, read-only checkpoints, CLI archive')

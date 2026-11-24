"""Replay500updatesofarchivedseed1340withoutreplacingitscompletedresult."""
import json
import math
from types import SimpleNamespace
from run_additional_seed import ROOT, study, run, verify, utc

verify()
study.CONFIRMATION_SEED=1340
out=ROOT/'compatibility/seed_1340'
args=SimpleNamespace(device='cpu',phase='confirmation',rate=0.4,steps=500,
                     data=ROOT/'data/input.txt',output=out)
study.worker(args)
job=out/'confirmation/rate_0.4/graft/seed_1340'
snap=run.torch.load(job/'last.pt',map_location='cpu',weights_only=False)
actual=snap['progress']['history'][-1]
reference=next(x for x in json.loads((ROOT/'existing_seed1340/result.json').read_text())['history'] if x['step']==500)
def avg(alpha):
    weights=[]; values=[]
    for row in alpha['matrices']:
        fit=row['fits'][0]
        assert fit['status']=='success'
        weights.append(math.sqrt(fit['num_pl_spikes'])/(fit['D']+.02)); values.append(fit['alpha'])
    assert len(weights)==18
    return sum(w*a for w,a in zip(weights,values))/sum(weights)
loss_delta=actual['validation_loss']-reference['validation_loss']
alpha_delta=avg(actual['alpha'])-avg(reference['alpha'])
record=dict(status='passed' if abs(loss_delta)<1e-6 and abs(alpha_delta)<1e-3 else 'review_required',
            completed_at=utc(),step=500,seed=1340,validation_loss=actual['validation_loss'],
            archived_validation_loss=reference['validation_loss'],loss_difference=loss_delta,
            weighted_alpha=avg(actual['alpha']),archived_weighted_alpha=avg(reference['alpha']),
            weighted_alpha_difference=alpha_delta,exact_packages=True)
run.base.write_json(ROOT/'compatibility.json',record)
print(json.dumps(record,indent=2),flush=True)
if record['status']!='passed':raise SystemExit(2)

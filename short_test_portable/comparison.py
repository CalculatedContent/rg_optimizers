"""Portable version of frozen AdamW/SGDM comparison; use a NEW output directory."""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import zipfile

SOURCE = Path(__file__).resolve().parent
BASE = SOURCE.parent / 'short_test'
sys.path.insert(0, str(BASE))
import portable_common as base
import runtime
import numpy as np
import torch

STOP = False
def stop_signal(*_):
    global STOP
    STOP = True

def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): base.write_json(path, value)
def freeze(path, value):
    if path.exists(): assert read(path) == value, f'Frozen record mismatch: {path}'
    else: write(path, value)
def lock(path):
    f = path.open('a'); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); return f
def hashes():
    return {'comparison': {p.name: sha(p) for p in sorted(SOURCE.glob('*.py'))},
            'baseline': base.source_hashes()}

def candidates():
    a = [dict(id=f'adamw_lr{lr:g}_wd{wd:g}', optimizer='AdamW', peak_lr=lr,
              weight_decay=wd, betas=[.9, .99])
         for lr in (.0003, .001, .003) for wd in (0., .1)]
    s = [dict(id=f'sgdm_lr{lr:g}_m{m:g}', optimizer='SGDM', peak_lr=lr,
              weight_decay=0., momentum=m, nesterov=False, dampening=0.)
         for lr in (.01, .03, .1) for m in (.9, .98)]
    # Alternate optimizers during calibration; identical seed and minibatch stream.
    return [c for pair in zip(a, s) for c in pair]

def prepare(root, data):
    root.mkdir(parents=True, exist_ok=True)
    assert sha(data) == base.INPUT_SHA
    design = dict(schema=1, study='five paired seeds, AdamW versus momentum SGD',
        calibration_seed=2027, confirmation_seeds=[1340,1341,1342,1343,1344],
        candidates=candidates(), screen_steps=4000, finalists_per_optimizer=2,
        max_steps=20000, warmup_steps=100, cosine_end=17000, floor_fraction=.1,
        eval_every=500, checkpoint_every=250, architecture=base.ARCHITECTURE,
        batch_size=12, threads=1, precision='float32', execution=runtime.environment(), grad_clip=1.,
        split='Fixed contiguous 90/5/5, same TinyShakespeare corpus as prior pilot',
        input_sha256=base.INPUT_SHA, source_sha256=hashes(),
        versions={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','weightwatcher','powerlaw')},
        ranking='Lowest full validation cross entropy from step500; endpoint validation breaks ties, then candidate id. At4000 select top2 per optimizer; extend these unchanged to20000; select one recipe per optimizer before confirmation.',
        selection_limit='Bounded single-seed successive-halving calibration, not a proven global optimum. Slow starters can be missed. Confirmation seeds are fresh and paired.',
        stopping='All final runs receive20000 updates; cosine reaches floor at17000 then3000 at floor. A cap does not prove convergence; validation and sampled training plateau diagnostics reported separately.',
        test_policy='No calibration test access or alpha computation. Freeze optimizer recipes before all confirmation test evaluation. Evaluate full train/validation/test at endpoint and best-validation checkpoint only. Same test text was previously exposed in the discovery pilot; no test-guided retuning.',
        alpha_policy='Raw WW0.7.7 defaults, all18 unique matrices, packed QKV; tied token/output matrix counted once. Mean requires all18 successful fits. Alpha never selects a recipe.',
        timing='Record optimizer update time, accumulated training plus validation time, and final evaluation time; interruptions excluded.',
        inference='Compare separately optimized recipes; regularization may differ, so differences do not identify a pure causal optimizer effect.',
        references=[{'url':'https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py',
                     'use':'Adam peak1e-3, beta2.99, warmup100, floorone-tenth are starting points; architecture differs.'},
                    {'url':'https://arxiv.org/abs/2506.12543',
                     'use':'Small batches, momentum.9/.98, raw-gradient norm clipping1, separate LR grids; published large-model settings are not optima for this804096-parameter model.'},
                    {'url':'https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html',
                     'use':'Standard PyTorch unnormalized momentum buffer; no Nesterov or custom rescaling.'}])
    assert design['versions']['weightwatcher']=='0.7.7'
    freeze(root/'design.json', design)
    return design

def verify(root):
    d=read(root/'design.json'); assert d['execution']==runtime.environment(), 'Execution environment changed; use a separate output directory'
    assert d['source_sha256']==hashes(), 'Frozen source changed'
    assert d['versions']=={n:importlib.metadata.version(n) for n in d['versions']}, 'Dependency versions changed'
    return d

def setup(seed, recipe):
    torch.set_num_threads(1); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    assert base.ARCHITECTURE['dropout']==0., 'Portable RNG contract requires zero dropout'
    model=base.GPT(base.GPTConfig(**base.ARCHITECTURE)).to(runtime.DEVICE)
    model.lm_head.weight = model.transformer.wte.weight
    params=list(model.parameters())
    groups=[dict(params=[p for p in params if p.ndim>=2],weight_decay=recipe['weight_decay']),
            dict(params=[p for p in params if p.ndim<2],weight_decay=0.)]
    if recipe['optimizer']=='AdamW':
        optimizer=torch.optim.AdamW(groups,lr=recipe['peak_lr'],betas=tuple(recipe['betas']),eps=1e-8,fused=False)
    else:
        optimizer=torch.optim.SGD(groups,lr=recipe['peak_lr'],momentum=recipe['momentum'],
                                  dampening=0.,nesterov=False)
    return model,optimizer,torch.Generator().manual_seed(seed+1)

def lr_at(step, recipe, design):
    peak=recipe['peak_lr']; warm=design['warmup_steps']; end=design['cosine_end']
    floor=peak*design['floor_fraction']
    if step<warm: return peak*(step+1)/(warm+1)
    if step<end:return floor+.5*(peak-floor)*(1+math.cos(math.pi*(step-warm)/(end-warm)))
    return floor

def update(model,opt,rng,data,step,recipe,design):
    for group in opt.param_groups:group['lr']=lr_at(step,recipe,design)
    x,y=base.batch(data,rng); _,loss=model(x.to(runtime.DEVICE),y.to(runtime.DEVICE))
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    opt.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),design['grad_clip'])
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite raw gradient norm')
    opt.step();return loss.item(),float(norm)

def worker(root,data_path,job_path,target):
    d=verify(root);job=read(job_path);recipe=job['recipe'];seed=job['seed'];out=job_path.parent
    run_lock=lock(out/'run.lock')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    data=base.load_data(data_path)
    # Calibration can access only these arrays, never the test partition.
    if job['phase']=='calibration':data.pop('test')
    model,opt,rng=setup(seed,recipe)
    config=dict(job=job,design_sha256=sha(root/'design.json'))
    last=out/'last.pt';step=0;training=0.;monitor_prior=0.;history=[];best=None;best_step=None;best_state=None
    if last.exists():
        c=torch.load(last,map_location='cpu',weights_only=False);assert c['config']==config
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);base.restore_rng(c,rng)
        step=c['step'];training=c['training_seconds'];monitor_prior=c['monitor_seconds'];history=c['history']
        best=c['best_loss'];best_step=c['best_step'];best_state=c['best_state'];del c
    session_start=time.perf_counter()
    def save():
        obj=dict(config=config,model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                 training_seconds=training,monitor_seconds=monitor_prior+time.perf_counter()-session_start,
                 history=history,best_loss=best,best_step=best_step,best_state=best_state,**base.rng_state(rng))
        tmp=last.with_suffix('.pt.tmp');torch.save(runtime.cpu_tree(obj),tmp);tmp.replace(last)
        write(out/'progress.json',dict(phase=job['phase'],seed=seed,recipe=recipe['id'],step=step,
              target=target,training_seconds=training,last_evaluation=history[-1] if history else None,utc=utc()))
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    print(json.dumps(dict(event='start',phase=job['phase'],recipe=recipe['id'],seed=seed,step=step)),flush=True)
    try:
        while step<target:
            runtime.synchronize();t=time.perf_counter();loss,norm=update(model,opt,rng,data['train'],step,recipe,d)
            runtime.synchronize();step+=1;training+=time.perf_counter()-t
            if step%d['eval_every']==0:
                val=base.full_eval(model,data['validation']);v=val['loss_nats_per_character']
                history.append(dict(step=step,train_sample_loss=base.sampled_loss(model,data['train']),
                                    validation=val,lr=opt.param_groups[0]['lr']))
                if best is None or v<best:best,best_step,best_state=v,step,runtime.cpu_tree(model.state_dict())
            if step%d['checkpoint_every']==0 or STOP or (root/'PAUSE').exists():
                save();print(json.dumps(dict(event='checkpoint',recipe=recipe['id'],seed=seed,step=step,loss=loss,training_seconds=training)),flush=True)
            if STOP or (root/'PAUSE').exists():return 75
    except FloatingPointError as e:
        write(out/'numerical_failure.json',dict(job=job,step=step,error=str(e),utc=utc()))
        return 0
    # Selection and summaries never consult test values or alpha.
    tail=history[-4:]
    plateau=dict(validation_last1500_gain=tail[0]['validation']['loss_nats_per_character']-min(h['validation']['loss_nats_per_character'] for h in tail),
                 train_sample_last1500_gain=tail[0]['train_sample_loss']-min(h['train_sample_loss'] for h in tail)) if len(tail)==4 else None
    summary=dict(job=job,step=step,best_validation_loss=best,best_step=best_step,
                 endpoint_validation=history[-1]['validation'],history=history,plateau=plateau,
                 training_seconds=training,monitor_seconds=monitor_prior+time.perf_counter()-session_start,
                 checkpoint_sha256=sha(last),utc=utc())
    write(out/f'summary_{target}.json',summary)
    if job['phase']=='calibration':return 0
    freeze(out/'selection.json',dict(recipe=recipe,seed=seed,endpoint_step=step,selected_step=best_step,
           selected_validation_loss=best,checkpoint_sha256=sha(last),recipes_sha256=sha(root/'recipes.json')))
    t=time.perf_counter()
    endpoint=dict(step=step,metrics={k:base.full_eval(model,v) for k,v in data.items()},alpha=base.spectrum(model.state_dict()))
    if best_step==step:selected=endpoint
    else:
        model.load_state_dict(best_state)
        selected=dict(step=best_step,metrics={k:base.full_eval(model,v) for k,v in data.items()},alpha=base.spectrum(model.state_dict()))
    assert abs(selected['metrics']['validation']['loss_nats_per_character']-best)<1e-7
    write(out/'result.json',dict(completed=True,job=job,endpoint=endpoint,validation_selected=selected,
          timing_seconds=dict(training_updates=training,training_with_monitoring=summary['monitor_seconds'],final_evaluation=time.perf_counter()-t),
          plateau=plateau,stop_reason='fixed20000_update_budget_not_a_convergence_guarantee',
          checkpoint_sha256=sha(last),design_sha256=config['design_sha256'],test_disclosure=d['test_policy'],utc=utc()))
    return 0

def admission():
    return runtime.admission()

def completed(root):return len(list((root/'confirmation').glob('*/result.json')))
def run_job(root,data,phase,recipe,seed,target):
    if STOP or (root/'PAUSE').exists():raise SystemExit(75)
    out=root/phase/f'{recipe["id"]}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    job=dict(phase=phase,recipe=recipe,seed=seed);freeze(out/'job.json',job)
    if (out/'numerical_failure.json').exists():return
    done=out/('result.json' if phase=='confirmation' else f'summary_{target}.json')
    if done.exists():return
    while not admission():
        write(root/'status.json',dict(state='waiting_for_memory',job=job,utc=utc()))
        if STOP:raise SystemExit(75)
        time.sleep(15)
    write(root/'status.json',dict(state='running',job=job,target=target,completed_confirmation=completed(root),planned_confirmation=10,utc=utc()))
    with (out/'training.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(SOURCE/'comparison.py'),'worker','--output',str(root),
                                '--data',str(data),'--job',str(out/'job.json'),'--target',str(target)],stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            if STOP:child.terminate()
            time.sleep(1)
        code=child.returncode
    if code:
        write(root/'status.json',dict(state='needs_resume',job=job,exit_code=code,utc=utc()))
        raise SystemExit(code)

def ranking(root,recipe,seed,target):
    p=root/'calibration'/f'{recipe["id"]}_seed{seed}'/f'summary_{target}.json'
    if not p.exists():return (float('inf'),float('inf'),recipe['id'])
    r=read(p);return (r['best_validation_loss'],r['endpoint_validation']['loss_nats_per_character'],recipe['id'])

def run(root,data,legacy):
    d=verify(root);batch_lock=lock(root/'batch.lock')
    shared_lock=lock(root.parent/'short_gpt_training.lock')
    # The superseded batch has no shared lock; refuse concurrent execution.
    legacy_lock=lock(legacy/'batch.lock') if legacy and legacy.exists() else None
    if legacy:
        for p in (legacy/'runs').glob('*/run.lock'):
            probe=lock(p);probe.close()
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    seed=d['calibration_seed'];all_candidates=d['candidates']
    for c in all_candidates:run_job(root,data,'calibration',c,seed,d['screen_steps'])
    groups={name:sorted([c for c in all_candidates if c['optimizer']==name],key=lambda c:ranking(root,c,seed,d['screen_steps']))[:d['finalists_per_optimizer']]
            for name in ['AdamW','SGDM']}
    assert all(math.isfinite(ranking(root,c,seed,d['screen_steps'])[0]) for g in groups.values() for c in g),'Insufficient stable candidates'
    freeze(root/'finalists.json',dict(screen_steps=d['screen_steps'],groups=groups))
    for i in range(d['finalists_per_optimizer']):
        for name in ['AdamW','SGDM']:run_job(root,data,'calibration',groups[name][i],seed,d['max_steps'])
    winners={name:min(group,key=lambda c:ranking(root,c,seed,d['max_steps'])) for name,group in groups.items()}
    assert all(math.isfinite(ranking(root,c,seed,d['max_steps'])[0]) for c in winners.values()),'No stable full-budget recipe'
    freeze(root/'recipes.json',dict(design_sha256=sha(root/'design.json'),selected=winners,
                calibration_evidence={name:dict(candidate=c['id'],ranking=ranking(root,c,seed,d['max_steps'])) for name,c in winners.items()},
                confirmation_seeds=d['confirmation_seeds'],selection='Validation only; frozen before confirmation test evaluation'))
    for i,seed in enumerate(d['confirmation_seeds']):
        for name in (['AdamW','SGDM'] if i%2==0 else ['SGDM','AdamW']):
            run_job(root,data,'confirmation',winners[name],seed,d['max_steps']);summarize(root)
    summarize(root)
    failures=len(list((root/'confirmation').glob('*/numerical_failure.json')))
    write(root/'status.json',dict(state='training_finished_analysis_pending',completed_confirmation=completed(root),numerical_failures=failures,planned_confirmation=10,utc=utc()))

def summarize(root):
    rows=[]
    for p in sorted((root/'confirmation').glob('*/result.json')):
        r=read(p)
        for endpoint in ['endpoint','validation_selected']:
            q=r[endpoint];row=dict(optimizer=r['job']['recipe']['optimizer'],seed=r['job']['seed'],checkpoint=endpoint,step=q['step'],
                mean_alpha=q['alpha']['mean_alpha'],below_two=q['alpha']['below_two'],successful_fits=q['alpha']['successful_fits'],
                training_seconds=r['timing_seconds']['training_updates'])
            for split,metrics in q['metrics'].items():
                for k,v in metrics.items():row[f'{split}_{k}']=v
            rows.append(row)
    write(root/'results.json',rows)
    if rows:
        with (root/'results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    paired=[]
    for checkpoint in ['endpoint','validation_selected']:
        for metric in ['test_loss_nats_per_character','validation_loss_nats_per_character','train_loss_nats_per_character','mean_alpha','training_seconds']:
            a={r['seed']:r[metric] for r in rows if r['optimizer']=='AdamW' and r['checkpoint']==checkpoint}
            s={r['seed']:r[metric] for r in rows if r['optimizer']=='SGDM' and r['checkpoint']==checkpoint}
            seeds=sorted(k for k in a.keys()&s.keys() if a[k] is not None and s[k] is not None)
            v=np.array([a[k]-s[k] for k in seeds]);n=len(v)
            from scipy.stats import t
            mean=float(v.mean()) if n else None;sd=float(v.std(ddof=1)) if n>1 else None
            half=float(t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
            paired.append(dict(checkpoint=checkpoint,metric=metric,seeds=seeds,n=n,planned_pairs=5,
                adamw_minus_sgdm_mean=mean,sd=sd,ci95_low=mean-half if half is not None else None,
                ci95_high=mean+half if half is not None else None,uncertainty='Paired Student-t interval across seeds; incomplete until n=5; does not capture corpus or hyperparameter-search uncertainty'))
    write(root/'paired_statistics.json',paired)

def pack(root,data):
    d=verify(root);archive_lock=lock(root/'archive.lock');base_entries={}
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.json','.txt','.csv','.png'):base_entries[p.name]=p.read_bytes()
    base_entries['input.txt']=data.read_bytes();assert hashlib.sha256(base_entries['input.txt']).hexdigest()==base.INPUT_SHA
    chunks=[];entries=dict(base_entries)
    for stage in ['calibration','confirmation','legacy_pilot']:
        for out in sorted((root/stage).glob('*')):
            if not out.is_dir():continue
            payload={str(p.relative_to(root)):p.read_bytes() for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.pt','.log')}
            if not payload:continue
            if sum(map(len,entries.values()))+sum(map(len,payload.values()))>38*2**20:
                chunks.append(entries);entries={}
            entries.update(payload)
    if entries:chunks.append(entries)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    paths=[]
    for i,entry in enumerate(chunks,1):
        path=root/f'GPT_AdamW_SGDM_{stamp}_Part{i:03d}.zip'
        integrity={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in entry.items()}
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for k,v in entry.items():z.writestr(k,v)
            z.writestr('integrity.json',json.dumps(integrity))
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        paths.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    write(root/'package_index.json',dict(created_utc=utc(),parts=paths,restore='Merge matching jobs by checkpoint step; never overwrite newer local states. Source revision is in repository_revision.json.'))
    print(json.dumps(paths),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','worker','summarize','package'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path)
    p.add_argument('--legacy',type=Path);p.add_argument('--job',type=Path);p.add_argument('--target',type=int)
    p.add_argument('--device', choices=['cpu','tpu'], default=os.environ.get('SHORT_GPT_DEVICE','cpu'))
    a=p.parse_args();runtime.DEVICE=a.device;os.environ['SHORT_GPT_DEVICE']=a.device;runtime.initialize()
    root=a.output.resolve();data=a.data.resolve() if a.data else None
    if a.command=='prepare':print(json.dumps(prepare(root,data)))
    elif a.command=='worker':raise SystemExit(worker(root,data,a.job,a.target))
    elif a.command=='run':run(root,data,a.legacy)
    elif a.command=='summarize':verify(root);summarize(root)
    else:pack(root,data)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Continue one completed Muon run from step 10k to 100k without replaying canaries."""
from __future__ import annotations
import argparse, copy, json, math, time
from collections import Counter
from pathlib import Path
import torch

from am_data import Dataset
from am_metrics import batches, suffix_losses
from am_runtime import baseline, imports, make_model, environment, model_hash, atomic_json, atomic_save, safe_clip
from am_train import audit, snapshot, restore


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,help='Completed 10-run study root')
    p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--target-step',type=int,default=100000)
    p.add_argument('--device',choices=['mps','cpu','cuda'],default='mps')
    p.add_argument('--checkpoint-every',type=int,default=5000)
    p.add_argument('--behavior-every',type=int,default=5000)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()

    root=Path(a.root).resolve(); original_cfg=json.loads((root/'protocol.json').read_text())
    start_run=root/'muon'/f'seed_{a.seed}'; source_checkpoint=start_run/'checkpoint_latest.pt'
    if not source_checkpoint.exists(): raise FileNotFoundError(source_checkpoint)
    source_saved=torch.load(source_checkpoint,map_location='cpu',weights_only=True)
    start_step=int(source_saved['step'])
    if start_step!=int(original_cfg['steps']): raise ValueError(f'Expected completed checkpoint at {original_cfg["steps"]}, found {start_step}')
    if a.target_step<=start_step: raise ValueError('target-step must exceed the source checkpoint step')

    # Freeze the original 0..5000 canary acquisition schedule while extending only ordinary training.
    cfg=copy.deepcopy(original_cfg)
    cfg['steps']=a.target_step
    cfg['withdrawal_step']=int(original_cfg['steps'])//2
    cfg['checkpoint_every']=a.checkpoint_every
    cfg['behavior_every']=a.behavior_every
    cfg['expensive_every']=max(a.behavior_every,25000)

    out=root/'extensions'/f'muon_seed_{a.seed}_to_{a.target_step}'
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'continuation_protocol.json',cfg)
    atomic_json(out/'source.json',{'source_run':str(start_run),'source_checkpoint':str(source_checkpoint),'source_step':start_step})

    runtime=environment(a.device); source=baseline(original_cfg)
    _,_,make_handles,set_lrs,zero_grad,optimizer_step=imports()
    model=make_model(source,original_cfg,a.seed,a.device)
    profile=copy.deepcopy(source['optimizer_profiles']['muon']); handles=make_handles(model,profile)
    data=Dataset(cfg,a.seed,source['training']['batch_size']*source['training']['grad_accum_steps'])

    original_manifest=json.loads((start_run/'manifest.json').read_text())
    if data.fingerprint!=original_manifest['data_sha256']:
        raise ValueError('Extended dataset changed the original canary/data schedule; refusing continuation.')

    latest=out/'checkpoint_latest.pt'; completed=start_step; attempted=int(source_saved['attempted']); counts=Counter(source_saved['counts'])
    if a.resume:
        if not latest.exists(): raise ValueError('No continuation checkpoint exists to resume.')
        saved=torch.load(latest,map_location='cpu',weights_only=True); completed=int(saved['step']); attempted=int(saved['attempted']); counts=Counter(saved['counts']); restore(saved,model,handles)
    else:
        if latest.exists(): raise ValueError('Continuation already exists; use --resume.')
        restore(source_saved,model,handles)
        if model_hash(model)!=source_saved['model_sha256']: raise ValueError('Source checkpoint tensor hash mismatch.')
        state=snapshot(model,handles,counts,completed,attempted); state.update(model_sha256=model_hash(model),source_step=start_step)
        atomic_save(latest,state); atomic_save(out/f'model_{completed:08d}.pt',{'model':state['model'],'step':completed,'source_step':start_step})

    # Preserve the original Muon LR schedule. At 10k it has reached its configured minimum,
    # so the continuation remains at that minimum rather than stretching/restarting the schedule.
    schedule_steps=math.ceil(source['dataset']['train_tokens']*profile['lr_schedule_epochs']/(data.batch_size*source['model']['block_size']))
    warmup=math.ceil(schedule_steps*profile['warmup_fraction'])
    print(f'Continuing Muon seed={a.seed}: {completed} -> {a.target_step} on {a.device}',flush=True)
    print(f'Canary withdrawal remains frozen at step {data.withdrawal}; no new canary presentations occur.',flush=True)
    print(f'Original LR schedule length={schedule_steps}; continuation uses the configured minimum LR after schedule end.',flush=True)
    print(f'Output: {out}',flush=True)
    started=time.monotonic()

    while completed<a.target_step:
        step=completed; attempted+=1; zero_grad(handles); model.train(); set_lrs(handles,update_index=step,total_steps=schedule_steps,warmup_steps=warmup)
        records=data.batch(step); last_loss=0.0
        for chunk in batches(records,source['training']['batch_size']):
            loss=suffix_losses(model,chunk)[0].mean(1).sum()/data.batch_size
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite forward loss at step {step}')
            loss.backward(); last_loss+=float(loss.detach())
        norm,overflow=safe_clip(model,source['training']['grad_clip']); optimizer_step(handles)
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()): raise FloatingPointError(f'Nonfinite parameter after step {step}')
        counts.update(r.id for r in records); completed=step+1
        if completed%100==0: print(f'muon seed={a.seed} step={completed}/{a.target_step} loss={last_loss:.6f} grad_norm={norm:.3g} norm_fallback={overflow}',flush=True)
        if completed%a.checkpoint_every==0 or completed==a.target_step:
            if counts!=data.planned_counts(completed): raise RuntimeError('Presentation counts changed during continuation.')
            state=snapshot(model,handles,counts,completed,attempted); state.update(model_sha256=model_hash(model),source_step=start_step)
            atomic_save(latest,state); atomic_save(out/f'model_{completed:08d}.pt',{'model':state['model'],'step':completed,'source_step':start_step})
        if completed%a.behavior_every==0 or completed==a.target_step:
            audit(model,data,cfg,source,out,completed,counts)

    result={'state':'complete','seed':a.seed,'source_step':start_step,'step':completed,'elapsed_seconds':time.monotonic()-started,'runtime':runtime}
    atomic_json(out/'complete.json',result); print(json.dumps(result),flush=True); print(out,flush=True)

if __name__=='__main__': main()

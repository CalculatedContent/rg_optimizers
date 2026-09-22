"""Fast ten-run memorization protocol: ordinary AdamW vs ordinary Muon, no online WW."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
import time
import torch
from am_data import Dataset,digest
from am_metrics import batches,suffix_losses,evaluate,exposure,compression_probes,compress_summary
from am_runtime import (baseline,imports,make_model,environment,code_hash,cpu_copy,model_hash,
                        atomic_json,atomic_save,safe_clip)


def snapshot(model,handles,counts,step,attempted):
    return {'model':cpu_copy(model.state_dict()),'optimizers':[cpu_copy(h.optimizer.state_dict()) for h in handles],
            'counts':dict(counts),'step':step,'attempted':attempted,'torch_rng':torch.get_rng_state(),
            'device_rng':torch.mps.get_rng_state() if next(model.parameters()).device.type=='mps' else
                         torch.cuda.get_rng_state_all() if next(model.parameters()).device.type=='cuda' else None}


def restore(saved,model,handles):
    model.load_state_dict(saved['model'])
    for h,state in zip(handles,saved['optimizers'],strict=True): h.optimizer.load_state_dict(state)
    torch.set_rng_state(saved['torch_rng'])
    device=next(model.parameters()).device.type
    if device=='mps': torch.mps.set_rng_state(saved['device_rng'])
    elif device=='cuda': torch.cuda.set_rng_state_all(saved['device_rng'])


def audit(model,data,cfg,source,run_dir,step,counts):
    target=run_dir/'behavior'/f'step_{step:08d}.json'
    if target.exists(): return
    before=model_hash(model)
    rows=evaluate(model,data.probes(final=step==cfg['steps']),source['training']['batch_size'])
    for r in rows: r['presentations']=counts.get(r['id'],0)
    result={'step':step,'model_sha256':before,'rows':rows,'withdrawal_step':data.withdrawal}
    if step==cfg['steps'] or step%cfg['expensive_every']==0:
        compressed=evaluate(model,compression_probes(data.canaries,cfg['prefix_grid']),source['training']['batch_size'])
        result['prefix_rows']=compressed; result['compression']=compress_summary(compressed)
        result['exposure']=[]
        for record in data.canaries:
            if record.cohort!='short': continue
            rank,scores=exposure(model,record,cfg['exposure_alphabet'],32)
            result['exposure'].append(dict(id=record.id,dose=record.dose,**rank))
            folder=run_dir/'candidate_scores'; folder.mkdir(exist_ok=True)
            import numpy as np
            np.savez_compressed(folder/f'{step:08d}_{record.id}.npz',nll=scores)
    if model_hash(model)!=before: raise RuntimeError('Behavioral audit mutated weights.')
    atomic_json(target,result)


def train(cfg,root,arm,seed,device,resume=False):
    import fcntl
    root=Path(root); run_dir=root/arm/f'seed_{seed}'
    run_dir.mkdir(parents=True,exist_ok=True)
    with (run_dir/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train(cfg,run_dir,arm,seed,device,resume)


def _train(cfg,run_dir,arm,seed,device,resume):
    if arm not in ('adamw','muon'): raise ValueError(f'Unsupported arm: {arm}')
    if cfg.get('online_weightwatcher',True): raise ValueError('This trainer requires online_weightwatcher=false.')
    runtime=environment(device); source=baseline(cfg)
    _,_,make_handles,set_lrs,zero_grad,optimizer_step=imports()
    model=make_model(source,cfg,seed,device)
    data=Dataset(cfg,seed,source['training']['batch_size']*source['training']['grad_accum_steps'])
    profile=copy.deepcopy(source['optimizer_profiles'][arm])
    overrides=cfg.get('optimizer_overrides',{}).get(arm,{})
    profile.update(copy.deepcopy(overrides))
    handles=make_handles(model,profile)
    manifest={'protocol':cfg,'source_model':asdict(model.cfg),'profile':profile,'arm':arm,'optimizer':arm,
              'seed':seed,'runtime':runtime,'initial_sha256':model_hash(model),
              'data_sha256':data.fingerprint,'code_sha256':code_hash()}
    manifest['fingerprint']=digest(manifest)
    peer=run_dir.parent.parent/('muon' if arm=='adamw' else 'adamw')/run_dir.name/'manifest.json'
    if peer.exists():
        other=json.loads(peer.read_text())
        for key in ('protocol','source_model','seed','runtime','initial_sha256','data_sha256','code_sha256'):
            if manifest[key]!=other[key]: raise ValueError('Paired arm mismatch: '+key)
    file=run_dir/'manifest.json'; latest=run_dir/'checkpoint_latest.pt'
    if file.exists():
        if json.loads(file.read_text())!=manifest: raise ValueError('Run fingerprint changed; use a new study root.')
        if (run_dir/'complete.json').exists(): return {'state':'complete','run':str(run_dir)}
        if not resume: raise ValueError('Run already exists. Request --resume explicitly; nothing was overwritten.')
        if not latest.exists(): raise ValueError('No safe checkpoint to resume; preserve this failure and start a new root.')
    else:
        atomic_json(file,manifest); data.save(run_dir)
    counts=Counter(); completed=0; attempted=0
    if resume and latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved['fingerprint']!=manifest['fingerprint']: raise ValueError('Checkpoint fingerprint mismatch.')
        restore(saved,model,handles); completed=saved['step']; counts=Counter(saved['counts']); attempted=saved['attempted']
        if model_hash(model)!=saved['model_sha256']: raise ValueError('Checkpoint tensor hash mismatch.')
    schedule_steps=math.ceil(source['dataset']['train_tokens']*profile['lr_schedule_epochs']/
                             (data.batch_size*source['model']['block_size']))
    warmup=math.ceil(schedule_steps*profile['warmup_fraction'])
    started=time.monotonic()
    try:
        if not latest.exists():
            state=snapshot(model,handles,counts,0,attempted)
            state.update(fingerprint=manifest['fingerprint'],model_sha256=model_hash(model))
            atomic_save(latest,state)
            atomic_save(run_dir/'model_00000000.pt',{'model':state['model'],'manifest':manifest,'step':0})
        if completed%cfg['behavior_every']==0: audit(model,data,cfg,source,run_dir,completed,counts)
        while completed<cfg['steps']:
            step=completed; attempted+=1; zero_grad(handles); model.train()
            set_lrs(handles,update_index=step,total_steps=schedule_steps,warmup_steps=warmup)
            records=data.batch(step); last_loss=0.
            try:
                for chunk in batches(records,source['training']['batch_size']):
                    loss=suffix_losses(model,chunk)[0].mean(1).sum()/data.batch_size
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite forward loss.')
                    loss.backward(); last_loss+=float(loss.detach())
                norm,overflow=safe_clip(model,source['training']['grad_clip'])
                if overflow:
                    with (run_dir/'numeric_events.jsonl').open('a') as f:
                        f.write(json.dumps({'step_index':step,'attempted_update':attempted,'kind':'finite_norm_overflow','cpu_float64_norm':norm})+'\n')
                optimizer_step(handles)
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError('Nonfinite parameter after optimizer update.')
            except (FloatingPointError,RuntimeError) as exc:
                state=snapshot(model,handles,counts,step,attempted)
                state.update(diagnostic_only=True,error=str(exc),records=[asdict(r) for r in records],
                             gradients={n:cpu_copy(p.grad) for n,p in model.named_parameters() if p.grad is not None})
                atomic_save(run_dir/'failure_state.pt',state); raise
            counts.update(r.id for r in records); completed=step+1
            if completed%25==0:
                print(f'{arm} seed={seed} step={completed}/{cfg["steps"]} loss={last_loss:.6f} '
                      f'grad_norm={norm:.3g} norm_fallback={overflow}',flush=True)
            checkpoint=(completed%cfg['checkpoint_every']==0 or completed in (data.withdrawal,cfg['steps']))
            if checkpoint:
                for r in data.canaries:
                    if counts[r.id]!=data.planned_counts(completed)[r.id]: raise RuntimeError('Realized dose disagrees with schedule.')
                state=snapshot(model,handles,counts,completed,attempted)
                state.update(fingerprint=manifest['fingerprint'],model_sha256=model_hash(model))
                atomic_save(latest,state)
                atomic_save(run_dir/f'model_{completed:08d}.pt',{'model':state['model'],'manifest':manifest,'step':completed})
            if completed%cfg['behavior_every']==0 or completed in (data.withdrawal,cfg['steps']):
                audit(model,data,cfg,source,run_dir,completed,counts)
        result={'state':'complete','step':completed,'attempted_updates':attempted,
                'elapsed_seconds':time.monotonic()-started,'fingerprint':manifest['fingerprint'],
                'online_weightwatcher':False}
        atomic_json(run_dir/'complete.json',result); return result
    except (Exception,KeyboardInterrupt) as exc:
        result={'state':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                'last_completed_step':completed,'error':str(exc),'attempted_updates':attempted}
        atomic_json(run_dir/'failure.json',result)
        if isinstance(exc,KeyboardInterrupt): raise
        return result

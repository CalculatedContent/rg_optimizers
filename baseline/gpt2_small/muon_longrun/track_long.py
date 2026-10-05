"""CPU spectral measurements; preserve explicit zero/unfit matrices at initialization."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import time
from common import atomic_json, sha, EARLY_WW, PERMANENT
import tracking


def measure(path):
    import numpy as np
    import torch
    import weightwatcher as ww
    torch.set_num_threads(1)
    began=time.monotonic(); payload=torch.load(path,map_location='cpu',weights_only=False)
    seed=1001340+payload['step']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    names=list(payload['matrices'])
    nonzero={n:v for n,v in payload['matrices'].items() if bool(torch.count_nonzero(v))}
    identity={k:payload['validation'].get(k) for k in
              ('evaluation_tokens','full_benchmark_evaluation','val_nll','val_perplexity',
               'val_token_error','val_accuracy','val_error_count','epoch','lr_factor',
               'scheduler_phase','muon_lr','adam_embedding_lr','adam_head_lr','adam_scalar_lr')}
    identity.update(step=payload['step'],tokens_seen=payload['tokens_seen'],run_id=payload['run_id'],
                    snapshot_sha256=sha(path),diagnostic_seed=seed,
                    weightwatcher_version=importlib.metadata.version('weightwatcher'))
    frame=ww.WeightWatcher(model=tracking.holder_from(nonzero)).analyze(**tracking.WW_OPTIONS)
    if not {'alpha','raw_alpha'}.issubset(frame.columns):
        raise RuntimeError('WeightWatcher raw/clipped fields missing')
    rows=tracking.normalize_rows(frame,names,identity)
    for row in rows:
        name=row['matrix_name']
        if name not in nonzero:
            row.update(status='zero_matrix',matrix_rank=0,
                       randomized_status='degenerate_zero_matrix')
        else:
            row['randomized_status']='available' if row.get('max_rand_eval') is not None else 'unavailable'
        # Keep all WW scalar fields, plus explicit missing values where a fit is undefined.
        for key in ('matrix_rank','alpha_weighted','log_alpha_norm','max_rand_eval',
                    'rand_distance','rand_mp_softrank','rand_num_spikes','num_fingers'):
            row.setdefault(key,None)
        row['weighted_metric_alpha_source']='clipped alpha (WeightWatcher native definition)'
    result=tracking.summary(rows,identity)
    result.update(weightwatcher_seconds=time.monotonic()-began,
                  zero_matrix_count=len(names)-len(nonzero))
    return {'layers':rows,'summary':result,'options':tracking.WW_OPTIONS}


def configure(root):
    version=importlib.metadata.version('weightwatcher')
    if version!='0.7.7': raise RuntimeError('Expected the existing weightwatcher==0.7.7')
    atomic_json(root/'TRACKING_CONFIG.json',dict(weightwatcher_version=version,
        options=tracking.WW_OPTIONS,matrix_count=72,matrix_roles=list(tracking.ROLES.values()),
        early_steps=sorted(EARLY_WW),later_interval=1000,always_steps=sorted(PERMANENT),
        overhead_fallback_interval=2000,execution='separate CPU process, one BLAS thread',
        raw_alpha_source='raw_alpha',clipped_alpha_source='alpha',
        zero_matrix_policy='Keep row, rank=0, undefined alpha/null fits explicitly unavailable',
        uncertainty='Across-matrix standard deviation is not uncertainty across seeds',
        token_error='Validation teacher-forced top-1 error, same tokens and weights as NLL',
        randomization='WeightWatcher randomize=True, all returned null/ERG fields retained'))


def watch(root,deadline):
    configure(root); folder=root/'tracking'
    while time.time()<deadline:
        for path in sorted((folder/'snapshots').glob('*.pt')):
            output=folder/'measurements'/(path.stem+'.json')
            failed=folder/'failures'/(path.stem+'.json')
            if output.exists() or failed.exists(): continue
            tracking.tracking_status(root,'measuring',current_step=int(path.stem))
            try:
                result=measure(path); atomic_json(output,result); tracking.collect(root)
                # CSV readers see complete atomic replacements, then durable contents.
                import os
                for name in ('summary.csv','layers.csv'):
                    with (folder/name).open('rb') as f: os.fsync(f.fileno())
                print('[weightwatcher] '+json.dumps(result['summary']),flush=True)
            except Exception as exc:
                atomic_json(failed,{'step':int(path.stem),'error':repr(exc)})
                print('[weightwatcher] FAILED '+repr(exc),flush=True)
        state=tracking.tracking_status(root,'waiting')
        if (folder/'TRAINING_DONE').exists() and not state['pending']:
            tracking.tracking_status(root,'complete' if not state['failed'] else 'completed_with_errors')
            return 0 if not state['failed'] else 1
        time.sleep(1)
    tracking.tracking_status(root,'deadline_reached'); return 1


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('deadline',type=float)
    a=p.parse_args(); raise SystemExit(watch(a.root,a.deadline))

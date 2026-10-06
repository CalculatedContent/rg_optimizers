"""CPU-only WeightWatcher measurements paired with immutable validation snapshots.

Only this separate process imports WeightWatcher. No training RNG, model, optimizer,
TPU graph, data iterator, or learning-rate state is touched by spectral analysis.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

INTERVAL = 100
PROBE_TOKENS = 131072
WW_OPTIONS = dict(ERG=True, randomize=True, plot=False, fix_fingers='clip_xmax',
                  max_fingers=10, min_evals=20)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def projection_matrices(banks):
    """50 active Q/K/V/O/MLP projections; never analyze padded or frozen bank slots."""
    from .gpt import (ATTN_BANK_ORDER, NO_MLP_LAYERS, PARALLEL_MLP_LAYER,
                      PARALLEL_MLP_SLOT, WIDE_QK_LAYERS, HALF_V_LAYERS)
    matrices={}; heads=6; dim=768; storage=128
    qk=banks['qk_bank'][:42].view(7,2*heads,storage,dim)
    for slot,layer in enumerate(ATTN_BANK_ORDER):
        qwidth=128 if layer in WIDE_QK_LAYERS else 64
        vwidth=64 if layer in HALF_V_LAYERS else 128
        matrices[f'L{layer:02d}_W_Q']=qk[slot,:heads,:qwidth].reshape(heads*qwidth,dim)
        matrices[f'L{layer:02d}_W_K']=qk[slot,heads:,:qwidth].reshape(heads*qwidth,dim)
        matrices[f'L{layer:02d}_W_V']=banks['vo_bank'][2*slot].view(heads,storage,dim)[:,:vwidth].reshape(heads*vwidth,dim)
        matrices[f'L{layer:02d}_W_O']=banks['vo_bank'][2*slot+1].view(dim,heads,storage)[:,:,:vwidth].reshape(dim,heads*vwidth)
    for layer in range(11):
        if layer in NO_MLP_LAYERS: continue
        matrices[f'L{layer:02d}_W_MLP_IN']=banks['mlp_bank'][layer,0]
        matrices[f'L{layer:02d}_W_MLP_OUT']=banks['mlp_bank'][layer,1].T
    matrices[f'L{PARALLEL_MLP_LAYER:02d}_PARALLEL_W_MLP_IN']=banks['mlp_bank'][PARALLEL_MLP_SLOT,0]
    matrices[f'L{PARALLEL_MLP_LAYER:02d}_PARALLEL_W_MLP_OUT']=banks['mlp_bank'][PARALLEL_MLP_SLOT,1].T
    return matrices


def queue_snapshot(root,model,validation):
    """One immutable CPU snapshot paired with validation of this exact weight state."""
    import torch
    root=Path(root); step=validation['step']
    banks={name:getattr(model,name).detach().cpu() for name in ('qk_bank','vo_bank','mlp_bank')}
    matrices={name:weight.contiguous().clone() for name,weight in projection_matrices(banks).items()}
    folder=root/'tracking/snapshots';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{step:07d}.pt'
    if path.exists(): raise RuntimeError('Refusing to overwrite spectral snapshot: '+str(path))
    payload={'step':step,'tokens_seen':validation['tokens_seen'],'run_id':root.name,
             'validation':validation,'matrices':matrices}
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        torch.save(payload,handle);handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)
    print('[weightwatcher] queued '+str(path),flush=True)


def holder_from(matrices):
    import torch
    holder = torch.nn.Module()
    for name, value in matrices.items():
        # No random initialization or allocation of a second model.
        layer = torch.nn.Linear(value.shape[1], value.shape[0], bias=False, device='meta')
        layer.weight = torch.nn.Parameter(value.detach().float().cpu(), requires_grad=False)
        holder.add_module(name, layer)
    return holder


def scalar(value):
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if isinstance(value, (str, bool, int, float, type(None))) else str(value)


def positive(value):
    try:
        x = float(value)
        return x if math.isfinite(x) and x > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_rows(frame, names, identity):
    """Match by the explicit layer name; never substitute clipped alpha for raw."""
    found = {}
    for source in frame.to_dict('records'):
        matches = [name for name in names if name in
                   str(source.get('longname', '')) or name == str(source.get('name', ''))]
        if len(matches) != 1 or matches[0] in found:
            raise RuntimeError('WeightWatcher returned ambiguous or duplicate matrix names')
        name = matches[0]
        raw = {key:scalar(value) for key, value in source.items()}
        good = source.get('status') == 'success'
        alpha_raw = positive(source.get('raw_alpha')) if good else None
        alpha_clip = positive(source.get('alpha')) if good else None
        found[name] = {**raw, **identity, 'matrix_name':name, 'block':int(name[1:3]),
                       'matrix_type':name.split('_W_', 1)[1],
                       'alpha_raw':alpha_raw, 'alpha_clip_xmax':alpha_clip,
                       'raw_fit_status':'success' if alpha_raw is not None else 'unavailable',
                       'clipped_fit_status':'success' if alpha_clip is not None else 'unavailable'}
    # Some versions skip an unfit/zero matrix. Keep explicit missing rows.
    for name in names:
        if name not in found:
            found[name] = {**identity, 'matrix_name':name, 'block':int(name[1:3]),
                           'matrix_type':name.split('_W_', 1)[1], 'status':'not_returned',
                           'alpha_raw':None, 'alpha_clip_xmax':None,
                           'raw_fit_status':'unavailable', 'clipped_fit_status':'unavailable'}
    return [found[name] for name in names]


def summary(rows, identity):
    import statistics
    result = {**identity, 'matrix_count':len(rows)}
    for field in ('alpha_raw', 'alpha_clip_xmax'):
        values = [row[field] for row in rows if row[field] is not None]
        result.update({field+'_valid_count':len(values),
                       field+'_mean':statistics.mean(values) if values else None,
                       field+'_min':min(values) if values else None,
                       field+'_std_across_matrices':statistics.stdev(values) if len(values) > 1 else None,
                       field+'_below_two':sum(x < 2 for x in values)})
    return result


def measure(path):
    import random
    import numpy as np
    import torch
    import weightwatcher as ww
    torch.set_num_threads(1)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if importlib.metadata.version('weightwatcher')!='0.7.7':
        raise RuntimeError('Expected weightwatcher==0.7.7')
    seed = 1_001_340+payload['step']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if digest is None:  # TPU host runs Python 3.10.
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024), b''):
                h.update(chunk)
        digest = h.hexdigest()
    validation = payload['validation']
    identity = {key:validation.get(key) for key in ('evaluation_tokens', 'full_benchmark_evaluation',
                'val_nll', 'val_perplexity', 'val_token_error', 'val_accuracy', 'val_error_count', 'weight_state', 'ws_short', 'ws_long')}
    identity.update(step=payload['step'], tokens_seen=payload['tokens_seen'], run_id=payload['run_id'],
                    snapshot_sha256=digest, diagnostic_seed=seed,
                    weightwatcher_version=importlib.metadata.version('weightwatcher'))
    frame = ww.WeightWatcher(model=holder_from(payload['matrices'])).analyze(**WW_OPTIONS)
    if not {'alpha', 'raw_alpha'}.issubset(frame.columns):
        raise RuntimeError('WeightWatcher must expose both raw_alpha and clipped alpha')
    rows = normalize_rows(frame, list(payload['matrices']), identity)
    return {'layers':rows, 'summary':summary(rows, identity), 'options':WW_OPTIONS}


def csv_write(path, rows):
    temp = path.with_suffix('.csv.tmp')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def collect(root):
    folder = root/'tracking'
    files = sorted((folder/'measurements').glob('*.json'))
    data = [json.loads(path.read_text()) for path in files]
    csv_write(folder/'layers.csv', [row for item in data for row in item['layers']])
    csv_write(folder/'summary.csv', [item['summary'] for item in data])
    return len(files)


def tracking_status(root, status, **extra):
    folder = root/'tracking'
    snapshots = list((folder/'snapshots').glob('*.pt'))
    done = {p.stem for p in (folder/'measurements').glob('*.json')}
    failed = {p.stem for p in (folder/'failures').glob('*.json')}
    result = dict(status=status, snapshots=len(snapshots), completed=len(done), failed=len(failed),
                  pending=sum(p.stem not in done|failed for p in snapshots), updated_unix=time.time(), **extra)
    write_json(root/'TRACKING_STATUS.json', result)
    return result


def watch(root, deadline):
    folder = root/'tracking'
    while time.time() < deadline:
        for path in sorted((folder/'snapshots').glob('*.pt')):
            output = folder/'measurements'/(path.stem+'.json')
            failure = folder/'failures'/(path.stem+'.json')
            if output.exists() or failure.exists():
                continue
            if time.time() >= deadline:
                break
            tracking_status(root, 'measuring', current_step=int(path.stem))
            try:
                result = measure(path)
                write_json(output, result)
                collect(root)
                print('[weightwatcher] '+json.dumps(result['summary']), flush=True)
            except Exception as exc:
                write_json(failure, {'step':int(path.stem), 'error':repr(exc)})
                print('[weightwatcher] FAILED update '+path.stem+': '+repr(exc), flush=True)
        state = tracking_status(root, 'waiting')
        if (folder/'TRAINING_DONE').exists() and not state['pending']:
            tracking_status(root, 'complete' if not state['failed'] else 'completed_with_errors')
            return 0 if not state['failed'] else 1
        time.sleep(1)
    tracking_status(root, 'deadline_reached')
    return 1


def check(root):
    root=Path(root)
    import torch
    import weightwatcher as ww
    torch.set_num_threads(1)
    version = importlib.metadata.version('weightwatcher')
    if version != '0.7.7':
        raise RuntimeError('Expected the existing weightwatcher==0.7.7; got '+version)
    holder = holder_from({'L00_W_Q':torch.randn(64, 64)})
    frame = ww.WeightWatcher(model=holder).analyze(**WW_OPTIONS)
    if not {'raw_alpha', 'alpha'}.issubset(frame.columns):
        raise RuntimeError('WeightWatcher raw/clipped fields unavailable')
    normalize_rows(frame, ['L00_W_Q'], {})
    write_json(root/'TRACKING_CONFIG.json', {'weightwatcher_version':version, 'options':WW_OPTIONS,
               'interval_updates':INTERVAL, 'extra_final_measurement':True, 'matrix_roles':['Q','K','V','O','MLP_IN','MLP_OUT','parallel MLP'],
               'matrix_count':50, 'probe_validation_tokens':PROBE_TOKENS,
               'excluded':['embedding tables','ngram table','small gates','frozen and padded bank slots'], 'execution':'separate CPU process',
               'token_error_units':'fraction of the same teacher-forced benchmark validation tokens',
               'snapshot_storage':'local results disk; no automatic cloud backup',
               'std_definition':'sample standard deviation across matrices; not uncertainty across seeds'})
    print('WeightWatcher CPU tracking ready: raw/clipped alpha and paired validation token error.', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('check', 'watch'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--deadline', type=float, default=float('inf'))
    args = p.parse_args()
    if args.action == 'check':
        check(args.root)
        return 0
    return watch(args.root, args.deadline)


if __name__ == '__main__':
    raise SystemExit(main())

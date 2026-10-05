"""25k continuous updates; validated numerical implementation imported unchanged."""
import argparse
from dataclasses import asdict
import importlib.util
import json
import math
import random
import signal
import time
import os
from pathlib import Path
import numpy as np
import torch
from common import (SHORT, STEPS, BATCH_TOKENS, VAL_TOKENS, MICROBATCH, CACHE,
                    PERMANENT, Schedule, atomic_json, ww_due, val_due, sha, HERE, adapt_tracking)
from long_data import FineWeb, Stream, corpus_metadata, prepare
import checkpoint
import model as architecture
from optim import make_optimizers, momentum
from runtime import Runtime
from tracking import queue_snapshot, ROLES

# Reuse the EXACT full benchmark evaluator (including token-error counting).
spec = importlib.util.spec_from_file_location('validated_speedrun', SHORT/'run.py')
reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)


def record(root, row):
    with (root/'metrics.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(row, allow_nan=False), flush=True)


def apply_update(muon, adam, rt, step, schedule):
    factor = schedule.factor(step)
    muon.step(rt.scalar(.04*factor), rt.scalar(momentum(step)))
    for group in adam.param_groups:
        group['lr'] = rt.scalar(group['peak_lr']*factor)
    adam.step()


def gradient_norm(model):
    # One FP32 scalar reduction, no per-tensor CPU scans, stacks, or clipping.
    return sum(p.grad.detach().float().square().sum()
               for p in model.parameters() if p.grad is not None).sqrt()


def update(model,muon,adam,stream,rt,step,schedule,logged):
    model.zero_grad(set_to_none=False)
    loss_sum=torch.zeros((),device=rt.device,dtype=torch.float32)
    for _ in range(8):
        x,y=stream.next_batch(); loss=model(rt.put(x),rt.put(y))
        (loss/8).backward(); loss_sum+=loss.detach()/8; rt.step()
    for p in model.parameters():
        if p.grad is not None: rt.replicate(p.grad)
    norm=gradient_norm(model) if logged else None
    apply_update(muon,adam,rt,step,schedule); rt.step(wait=True)
    loss_value=float(loss_sum.cpu()); norm_value=float(norm.cpu()) if norm is not None else None
    if not math.isfinite(loss_value) or (norm_value is not None and not math.isfinite(norm_value)):
        raise RuntimeError(f'Nonfinite training measurement at update {step+1}')
    return loss_value,norm_value


def verify_startup_replay(root,model,muon,adam,stream,rt,schedule,identity,base_step=0):
    """Replay the first two updates of this process, then continue this SAME process."""
    began=time.monotonic()
    expected=checkpoint.state(model,muon,adam,stream,rt,base_step+2,schedule,identity)
    initial=torch.load(root/f'checkpoints/step_{base_step:07d}.pt',map_location='cpu',weights_only=False)
    checkpoint.restore(initial,model,muon,adam,stream,rt,schedule,identity); del initial
    for step in range(base_step,base_step+2):
        update(model,muon,adam,stream,rt,step,schedule,step<5 or (step+1)%10==0)
    actual=checkpoint.state(model,muon,adam,stream,rt,base_step+2,schedule,identity)
    checkpoint.assert_same(expected,actual)
    atomic_json(root/'RESUME_PARITY.json',{'status':'passed','exact_equal':True,
        'device':str(rt.device),'updates_replayed':2,'from_step':base_step,'seconds':time.monotonic()-began,
        'fields':['model','muon','adam','RNG','data_cursor','scheduler'],
        'scope':'This initial checkpoint and two updates; later-point replay still requires verification',
        'same_process_continues_at_step':base_step+2})
    print(f'Full-state startup replay passed exactly; continuing this process at step {base_step+2}.',flush=True)


def spectral_payload(model, step, validation, manifest):
    # Transfer only the 72 hidden matrices; optimizer state is NOT copied here.
    state = model.state_dict()
    selected = {f'transformer.h.{i}.{suffix}.weight':
                state[f'transformer.h.{i}.{suffix}.weight'].detach().cpu()
                for i in range(model.config.n_layer) for suffix in ROLES}
    return dict(model=selected, step=step, tokens_seen=step*BATCH_TOKENS,
                validation=validation, manifest=manifest, config=asdict(model.config))


def evaluate(model, val, rt, root, step, deadline, started, schedule, corpus):
    # Suppress only the old evaluator's writer so enriched rows are fsynced once.
    original = reference.record
    reference.record = lambda *_: None
    try:
        row = reference.evaluate(model, val, rt, root, step, deadline, MICROBATCH, started)
    finally:
        reference.record = original
    row.update(epoch=step*BATCH_TOKENS/corpus['usable_tokens_per_epoch'],
               **schedule.values(max(0, step-1)), next_update=schedule.values(step))
    # target_met remains a descriptive comparison; it never controls this loop.
    record(root, row); atomic_json(root/'latest_validation.json', row)
    return row


def wait_initial_tracking(root, deadline):
    result = root/'tracking/measurements/0000000.json'
    cutoff = min(deadline, time.time()+600)
    while not result.exists():
        if (root/'tracking/failures/0000000.json').exists():
            raise RuntimeError('Step-0 WeightWatcher failed; inspect tracking/failures')
        if (root/'STOP').exists(): raise RuntimeError('Stop requested during initial measurement')
        if time.time() >= cutoff: raise RuntimeError('Step-0 WeightWatcher exceeded 10 minutes')
        time.sleep(1)
    data = json.loads(result.read_text())
    if (len(data['layers']) != 72 or data['summary']['alpha_raw_valid_count'] == 0
            or data['summary']['alpha_clip_xmax_valid_count'] == 0):
        raise RuntimeError('Initial 72-matrix/raw-alpha measurement unavailable')
    for row in data['layers']:
        if not {'alpha_raw','alpha_clip_xmax','randomized_status','matrix_rank'}.issubset(row):
            raise RuntimeError('Initial WeightWatcher fields missing')
    if not any(r['randomized_status']=='available' for r in data['layers']):
        raise RuntimeError('Randomized/null statistics unavailable')
    print('Step-0 tracking complete: 72 rows; zero matrices have explicit unavailable fits.', flush=True)


def train(root, deadline, resume=None):
    torch.set_num_threads(4)
    torch.manual_seed(1337); np.random.seed(1337); random.seed(1337)
    started = time.time(); schedule = Schedule()
    approved = json.loads((root/'REFERENCE.json').read_text())
    source = FineWeb(CACHE, deadline); corpus = corpus_metadata(source)
    identity = {'source_sha256':approved['unchanged_source_sha256'],
                'longrun_source_sha256':{name:sha(HERE/name) for name in
                    ('train_long.py','checkpoint.py','long_data.py','common.py')},
                'corpus':corpus, 'microbatch':MICROBATCH, 'batch_tokens':BATCH_TOKENS,
                'attention':'flash', 'seed':1337, 'precision':'BF16 activations/embedding/scalars; FP32 linear weights',
                'runtime_versions':json.loads((root/'PALLAS_DEPENDENCIES.json').read_text())['core_after']}
    manifest = dict(approved['manifest'])
    manifest.update(run_id=root.name, steps=STEPS, target_val_nll=None, stop_on_target=False,
                    warmdown_updates=7500, scheduler=asdict(schedule), identity=identity,
                    scalar_interval=10, full_validation_interval=500,
                    tracking='See TRACKING_CONFIG.json', checkpoint_interval=2500,
                    permanent_checkpoint_steps=sorted(PERMANENT),
                    reference_run=approved['reference_run'], fresh_initialization=resume is None,
                    resume_source=str(resume) if resume else None,
                    resume_validation='Requires exact two-update TPU startup replay; see RESUME_PARITY.json.')
    atomic_json(root/'manifest.json', manifest)
    rt = Runtime('tpu', root/'xla-cache'); architecture.ATTENTION = rt.attention('flash')
    model = architecture.GPT(architecture.GPTConfig()).bfloat16()
    for module in model.modules():
        if isinstance(module, architecture.CastedLinear): module.float()
    model = model.to(rt.device)
    for value in (*model.parameters(), *model.buffers()): rt.replicate(value)
    muon, adam = make_optimizers(model, rt, 'muon')
    stream = Stream(source); val = source.array('fineweb_val_000000.bin')
    step = 0
    if resume:
        step = checkpoint.restore(torch.load(resume, map_location='cpu', weights_only=False),
                                  model, muon, adam, stream, rt, schedule, identity)
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    rt.step(wait=True)
    initial = evaluate(model, val, rt, root, step, deadline-180, started, schedule, corpus)
    if not initial['full_benchmark_evaluation']: raise RuntimeError('Initial validation incomplete')
    payload = checkpoint.state(model, muon, adam, stream, rt, step, schedule, identity)
    payload.update(manifest=manifest, validation=initial)
    checkpoint.save(root, payload); queue_snapshot(root, payload); del payload
    if step == 0: wait_initial_tracking(root, deadline)
    atomic_json(root/'STARTUP.json', {'status':'numerical_startup', 'chips':8, 'optimizer':'muon',
                'initial_validation':initial, 'step_3000_next_update':schedule.values(3000)})
    validation = initial; timings = []; ww_foreground = 0.; sparse = False
    last_saved = step; training_started = time.time()
    while step < STEPS and time.time() < deadline-180 and not stopped[0] and not (root/'STOP').exists():
        began = time.monotonic()
        logged = step < 5 or (step+1)%10 == 0
        loss_value,norm_value=update(model,muon,adam,stream,rt,step,schedule,logged)
        step += 1
        seconds = time.monotonic()-began; timings.append(seconds)
        if step==initial['step']+2:
            verify_startup_replay(root,model,muon,adam,stream,rt,schedule,identity,initial['step'])
        if logged:
            row = dict(kind='train', step=step, tokens_seen=step*BATCH_TOKENS,
                epoch=step*BATCH_TOKENS/corpus['usable_tokens_per_epoch'], train_nll=loss_value,
                gradient_norm=norm_value, seconds=seconds, tokens_per_second=BATCH_TOKENS/seconds,
                elapsed_seconds=time.time()-started, recorded_unix=time.time(),
                end_to_end_tokens_per_second=(step-(initial['step']))*BATCH_TOKENS/(time.time()-training_started),
                **schedule.values(step-1), next_update=schedule.values(step))
            if step >= 100:
                row['training_seconds_remaining_estimate'] = float(np.median(timings[-100:]))*(STEPS-step)
            record(root, row); atomic_json(root/'status.json', {'status':'training', **row})
        if step == initial['step']+100:
            rate = float(np.median(timings[-50:]))
            baseline = approved.get('median_update_seconds', 1.28)
            atomic_json(root/'STARTUP.json', {'status':'passed', 'step':step, 'chips':8,
                'optimizer':'muon','finite_loss':loss_value, 'median_update_seconds':rate,
                'reference_median_update_seconds':baseline, 'throughput_ratio':baseline/rate,
                'schedule_at_3000':schedule.values(3000), 'same_process_continues':True})
            if rate > baseline*1.5:
                raise RuntimeError('Post-compile throughput >50% slower than reference; inspect STARTUP.json')
        should_ww = ww_due(step, sparse)
        if val_due(step, sparse):
            validation = evaluate(model, val, rt, root, step, deadline-120, started, schedule, corpus)
            if not validation['full_benchmark_evaluation']: break
            if step == 3000:
                atomic_json(root/'COMPARISON_3000.json', {'reference':approved['final_validation'],
                    'long_run':validation,'explanation':'Long-run LR is still at its peak; reference finished cooldown.'})
        saved_payload = None
        if step%2500 == 0 or step in PERMANENT:
            saved_payload = checkpoint.state(model,muon,adam,stream,rt,step,schedule,identity)
            saved_payload.update(manifest=manifest, validation=validation if validation['step']==step else None)
            checkpoint.save(root,saved_payload); last_saved=step
        if should_ww:
            began = time.monotonic()
            queue_snapshot(root, saved_payload or spectral_payload(model,step,validation,manifest))
            overhead=time.monotonic()-began; ww_foreground += overhead
            fraction = ww_foreground/max(1.,time.time()-training_started)
            tracking=json.loads((root/'TRACKING_STATUS.json').read_text()) if (root/'TRACKING_STATUS.json').exists() else {}
            # CPU analysis wall time is not TPU blocking time. Report both separately.
            if adapt_tracking(step,fraction,float(np.median(timings[-100:])),
                              approved.get('median_update_seconds',1.28),tracking): sparse=True
            record(root, dict(kind='tracking_overhead',step=step,foreground_seconds=overhead,
                cumulative_foreground_fraction=fraction, later_interval=2000 if sparse else 1000,
                slowdown_interpretation='Foreground time is measured; concurrent CPU contention is a proxy, not causal proof',
                cpu_tracker=tracking, elapsed_seconds=time.time()-started))
        del saved_payload
    if validation['step'] != step:
        validation = evaluate(model,val,rt,root,step,deadline-60,started,schedule,corpus)
    if last_saved != step:
        payload=checkpoint.state(model,muon,adam,stream,rt,step,schedule,identity)
        payload.update(manifest=manifest,validation=validation); checkpoint.save(root,payload)
        del payload
    if not (root/f'tracking/snapshots/{step:07d}.pt').exists():
        queue_snapshot(root,spectral_payload(model,step,validation,manifest))
    result={'status':'schedule_complete' if step==STEPS else 'stopped_before_schedule_complete',
            'step':step,'validation':validation,'elapsed_seconds':time.time()-started,
            'automatic_restart':False,'next_update':schedule.values(step)}
    atomic_json(root/'status.json',result); print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('prepare','train'))
    p.add_argument('--root',type=Path,required=True); p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--resume',type=Path,help='Explicit full-state recovery into a NEW directory; never automatic')
    a=p.parse_args()
    try:
        if a.action=='prepare': prepare(a.root,a.deadline)
        else: train(a.root,a.deadline,a.resume)
    except Exception as exc:
        atomic_json(a.root/'FAILURE.json',{'error':repr(exc),'attribution':'unconfirmed','unix_time':time.time()})
        raise

if __name__=='__main__': main()

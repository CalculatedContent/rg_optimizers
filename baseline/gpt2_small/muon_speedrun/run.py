"""Pinned Muon speedrun with full validation, atomic checkpoints and explicit limits."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch

import model as architecture
from data import FineWeb, TrainStream, write_json, prepare
from optim import make_optimizers, apply_update, schedule, momentum
from runtime import Runtime, attention_check
from tracking import queue_snapshot

TOTAL_STEPS = 3000
BATCH_TOKENS = 524288
VAL_TOKENS = 10485760
TARGET = 3.28
HERE = Path(__file__).resolve().parent


def record(root, row):
    with (root/'metrics.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def save_checkpoint(root, model, muon, adam, stream, step, manifest, rt, validation, best):
    rt.step(wait=True)
    payload = {'schema':1, 'step':step, 'tokens_seen':step*BATCH_TOKENS,
               'model':to_cpu(model.state_dict()), 'config':asdict(model.config),
               'adam':to_cpu(adam.state_dict()),
               'muon':to_cpu(muon.state_dict()) if muon is not None else None,
               'data_cursor':{'shard':stream.shard, 'position':stream.position},
               'rng':{'torch':torch.get_rng_state(), 'numpy':np.random.get_state(),
                      'python':random.getstate()},
               'manifest':manifest, 'validation':validation,
               'next_lr_factor':schedule(step), 'next_momentum':momentum(step),
               'resume_validation':'Full state saved; automatic resume disabled. TPU resume parity not yet validated.'}
    checkpoint = root/'checkpoint_latest.pt'
    temporary = checkpoint.with_suffix('.tmp')
    with temporary.open('wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(checkpoint)
    passed = target_met(validation)
    aliases = []
    if validation and validation.get('full_benchmark_evaluation') and validation['val_nll'] <= best:
        aliases.append('checkpoint_best.pt')
    if passed:
        aliases.append('checkpoint_target.pt')
    for name in aliases:
        temp = root/(name+'.tmp')
        temp.unlink(missing_ok=True)
        os.link(checkpoint, temp)
        temp.replace(root/name)
    write_json(root/'checkpoint_latest.json', {'file':checkpoint.name, 'step':step,
                                              'target_met':passed, 'validation':validation})
    if validation is not None:
        queue_snapshot(root, payload)
    print('Checkpoint saved at update '+str(step), flush=True)


def target_met(validation):
    return bool(validation and validation.get('full_benchmark_evaluation')
                and validation.get('evaluation_tokens') == VAL_TOKENS
                and validation.get('val_nll') is not None
                and math.isfinite(validation['val_nll']) and validation['val_nll'] <= TARGET)


@torch.no_grad()
def evaluate(model, tokens, rt, root, step, deadline, microbatch, started):
    model.eval()
    size = microbatch*1024
    total = torch.zeros((), device=rt.device, dtype=torch.float32)
    errors = torch.zeros((), device=rt.device, dtype=torch.int32)
    evaluated = 0
    for offset in range(0, VAL_TOKENS, size):
        if time.time() >= deadline:
            break
        buf = torch.from_numpy(np.array(tokens[offset:offset+size+1], dtype=np.int64))
        batch_loss, batch_errors = model(rt.put(buf[:-1].reshape(microbatch, 1024)),
                       rt.put(buf[1:].reshape(microbatch, 1024)), return_token_errors=True)
        total += batch_loss.detach().float()
        errors += batch_errors
        rt.step()
        evaluated += size
    rt.step(wait=True)
    loss = float(total.cpu())/(evaluated/size) if evaluated else None
    error_count = int(errors.cpu())
    if loss is not None and not math.isfinite(loss):
        raise RuntimeError('Nonfinite validation NLL')
    row = {'kind':'validation', 'step':step, 'tokens_seen':step*BATCH_TOKENS,
           'evaluation_tokens':evaluated, 'full_benchmark_evaluation':evaluated == VAL_TOKENS,
           'val_nll':loss, 'val_perplexity':math.exp(loss) if loss is not None else None,
           'val_error_count':error_count, 'val_token_error':error_count/evaluated if evaluated else None,
           'val_accuracy':1-error_count/evaluated if evaluated else None,
           'elapsed_seconds':time.time()-started, 'recorded_unix':time.time()}
    reference = json.loads((HERE/'reference_val.json').read_text())
    row['published_at_same_step'] = next((r for r in reference if r['step'] == step), None)
    row['target_met'] = target_met(row)
    record(root, row)
    write_json(root/'latest_validation.json', row)
    model.train()
    return row


def train(a):
    root = a.root
    seed = 1337
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    started = time.time()
    rt = Runtime(a.device, root/'xla-cache')
    architecture.ATTENTION = rt.attention(a.attention)
    model = architecture.GPT(architecture.GPTConfig()).bfloat16()
    for module in model.modules():
        if isinstance(module, architecture.CastedLinear):
            module.float()
    model = model.to(rt.device)
    for value in (*model.parameters(), *model.buffers()):
        rt.replicate(value)
    muon, adam = make_optimizers(model, rt, a.optimizer)
    source = FineWeb(a.cache, a.deadline-180)
    stream = TrainStream(source, a.microbatch, 1024)
    val = source.array('fineweb_val_000000.bin')
    manifest = {'recipe':'2024-11-10_UNetDoubleLr', 'optimizer':a.optimizer, 'seed':seed,
                'config':asdict(model.config), 'parameters':sum(p.numel() for p in model.parameters()),
                'steps':TOTAL_STEPS, 'batch_tokens':BATCH_TOKENS,
                'global_microbatch_sequences':a.microbatch, 'accumulation':512//a.microbatch,
                'target_val_nll':TARGET, 'validation_tokens':VAL_TOKENS,
                'muon_lr':0.04 if muon else None, 'adam_embedding_lr':0.6,
                'adam_head_lr':0.008, 'adam_scalar_lr':0.04,
                'adam_control_matrix_lr':0.0006 if not muon else None,
                'warmup_updates':0, 'warmdown_updates':900, 'weight_decay':0,
                'gradient_clipping':False, 'attention':a.attention,
                'data_repo':source.manifest['repo'], 'data_revision':source.manifest['revision'],
                'record_source_sha256':hashlib.sha256((HERE/'vendor/record_source.py').read_bytes()).hexdigest(),
                'torch_version':torch.__version__, 'automatic_restart':False,
                'tracking':{'interval_updates':125, 'extra_final_measurement':True,
                            'matrices':'Q,K,V,O,MLP_IN,MLP_OUT in all 12 blocks',
                            'execution':'separate CPU process on immutable snapshots',
                            'token_error':'teacher-forced top-1 error on the same benchmark validation tokens'},
                'differences':['TPU SPMD instead of CUDA DDP', 'batched matrix-partitioned Muon',
                               'microbatch accumulation and shard-boundary ordering',
                               'fixed seed 1337; original record did not pin a seed',
                               'CPU-precomputed BF16 rotary buffers; hardware rounding differs'],
                'optimality':'Published GPU recipe; TPU convergence/performance unvalidated'}
    write_json(root/'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)
    step, best, validation = 0, float('inf'), None
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_:stopped.__setitem__(0, True))
    write_json(root/'status.json', {'status':'training', 'step':0})
    rt.step(wait=True)
    save_checkpoint(root, model, muon, adam, stream, step, manifest, rt, None, best)
    timings = []
    while step < TOTAL_STEPS and time.time() < a.deadline-180 and not stopped[0] and not (root/'STOP').exists():
        began = time.monotonic()
        model.zero_grad(set_to_none=False)
        loss_sum = torch.zeros((), device=rt.device, dtype=torch.float32)
        for _ in range(512//a.microbatch):
            x, y = stream.next_batch()
            loss = model(rt.put(x), rt.put(y))
            (loss/(512//a.microbatch)).backward()
            loss_sum += loss.detach()/(512//a.microbatch)
            rt.step()
        for p in model.parameters():
            if p.grad is not None:
                rt.replicate(p.grad)
        apply_update(muon, adam, rt, step)
        rt.step(wait=True)
        step += 1
        loss_value = float(loss_sum.cpu())
        if not math.isfinite(loss_value):
            raise RuntimeError('Nonfinite training NLL at update '+str(step))
        seconds = time.monotonic()-began
        timings.append(seconds)
        row = {'kind':'train', 'step':step, 'tokens_seen':step*BATCH_TOKENS,
               'train_nll':loss_value, 'seconds':seconds, 'tokens_per_second':BATCH_TOKENS/seconds,
               'elapsed_seconds':time.time()-started, 'lr_factor':schedule(step-1)}
        if step <= 5 or step % 10 == 0:
            if step >= 20:
                row['training_seconds_remaining_estimate'] = float(np.median(timings[-50:]))*(TOTAL_STEPS-step)
            record(root, row)
            write_json(root/'status.json', {'status':'training', **row})
        if step in (1,5):
            save_checkpoint(root, model, muon, adam, stream, step, manifest, rt, None, best)
        if step % 125 == 0:
            validation = evaluate(model, val, rt, root, step, a.deadline-90, a.microbatch, started)
            save_checkpoint(root, model, muon, adam, stream, step, manifest, rt, validation, best)
            if validation['full_benchmark_evaluation']:
                best = min(best, validation['val_nll'])
            if target_met(validation):
                break
    if validation is None or validation['step'] != step:
        validation = evaluate(model, val, rt, root, step, a.deadline-60, a.microbatch, started)
        save_checkpoint(root, model, muon, adam, stream, step, manifest, rt, validation, best)
    outcome = 'target_reached' if target_met(validation) else (
        'schedule_complete_target_not_met' if step == TOTAL_STEPS else 'stopped_before_schedule_complete')
    result = {'status':outcome, 'step':step, 'validation':validation,
              'target_met':target_met(validation), 'elapsed_seconds':time.time()-started,
              'checkpoint':str(root/'checkpoint_latest.pt')}
    write_json(root/'status.json', result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'attention-check', 'train'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('/mnt/disks/rg-data/benchmark-fineweb10B-889765ea'))
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--device', choices=('tpu', 'cpu'), default='tpu')
    p.add_argument('--attention', choices=('flash', 'math'), default='flash')
    p.add_argument('--microbatch', type=int, choices=(32,64,128), default=64)
    p.add_argument('--optimizer', choices=('muon', 'adam'), default='muon')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.cache, a.deadline, a.root, a.microbatch)
    elif a.action == 'attention-check':
        attention_check(a.root, a.microbatch)
    else:
        try:
            train(a)
        except Exception as exc:
            write_json(a.root/'FAILURE.json', {'error':str(exc), 'attribution':'unconfirmed'})
            raise


if __name__ == '__main__':
    main()

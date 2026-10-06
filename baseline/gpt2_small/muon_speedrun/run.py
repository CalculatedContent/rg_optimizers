"""Stock GPT-2 Small Muon/AdamW runs with paired validation and spectral tracking."""
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

import stock_model as architecture
from data import FineWeb, TrainStream, write_json, prepare
from optim import make_optimizers, apply_update, momentum, optimizer_metadata, clip_gradients
from runtime import Runtime, attention_check
from tracking import queue_snapshot

from benchmark_config import (BENCHMARK, TOTAL_STEPS, BATCH_TOKENS, VAL_TOKENS,
                              MEASUREMENT_INTERVAL, WARMUP, GRAD_CLIP,
                              lr_factor, protocol)

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
               'next_lr_factor':lr_factor(step), 'next_momentum':getattr(muon, 'beta', momentum(step)) if muon else None,
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
        targets = rt.put(buf[1:].reshape(microbatch, 1024))
        with rt.autocast():
            logits, batch_loss = model(rt.put(buf[:-1].reshape(microbatch, 1024)), targets)
        batch_errors = (logits.argmax(dim=-1) != targets).sum(dtype=torch.int32)
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
    row['historical_reference_architecture'] = '2024-11-10_UNetDoubleLr'
    row['historical_reference_at_same_step'] = next((r for r in reference if r['step'] == step), None)
    row['reference_note'] = 'This curve uses a different architecture. The original GPT-2/FineWeb baseline reached approximately 3.28 after 19560 updates; no TPU convergence guarantee.'
    row['target_met'] = target_met(row)
    record(root, row)
    write_json(root/'latest_validation.json', row)
    model.train()
    return row



def preflight_identity(a):
    return dict(architecture=architecture.ARCHITECTURE,
                config=asdict(architecture.GPTConfig()),
                source_sha256=architecture.SOURCE_SHA256,
                optimizer=a.optimizer, microbatch=a.microbatch, device=a.device,
                attention=a.attention, seed=a.seed, benchmark=BENCHMARK)


def require_preflight(a):
    proof=json.loads((a.root/'MODEL_PREFLIGHT.json').read_text())
    if proof.get('status') != 'passed' or proof.get('identity') != preflight_identity(a):
        raise RuntimeError('Missing or mismatched full-model/optimizer TPU preflight')
    return proof


def model_preflight(a):
    """A disposable full-sized accumulated update on the actual device/kernel."""
    rt=Runtime(a.device,a.root/'xla-cache')
    architecture.configure_attention(rt.attention(a.attention))
    model=architecture.make_model(seed=a.seed,device=rt.device)
    for p in (*model.parameters(),*model.buffers()): rt.replicate(p)
    muon,adam=make_optimizers(model,rt,a.optimizer)
    generator=torch.Generator().manual_seed(a.seed)
    model.zero_grad(set_to_none=False)
    accumulated=torch.zeros((),device=rt.device)
    count=512//a.microbatch
    for _ in range(count):
        tokens=torch.randint(model.config.vocab_size,(a.microbatch,1025),generator=generator)
        with rt.autocast():
            _,loss=model(rt.put(tokens[:,:-1].contiguous()),rt.put(tokens[:,1:].contiguous()),return_logits=False)
        (loss/count).backward(); accumulated+=loss.detach()/count; rt.step()
    for p in model.parameters():
        if p.grad is not None: rt.replicate(p.grad)
    norm=clip_gradients(model)
    apply_update(muon,adam,rt,0)
    finite=torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
    rt.step(wait=True)
    loss_value, norm_value=float(accumulated.cpu()),float(norm.cpu())
    if not bool(finite.cpu()) or not math.isfinite(loss_value) or not math.isfinite(norm_value):
        raise RuntimeError('Full-model optimizer preflight produced nonfinite values')
    if model.lm_head.weight is not model.transformer.wte.weight:
        raise RuntimeError('Upstream embedding/output alias was lost')
    proof=dict(status='passed',identity=preflight_identity(a),parameters=sum(p.numel() for p in model.parameters()),
               global_batch_tokens=BATCH_TOKENS,accumulation=count,loss=loss_value,gradient_norm=norm_value,
               torch_version=torch.__version__,completed_unix=time.time(),
               note='Disposable synthetic update; training starts from fresh upstream initialization')
    write_json(a.root/'MODEL_PREFLIGHT.json',proof)
    print(json.dumps(proof),flush=True)


def train(a):
    root = a.root
    if a.device == 'tpu':
        require_preflight(a)
    seed = a.seed
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    started = time.time()
    rt = Runtime(a.device, root/'xla-cache')
    architecture.configure_attention(rt.attention(a.attention))
    model = architecture.make_model(seed=seed, device=rt.device)
    for value in (*model.parameters(), *model.buffers()):
        rt.replicate(value)
    muon, adam = make_optimizers(model, rt, a.optimizer)
    source = FineWeb(a.cache, a.deadline-180)
    stream = TrainStream(source, a.microbatch, 1024)
    val = source.array('fineweb_val_000000.bin')
    manifest = {'recipe':BENCHMARK+'-'+a.optimizer,
                'benchmark':BENCHMARK, 'protocol':protocol(),
                'architecture':architecture.ARCHITECTURE,
                'matrix_inventory':architecture.matrix_inventory(model),
                'optimizer':a.optimizer, 'seed':seed,
                'full_budget':a.full_budget,
                'config':asdict(model.config), 'parameters':sum(p.numel() for p in model.parameters()),
                'steps':TOTAL_STEPS, 'batch_tokens':BATCH_TOKENS,
                'global_microbatch_sequences':a.microbatch, 'accumulation':512//a.microbatch,
                'target_val_nll':TARGET, 'validation_tokens':VAL_TOKENS,
                'muon_lr':getattr(muon, 'peak_lr', 0.04) if muon else None, 'adam_embedding_lr':0.0006,
                'adam_head_lr':0.0006, 'adam_scalar_lr':0.0006,
                'head_tied_to_embedding':True, 'dropout':0.0,
                'adam_control_matrix_lr':0.0006 if not muon else None,
                'warmup_updates':WARMUP, 'lr_schedule':'cosine_to_zero',
                'weight_decay':0.0 if a.optimizer == 'adam' else 0.1,
                'weight_decay_scope':('auxiliary_matrices_only' if a.optimizer == 'muon' else 'all_matrices')
                                     if a.optimizer != 'adam' else 'none',
                'muon_weight_decay':getattr(muon, 'weight_decay', 0.0) if muon else None,
                'muonclip':dict(threshold=muon.threshold, balance=muon.balance, rms_scale=muon.rms_scale, momentum=muon.beta) if a.optimizer == 'muon_clip' else None,
                'adam_implementation':optimizer_metadata(adam),
                'historical_reference':'muon-speedrun-muon-20261005-030026',
                'gradient_clipping':True, 'gradient_clip_norm':GRAD_CLIP, 'attention':a.attention,
                'data_repo':source.manifest['repo'], 'data_revision':source.manifest['revision'],
                'model_source_sha256':hashlib.sha256(architecture.SOURCE.read_bytes()).hexdigest(),
                'legacy_record_source_sha256':hashlib.sha256((HERE/'vendor/record_source.py').read_bytes()).hexdigest(),
                'torch_version':torch.__version__, 'automatic_restart':False,
                'tracking':{'interval_updates':MEASUREMENT_INTERVAL, 'extra_final_measurement':True,
                            'matrices':'Q,K,V,O,MLP_IN,MLP_OUT in all 12 blocks',
                            'execution':'separate CPU process on immutable snapshots',
                            'token_error':'teacher-forced top-1 error on the same benchmark validation tokens'},
                'implementation':['Byte-identical pinned upstream GPT class and packed QKV; observation-only optimizer hooks',
                                  'TPU SPMD with BF16 activations and FP32 parameters/optimizer states'],
                'optimality':'Pinned original GPT-2/FineWeb core hyperparameters; Muon is an experimental optimizer substitution. See BENCHMARK.md for TPU port differences.',
                'comparison_note':('Only compare new stock-model seed pairs as same-architecture controls. '
                                   'Historical six-head runs used a different model and auxiliary learning rates.')}
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
            with rt.autocast():
                _, loss = model(rt.put(x), rt.put(y), return_logits=False)
            (loss/(512//a.microbatch)).backward()
            loss_sum += loss.detach()/(512//a.microbatch)
            rt.step()
        for p in model.parameters():
            if p.grad is not None:
                rt.replicate(p.grad)
        grad_norm = clip_gradients(model)
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
               'elapsed_seconds':time.time()-started, 'lr_factor':lr_factor(step-1),
               'gradient_norm_before_clip':float(grad_norm.cpu())}
        if a.optimizer == 'muon_clip':
            row['muonclip'] = {k:float(v.cpu()) for k,v in muon.last_diagnostics.items()}
        if step <= 5 or step % 10 == 0:
            if step >= 20:
                row['training_seconds_remaining_estimate'] = float(np.median(timings[-50:]))*(TOTAL_STEPS-step)
            record(root, row)
            write_json(root/'status.json', {'status':'training', **row})
        if step in (1,5):
            save_checkpoint(root, model, muon, adam, stream, step, manifest, rt, None, best)
        if step % MEASUREMENT_INTERVAL == 0:
            validation = evaluate(model, val, rt, root, step, a.deadline-90, a.microbatch, started)
            save_checkpoint(root, model, muon, adam, stream, step, manifest, rt, validation, best)
            if validation['full_benchmark_evaluation']:
                best = min(best, validation['val_nll'])
            if target_met(validation) and not a.full_budget:
                break
    if validation is None or validation['step'] != step:
        validation = evaluate(model, val, rt, root, step, a.deadline-60, a.microbatch, started)
        save_checkpoint(root, model, muon, adam, stream, step, manifest, rt, validation, best)
    outcome = training_outcome(step, validation, a.full_budget)
    result = {'status':outcome, 'step':step, 'validation':validation,
              'target_met':target_met(validation), 'full_training_recipe_completed':step == TOTAL_STEPS,
              'elapsed_seconds':time.time()-started,
              'checkpoint':str(root/'checkpoint_latest.pt')}
    write_json(root/'status.json', result)
    print(json.dumps(result), flush=True)



def training_outcome(step, validation, full_budget):
    # A deadline/STOP must not masquerade as completion just because NLL passed.
    if step == TOTAL_STEPS:
        if not (validation and validation.get('full_benchmark_evaluation')
                and validation.get('evaluation_tokens') == VAL_TOKENS):
            return 'schedule_complete_evaluation_incomplete'
        return 'target_reached' if target_met(validation) else 'schedule_complete_target_not_met'
    if not full_budget and target_met(validation):
        return 'target_reached'
    return 'stopped_before_schedule_complete'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'attention-check', 'model-preflight', 'train'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('/mnt/disks/rg-data/benchmark-fineweb10B-889765ea'))
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--device', choices=('tpu', 'cpu'), default='tpu')
    p.add_argument('--attention', choices=('flash', 'math'), default='flash')
    p.add_argument('--microbatch', type=int, choices=(32,64,128), default=64)
    p.add_argument('--optimizer', choices=('muon', 'muon_clip', 'adam', 'adamw'), default='muon_clip')
    p.add_argument('--seed', type=int, default=42)
    budget = p.add_mutually_exclusive_group()
    budget.add_argument('--full-budget', action='store_true', default=True, help='Default: complete all 19560 updates')
    budget.add_argument('--stop-at-target', action='store_false', dest='full_budget', help='Explicit non-default early stop at NLL <= 3.28')
    p.add_argument('--legacy-attention-check', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.cache, a.deadline, a.root, a.microbatch, updates=TOTAL_STEPS)
    elif a.action == 'model-preflight':
        model_preflight(a)
    elif a.action == 'attention-check':
        cfg = architecture.GPTConfig()
        attention_check(a.root, a.microbatch, n_head=6 if a.legacy_attention_check else cfg.n_head,
                        head_dim=128 if a.legacy_attention_check else cfg.n_embd // cfg.n_head)
    else:
        if a.legacy_attention_check:
            p.error('--legacy-attention-check is only valid for attention-check')
        try:
            train(a)
        except Exception as exc:
            write_json(a.root/'FAILURE.json', {'error':str(exc), 'attribution':'unconfirmed'})
            raise


if __name__ == '__main__':
    main()

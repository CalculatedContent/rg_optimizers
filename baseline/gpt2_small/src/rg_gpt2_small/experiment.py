"""Isolated, token-budgeted GPT-2 validation; never downloads data or allocates TPUs."""
from __future__ import annotations
import argparse
import copy
import fcntl
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml
from rg_nanogpt_one_head.model import GPT, GPTConfig, transformer_matrix_items
from rg_nanogpt_one_head import runtime as rt, tpu_spmd as spmd
from rg_nanogpt_one_head.data import load_memmaps
from rg_nanogpt_one_head.muonclip import install_muonclip_extension
from rg_nanogpt_one_head import optimizers
from rg_nanogpt_one_head.spectral import WeightMatrixHolder, _attach_matrix_metadata
from . import port_debug
from . import execution_checks


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, default=str, allow_nan=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def append_record(path, row):
    """One immutable JSON per step: no truncation, silent overwrite, or duplicate."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Already recorded: {path}')
    atomic_json(path, row)


def due(step, spec):
    return (step in spec.get('steps', []) or
            bool(spec.get('interval', 0) and step % spec['interval'] == 0) or
            bool(spec.get('logarithmic', False) and step > 0 and
                 step in {int(k * 10**p) for p in range(12) for k in (1, 2, 5)}))


def batch(array, generator, size, context, device, trace=None):
    if len(array) <= context:
        raise ValueError('Split is shorter than one context')
    starts = torch.randint(len(array) - context, (size,), generator=generator).tolist()
    if trace is not None:
        trace.append(starts)
    x = torch.from_numpy(np.stack([np.array(array[i:i+context], dtype=np.int64) for i in starts]))
    y = torch.from_numpy(np.stack([np.array(array[i+1:i+context+1], dtype=np.int64) for i in starts]))
    return spmd.batch_to_device(x, device), spmd.batch_to_device(y, device)


@torch.no_grad()
def evaluate(model, arrays, cfg, device):
    model.eval(); result = {}
    for j, split in enumerate(('train', 'val', 'test')):
        gen = torch.Generator().manual_seed(cfg['seed'] + 20000 + j)
        losses, correct = [], []
        for _ in range(cfg['eval_batches']):
            x, y = batch(arrays[split], gen, cfg['training']['batch_size'], cfg['model']['block_size'], device)
            logits, loss = model(x, y)
            losses.append(loss.detach()); correct.append((logits.argmax(-1) == y).float().mean())
            rt.mark_step(device)
        nll, accuracy = torch.stack(losses).mean().item(), torch.stack(correct).mean().item()
        if not math.isfinite(nll):
            raise RuntimeError(f'Nonfinite {split} NLL')
        result.update({f'{split}_nll': nll, f'{split}_perplexity': math.exp(min(nll, 700)),
                       f'{split}_accuracy': accuracy, f'{split}_token_error': 1 - accuracy})
    model.train(); return result


def make_handles(model, cfg):
    install_muonclip_extension()
    profile = copy.deepcopy(cfg['optimizer'])
    handles = optimizers.make_optimizer_handles(model, profile)
    if profile['family'] == 'muon_clip':
        aux = next(h for h in handles if h.role == 'auxiliary')
        aux.peak_lr = profile['aux_learning_rate']; aux.min_lr = profile['aux_min_learning_rate']
        for group in aux.optimizer.param_groups:
            group['lr'] = aux.peak_lr
    return handles


def measure_ww(model, cfg, identity, metrics):
    """Existing clip_xmax + randomized null; never substitute clipped for raw."""
    import weightwatcher as ww
    started = time.monotonic()
    device = rt.model_device(model)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(), rt.capture_accelerator_rng_state(device))
    try:
        seed = cfg['seed'] + 1000003 + identity['step']
        random.seed(seed); np.random.seed(seed % (2**32-1)); torch.manual_seed(seed)
        holder = WeightMatrixHolder(model)
        frame = ww.WeightWatcher(model=holder).analyze(
            ERG=True, randomize=True, plot=False, fix_fingers='clip_xmax',
            max_fingers=10, min_evals=cfg['ww'].get('min_evals', 20))
        frame = _attach_matrix_metadata(frame, holder.matrix_metadata)
        expected = {m['matrix_name'] for m in holder.matrix_metadata}
        if len(frame) != len(expected) or set(frame.matrix_name) != expected:
            raise RuntimeError('Incomplete/duplicated WeightWatcher matrix inventory')
        records = []
        for raw in frame.to_dict('records'):
            def number(key):
                try: return float(raw.get(key, float('nan')))
                except (ValueError, TypeError): return float('nan')
            a, c, null = number('raw_alpha'), number('alpha'), number('rand_distance')
            fit_ok = str(raw.get('status', 'success')) == 'success'
            valid = math.isfinite(a) and a > 0 and fit_ok
            clipped_valid = math.isfinite(c) and c > 0 and fit_ok
            records.append({**raw, **identity, **metrics,
                'layer': f"L{int(raw['block']):02d}",
                'matrix_type': str(raw['matrix_type']).removeprefix('W_'),
                'alpha_raw': a if valid else float('nan'),
                'alpha_clip_xmax': c if clipped_valid else float('nan'),
                'raw_fit_status': 'success' if valid else 'failed',
                'raw_fit_reason': '' if valid else str(raw.get('warning', 'raw_alpha missing/nonfinite/nonpositive')),
                'clipped_fit_status': 'success' if clipped_valid else 'failed',
                'null_status': 'success' if math.isfinite(null) else 'failed',
                'randomized_distance': null})
        return {'records': records, 'seconds': time.monotonic() - started}
    finally:
        random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
        rt.restore_accelerator_rng_state(states[3], device)


def require_finite_update(model, norm, losses, output, step, verbose=True):
    """Diagnostic pre-update barrier for validation. Never apply known-invalid gradients."""
    if verbose: print(json.dumps({'checking_before_update':step}),flush=True)
    # Materialize the pending XLA graph once before any individual host reads.
    rt.synchronize(norm.device)
    value=float(norm.detach().cpu())
    loss_values=[float(x.detach().cpu()) for x in losses]
    if math.isfinite(value) and all(math.isfinite(x) for x in loss_values):
        if verbose: print(json.dumps({'before_update':step,'microbatch_losses':loss_values,'gradient_norm':value}),flush=True)
        return
    # Persist the known failure immediately. Copying every gradient to the host
    # previously stalled here and hid the nonfinite result for over two hours.
    report={'status':'invalid_before_update','before_update':step,
            'gradient_norm':value,'microbatch_losses':loss_values,
            'gradient_details_collected':False,
            'reason':'Nonfinite loss or aggregate gradient norm; invalid optimizer update was not applied.'}
    atomic_json(Path(output)/'nonfinite_diagnostics.json',report)
    print(json.dumps(report),flush=True)
    raise RuntimeError(f'Invalid loss/gradient BEFORE update {step}; see nonfinite_diagnostics.json')


def save_checkpoint(root, payload, keep=3, milestone=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    path = root / f"step_{payload['step']:09d}.pt"
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(rt.tree_to_cpu(payload), f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    atomic_json(root / 'latest.json', {'file': path.name, 'step': payload['step']})
    if milestone:
        import shutil
        (root / 'milestones').mkdir(exist_ok=True)
        shutil.copy2(path, root / 'milestones' / path.name)
    for old in sorted(root.glob('step_*.pt'))[:-max(2, keep)]:
        old.unlink()
    return path


def architecture(model):
    c = model.cfg
    return {'parameters': model.parameter_count(), 'layers': c.n_layer, 'heads': c.n_head,
            'd_model': c.n_embd, 'head_dim': c.n_embd // c.n_head, 'context': c.block_size,
            'vocab_size': c.vocab_size, 'tied': model.lm_head.weight is model.token_embedding.weight,
            'logical_matrices': len(transformer_matrix_items(model)), **spmd.metadata()}


def train(cfg, data_root, output, *, device='cpu', resume=False, stop_after=None, deadline=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reporting = cfg.get('progress_reporting', False)
        diagnostic = cfg.get('validation_gradient_checks', False) or reporting
        port_checks = cfg.get('validation_tensor_checks', False)
        synced_checks = cfg.get('synchronized_finite_checks', False)
        if port_checks or reporting or synced_checks:
            port_debug.environment(output)
        if diagnostic:
            faulthandler.dump_traceback_later(300, repeat=True)
        try:
            return _train(cfg, data_root, output, device, resume, stop_after, deadline)
        except Exception as exc:
            if port_checks or reporting or synced_checks:
                port_debug.failure(output,exc)
            raise
        finally:
            if diagnostic:
                faulthandler.cancel_dump_traceback_later()


def _train(cfg, data_root, output, device, resume, stop_after, deadline):
    cfg = copy.deepcopy(cfg)
    t = cfg['training']; context = cfg['model']['block_size']
    port_checks = cfg.get('validation_tensor_checks', False)
    synced_checks = cfg.get('synchronized_finite_checks', False)
    if port_checks and synced_checks:
        raise ValueError('Choose one tensor-check implementation, not both.')
    def progress(stage, completed_step):
        if cfg.get('progress_reporting', False):
            atomic_json(output/'progress.json', {'stage':stage, 'completed_step':completed_step,
                                                'unix_time':time.time()})
    progress('initializing_runtime_and_data', 0)
    cloud = None
    if cfg.get('cloud_checkpoints', False):
        from .run_backup import RunBackup
        cloud = RunBackup(output, os.environ['RG_GPT2_GCS_URI'])
    if port_checks:
        port_debug.stage(output,'initializing_runtime_and_data',0)
    step_tokens = t['batch_size'] * t['grad_accum_steps'] * context
    total = min(t.get('max_steps', 10**12), math.ceil(t['max_tokens'] / step_tokens))
    if not 0 <= t['warmup_steps'] < t['schedule_steps'] or total < 1:
        raise ValueError('Invalid training horizon/warmup')
    spmd.initialize(cfg, device); dev = rt.choose_device(device); rt.configure_runtime(dev, cfg)
    rt.seed_everything(cfg['seed'], dev)
    metadata, arrays = load_memmaps(data_root, cfg)  # hashes and document isolation; NO download
    versions = {'torch': torch.__version__, 'numpy': np.__version__}
    fingerprint = hashlib.sha256(json.dumps({'config': cfg, 'data': metadata, 'versions': versions}, sort_keys=True).encode()).hexdigest()
    model = GPT(GPTConfig(**cfg['model'])).to(dev); spmd.replicate_model(model)
    handles = make_handles(model, cfg)
    gen = torch.Generator().manual_seed(cfg['seed'] + 11)
    step = 0; elapsed = 0.; last_grad = None; compile_seconds = 0.; steady_seconds = 0.; steady_steps = 0
    measurement_pending = False
    latest = output / 'checkpoints/latest.json'
    if resume:
        pointer = json.loads(latest.read_text())
        state = torch.load(latest.parent / pointer['file'], map_location='cpu', weights_only=False)
        if state['fingerprint'] != fingerprint or state['run_id'] != cfg['run_id']:
            raise RuntimeError('Resume config/data/run ID mismatch')
        model.load_state_dict(state['model']); spmd.replicate_model(model)
        optimizers.load_optimizer_state_dict(handles, state['optimizers'])
        gen.set_state(state['data_rng']); torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        rt.restore_accelerator_rng_state(state['accelerator_rng'], dev)
        step = state['step']; elapsed = state['wall_time']; last_grad = state['gradient_norm']
        measurement_pending = state.get('measurement_pending', False)
        if state['tokens_seen'] != step * step_tokens or state['scheduler_step'] != step:
            raise RuntimeError('Checkpoint step/token/scheduler mismatch')
        for folder in ('metrics', 'ww_metrics'):
            if any(int(p.stem) > step for p in (output / folder).glob('*.json')):
                raise RuntimeError('History ahead of checkpoint; refusing to erase scientific records')
        # Complete the checkpoint-first measurement transaction after interrupted writes.
        for folder, row in state['pending_records'].items():
            path = output / folder / f'{step:09d}.json'
            if not path.exists(): append_record(path, row)
        atomic_json(output / 'resume_verified.json', {'step': step, 'tokens_seen': step * step_tokens,
                    'optimizer_states': [len(h.optimizer.state) for h in handles], 'fingerprint': fingerprint})
    elif latest.exists() or (output / 'manifest.json').exists():
        raise FileExistsError('Use a fresh output directory or explicit --resume')
    atomic_json(output / 'manifest.json', {'config': cfg, 'data': metadata, 'data_path': str(Path(data_root).resolve()),
                'fingerprint': fingerprint, 'architecture': architecture(model), 'runtime': rt.runtime_metadata(dev),
                'effective_batch_tokens': step_tokens})
    print(json.dumps({'architecture': architecture(model), 'batch_tokens': step_tokens}), flush=True)
    started = time.monotonic(); initial_step = step

    def payload(step, pending, measurement_pending=False):
        return {'run_id': cfg['run_id'], 'config': cfg, 'fingerprint': fingerprint, 'model': model.state_dict(),
                'optimizers': optimizers.optimizer_state_dict(handles), 'step': step, 'tokens_seen': step*step_tokens,
                'scheduler_step': step, 'data_rng': gen.get_state(), 'torch_rng': torch.get_rng_state(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'accelerator_rng': rt.capture_accelerator_rng_state(dev), 'wall_time': elapsed + time.monotonic()-started,
                'gradient_norm': last_grad, 'pending_records': pending, 'measurement_pending': measurement_pending}

    def record(step, final=False):
        nonlocal elapsed
        if cfg.get('checkpoint_before_evaluation', False):
            progress('saving_pre_evaluation_checkpoint', step)
            save_checkpoint(output/'checkpoints', payload(step, {}, measurement_pending=True))
        progress('evaluating', step)
        if synced_checks:
            measured = execution_checks.evaluate_splits(model, arrays, cfg, dev, output, step, 'evaluation')
            metrics = dict(measured)
            for split in ('train', 'val', 'test'):
                metrics[split+'_accuracy'] = 1-measured[split+'_token_error']
                metrics[split+'_perplexity'] = math.exp(min(measured[split+'_nll'], 700))
        else:
            metrics = evaluate(model, arrays, cfg, dev)
        wall = elapsed + time.monotonic() - started
        identity = {'run_id': cfg['run_id'], 'optimizer': cfg['optimizer']['family'], 'seed': cfg['seed'],
                    'step': step, 'tokens_seen': step * step_tokens, 'wall_time': wall,
                    'learning_rate': handles[0].optimizer.param_groups[0]['lr'] if step else 0., 'gradient_norm': last_grad}
        row = {**identity, **metrics, 'first_two_updates_seconds_including_compile': compile_seconds,
               'steady_training_tokens_per_second': steady_steps * step_tokens / max(steady_seconds, 1e-9),
               'end_to_end_tokens_per_second': (step-initial_step) * step_tokens / max(time.monotonic()-started, 1e-9)}
        pending = {'metrics': row}
        if cfg['ww']['enabled'] and due(step, cfg['ww']):
            progress('weightwatcher', step)
            measured = measure_ww(model, cfg, identity, metrics)
            measured['recommended_interval_seconds_for_10pct'] = 9 * measured['seconds']
            pending['ww_metrics'] = measured
        # Checkpoint FIRST includes the pending scalar/WW transaction. On resume, finish missing rows.
        progress('saving_checkpoint', step)
        checkpoint = save_checkpoint(output / 'checkpoints', payload(step, pending),
            milestone=step in cfg.get('milestones', []))
        for folder, value in pending.items(): append_record(output / folder / f'{step:09d}.json', value)
        print(json.dumps(row), flush=True)
        if cloud:
            progress('cloud_backup', step)
            cloud.publish(checkpoint, step)
        progress('measurement_completed', step)

    if synced_checks:
        execution_checks.check_finite(port_debug.optimizer_tensors(model, handles), output, 'initial_state', step, dev)
    if not resume or measurement_pending: record(step)
    training_window = time.monotonic(); last_timed_step = step
    while step < total:
        if (stop_after is not None and step >= stop_after) or (deadline and time.time() >= deadline) or (output / 'STOP').exists():
            break
        optimizers.zero_grad(handles)
        progress('update_started', step)
        for handle in handles:
            lr = optimizers.cosine_learning_rate(step, total_steps=t['schedule_steps'], warmup_steps=t['warmup_steps'],
                                                 peak_lr=handle.peak_lr, min_lr=handle.min_lr)
            handle.set_lr(lr)
        losses = []
        retain_diagnostics = (step < 4 or (step+1) % cfg['metrics_interval'] == 0 or due(step+1, cfg['ww']))
        input_windows = [] if port_checks or synced_checks else None
        if port_checks:
            port_debug.stage(output,'building_forward_backward_graph',step+1)
        for _ in range(t['grad_accum_steps']):
            x, y = batch(arrays['train'], gen, t['batch_size'], context, dev,trace=input_windows)
            _, loss = model(x, y); losses.append(loss.detach())
            (loss / t['grad_accum_steps']).backward()
        if port_checks or synced_checks:
            prefix=f'{step+1:06d}' if retain_diagnostics or port_checks else 'latest'
            atomic_json(output/'diagnostics'/f'{prefix}-input-windows.json',
                        {'update':step+1,'split':'train','context':context,'microbatch_offsets':input_windows,
                         'corpus_identity':'manifest.json/data','config_fingerprint':fingerprint})
        spmd.replicate_gradients(model)
        norm = rt.gradient_norm(model.parameters())
        if synced_checks:
            progress('checking_gradients', step)
            execution_checks.check_finite(
                [('gradient/'+name,p.grad) for name,p in model.named_parameters() if p.grad is not None]
                + [('loss/'+str(i),loss) for i,loss in enumerate(losses)] + [('gradient_norm',norm)],
                output, 'before_clipping', step+1, dev, retain=retain_diagnostics)
        if port_checks:
            port_debug.check_tensors(
                [('gradient/'+name,p.grad) for name,p in model.named_parameters() if p.grad is not None]
                + [('microbatch_loss/'+str(i),loss) for i,loss in enumerate(losses)]
                + [('global_gradient_norm',norm)],output,'before_clipping',step+1,dev)
        if cfg.get('validation_gradient_checks', False) or cfg.get('finite_update_guard', False):
            require_finite_update(model, norm, losses, output, step + 1,
                                  verbose=cfg.get('validation_gradient_checks', False) or step < 4)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'], foreach=False)
        if synced_checks:
            progress('checking_clipped_gradients', step)
            execution_checks.check_finite(
                [('gradient/'+name,p.grad) for name,p in model.named_parameters() if p.grad is not None],
                output, 'after_clipping', step+1, dev, retain=retain_diagnostics)
        if port_checks:
            port_debug.check_tensors(
                [('gradient/'+name,p.grad) for name,p in model.named_parameters() if p.grad is not None],
                output,'after_clipping',step+1,dev)
            port_debug.stage(output,'optimizer_update_started',step+1)
        if synced_checks:
            for handle in handles:
                progress('applying_'+handle.role, step)
                port_debug.stage(output, 'applying_'+handle.role, step+1)
                handle.optimizer.step(); rt.mark_step(dev); rt.synchronize(dev)
                execution_checks.check_finite(port_debug.optimizer_tensors(model, handles),
                    output, 'after_'+handle.role, step+1, dev, retain=retain_diagnostics)
        else:
            optimizers.optimizer_step(handles); rt.mark_step(dev)
        if port_checks:
            port_debug.check_tensors(port_debug.optimizer_tensors(model,handles),
                                     output,'after_optimizer',step+1,dev)
            port_debug.stage(output,'optimizer_update_completed',step+1)
        step += 1
        progress('update_submitted', step-1)
        measurement_due = (step % cfg['metrics_interval'] == 0 or due(step, cfg['ww'])
                           or step in cfg.get('metrics_steps', []) or step == total or step == stop_after)
        if step <= initial_step + 2 or measurement_due or cfg.get('benchmark_sync_every_step', False):
            rt.synchronize(dev)
            progress('update_completed', step)
            if cfg.get('progress_reporting', False) and (step <= 4 or step % 25 == 0):
                print(json.dumps({'completed_update':step,'tokens_seen':step*step_tokens}),flush=True)
            seconds = time.monotonic() - training_window
            if step <= initial_step + 2: compile_seconds += seconds
            else: steady_seconds += seconds; steady_steps += step - last_timed_step
            last_timed_step = step
            if step == initial_step + 2 and dev.type == 'xla':
                from torch_xla.debug import metrics as xla_metrics
                log = output / 'logs' / f'xla_compile_metrics_after_step_{step}.txt'
                log.parent.mkdir(exist_ok=True)
                log.write_text(xla_metrics.metrics_report())
            if measurement_due:
                last_grad = norm.item()
                if not math.isfinite(last_grad): raise RuntimeError('Nonfinite gradient norm')
                record(step)
            training_window = time.monotonic()
    if not (output / 'metrics' / f'{step:09d}.json').exists():
        last_grad = norm.item(); record(step, final=True)
    atomic_json(output / 'status.json', {'step': step, 'tokens_seen': step*step_tokens,
                'completed': step >= total, 'stopped': step < total, 'long_run_launched': cfg.get('long_run', False)})
    progress('stopped' if step < total else 'completed', step)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='tpu')
    p.add_argument('--resume', action='store_true'); p.add_argument('--stop-after', type=int)
    p.add_argument('--deadline-unix', type=float); p.add_argument('--allow-long-run', action='store_true')
    a = p.parse_args(); cfg = yaml.safe_load(Path(a.config).read_text())
    if cfg.get('long_run') and not a.allow_long_run: p.error('Long experiment requires explicit --allow-long-run')
    train(cfg, a.data_root, a.output, device=a.device, resume=a.resume, stop_after=a.stop_after, deadline=a.deadline_unix)

if __name__ == '__main__': main()

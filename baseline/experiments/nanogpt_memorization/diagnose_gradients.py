#!/usr/bin/env python3
"""Replay a saved baseline checkpoint into NEW diagnostic files, never resume a study.

The original run.py and its fingerprint are deliberately unchanged. This program
stops BEFORE clipping when the native gradient norm is nonfinite. It preserves
unmodified gradients, reports CPU-float64 norms, and can recompute the failing
update on CPU using the same pre-update weights and records. A CPU/MPS difference
is diagnostic evidence, not proof of a backend bug or an optimizer effect.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent


def encoded_number(value: float) -> float | str:
    value = float(value)
    return value if math.isfinite(value) else str(value)


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select_run(args) -> Path:
    if args.run_dir:
        candidates = [Path(args.run_dir).expanduser().resolve()]
    else:
        roots = ([Path(args.root).expanduser().resolve()] if args.root else
                 sorted(Path('/tmp').glob('nanogpt_*')))
        relative = Path('full') / 'repository' / args.condition / args.optimizer / f'seed_{args.seed}'
        candidates = [r / relative for r in roots if r.is_dir()]
        candidates = [r for r in candidates if not (r / 'complete.json').exists()]
    candidates = [r for r in candidates if (r / 'checkpoint_latest.pt').is_file()
                  and (r / 'manifest.json').is_file()]
    if not candidates:
        raise ValueError('No saved incomplete run found. Supply --run-dir /tmp/.../seed_1337.')
    return max(candidates, key=lambda r: (r / 'checkpoint_latest.pt').stat().st_mtime).resolve()


def manifest_optimizer(manifest: dict) -> str:
    """Historical run.py stored profile.family but omitted top-level optimizer.

    Never add a default to the fingerprinted manifest. The caller still compares
    the entire recovered source profile with the saved profile before replay.
    """
    name = manifest.get('optimizer', manifest.get('profile', {}).get('family'))
    if name not in ('adamw', 'muon'):
        raise ValueError('Saved manifest has no supported AdamW/Muon optimizer identity.')
    if name != manifest.get('profile', {}).get('family'):
        raise ValueError('Saved optimizer name conflicts with profile.family.')
    return name


def tensor_statistics(tensor) -> dict:
    """Inspect on CPU; scale before squaring to avoid overflow even in float64."""
    import torch
    value = tensor.detach().to(device='cpu', dtype=torch.float64)
    finite = torch.isfinite(value)
    good = value[finite]
    maximum = float(good.abs().max()) if good.numel() else 0.0
    norm = maximum * float(torch.linalg.vector_norm(good / maximum)) if maximum else 0.0
    all_finite = bool(finite.all())
    return {'elements': value.numel(), 'all_finite': all_finite,
            'nan_count': int(torch.isnan(value).sum()),
            'inf_count': int(torch.isinf(value).sum()),
            'finite_abs_max': maximum,
            'l2_cpu_float64': encoded_number(norm) if all_finite else None}


def gradient_statistics(model) -> dict:
    per_parameter = {name: tensor_statistics(p.grad) for name, p in model.named_parameters()
                     if p.grad is not None}
    finite = bool(per_parameter) and all(r['all_finite'] for r in per_parameter.values())
    norm = math.hypot(*(float(r['l2_cpu_float64']) for r in per_parameter.values())) if finite else None
    return {'all_gradients_finite': finite, 'parameters_with_grad': len(per_parameter),
            'total_l2_cpu_float64': encoded_number(norm) if norm is not None else None,
            'nonfinite_parameters': [name for name, r in per_parameter.items() if not r['all_finite']],
            'per_parameter': per_parameter}


def checked_clip(model, max_norm: float):
    import torch
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)


def accumulate(model, records, source, driver, device: str) -> list[float]:
    import torch
    batch_size = int(source['training']['batch_size'])
    accumulation = int(source['training']['grad_accum_steps'])
    if len(records) != batch_size * accumulation:
        raise ValueError('Replay effective batch differs from the saved training protocol.')
    losses = []
    for index in range(0, len(records), batch_size):
        x, y = driver.pack(records[index:index + batch_size], source['model']['block_size'])
        logits, loss = model(torch.as_tensor(x, device=device), torch.as_tensor(y, device=device))
        losses.append(float(loss.detach()))
        if not math.isfinite(losses[-1]):
            raise FloatingPointError(f'Nonfinite forward loss in microbatch {index // batch_size}.')
        (loss / accumulation).backward()
        del logits, loss
    return losses


def same_state_cpu_check(model_factory, model_config, model_state, records, source, driver) -> dict:
    import torch
    with torch.random.fork_rng(devices=[]):
        model = model_factory(model_config).cpu().train()
        model.load_state_dict(model_state)
        try:
            losses = accumulate(model, records, source, driver, 'cpu')
            return {'status': 'completed', 'microbatch_losses': losses,
                    'gradients': gradient_statistics(model),
                    'interpretation': 'Same pre-update tensors and records; CPU numerical path, not a training continuation.'}
        except (FloatingPointError, RuntimeError) as exc:
            return {'status': 'failed', 'error': str(exc)}


def runtime_differences(saved: dict, device: str) -> list[str]:
    import torch
    import numpy as np
    current = {'device': device, 'torch': str(torch.__version__), 'numpy': np.__version__,
               'python': platform.python_version(), 'platform': platform.platform(),
               'machine': platform.machine()}
    return [f'{key}: saved={saved.get(key)!r}; current={value!r}'
            for key, value in current.items() if saved.get(key) != value]


def diagnose(args) -> Path:
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import fcntl
    run_dir = select_run(args)
    with (run_dir / '.lock').open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('That training run is still active. Stop it before diagnostic replay.') from exc
        return replay_selected(args, run_dir)


def replay_selected(args, run_dir: Path) -> Path:
    import torch
    import run as driver
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    identity = dict(manifest)
    fingerprint = identity.pop('fingerprint')
    if driver.digest(identity) != fingerprint:
        raise ValueError('Manifest fingerprint is invalid; no replay started.')
    if sha256(HERE / 'run.py') != manifest['runner_sha256']:
        raise ValueError('Original run.py changed. Replay requires the exact saved trainer source.')
    cfg = manifest['suite']
    source = driver.load_source(cfg)
    if source['model'] != manifest['source_model']:
        raise ValueError('Source model differs from the saved model.')
    optimizer_name = manifest_optimizer(manifest)
    if driver.resolve_profile(source, optimizer_name, manifest['recipe']) != manifest['profile']:
        raise ValueError('Source optimizer profile differs from the saved profile.')
    if float(source['model'].get('dropout', 0.0)) != 0.0:
        raise ValueError('This replay supports only the original zero-dropout baseline.')
    device = args.device or manifest['device']['device']
    if device not in ('cpu', 'mps', 'cuda'):
        raise ValueError(f'Unsupported saved device: {device}')
    differences = runtime_differences(manifest['device'], device)
    if differences and not args.allow_runtime_mismatch:
        raise ValueError('Runtime differs from original. Use the original environment, or explicitly '
                         'label a different-runtime replay with --allow-runtime-mismatch.\n' + '\n'.join(differences))
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable in this Python environment.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this Python environment.')
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(driver.REPO / 'baseline/nanogpt_one_head/src'))
    from rg_nanogpt_one_head.model import GPT, GPTConfig
    from rg_nanogpt_one_head.optimizers import make_optimizer_handles, set_learning_rates, zero_grad, optimizer_step
    saved = run_dir / 'checkpoint_latest.pt'
    saved_hash = sha256(saved)
    checkpoint = torch.load(saved, map_location='cpu', weights_only=True)
    if checkpoint['fingerprint'] != fingerprint:
        raise ValueError('Checkpoint fingerprint does not match the manifest.')
    first = int(checkpoint['step'])
    batch_size = int(source['training']['batch_size']) * int(source['training']['grad_accum_steps'])
    study = driver.Study(cfg, manifest['condition'], manifest['stage'], int(manifest['seed']), batch_size)
    if study.identity() != manifest['data_sha256']:
        raise ValueError('Regenerated data/schedule does not match the saved run.')
    stop = min(study.steps, first + args.updates)
    model_config = GPTConfig(**source['model'])
    model = GPT(model_config).to(device).train()
    model.load_state_dict(checkpoint['model'])
    state_hash = driver.state_digest(model)
    rows = driver.read_rows(run_dir / 'metrics.jsonl')
    matching = [r for r in rows if r['step'] == first]
    if not matching or any(r['model_sha256'] != state_hash for r in matching):
        raise ValueError('Checkpoint tensors do not match the saved audited state.')
    handles = make_optimizer_handles(model, manifest['profile'])
    for handle, state in zip(handles, checkpoint['optimizers'], strict=True):
        handle.optimizer.load_state_dict(state)
    torch.set_rng_state(checkpoint['torch_rng'])
    if device == manifest['device']['device']:
        if device == 'mps':
            torch.mps.set_rng_state(checkpoint['device_rng'])
        elif device == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['device_rng'])
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    destination = run_dir / 'diagnostics' / f'gradient_replay_{stamp}_{os.getpid()}'
    destination.mkdir(parents=True, exist_ok=False)
    import importlib.metadata
    evidence = {'diagnostic_only': True, 'original_run': str(run_dir), 'checkpoint_step': first,
                'checkpoint_sha256': saved_hash, 'fingerprint': fingerprint,
                'optimizer': optimizer_name,
                'optimizer_identity_source': 'manifest.optimizer' if 'optimizer' in manifest else 'validated profile.family',
                'checkpoint_model_sha256': state_hash, 'device': device,
                'runtime_differences': differences, 'planned_stop_step': stop,
                'current_packages': sorted((d.metadata.get('Name', 'unknown'), d.version)
                                           for d in importlib.metadata.distributions()),
                'spectral_audits_replayed': False, 'status': 'running',
                'note': 'Original files are read-only. Replaying training updates without WeightWatcher; not a new scientific run.'}
    write_json(destination / 'report.json', evidence)
    print(f'Original run: {run_dir}\nSaved checkpoint: {first}\nReplay: {first} through {stop - 1} (zero-based update indices)\n'
          f'New diagnostic files: {destination}\nNo original checkpoint or metric file will be overwritten.', flush=True)
    del checkpoint
    for step in range(first, stop):
        zero_grad(handles)
        learning_rates = set_learning_rates(handles, update_index=step,
                                            total_steps=manifest['schedule_steps'],
                                            warmup_steps=manifest['warmup_steps'])
        records = study.sample(step)
        losses = []
        phase = 'forward_backward'
        try:
            losses = accumulate(model, records, source, driver, device)
            phase = 'native_gradient_norm'
            norm = checked_clip(model, float(source['training']['grad_clip']))
        except (FloatingPointError, RuntimeError) as exc:
            gradients = gradient_statistics(model)
            all_finite = gradients['all_gradients_finite']
            cause = ('native_norm_nonfinite_with_finite_gradients'
                     if phase == 'native_gradient_norm' and all_finite and 'non-finite' in str(exc)
                     else 'nonfinite_gradient_entries' if gradients['nonfinite_parameters']
                     else 'forward_or_backend_error')
            evidence.update(status='failure_observed', failure_step_index=step,
                            failure_phase=phase, observed_failure=cause, error=str(exc),
                            microbatch_losses=[encoded_number(x) for x in losses],
                            learning_rates=learning_rates, gradients=gradients)
            evidence['parameters'] = {n: tensor_statistics(p) for n, p in model.named_parameters()}
            cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            x, y = driver.pack(records, source['model']['block_size'])
            snapshot = {'diagnostic_only': True, 'step_index': step, 'model': cpu_state,
                        'model_config': source['model'], 'manifest_fingerprint': fingerprint,
                        'gradients': {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                                      if p.grad is not None},
                        'inputs': torch.as_tensor(x), 'targets': torch.as_tensor(y),
                        'optimizer_states': [h.optimizer.state_dict() for h in handles]}
            temporary = destination / 'failure_state.pt.tmp'
            torch.save(snapshot, temporary)
            temporary.replace(destination / 'failure_state.pt')
            write_json(destination / 'report.json', evidence)
            print(f'Failure at update index {step}: {cause}', flush=True)
            print('Affected parameters: ' + ', '.join(gradients['nonfinite_parameters']), flush=True)
            if device != 'cpu' and not args.no_cpu_check and all(
                    row['all_finite'] for row in evidence['parameters'].values()):
                print('Recomputing this ONE update on CPU from the same weights and records...', flush=True)
                evidence['same_state_cpu_check'] = same_state_cpu_check(
                    GPT, model_config, cpu_state, records, source, driver)
            break
        optimizer_step(handles)
        row = {'step_index': step, 'completed_updates': step + 1,
               'train_loss': sum(losses) / len(losses), 'native_gradient_norm': float(norm),
               'learning_rates': learning_rates}
        with (destination / 'replay_trace.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        if (step + 1 - first) % 25 == 0:
            print(f'Replayed to {step + 1}: loss={row["train_loss"]:.6f}', flush=True)
    else:
        evidence.update(status='no_failure_in_replay_window', completed_updates=stop)
    evidence['original_checkpoint_unchanged'] = sha256(saved) == saved_hash
    write_json(destination / 'report.json', evidence)
    print(f'\n{evidence["status"]}\nReport: {destination / "report.json"}', flush=True)
    if not evidence['original_checkpoint_unchanged']:
        print('WARNING: original checkpoint changed externally while diagnosis ran; check for an active trainer.')
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    where = parser.add_mutually_exclusive_group()
    where.add_argument('--run-dir', help='Exact original seed directory.')
    where.add_argument('--root', help='Study root; otherwise select latest incomplete matching run under /tmp.')
    parser.add_argument('--condition', default='rule_random')
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', choices=['mps', 'cuda', 'cpu'])
    parser.add_argument('--updates', type=int, default=400)
    parser.add_argument('--allow-runtime-mismatch', action='store_true')
    parser.add_argument('--no-cpu-check', action='store_true')
    args = parser.parse_args(argv)
    if args.updates < 1:
        parser.error('--updates must be positive')
    try:
        diagnose(args)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'Diagnostic error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nDiagnostic replay stopped; original results were not modified.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())

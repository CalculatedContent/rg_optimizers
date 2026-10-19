"""Eight-device XLA data parallel training with exact masked metrics and resumable state."""
import argparse
import fcntl
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import (atomic_json, atomic_torch, batch_indices, build_model, config_dict,
                    digest, file_hash, lr_at, make_optimizer, read_config,
                    restore_rng, rng_state, seed_all)


class Runtime:
    def __init__(self, backend):
        self.xm = None
        self.rank, self.world = 0, 1
        self.device = torch.device('cpu')
        if backend == 'xla':
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr
            self.xm = xm
            self.device = xm.xla_device()
            self.rank, self.world = xr.global_ordinal(), xr.world_size()
            if self.world != 8:
                raise RuntimeError(f'Expected exactly eight TPU devices, got {self.world}')

    def barrier(self, name):
        if self.xm:
            self.xm.rendezvous(name)

    def step(self):
        if self.xm:
            self.xm.mark_step()

    def sum(self, value):
        if self.xm:
            value = self.xm.all_reduce(self.xm.REDUCE_SUM, value)
        self.step()
        return value

    def any(self, value):
        return bool(self.sum(torch.tensor(int(value), device=self.device)).cpu().item())

    def rng(self):
        out = rng_state()
        if self.xm:
            out['xla'] = self.xm.get_rng_state(self.device)
        return out

    def restore(self, state):
        restore_rng(state)
        if self.xm:
            self.xm.set_rng_state(state['xla'], self.device)


def autocast(c, rt):
    return torch.autocast(device_type=rt.device.type, dtype=torch.bfloat16) if c.precision == 'bf16' else nullcontext()


def batches(arrays, indices, c, rt, epoch=0, shuffle=False):
    for rows, weights, real in batch_indices(len(indices), c.per_device_batch, rt.world, rt.rank, epoch, c.seed, shuffle):
        selected = indices[rows]
        inputs = {k: torch.from_numpy(arrays[k][selected]).to(rt.device) for k in ('input_ids', 'attention_mask')}
        labels = torch.from_numpy(arrays['labels'][selected]).to(rt.device)
        yield inputs, labels, torch.from_numpy(weights).to(rt.device), real


def evaluate(model, arrays, indices, c, rt):
    model.eval()
    sums = torch.zeros(3, dtype=torch.float32, device=rt.device)
    with torch.no_grad():
        for inputs, labels, weights, _ in batches(arrays, indices, c, rt):
            with autocast(c, rt):
                logits = model(**inputs).logits
            loss = F.cross_entropy(logits.float(), labels, reduction='none')
            sums += torch.stack([(loss * weights).sum(), ((logits.argmax(-1) == labels) * weights).sum(), weights.sum()])
            rt.step()
    sums = rt.sum(sums).cpu().tolist()
    if int(sums[2]) != len(indices):
        raise RuntimeError(f'Evaluation counted {sums[2]} examples, expected {len(indices)}')
    return {'loss': sums[0] / sums[2], 'accuracy': sums[1] / sums[2],
            'errors': int(sums[2] - sums[1]), 'examples': int(sums[2])}


def save(root, model, optimizer, state, c, data_hash, rt):
    rt.step()
    rt.barrier('save-start')
    generation = root / 'checkpoints' / f'step_{state["step"]:06d}'
    latest = root / 'latest_checkpoint.json'
    if latest.exists() and json.loads(latest.read_text())['step'] == state['step']:
        rt.barrier('save-already-committed')
        return
    generation.mkdir(parents=True, exist_ok=True)
    atomic_torch(generation / f'rng_rank{rt.rank}.pt', rt.rng())
    if rt.rank == 0:
        atomic_torch(generation / 'state.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'cursor': state, 'config': config_dict(c), 'data_manifest_sha256': data_hash, 'world': rt.world})
    rt.barrier('save-files')
    if rt.rank == 0:
        files = {p.name: file_hash(p) for p in sorted(generation.glob('*.pt'))}
        if len(files) != rt.world + 1:
            raise RuntimeError('Missing rank RNG state')
        manifest = {'step': state['step'], 'generation': generation.relative_to(root).as_posix(), 'sha256': files}
        atomic_json(generation / 'manifest.json', manifest)
        atomic_json(latest, manifest)
        # Two committed full states remain; spectral snapshots are independent files.
        committed = sorted((root / 'checkpoints').glob('step_*/manifest.json'))
        for old in committed[:-2]:
            shutil.rmtree(old.parent)
    rt.barrier('save-committed')


def restore(root, model, optimizer, c, data_hash, rt):
    latest = root / 'latest_checkpoint.json'
    if not latest.exists():
        return {'step': 0, 'epoch': 0, 'batch': 0}
    manifest = json.loads(latest.read_text())
    generation = root / manifest['generation']
    # Rank zero validates the large shared state once; each rank verifies its RNG.
    if rt.rank == 0 and file_hash(generation / 'state.pt') != manifest['sha256']['state.pt']:
        raise RuntimeError('Checkpoint hash mismatch')
    rng_name = f'rng_rank{rt.rank}.pt'
    if file_hash(generation / rng_name) != manifest['sha256'][rng_name]:
        raise RuntimeError('Rank RNG hash mismatch')
    rt.barrier('restore-verified')
    payload = torch.load(generation / 'state.pt', map_location='cpu', weights_only=False)
    if payload['config'] != config_dict(c) or payload['world'] != rt.world or payload['data_manifest_sha256'] != data_hash:
        raise RuntimeError('Resume requires identical configuration, data, and device count')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    state = payload['cursor']
    del payload
    rt.step()
    rt.restore(torch.load(generation / rng_name, map_location='cpu', weights_only=False))
    return state


def measure(root, model, arrays, split, c, rt, state, started, test_arrays):
    step = state['step']
    path = root / 'measurements' / f'step_{step:06d}.json'
    # Resume after an already-completed measurement doesn't duplicate it.
    if path.exists() and json.loads(path.read_text()).get('test') is not None:
        return
    saved_rng = rt.rng()
    try:
        train = evaluate(model, arrays, split['train'], c, rt)
        validation = evaluate(model, arrays, split['validation'], c, rt)
        test = evaluate(model, test_arrays, np.arange(len(test_arrays['labels'])), c, rt)
    finally:
        rt.restore(saved_rng)
    if rt.rank == 0:
        snapshot = root / 'spectral_snapshots' / f'step_{step:06d}.pt'
        matrices = {name: p for name, p in model.named_parameters() if p.ndim == 2 and 'embeddings' not in name}
        atomic_torch(snapshot, matrices)
        atomic_json(path, {'step': step, 'epoch': state['epoch'], 'train': train, 'validation': validation, 'test': test,
                          'generalization_gap_pp': 100 * (train['accuracy'] - validation['accuracy']),
                          'train_test_gap_pp': 100 * (train['accuracy'] - test['accuracy']),
                          'test_evaluated': True, 'official_test_accessed': not c.synthetic,
                          'test_used_for_selection': False,
                          'elapsed_this_process_seconds': time.monotonic() - started,
                          'spectral_snapshot': str(snapshot.relative_to(root)),
                          'snapshot_sha256': file_hash(snapshot), 'synthetic_engineering_only': c.synthetic})
    rt.barrier('measurement-written')


def worker(index, args):
    torch.set_num_threads(1)
    c = read_config(args.config)
    root, data = Path(args.root), Path(args.data)
    rt = Runtime(args.backend)
    root.mkdir(parents=True, exist_ok=True)
    data_hash = file_hash(data / 'prepared.json')
    manifest = json.loads((data / 'prepared.json').read_text())
    identity = {k: config_dict(c)[k] for k in manifest['identity']}
    if identity != manifest['identity']:
        raise RuntimeError('Config differs from prepared data')
    arrays = dict(np.load(data / 'tokens.npz'))
    split = dict(np.load(data / 'split.npz'))
    test_manifest = json.loads((data / 'test_prepared.json').read_text())
    if test_manifest['identity']['training_manifest_sha256'] != data_hash or file_hash(data / 'test_tokens.npz') != test_manifest['sha256']:
        raise RuntimeError('Test monitoring manifest/hash mismatch')
    test_arrays = dict(np.load(data / 'test_tokens.npz'))
    seed_all(c.seed)  # Identical pretrained encoder + random classifier on all ranks.
    model = build_model(c, data / 'model').to(rt.device)
    optimizer = make_optimizer(model, c)
    if rt.xm:
        rt.xm.set_rng_state(c.seed + rt.rank * 100003, rt.device)
    state = restore(root, model, optimizer, c, data_hash, rt)
    start = time.monotonic()
    steps_per_epoch = math.ceil(len(split['train']) / (c.per_device_batch * rt.world))
    total = steps_per_epoch * c.epochs
    if rt.rank == 0:
        atomic_json(root / 'config.json', config_dict(c))
        atomic_json(root / 'runtime.json', {'world': rt.world, 'backend': args.backend, 'total_steps': total,
                    'global_batch': c.per_device_batch * rt.world, 'resumed_step': state['step'],
                    'data_manifest_sha256': data_hash, 'precision': c.precision,
                    'decay_semantics': 'decoupled' if c.optimizer == 'adamw' else 'coupled_L2',
                    'no_decay_parameters': 'All vectors: biases and LayerNorm scale/offset',
                    'test_accessed': not c.synthetic, 'test_monitoring_enabled': True,
                    'test_manifest_sha256': file_hash(data / 'test_prepared.json'),
                    'test_used_for_selection': False})
    if state['step'] == 0:
        measure(root, model, arrays, split, c, rt, state, start, test_arrays)
        save(root, model, optimizer, state, c, data_hash, rt)
    # Revisit the epoch-boundary measurement if interrupted between checkpoint and evaluation.
    if state['batch'] == 0 and state['step']:
        measure(root, model, arrays, split, c, rt, state, start, test_arrays)
    for epoch in range(state['epoch'], c.epochs):
        model.train()
        skip = state['batch'] if epoch == state['epoch'] else 0
        for batch_num, (inputs, labels, weights, real) in enumerate(batches(arrays, split['train'], c, rt, epoch, True)):
            if batch_num < skip:
                continue
            if rt.any(time.time() >= args.deadline or (args.stop_after and state['step'] >= args.stop_after)):
                save(root, model, optimizer, state, c, data_hash, rt)
                if rt.rank == 0:
                    atomic_json(root / 'status.json', {'status': 'paused_resumable', **state, 'time': time.time()})
                return
            optimizer.zero_grad(set_to_none=True)
            current_lr = lr_at(state['step'], total, c)
            for group in optimizer.param_groups:
                group['lr'] = torch.tensor(current_lr, device=rt.device) if rt.xm else current_lr
            with autocast(c, rt):
                logits = model(**inputs).logits
            # Averaged gradients across ranks represent exactly the real global examples.
            losses = F.cross_entropy(logits.float(), labels, reduction='none')
            loss = (losses * weights).sum() * (rt.world / real)
            loss.backward()
            if rt.xm:
                rt.xm.reduce_gradients(optimizer)  # Average BEFORE global gradient clipping.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip, foreach=False)
            if rt.any((~torch.isfinite(norm)) | (~torch.isfinite(loss))):
                raise FloatingPointError('Nonfinite loss/gradient: latest committed checkpoint preserved')
            optimizer.step()
            rt.step()
            state = {'step': state['step'] + 1, 'epoch': epoch, 'batch': batch_num + 1}
            end_epoch = state['batch'] == steps_per_epoch
            if end_epoch:
                state.update(epoch=epoch + 1, batch=0)
            if rt.rank == 0:
                atomic_json(root / 'status.json', {'status': 'training', **state, 'time': time.time(),
                            'lr': current_lr, 'total_steps': total})
            if state['step'] % c.checkpoint_steps == 0 or end_epoch:
                save(root, model, optimizer, state, c, data_hash, rt)
            if end_epoch:
                measure(root, model, arrays, split, c, rt, state, start, test_arrays)
    save(root, model, optimizer, state, c, data_hash, rt)
    if rt.rank == 0:
        atomic_json(root / 'status.json', {'status': 'training_complete', **state, 'time': time.time(),
                    'elapsed_this_process_seconds': time.monotonic() - start})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--backend', choices=('cpu', 'xla'), default='xla')
    p.add_argument('--deadline', type=float, default=float('inf'))
    p.add_argument('--stop-after', type=int, default=0, help='Pause after this global step; full-state resume supported')
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.backend == 'xla':
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(worker, args=(args,), start_method='spawn')
        else:
            worker(0, args)


if __name__ == '__main__':
    main()

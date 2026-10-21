"""Random-init DistilBERT MLM on native TorchTPU; CPU mode is engineering-only."""
import argparse
from dataclasses import asdict
import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from common import (atomic_json, batch_indices, build_model, cpu_tree, digest,
                    encoder_matrices, file_hash, learning_rate, make_optimizer,
                    masked_batch, read_config, save_torch, source_hash, sync_dir)


class LossModel(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, ids, labels):
        # All prepared blocks are full length; there are no padding tokens to mask.
        logits = self.core(input_ids=ids).logits.float()
        loss_sum = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                   ignore_index=-100, reduction='sum')
        keep = labels != -100
        correct = ((logits.argmax(-1) == labels) & keep).sum()
        return loss_sum, correct, keep.sum()


class Runtime:
    def __init__(self, backend, expected):
        self.backend = backend
        if backend == 'tpu':
            import torch_tpu  # noqa: F401; registers native PyTorch backend
            if int(os.environ.get('WORLD_SIZE', '0')) != expected:
                raise RuntimeError('Use launch.py; expected exactly eight TorchTPU workers')
            dist.init_process_group('tpu_dist', timeout=dt.timedelta(minutes=20))
            self.device = torch.device('tpu')  # each process owns its assigned device
            torch.set_float32_matmul_precision('medium')
        else:
            if int(os.environ.get('WORLD_SIZE', '1')) > 1:
                dist.init_process_group('gloo', timeout=dt.timedelta(minutes=5))
            self.device = torch.device('cpu')
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.world != expected:
            raise RuntimeError(f'Expected {expected} workers; got {self.world}')

    def sum(self, x):
        if self.world > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def rng(self):
        state = dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())
        if self.backend == 'tpu':
            # TorchTPU 0.2 makes the legacy getter internal on torch >=2.13.
            api = getattr(torch.accelerator, 'random', None)
            getter = getattr(api, 'get_rng_state', None)
            state['tpu'] = (getter() if getter is not None else torch.tpu.get_rng_state()).cpu()
        return state

    def restore_rng(self, state):
        torch.set_rng_state(state['torch'])
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        if self.backend == 'tpu':
            torch.tpu.set_rng_state(state['tpu'])

    def stop(self, deadline):
        flag = torch.tensor(float(time.time() >= deadline), device=self.device)
        return self.sum(flag).item() > 0


def probe(rt, output):
    # Matmul and collectives on every worker, before any model/data allocation.
    x = torch.full((128, 128), float(rt.rank + 1), device=rt.device)
    y = x @ torch.ones_like(x)
    value = rt.sum(y[0, 0].clone()).cpu().item()
    expected = 128 * rt.world * (rt.world + 1) / 2
    if value != expected:
        raise RuntimeError(f'Collective probe failed: {value} != {expected}')
    state = rt.rng()
    sample = torch.rand(16, device=rt.device).cpu()
    rt.restore_rng(state)
    replay = torch.rand(16, device=rt.device).cpu()
    rt.restore_rng(state)
    if not torch.equal(sample, replay):
        raise RuntimeError('RNG checkpoint/restore probe failed')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / f'rank_{rt.rank}.json', dict(rank=rt.rank, world=rt.world,
                backend=rt.backend, device=str(rt.device), collective_value=value, rng_roundtrip=True,
                torch=torch.__version__, torch_tpu=importlib.metadata.version('torch_tpu') if rt.backend == 'tpu' else None))
    rt.barrier()
    if rt.rank == 0:
        records = [json.loads((root / f'rank_{i}.json').read_text()) for i in range(rt.world)]
        atomic_json(root / 'probe.json', dict(passed=True, ranks=records))


def checkpoint(root, step, model, opt, rt, identity):
    checkpoints = root / 'checkpoints'
    target = checkpoints / f'step_{step:08d}'
    partial = checkpoints / f'step_{step:08d}.partial'
    if rt.rank == 0:
        checkpoints.mkdir(exist_ok=True)
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
    rt.barrier()
    save_torch(partial / f'rng_{rt.rank}.pt', rt.rng())
    if rt.rank == 0:
        save_torch(partial / 'state.pt', dict(step=step, identity=identity,
                   model=cpu_tree(model.state_dict()), optimizer=cpu_tree(opt.state_dict())))
    rt.barrier()
    if rt.rank == 0:
        atomic_json(partial / 'manifest.json', {p.name: file_hash(p) for p in partial.glob('*.pt')})
        pointer = root / 'latest_checkpoint.json'
        published = json.loads(pointer.read_text())['path'] if pointer.exists() else None
        # Reuse only a published generation. A crash between rename and pointer
        # publication can leave an orphan from a different replay of this step.
        if target.exists() and published == str(target.relative_to(root)):
            shutil.rmtree(partial)
        else:
            if target.exists():
                shutil.rmtree(target)
            partial.rename(target)
        sync_dir(checkpoints)
        atomic_json(root / 'latest_checkpoint.json', dict(path=str(target.relative_to(root)), step=step))
        # Future generations left by an interrupted replay were never published
        # on this trajectory; they must not displace either retained checkpoint.
        committed = sorted((p for p in checkpoints.glob('step_*')
                            if p.is_dir() and p.name[5:].isdigit() and int(p.name[5:]) <= step),
                           key=lambda p: int(p.name[5:]))
        for old in committed[:-2]:
            shutil.rmtree(old)
    rt.barrier()


def restore(root, model, opt, rt, identity):
    target = root / json.loads((root / 'latest_checkpoint.json').read_text())['path']
    manifest = json.loads((target / 'manifest.json').read_text())
    if set(manifest) != {'state.pt', *(f'rng_{i}.pt' for i in range(rt.world))}:
        raise RuntimeError('Incomplete checkpoint')
    for name, sha in manifest.items():
        if file_hash(target / name) != sha:
            raise RuntimeError('Checkpoint hash mismatch: ' + name)
    # Only load files created by this experiment, after verifying their manifest.
    state = torch.load(target / 'state.pt', map_location='cpu', weights_only=False)
    if state['identity'] != identity:
        raise RuntimeError('Resume requires identical source, config, prepared data, packages and world size')
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['optimizer'])
    if rt.backend == 'tpu':
        for group in opt.param_groups:
            group['lr'] = torch.as_tensor(group['lr'], device=rt.device)
    rt.restore_rng(torch.load(target / f'rng_{rt.rank}.pt', map_location='cpu', weights_only=False))
    return state['step']


def evaluate(net, arrays, meta, c, rt):
    state = rt.rng()
    net.eval()
    results = {}
    with torch.no_grad():
        for name, salt in [('train', 991), ('validation', 992), ('test', 993)]:
            order = np.arange(len(arrays[name]))
            if name == 'train':
                order = np.random.default_rng(c.data_seed).permutation(order)[:c.train_eval_sequences]
            totals = torch.zeros(3, device=rt.device)
            batches = math.ceil(len(order) / (rt.world * c.per_device_batch))
            for j in range(batches):
                idx, valid = batch_indices(order, j, rt.rank, rt.world, c.per_device_batch)
                ids, labels = masked_batch(arrays[name], idx, valid, meta, c, salt, evaluation=True)
                with torch.autocast(rt.device.type, dtype=torch.bfloat16,
                                    enabled=c.precision == 'bf16'):
                    loss_sum, correct, count = net(ids.to(rt.device), labels.to(rt.device))
                totals += torch.stack((loss_sum.detach(), correct.float(), count.float()))
            nll, correct, count = rt.sum(totals).cpu().tolist()
            if count <= 0 or not math.isfinite(nll):
                raise RuntimeError('Invalid evaluation')
            results[name] = dict(mlm_loss=nll / count, masked_token_accuracy=correct / count,
                                 masked_tokens=int(count), sequences=len(order))
    rt.restore_rng(state)
    net.train()
    return results


def preserve_best_validation(root, step, model, results, identity):
    """Retain full weights selected only by the fixed validation MLM loss."""
    root = Path(root)
    pointer = root / 'best_validation.json'
    loss = results['validation']['mlm_loss']
    if not math.isfinite(loss):
        raise ValueError('Best-checkpoint selection requires finite validation loss')
    if pointer.exists() and loss >= json.loads(pointer.read_text())['validation_mlm_loss']:
        return
    folder = root / 'best_validation'
    folder.mkdir(exist_ok=True)
    target = folder / f'step_{step:08d}.pt'
    partial = target.with_suffix('.partial')
    save_torch(partial, dict(step=step, identity=identity,
               model=cpu_tree(model.state_dict()), model_config=model.config.to_dict(),
               validation_mlm_loss=loss))
    partial.replace(target)
    sync_dir(folder)
    atomic_json(pointer, dict(step=step, validation_mlm_loss=loss,
                path=str(target.relative_to(root)), sha256=file_hash(target),
                criterion='minimum fixed-mask validation MLM loss; first checkpoint wins ties',
                test_used_for_selection=False, alpha_used_for_selection=False,
                purpose='full model weights for evaluation; use checkpoints/ for exact training resume'))
    for old in folder.glob('step_*.pt'):
        if old != target:
            old.unlink()


def measure(root, step, net, model, arrays, meta, c, rt, identity):
    results = evaluate(net, arrays, meta, c, rt)
    if rt.rank == 0:
        preserve_best_validation(root, step, model, results, identity)
        snapshots = root / 'spectral_snapshots'
        snapshots.mkdir(exist_ok=True)
        path = snapshots / f'step_{step:08d}.pt'
        mats = encoder_matrices(model)
        expected = 12 if c.synthetic else 36
        if len(mats) != expected:
            raise RuntimeError(f'Expected {expected} encoder matrices; got {len(mats)}')
        save_torch(path, mats)
        atomic_json(root / 'measurements' / f'step_{step:08d}.json', dict(
            step=step, **results, synthetic=c.synthetic,
            validation_train_loss_gap=results['validation']['mlm_loss'] - results['train']['mlm_loss'],
            snapshot=str(path.relative_to(root)), sha256=file_hash(path), expected_layers=expected,
            test_used_for_selection=False, evaluation_mask_probability=c.evaluation_mask_probability,
            metric='masked-token accuracy, not downstream classification'))
    rt.barrier()


def train(a, rt):
    c = read_config(a.config)
    if rt.backend == 'cpu' and not c.synthetic:
        raise RuntimeError('CPU execution is only for synthetic engineering checks')
    if rt.backend == 'tpu' and c.synthetic:
        raise RuntimeError('Production TPU launch requires real data and full DistilBERT')
    data, root = Path(a.data), Path(a.output)
    meta = json.loads((data / 'prepared.json').read_text())
    if bool(meta['synthetic']) != c.synthetic or meta['sequence_length'] != c.sequence_length:
        raise RuntimeError('Data/config mismatch')
    arrays = {}
    for split, item in meta['splits'].items():
        path = data / item['file']
        if file_hash(path) != item['sha256']:
            raise RuntimeError('Prepared data hash mismatch')
        arrays[split] = np.memmap(path, dtype=np.int32, mode='r', shape=(item['rows'], c.sequence_length))
    packages = {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'numpy')}
    if rt.backend == 'tpu':
        packages['torch_tpu'] = importlib.metadata.version('torch_tpu')
        packages['libtpu'] = importlib.metadata.version('libtpu')
    identity = dict(config=asdict(c), data=digest(meta), source=source_hash(), world=rt.world,
                    backend=rt.backend, packages=packages)
    if rt.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'identity.json').exists():
            if not a.resume:
                raise RuntimeError('Existing run: use --resume')
            if json.loads((root / 'identity.json').read_text()) != identity:
                raise RuntimeError('Run identity mismatch')
        else:
            if a.resume:
                raise RuntimeError('No run to resume')
            atomic_json(root / 'identity.json', identity)
    rt.barrier()
    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)
    model = build_model(c, meta['vocab_size']).to(rt.device)
    opt = make_optimizer(model, c, rt.device)
    wrapped = LossModel(model)
    if c.compile:
        wrapped = torch.compile(wrapped, backend='tpu' if rt.backend == 'tpu' else 'aot_eager', dynamic=False)
    net = DDP(wrapped, broadcast_buffers=False) if rt.world > 1 else wrapped
    # Same initialization, independent per-rank dropout; data/masks have a separate seed.
    torch.manual_seed(c.seed + 100000 + rt.rank)
    step = restore(root, model, opt, rt, identity) if a.resume else 0
    if rt.rank == 0:
        atomic_json(root / 'runtime.json', dict(world=rt.world, backend=rt.backend, resumed_step=step,
                    experiment='BERT-6 MLM from scratch', architecture='DistilBertForMaskedLM',
                    distillation=False,
                    parameters=sum(p.numel() for p in model.parameters()), pretrained_weights=False,
                    global_batch=c.per_device_batch * rt.world, packages=packages, config=asdict(c)))
    if step == 0:
        checkpoint(root, step, model, opt, rt, identity)
    # A crash after committing a measurement cursor must not skip its diagnostics
    # on resume. Publish diagnostics only after the corresponding state is saved.
    if (step == 0 or step % c.measure_steps == 0) and not (
            root / 'measurements' / f'step_{step:08d}.json').exists():
        measure(root, step, net, model, arrays, meta, c, rt, identity)
    batches = math.ceil(len(arrays['train']) / (rt.world * c.per_device_batch))
    epoch, order = None, None
    start, start_step = time.monotonic(), step
    last_measured = step if (root / 'measurements' / f'step_{step:08d}.json').exists() else -1
    net.train()
    limit = min(c.steps, a.stop_after) if a.stop_after is not None else c.steps
    while step < limit and not rt.stop(a.deadline):
        current_epoch, batch = divmod(step, batches)
        if epoch != current_epoch:
            epoch = current_epoch
            order = np.random.default_rng(c.data_seed + epoch).permutation(len(arrays['train']))
        idx, valid = batch_indices(order, batch, rt.rank, rt.world, c.per_device_batch)
        ids, labels = masked_batch(arrays['train'], idx, valid, meta, c, 901, step)
        ids, labels = ids.to(rt.device), labels.to(rt.device)
        global_count = rt.sum((labels != -100).sum().float())
        lr = learning_rate(c, step)
        for group in opt.param_groups:
            if isinstance(group['lr'], torch.Tensor):
                group['lr'].copy_(torch.tensor(lr, device=rt.device))
            else:
                group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast(rt.device.type, dtype=torch.bfloat16, enabled=c.precision == 'bf16'):
            loss_sum, correct, count = net(ids, labels)
        # DDP averages gradients; this yields the exact GLOBAL masked-token mean,
        # even for uneven mask counts and padded final batches.
        loss = loss_sum * rt.world / global_count.clamp_min(1)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip, foreach=False)
        totals = rt.sum(torch.stack((loss_sum.detach(), correct.float(), count.float(), norm.detach().float())))
        values = totals.cpu().tolist()
        if values[2] <= 0 or not all(math.isfinite(x) for x in values):
            raise RuntimeError('Nonfinite loss/gradients or no targets; no update applied')
        opt.step()
        step += 1
        if rt.rank == 0:
            status = dict(step=step, target_steps=c.steps, epoch=current_epoch, lr=lr,
                          mlm_loss=values[0] / values[2], masked_token_accuracy=values[1] / values[2],
                          mean_gradient_norm=values[3] / rt.world,
                          elapsed_seconds=time.monotonic() - start,
                          steps_this_process=step - start_step, state='training')
            atomic_json(root / 'status.json', status)
            if step <= 3 or step % 10 == 0:
                print(json.dumps(status), flush=True)
        if step % c.checkpoint_steps == 0 or step % c.measure_steps == 0:
            checkpoint(root, step, model, opt, rt, identity)
        if step % c.measure_steps == 0:
            measure(root, step, net, model, arrays, meta, c, rt, identity)
            last_measured = step
    # Commit training state before potentially expensive final diagnostics.
    checkpoint(root, step, model, opt, rt, identity)
    if last_measured != step and not rt.stop(a.deadline):
        measure(root, step, net, model, arrays, meta, c, rt, identity)
    if rt.rank == 0:
        atomic_json(root / 'status.json', dict(step=step, target_steps=c.steps,
                    state='completed' if step == c.steps else 'paused', checkpoint_saved=True))
        atomic_json(root / 'RESULT.json', dict(step=step, completed=step == c.steps,
                    synthetic=c.synthetic, hardware=rt.backend, reason='step_limit' if step == limit else 'deadline'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['tpu', 'cpu'], default='tpu')
    p.add_argument('--expected-world', type=int, default=8)
    p.add_argument('--probe-only', action='store_true')
    p.add_argument('--config')
    p.add_argument('--data')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int)
    p.add_argument('--deadline', type=float, default=float('inf'))
    args = p.parse_args()
    torch.set_num_threads(1)
    runtime = Runtime(args.backend, args.expected_world)
    try:
        if args.probe_only:
            probe(runtime, args.output)
        else:
            train(args, runtime)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

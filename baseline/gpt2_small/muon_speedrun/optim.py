"""Pinned-record Muon math, batched and partitioned over TPU matrix groups."""
from collections import defaultdict
import math
import torch
from stock_config import ARCHITECTURE


def schedule(step):
    # Record: 3,000 updates, no LR warmup, 900-update linear warmdown.
    return min(1., max(0., (3000-step)/900))


def momentum(step):
    return 0.85 + 0.10 * min(step/500, 1.)


def orthogonalize(g, steps=5):
    x = g.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        a = x @ x.transpose(-2, -1)
        # Preserve the source's BF16 operation order: (c * A) @ A.
        b = -4.7750 * a + (2.0315 * a) @ a
        x = 3.4445 * x + b @ x
    return x.transpose(-2, -1) if transposed else x


class Muon:
    def __init__(self, named_parameters, rt):
        self.rt = rt
        grouped = defaultdict(list)
        for name, p in named_parameters:
            if p.ndim != 2:
                raise ValueError('Muon requires matrix parameters')
            grouped[tuple(p.shape)].append((name, p))
        self.groups = []
        for shape, entries in sorted(grouped.items()):
            padded = math.ceil(len(entries)/8)*8 if rt.tpu else len(entries)
            # CPU allocation then one transfer, independent of the lazy training graph.
            buf = torch.zeros((padded, *shape), dtype=torch.float32).to(rt.device)
            rt.shard_matrices(buf)
            self.groups.append({'entries':entries, 'buffer':buf, 'shape':shape, 'padded':padded})

    @torch.no_grad()
    def step(self, lr, beta):
        for group in self.groups:
            params = [p for _, p in group['entries']]
            if any(p.grad is None for p in params):
                raise RuntimeError('Missing Muon gradient')
            grads = torch.stack([p.grad for p in params])
            if group['padded'] > len(params):
                grads = torch.cat([grads, torch.zeros((group['padded']-len(params), *group['shape']),
                                                     dtype=grads.dtype, device=grads.device)])
            self.rt.shard_matrices(grads)
            buf = group['buffer']
            buf.mul_(beta).add_(grads)
            updates = orthogonalize(grads + beta * buf)
            updates *= max(1, group['shape'][0]/group['shape'][1])**0.5
            # NS work is partitioned across chips; updates are then gathered for
            # replicated model weights. This replaces upstream rank-local NS + SUM.
            self.rt.replicate(updates)
            for i, p in enumerate(params):
                p.add_(updates[i].to(p.dtype) * (-lr))

    def state_dict(self):
        return [{'names':[name for name, _ in group['entries']],
                 'momentum_buffer':group['buffer']} for group in self.groups]

    def load_state_dict(self, state):
        if len(state) != len(self.groups):
            raise ValueError('Different Muon group count')
        for saved, group in zip(state, self.groups):
            if saved['names'] != [n for n, _ in group['entries']]:
                raise ValueError('Different Muon parameter ordering')
            group['buffer'].copy_(saved['momentum_buffer'].to(self.rt.device))


def make_optimizers(model, rt, kind='muon'):
    if kind not in ('muon', 'adam', 'adamw'):
        raise ValueError('Unknown optimizer: '+kind)
    if getattr(model, 'architecture_id', None) == ARCHITECTURE:
        return make_stock_optimizers(model, rt, kind)
    matrices = [(n, p) for n, p in model.transformer.h.named_parameters() if p.ndim == 2]
    scalars = [p for p in model.transformer.h.parameters() if p.ndim < 2] + [model.skip_weights]
    groups = [
        {'params':[model.transformer.wte.weight], 'peak_lr':0.6, 'role':'embedding'},
        {'params':[model.lm_head.weight], 'peak_lr':0.008, 'role':'head'},
        {'params':scalars, 'peak_lr':0.04, 'role':'scalars'},
    ]
    muon = Muon(matrices, rt) if kind == 'muon' else None
    if kind != 'muon':
        # Same-model control, not an independently tuned Adam(W) speed record.
        # Keep auxiliary updates identical; decoupled decay is confined to the
        # transformer matrices replacing Muon. No 0.1 decay at embedding LR 0.6.
        groups.append({'params':[p for _, p in matrices], 'peak_lr':0.0006,
                       'role':'hidden_matrices', 'weight_decay':0.1 if kind == 'adamw' else 0.})
    for group in groups:
        group['lr'] = group['peak_lr']
    optimizer_class = torch.optim.AdamW if kind == 'adamw' else torch.optim.Adam
    adam = optimizer_class(groups, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0., foreach=False, fused=False,
                            capturable=rt.tpu)
    return muon, adam


def make_stock_optimizers(model, rt, kind):
    """One owner per parameter, including the tied token embedding/output head.

    Muon retains its hidden-matrix recipe; embeddings, biases and LayerNorm use
    auxiliary AdamW at 6e-4. The control uses AdamW at 6e-4 for the entire model.
    The legacy model's distinct 0.6 embedding / 0.008 head LRs cannot apply to a
    tied parameter. Decay applies to AdamW matrices, never biases or LayerNorm.
    """
    matrices = [(n, p) for n, p in model.transformer.h.named_parameters() if p.ndim == 2]
    muon = Muon(matrices, rt) if kind == 'muon' else None
    hidden_ids = {id(p) for _, p in matrices} if muon is not None else set()
    auxiliary = [p for p in model.parameters() if id(p) not in hidden_ids]
    groups = [dict(params=[p for p in auxiliary if p.ndim >= 2], role='matrices',
                   weight_decay=0.0 if kind == 'adam' else 0.1),
              dict(params=[p for p in auxiliary if p.ndim < 2], role='bias_and_layernorm',
                   weight_decay=0.0)]
    for group in groups:
        group.update(lr=0.0006, peak_lr=0.0006)
    cls = torch.optim.Adam if kind == 'adam' else torch.optim.AdamW
    adam = cls(groups, betas=(0.9, 0.95), eps=1e-8,
               foreach=False, fused=False, capturable=rt.tpu)
    return muon, adam


def optimizer_metadata(adam):
    return {'class':'torch.optim.'+type(adam).__name__,
            'groups':[{'role':g['role'], 'peak_lr':g['peak_lr'],
                       'weight_decay':g['weight_decay'], 'betas':list(g['betas']),
                       'eps':g['eps'], 'parameters':sum(p.numel() for p in g['params'])}
                      for g in adam.param_groups]}


def apply_update(muon, adam, rt, step):
    factor = schedule(step)
    if muon is not None:
        muon.step(rt.scalar(0.04 * factor), rt.scalar(momentum(step)))
    for group in adam.param_groups:
        group['lr'] = rt.scalar(group['peak_lr'] * factor)
    adam.step()

"""Frozen configuration, deterministic batches, and durable writes."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    model: str = 'distilbert/distilbert-base-uncased'
    dataset: str = 'stanfordnlp/imdb'
    model_revision: str = '12040accade4e8a0f71eabdb258fecc2e7e948be'
    dataset_revision: str = 'e6281661ce1c48d982bc483cf8a173c1bbeb5d31'
    seed: int = 401
    split_seed: int = 1701
    validation_per_class: int = 1250
    epochs: int = 3
    max_length: int = 128
    per_device_batch: int = 16
    optimizer: str = 'adamw'
    lr: float = 2e-5
    weight_decay: float = 0.01
    momentum: float = 0.9
    nesterov: bool = False
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    classifier_dropout: float = 0.2
    warmup_fraction: float = 0.1
    schedule: str = 'linear'
    grad_clip: float = 1.0
    precision: str = 'bf16'
    checkpoint_steps: int = 50
    # Engineering-only: tiny random model and synthetic examples, never a result.
    synthetic: bool = False

    def validate(self):
        if self.optimizer not in ('adamw', 'adam', 'sgd'):
            raise ValueError('optimizer must be adamw, adam, or sgd')
        if self.schedule not in ('linear', 'cosine', 'constant'):
            raise ValueError('unknown schedule')
        if self.precision not in ('fp32', 'bf16'):
            raise ValueError('unknown precision')
        if not (self.lr > 0 and self.weight_decay >= 0 and self.grad_clip > 0):
            raise ValueError('invalid optimization settings')
        if not (0 <= self.momentum < 1 and 0 <= self.beta1 < 1 and 0 <= self.beta2 < 1 and self.eps > 0):
            raise ValueError('invalid momentum/Adam coefficients')
        if self.nesterov and (self.optimizer != 'sgd' or self.momentum <= 0):
            raise ValueError('Nesterov requires SGD with positive momentum')
        if not all(0 <= p < 1 for p in (self.dropout, self.attention_dropout, self.classifier_dropout, self.warmup_fraction)):
            raise ValueError('invalid dropout/warmup')
        if min(self.epochs, self.per_device_batch, self.checkpoint_steps, self.validation_per_class) < 1 or self.max_length < 8:
            raise ValueError('invalid sizes')
        return self


def read_config(path):
    return Config(**json.loads(Path(path).read_text())).validate()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(str(path), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(cpu_tree(value), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def stratified_split(labels, seed, per_class):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        if len(indices) <= per_class:
            raise ValueError('validation would exhaust class')
        val.extend(indices[:per_class])
        train.extend(indices[per_class:])
    return np.array(sorted(train)), np.array(sorted(val))


def batch_indices(size, batch, world, rank, epoch, seed, shuffle):
    """Each real example exactly once; mask padding, with constant TPU shapes."""
    indices = np.arange(size)
    if shuffle:
        np.random.default_rng(seed + epoch).shuffle(indices)
    width = batch * world
    for start in range(0, size, width):
        real = min(width, size - start)
        block = np.zeros(width, dtype=np.int64)
        mask = np.zeros(width, dtype=np.float32)
        block[:real] = indices[start:start + real]
        mask[:real] = 1
        sl = slice(rank * batch, (rank + 1) * batch)
        yield block[sl], mask[sl], real


def lr_at(step, total, c):
    warm = max(1, int(total * c.warmup_fraction)) if c.warmup_fraction else 0
    if step < warm:
        return c.lr * (step + 1) / warm
    if c.schedule == 'constant':
        return c.lr
    progress = (step - warm) / max(1, total - warm)
    factor = max(0.0, 1 - progress) if c.schedule == 'linear' else .5 * (1 + math.cos(math.pi * progress))
    return c.lr * factor


class TensorSGD(torch.optim.Optimizer):
    """Reference SGD arithmetic with a tensor LR to avoid XLA scalar recompilation."""
    def __init__(self, groups, lr, momentum, nesterov):
        super().__init__(groups, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=0.))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are not used in this experiment')
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                if group['weight_decay']:
                    grad = grad.add(param, alpha=group['weight_decay'])
                momentum = group['momentum']
                if momentum:
                    state = self.state[param]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = grad.detach().clone()
                    else:
                        state['momentum_buffer'].mul_(momentum).add_(grad)
                    buf = state['momentum_buffer']
                    grad = grad.add(buf, alpha=momentum) if group['nesterov'] else buf
                param.add_(-group['lr'] * grad)


def make_optimizer(model, c):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{'params': decay, 'weight_decay': c.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
    device = next(model.parameters()).device
    xla = device.type == 'xla'
    lr = torch.tensor(c.lr, device=device) if xla else c.lr
    if c.optimizer == 'sgd':
        if xla:
            return TensorSGD(groups, lr=lr, momentum=c.momentum, nesterov=c.nesterov)
        return torch.optim.SGD(groups, lr=c.lr, momentum=c.momentum, nesterov=c.nesterov, foreach=False)
    cls = torch.optim.AdamW if c.optimizer == 'adamw' else torch.optim.Adam
    return cls(groups, lr=lr, betas=(c.beta1, c.beta2), eps=c.eps, foreach=False, capturable=xla)


def build_model(c, model_dir=None):
    from transformers import DistilBertConfig, DistilBertForSequenceClassification
    kw = dict(dropout=c.dropout, attention_dropout=c.attention_dropout,
              seq_classif_dropout=c.classifier_dropout, num_labels=2, attn_implementation='eager')
    if c.synthetic:
        return DistilBertForSequenceClassification(DistilBertConfig(
            vocab_size=64, dim=32, hidden_dim=64, n_layers=2, n_heads=4,
            max_position_embeddings=128, **kw))
    return DistilBertForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True, **kw)


def config_dict(c):
    return asdict(c)

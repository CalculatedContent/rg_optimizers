"""Independent, reproducible masked-LM protocol. No torch_xla dependency."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    seed: int = 401
    data_seed: int = 1701
    steps: int = 10000
    warmup_steps: int = 1000
    per_device_batch: int = 32
    sequence_length: int = 128
    optimizer: str = 'adamw'
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-6
    momentum: float = 0.9
    grad_clip: float = 1.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    mask_probability: float = 0.15
    evaluation_mask_probability: float = 0.15
    precision: str = 'bf16'
    compile: bool = True
    checkpoint_steps: int = 100
    measure_steps: int = 500
    train_eval_sequences: int = 2048
    # Only used by local engineering tests, never the TPU launcher.
    synthetic: bool = False

    def validate(self):
        if self.optimizer not in ('adamw', 'adam', 'sgd'):
            raise ValueError('Unknown optimizer')
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError('Invalid warmup')
        if min(self.steps, self.per_device_batch, self.checkpoint_steps,
               self.measure_steps, self.train_eval_sequences) < 1:
            raise ValueError('Positive sizes required')
        if not 8 <= self.sequence_length <= 512:
            raise ValueError('Sequence length must be 8..512')
        if not (self.lr > 0 and self.weight_decay >= 0 and self.grad_clip > 0 and self.eps > 0):
            raise ValueError('Invalid optimizer settings')
        if not all(0 <= x < 1 for x in (self.beta1, self.beta2, self.momentum,
                                      self.dropout, self.attention_dropout)):
            raise ValueError('Invalid probabilities/momenta')
        if not all(0 < p < 1 for p in (self.mask_probability, self.evaluation_mask_probability)) or self.precision not in ('bf16', 'fp32'):
            raise ValueError('Invalid masking/precision')
        return self


def read_config(path):
    return Config(**json.loads(Path(path).read_text())).validate()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sync_dir(path):
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
    tmp.replace(path)
    sync_dir(path.parent)


def save_torch(path, value):
    path = Path(path)
    with path.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def learning_rate(c, step):
    """step is the zero-based update about to execute; warmup reaches peak."""
    if step < c.warmup_steps:
        return c.lr * (step + 1) / c.warmup_steps
    return c.lr * (c.steps - step) / (c.steps - c.warmup_steps)


def batch_indices(order, batch_index, rank, world, local_batch):
    start = batch_index * world * local_batch + rank * local_batch
    positions = np.arange(start, start + local_batch)
    valid = positions < len(order)
    # Padding never contributes a target, gradient, or metric.
    return order[positions % len(order)], valid


def masked_batch(tokens, indices, valid, meta, c, salt, step=0, *, evaluation=False):
    ids = np.array(tokens[indices], dtype=np.int64, copy=True)
    labels = np.full(ids.shape, -100, dtype=np.int64)
    special = np.isin(ids, meta['special_ids'])
    probability = c.evaluation_mask_probability if evaluation else c.mask_probability
    for row, (idx, good) in enumerate(zip(indices, valid)):
        if not good:
            continue
        g = np.random.default_rng(np.random.SeedSequence([c.data_seed, salt, step, int(idx)]))
        eligible = np.flatnonzero(~special[row])
        if not len(eligible):
            raise ValueError('Prepared sequence has no maskable tokens')
        chosen = eligible[g.random(len(eligible)) < probability]
        if not len(chosen):
            chosen = g.choice(eligible, 1)  # documented nonempty-target safeguard
        labels[row, chosen] = ids[row, chosen]
        kind = g.random(len(chosen))
        ids[row, chosen[kind < 0.8]] = meta['mask_id']
        random_positions = chosen[(kind >= 0.8) & (kind < 0.9)]
        ids[row, random_positions] = g.integers(0, meta['vocab_size'], len(random_positions))
    return torch.from_numpy(ids), torch.from_numpy(labels)


def build_model(c, vocab_size):
    from transformers import DistilBertConfig, DistilBertForMaskedLM
    shape = dict(n_layers=2, dim=32, hidden_dim=64, n_heads=4) if c.synthetic else dict(
        n_layers=6, dim=768, hidden_dim=3072, n_heads=12)
    config = DistilBertConfig(vocab_size=vocab_size, max_position_embeddings=512,
                             dropout=c.dropout, attention_dropout=c.attention_dropout,
                             activation='gelu', **shape)
    config._attn_implementation = 'eager'
    # This constructor initializes every parameter randomly. Never load model weights.
    return DistilBertForMaskedLM(config)


def make_optimizer(model, c, device):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [dict(params=decay, weight_decay=c.weight_decay),
              dict(params=no_decay, weight_decay=0.0)]
    lr = torch.tensor(c.lr, device=device) if device.type == 'tpu' else c.lr
    if c.optimizer in ('adamw', 'adam'):
        cls = torch.optim.AdamW if c.optimizer == 'adamw' else torch.optim.Adam
        return cls(groups, lr=lr, betas=(c.beta1, c.beta2), eps=c.eps,
                   foreach=False, capturable=device.type == 'tpu')
    return torch.optim.SGD(groups, lr=lr, momentum=c.momentum, foreach=False)


def encoder_matrices(model):
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()
            if name.startswith('distilbert.transformer.layer.') and p.ndim == 2}


def source_hash():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*'))
                   if p.is_file() and (p.suffix == '.py' or p.name.startswith('requirements'))})

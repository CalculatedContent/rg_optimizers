"""Shared numerical recipe, evaluation, and raw WeightWatcher diagnostics."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from vendor.model import GPT, GPTConfig

SOURCE = Path(__file__).resolve().parent
INPUT_SHA = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
MODEL_SHA = '7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c'
ARCHITECTURE = dict(n_layer=4, n_head=4, n_embd=128, block_size=64, vocab_size=65, dropout=0., bias=False)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def source_hashes():
    return {str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob('*.py'))}

def setup(seed):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = GPT(GPTConfig(**ARCHITECTURE))
    optimizer = model.configure_optimizers(.1, .001, (.9, .99), 'cpu')
    rng = torch.Generator().manual_seed(seed + 1)
    return model, optimizer, rng

def load_data(path):
    assert sha(path) == INPUT_SHA, 'Dataset SHA256 mismatch'
    assert sha(SOURCE / 'vendor/model.py') == MODEL_SHA, 'Upstream model changed'
    text = Path(path).read_text()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    assert len(text) == 1115394 and len(chars) == 65
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    a, b = int(.90 * len(ids)), int(.95 * len(ids))
    return dict(train=ids[:a], validation=ids[a:b], test=ids[b:])

def batch(data, rng):
    ix = torch.randint(len(data) - 64, (12,), generator=rng)
    return (torch.stack([torch.from_numpy(data[i:i+64].astype(np.int64)) for i in ix]),
            torch.stack([torch.from_numpy(data[i+1:i+65].astype(np.int64)) for i in ix]))

def learning_rate(step, continuation_lr=.0001):
    if step < 100:
        return .001 * (step + 1) / 101
    if step < 2000:
        return .0001 + .5 * (1 + math.cos(math.pi * (step - 100) / 1900)) * .0009
    return continuation_lr

def update(model, optimizer, data, rng, step, continuation_lr=.0001):
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(step, continuation_lr)
    x, y = batch(data, rng)
    _, loss = model(x, y)
    assert torch.isfinite(loss), 'Nonfinite loss'
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def sampled_loss(model, data, seed=10001):
    mode = model.training; model.eval()
    rng = torch.Generator().manual_seed(seed)
    value = sum(model(*batch(data, rng))[1].item() for _ in range(20)) / 20
    model.train(mode)
    return value

@torch.no_grad()
def full_eval(model, data):
    mode = model.training; model.eval()
    total_loss, correct, count = 0., 0, 0
    for start in range(0, len(data) - 1, 12 * 64):
        length = min(12 * 64, len(data) - 1 - start)
        whole = length // 64
        if whole:
            size = whole * 64
            x = torch.from_numpy(data[start:start+size].astype(np.int64)).view(whole, 64)
            y = torch.from_numpy(data[start+1:start+size+1].astype(np.int64)).view(whole, 64)
            logits, loss = model(x, y)
            total_loss += loss.item() * size
            correct += (logits.argmax(-1) == y).sum().item(); count += size
        if length % 64:
            offset, tail = start + whole * 64, length % 64
            x = torch.from_numpy(data[offset:offset+tail].astype(np.int64))[None, :]
            y = torch.from_numpy(data[offset+1:offset+tail+1].astype(np.int64))[None, :]
            logits, loss = model(x, y)
            total_loss += loss.item() * tail
            correct += (logits.argmax(-1) == y).sum().item(); count += tail
    assert count == len(data) - 1
    model.train(mode); loss = total_loss / count
    return dict(loss_nats_per_character=loss, perplexity=math.exp(loss),
                next_character_accuracy=correct/count, evaluated_targets=count)

def rng_state(rng):
    return dict(torch_rng=torch.get_rng_state(), train_rng=rng.get_state(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate())

def restore_rng(state, rng):
    torch.set_rng_state(state['torch_rng']); rng.set_state(state['train_rng'])
    np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])

def spectrum(state):
    assert importlib.metadata.version('weightwatcher') == '0.7.7'
    import weightwatcher as ww
    start = time.perf_counter(); rows = []
    saved = rng_state(torch.Generator())
    try:
        for name, w in state.items():
            if w.ndim != 2 or name == 'lm_head.weight': continue
            aliases = [name]
            if name == 'transformer.wte.weight':
                assert torch.equal(w, state['lm_head.weight'])
                aliases.append('lm_head.weight')
            m = torch.nn.Sequential(torch.nn.Linear(w.shape[1], w.shape[0], bias=False))
            with torch.no_grad(): m[0].weight.copy_(w)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                f = ww.WeightWatcher(m).analyze()
            rows.append(dict(matrix=name, aliases=aliases, shape=list(w.shape),
                             fits=json.loads(f.to_json(orient='records'))))
    finally:
        torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    values = [r['fits'][0]['alpha'] for r in rows if len(r['fits']) == 1
              and r['fits'][0].get('status') == 'success'
              and isinstance(r['fits'][0].get('alpha'), (float, int))
              and math.isfinite(r['fits'][0]['alpha'])]
    assert len(rows) == 18
    return dict(unique_matrices=18, successful_fits=len(values),
                mean_alpha=sum(values)/18 if len(values)==18 else None,
                mean_successful_fits=sum(values)/len(values) if values else None,
                below_two=sum(a<2 for a in values), analysis_seconds=time.perf_counter()-start,
                method='Unmodified WW0.7.7 analyze per matrix; packed QKV; tied output/token matrix once; LayerNorm vectors ineligible.',
                matrices=rows)

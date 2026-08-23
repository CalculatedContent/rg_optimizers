from __future__ import annotations

from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_tiny_muonclip_training_writes_qk_diagnostics(tmp_path) -> None:
    code = r'''
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from rg_nanogpt_one_head.muonclip import install_muonclip_extension
install_muonclip_extension()

from rg_nanogpt_one_head.config import load_config
from rg_nanogpt_one_head.data import TOKEN_DTYPE
import rg_nanogpt_one_head.train_loop as train_loop
import rg_nanogpt_one_head.run_utils as run_utils
from rg_nanogpt_one_head.training import run_one

root = Path(sys.argv[1])
cfg = deepcopy(load_config(Path.cwd() / 'configs' / 'muonclip_reference.yaml'))
cfg['dataset'].update({
    'name': 'unit/fineweb',
    'config': 'unit',
    'revision': 'unit-revision',
    'train_tokens': 2048,
    'val_tokens': 512,
    'test_tokens': 512,
})
cfg['model'].update({
    'vocab_size': 64,
    'block_size': 8,
    'n_layer': 1,
    'n_head': 1,
    'n_embd': 16,
})
cfg['training'].update({
    'seeds': [13],
    'batch_size': 2,
    'grad_accum_steps': 2,
    'target_epochs': 0.04,
    'epoch_interval': 1.0,
    'eval_interval_steps': 1,
    'eval_batches': 1,
    'checkpoint_interval_steps': 1,
})
cfg['evaluation'].update({
    'bleu_examples': 2,
    'bleu_prompt_tokens': 3,
    'bleu_continuation_tokens': 2,
    'bleu_batch_size': 2,
})
cfg['optimizer_profiles']['muon_clip']['qk_diagnostics_interval'] = 1

data_root = root / 'data'
results_root = root / 'results'
data_root.mkdir(parents=True)
rng = np.random.default_rng(7)
splits = {
    'train': int(cfg['dataset']['train_tokens']),
    'val': int(cfg['dataset']['val_tokens']),
    'test': int(cfg['dataset']['test_tokens']),
}
files = {}
for split, size in splits.items():
    path = data_root / f'{split}.bin'
    rng.integers(0, cfg['model']['vocab_size'], size=size, dtype=np.uint16).tofile(path)
    files[split] = {
        'path': path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'bytes': path.stat().st_size,
    }
(data_root / 'meta.json').write_text(json.dumps({
    'schema_version': 2,
    'tokenizer': 'gpt2',
    'vocab_size': cfg['model']['vocab_size'],
    'dtype': TOKEN_DTYPE.name,
    'splits': splits,
    'document_disjoint_splits': True,
    'dataset_name': cfg['dataset']['name'],
    'dataset_config': cfg['dataset']['config'],
    'dataset_split': cfg['dataset'].get('split', 'train'),
    'dataset_revision': cfg['dataset']['revision'],
    'eot_token': 0,
    'files': files,
}), encoding='utf-8')

train_loop.evaluate_bleu = lambda *args, **kwargs: {'bleu': 0.0}
run_utils.evaluate_bleu = lambda *args, **kwargs: {'bleu': 0.0}
train_loop.run_weightwatcher = lambda *args, **kwargs: {
    'alpha_median': 2.0,
    'rand_distance_median': 0.1,
    'ERG_gap_median': 0.0,
    'num_traps_mean': 0.0,
}

run_dir = run_one(
    cfg=cfg,
    data_root=data_root,
    results_root=results_root,
    optimizer_name='muon_clip',
    seed=13,
    device='cpu',
    resume=True,
    progress=False,
)
assert (run_dir / 'run_complete.json').is_file()
assert (run_dir / 'checkpoint_final.pt').is_file()
qk = pd.read_csv(run_dir / 'muonclip_qk.csv')
assert len(qk) >= 1
assert qk['head_observations'].gt(0).all()
assert qk['max_logit'].notna().all()
print(run_dir)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=EXPERIMENT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "muon_clip/seed_13" in completed.stdout

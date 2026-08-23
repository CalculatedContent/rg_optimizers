from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.muonclip_walk import _validate_walk_profile


def test_walk_capture_profile_is_hard_capped_at_twenty() -> None:
    _validate_walk_profile(
        {
            "walk_capture_steps": 20,
            "walk_capture_root": "/tmp/test-walk",
            "walk_save_full_model": True,
            "walk_save_weightwatcher": True,
            "walk_save_optimizer_tensors": True,
        }
    )
    with pytest.raises(ValueError, match="between 0 and 20"):
        _validate_walk_profile(
            {
                "walk_capture_steps": 21,
                "walk_capture_root": "/tmp/test-walk",
            }
        )


def test_committed_walk_config_is_one_epoch_and_first_twenty_steps() -> None:
    cfg = yaml.safe_load(
        (EXPERIMENT_ROOT / "configs" / "muonclip_walk20.yaml").read_text()
    )
    profile = cfg["optimizer_profiles"]["muon_clip"]
    assert cfg["training"]["target_epochs"] == 1.0
    assert cfg["training"]["seeds"] == [1337]
    assert profile["walk_capture_steps"] == 20
    assert profile["walk_capture_root"].startswith("/tmp/")
    assert profile["walk_save_full_model"] is True
    assert profile["walk_save_weightwatcher"] is True
    assert profile["walk_save_optimizer_tensors"] is True


def test_tiny_walk_run_writes_loadable_append_only_artifacts(tmp_path) -> None:
    code = r'''
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import weightwatcher as ww

from rg_nanogpt_one_head.muonclip_walk import (
    install_muonclip_walk_extension,
    load_full_model_checkpoint,
    load_weightwatcher_checkpoint,
)
install_muonclip_walk_extension()

from rg_nanogpt_one_head.config import load_config
from rg_nanogpt_one_head.data import TOKEN_DTYPE
import rg_nanogpt_one_head.train_loop as train_loop
import rg_nanogpt_one_head.run_utils as run_utils
from rg_nanogpt_one_head.training import run_one

root = Path(sys.argv[1])
cfg = deepcopy(load_config(Path.cwd() / 'configs' / 'muonclip_walk20.yaml'))
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
profile = cfg['optimizer_profiles']['muon_clip']
profile['walk_capture_steps'] = 2
profile['walk_capture_root'] = str(root / 'captures')
profile['qk_diagnostics_interval'] = 1

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

pointer = json.loads((run_dir / 'muonclip_walk_location.json').read_text())
capture_dir = Path(pointer['capture_dir'])
assert capture_dir.is_dir()

model_paths = sorted((capture_dir / 'model_checkpoints').glob('*.pt'))
ww_paths = sorted((capture_dir / 'weightwatcher_checkpoints').glob('*.pt'))
trace_paths = sorted((capture_dir / 'step_traces').glob('*.pt'))
assert [path.name for path in model_paths] == [
    'model_step_0000000.pt',
    'model_step_0000001.pt',
    'model_step_0000002.pt',
]
assert [path.name for path in ww_paths] == [
    'ww_step_0000000.pt',
    'ww_step_0000001.pt',
    'ww_step_0000002.pt',
]
assert [path.name for path in trace_paths] == [
    'step_0000001.pt',
    'step_0000002.pt',
]

model, model_payload = load_full_model_checkpoint(model_paths[-1])
assert model_payload['step'] == 2
assert model.cfg.n_embd == 16

holder, ww_payload = load_weightwatcher_checkpoint(ww_paths[-1])
assert ww_payload['step'] == 2
assert len(ww_payload['matrices']) == 6
watcher = ww.WeightWatcher(model=holder)
details = watcher.analyze(plot=False, min_evals=5)
assert len(details) == 6

first = torch.load(trace_paths[0], map_location='cpu', weights_only=False)
second = torch.load(trace_paths[1], map_location='cpu', weights_only=False)
assert first['purpose'] == 'muonclip_optimizer_step_trace'
assert len(first['matrices']) == 6
for name in first['matrices']:
    left = first['matrices'][name]
    right = second['matrices'][name]
    assert torch.equal(left['weight_after'], right['weight_before'])
    for key in (
        'gradient_post_clip',
        'momentum_before',
        'momentum_after',
        'rms_matched_orthogonal_update',
        'parameter_delta',
        'spectra_before',
        'spectra_after',
        'spectra_delta',
    ):
        assert key in left

checkpoints = pd.read_csv(capture_dir / 'checkpoint_index.csv')
steps = pd.read_csv(capture_dir / 'step_trajectory.csv')
matrices = pd.read_csv(capture_dir / 'matrix_trajectory.csv')
assert checkpoints['step'].tolist() == [0, 1, 2]
assert steps['step'].tolist() == [1, 2]
assert len(matrices) == 12
assert set(matrices['matrix_name']) == {
    'L00_W_Q', 'L00_W_K', 'L00_W_V',
    'L00_W_O', 'L00_W_MLP_IN', 'L00_W_MLP_OUT',
}
assert (capture_dir / 'walk_manifest.json').is_file()
assert (capture_dir / 'README.md').is_file()
print(capture_dir)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=EXPERIMENT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise AssertionError(
            "MuonClip walk subprocess failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    assert "captures" in completed.stdout

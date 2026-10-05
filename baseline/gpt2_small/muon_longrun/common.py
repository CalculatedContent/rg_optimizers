"""Fixed long-run plan; reuse the validated speedrun implementation."""
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SHORT = HERE.parent/'muon_speedrun'
sys.path.append(str(SHORT))
STEPS = 25000
BATCH_TOKENS = 524288
VAL_TOKENS = 10485760
MICROBATCH = 64
CONTEXT = 1024
REFERENCE_NAME = 'muon-speedrun-muon-20261005-030026'
CACHE = Path('/mnt/disks/rg-data/benchmark-fineweb10B-889765ea')
PERMANENT = {0, 3000, 10000, 17500, STEPS}
EARLY_WW = {0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000}


@dataclass(frozen=True)
class Schedule:
    total_steps: int = STEPS
    warmdown_steps: int = 7500
    warmup_steps: int = 0

    @property
    def cooldown_start(self):
        return self.total_steps-self.warmdown_steps

    def factor(self, update_index):
        return min(1., max(0., (self.total_steps-update_index)/self.warmdown_steps))

    def phase(self, update_index):
        return 'cooldown' if update_index >= self.cooldown_start else 'main'

    def values(self, update_index):
        factor = self.factor(update_index)
        return dict(lr_factor=factor, scheduler_phase=self.phase(update_index),
                    muon_lr=.04*factor, adam_embedding_lr=.6*factor,
                    adam_head_lr=.008*factor, adam_scalar_lr=.04*factor)


def ww_due(step, sparse=False):
    interval = 2000 if sparse else 1000
    return step in EARLY_WW or step in PERMANENT or (step > 3000 and step % interval == 0)


def val_due(step, sparse=False):
    return step % 500 == 0 or ww_due(step, sparse)


def adapt_tracking(step, foreground_fraction, median_seconds, reference_seconds, tracker_state):
    """Conservative cadence reduction; observed host slowdown is NOT causal proof."""
    if step < 3000: return False
    return foreground_fraction > .10 or (
        median_seconds > reference_seconds*1.10 and tracker_state.get('status')=='measuring')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)
    sync_dir(path.parent)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def verify_reference(reference):
    manifest=json.loads((reference/'manifest.json').read_text())
    val=json.loads((reference/'latest_validation.json').read_text())
    expected={'recipe':'2024-11-10_UNetDoubleLr','optimizer':'muon','seed':1337,
        'config':{'vocab_size':50304,'n_layer':12,'n_head':6,'n_embd':768},
        'batch_tokens':BATCH_TOKENS,'global_microbatch_sequences':64,'accumulation':8,
        'muon_lr':.04,'adam_embedding_lr':.6,'adam_head_lr':.008,'adam_scalar_lr':.04,
        'weight_decay':0,'gradient_clipping':False,'attention':'flash','warmup_updates':0,
        'data_repo':'kjj0/fineweb10B-gpt2','data_revision':'889765ea1f903759787add96995d81171b632d0c'}
    for key,value in expected.items():
        if manifest.get(key)!=value:
            raise RuntimeError(f'Reference setting differs: {key}: {manifest.get(key)!r}')
    if not (val['step']==3000 and val['full_benchmark_evaluation']
            and val['evaluation_tokens']==VAL_TOKENS and 3 < val['val_nll'] < 3.3):
        raise RuntimeError('Reference full 3,000-step validation is not available/healthy')
    hashes={}
    for relative in ('muon_speedrun/model.py','muon_speedrun/optim.py','muon_speedrun/runtime.py',
                     'muon_speedrun/data.py','muon_speedrun/pallas_dependencies.py',
                     'speedrun30/train.py','speedrun30/data_manifest.json'):
        old=reference/'repo/baseline/gpt2_small'/relative
        new=HERE.parent/relative
        if sha(old)!=sha(new):
            raise RuntimeError('Validated implementation changed: '+relative)
        hashes[relative]=sha(new)
    return {'reference_run':str(reference),'manifest':manifest,'final_validation':val,
            'unchanged_source_sha256':hashes}

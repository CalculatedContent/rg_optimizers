#!/usr/bin/env bash
# Invoked once by systemd, with Restart=no. Scientific process never resumes.
set -Eeuo pipefail
BASE=/mnt/disks/rg-data/continuous8
REPO="$BASE/repo"
EXP="$REPO/baseline/nanogpt_one_head"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PJRT_DEVICE=TPU TPU_ACCELERATOR_TYPE=v5litepod-8
export RG_TPU_PERSISTENT_ROOT=/mnt/disks/rg-data
export RG_CONTINUOUS_RUN_LOG="$BASE/run.log"
export HF_HOME="$BASE/hf-cache"
unset XLA_USE_BF16 XLA_DOWNCAST_BF16 TPU_VISIBLE_CHIPS
cd "$EXP"
python3 -m venv "$BASE/venv"
PY="$BASE/venv/bin/python"
"$PY" -m pip install --upgrade 'pip==25.0.1' 'setuptools==75.8.2' 'wheel==0.45.1'
"$PY" -m pip install 'torch==2.6.0' 'torch_xla[tpu]==2.6.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
"$PY" -m pip install -r continuous8/requirements.txt
"$PY" -m pip install --no-deps -e .
"$PY" -m pip freeze > "$BASE/environment.lock.txt"
git rev-parse HEAD > "$BASE/source_commit.txt"
# Upload credentials test before data download or training.
"$PY" - <<'PY'
import os
from pathlib import Path
from rg_nanogpt_one_head.continuous_support import CloudPublisher
p=CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI'])
base=Path('/mnt/disks/rg-data/continuous8')
p.claim({'commit':(base/'source_commit.txt').read_text().strip(),
         'seed':int(os.environ['RG_CONTINUOUS_SEED']), 'automatic_restart':False})
for name in ('environment.lock.txt','source_commit.txt'):
    p.file(base/name, name)
p.json({'status':'preflight'}, 'SETUP_STATUS.json')
PY
# Download the one shared CPU-prepared corpus and verify every file hash.
"$PY" - <<'PY'
import json, os
from pathlib import Path
import yaml
from rg_nanogpt_one_head.continuous_support import CloudPublisher, sha_file
from rg_nanogpt_one_head.data import validate_prepared_data
source=CloudPublisher(os.environ['RG_CONTINUOUS_DATA_URI'])
manifest=json.loads(source.bucket.blob(source.prefix+'/COMPLETE.json').download_as_text())
cfg=yaml.safe_load(Path('configs/muonclip_continuous8.yaml').read_text())
if manifest['dataset'] != cfg['dataset']:
    raise RuntimeError('Shared cloud corpus does not match experiment configuration')
base=Path('/mnt/disks/rg-data/continuous8/data')
base.mkdir(exist_ok=True)
for name in ('train.bin','val.bin','test.bin','meta.json'):
    path=base/name
    source.bucket.blob(source.prefix+'/'+name).download_to_filename(str(path), checksum='crc32c', timeout=600)
    entry=manifest['files'][name]
    if path.stat().st_size != entry['bytes'] or sha_file(path) != entry['sha256']:
        raise RuntimeError('Shared data checksum failed: '+name)
validate_prepared_data(base,cfg)
p=CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI'])
p.json({'uri':os.environ['RG_CONTINUOUS_DATA_URI'], 'manifest':manifest}, 'DATA_SOURCE.json')
print('Shared data downloaded and SHA256 verified.', flush=True)
PY
# Tiny, independent test: global gradients, clipping, metrics, optimizer + RNG restore.
# Also measures the full proposed model shape before the long run.
"$PY" -m rg_nanogpt_one_head.tpu_spmd_check --backend tpu --chips 8 \
  --benchmark-config configs/muonclip_continuous8.yaml --benchmark-steps 10 \
  --output "$BASE/preflight.json"
"$PY" - <<'PY'
import json, os, time
from pathlib import Path
from rg_nanogpt_one_head.continuous_support import CloudPublisher
p=CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI'])
path=Path('/mnt/disks/rg-data/continuous8/preflight.json')
p.file(path, 'preflight.json')
report=json.loads(path.read_text())
remaining=max(0,float(os.environ['RG_CONTINUOUS_DEADLINE_UNIX'])-time.time())
rate=report['benchmark']['tokens_per_second']
projection={'remaining_training_window_seconds':remaining,
            'benchmark_tokens_per_second':rate,
            'optimistic_token_presentations':int(rate*remaining),
            'optimistic_updates':int(rate*remaining/report['benchmark']['global_tokens_per_update']),
            'caveat':'Synthetic training throughput; excludes evaluation, spectra, initialization and checkpoint upload overhead.'}
p.json(projection, 'BENCHMARK_PROJECTION.json')
print('BENCHMARK PROJECTION:', json.dumps(projection), flush=True)
p.json({'status':'training'}, 'SETUP_STATUS.json')
PY
# No resilient supervisor, --resume, or automatic retry anywhere in this path.
"$PY" -u -m rg_nanogpt_one_head.continuous_run \
  --config configs/muonclip_continuous8.yaml --data-root "$BASE/data" \
  --results-root "$BASE/results" --device tpu --seed "$RG_CONTINUOUS_SEED" \
  --deadline-unix "$RG_CONTINUOUS_DEADLINE_UNIX"

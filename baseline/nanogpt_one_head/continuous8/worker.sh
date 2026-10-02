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
# Upload credentials test before corpus preparation or training.
"$PY" - <<'PY'
import os
from pathlib import Path
from rg_nanogpt_one_head.continuous_support import CloudPublisher
p=CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI'])
base=Path('/mnt/disks/rg-data/continuous8')
p.claim({'commit':(base/'source_commit.txt').read_text().strip(), 'automatic_restart':False})
for name in ('environment.lock.txt','source_commit.txt'):
    p.file(base/name, name)
p.json({'status':'preflight'}, 'SETUP_STATUS.json')
PY
# Tiny, independent test: global gradients, clipping, metrics, optimizer + RNG restore.
# Also measures the full proposed model shape before the long run.
"$PY" -m rg_nanogpt_one_head.tpu_spmd_check --backend tpu --chips 8 \
  --benchmark-config configs/muonclip_continuous8.yaml --benchmark-steps 10 \
  --output "$BASE/preflight.json"
"$PY" - <<'PY'
import os
from rg_nanogpt_one_head.continuous_support import CloudPublisher
CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI']).file('/mnt/disks/rg-data/continuous8/preflight.json', 'preflight.json')
PY
# Pinned dataset revision; exact document-disjoint split sizes and content hashes.
"$PY" -m rg_nanogpt_one_head.data --config configs/muonclip_continuous8.yaml \
  --output-dir "$BASE/data"
"$PY" - <<'PY'
import os
from pathlib import Path
from rg_nanogpt_one_head.continuous_support import CloudPublisher, sha_file
p=CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI'])
base=Path('/mnt/disks/rg-data/continuous8/data')
receipts={}
for path in sorted(base.iterdir()):
    if path.is_file():
        receipts[path.name]=p.file(path, 'data/'+path.name)
        receipts[path.name]['sha256']=sha_file(path)
p.json(receipts, 'data/COMPLETE.json')
p.json({'status':'training'}, 'SETUP_STATUS.json')
PY
# No resilient supervisor, --resume, or automatic retry anywhere in this path.
"$PY" -u -m rg_nanogpt_one_head.continuous_run \
  --config configs/muonclip_continuous8.yaml --data-root "$BASE/data" \
  --results-root "$BASE/results" --device tpu

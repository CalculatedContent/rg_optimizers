#!/usr/bin/env bash
# Invoked by systemd, with Restart=no. A failed dependency install may be retried
# manually before training starts. The scientific process never resumes.
set -Eeuo pipefail
BASE="${RG_CONTINUOUS_BASE:-/mnt/disks/rg-data/continuous8}"
SHARED="${RG_CONTINUOUS_SHARED_BASE:-$BASE}"
export RG_CONTINUOUS_BASE="$BASE"
export RG_CONTINUOUS_DATA_ROOT="$SHARED/data"
REPO="$BASE/repo"
EXP="$REPO/baseline/nanogpt_one_head"
CONFIG="${RG_CONTINUOUS_CONFIG:-$EXP/configs/muonclip_continuous8.yaml}"
export RG_CONTINUOUS_CONFIG="$CONFIG"
# Serialize manual setup retries and reject all scientific-run reuse before pip.
exec 9>"$BASE/worker.lock"
flock -n 9 || { echo 'Another worker is running; refusing duplicate setup.'; exit 1; }
if [ -e "$BASE/results/CONTINUOUS_STARTED.json" ]; then
  echo 'Scientific training already started; refusing setup or training restart.'
  exit 1
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PJRT_DEVICE=TPU TPU_ACCELERATOR_TYPE=v5litepod-8
export RG_TPU_PERSISTENT_ROOT=/mnt/disks/rg-data
export RG_CONTINUOUS_RUN_LOG="$BASE/run.log"
export HF_HOME="$SHARED/hf-cache"
export PIP_CACHE_DIR="$SHARED/pip-cache"
unset XLA_USE_BF16 XLA_DOWNCAST_BF16 TPU_VISIBLE_CHIPS
cd "$EXP"
python3 -m venv "$SHARED/venv"
PY="$SHARED/venv/bin/python"
finish_worker() {
  local result=$?
  trap - EXIT
  set +e
  echo "Worker exit code: $result"
  "$PY" - "$result" <<'PYCODE'
import json, os, sys, time
from pathlib import Path
code=int(sys.argv[1])
status={'status':'finished' if code in (0,75) else 'failed','exit_code':code,'ended_unix':time.time()}
base=Path(os.environ['RG_CONTINUOUS_BASE'])
(base/'WORKER_STATUS.json').write_text(json.dumps(status)+'\n')
try:
    from rg_nanogpt_one_head.continuous_support import CloudPublisher
    p=CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI'])
    p.json(status,'WORKER_STATUS.json')
    p.snapshot_text_file(base/'run.log','run.log')
except Exception as exc:
    print(f'Cloud exit report unavailable: {type(exc).__name__}: {exc}', flush=True)
    print(f'Local status and log retained in {base}', flush=True)
PYCODE
  exit "$result"
}
trap finish_worker EXIT
# Only package installation is retried. Preflight, data preparation and scientific
# training remain single-attempt, with the original allocation deadline.
source "$EXP/continuous8/install_dependencies.sh"
"$PY" -m pip freeze > "$BASE/environment.lock.txt"
git rev-parse HEAD > "$BASE/source_commit.txt"
# Upload credentials test before data download or training.
"$PY" - <<'PY'
import os
from pathlib import Path
from rg_nanogpt_one_head.continuous_support import CloudPublisher
p=CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI'])
base=Path(os.environ['RG_CONTINUOUS_BASE'])
p.claim({'commit':(base/'source_commit.txt').read_text().strip(),
         'seed':int(os.environ['RG_CONTINUOUS_SEED']), 'automatic_restart':False})
for name in ('environment.lock.txt','source_commit.txt'):
    p.file(base/name, name)
p.json({'status':'preflight'}, 'SETUP_STATUS.json')
PY
# Tiny, independent test: global gradients, clipping, metrics, optimizer + RNG restore.
# Also measures the full proposed model shape before the long run.
"$PY" -m rg_nanogpt_one_head.tpu_spmd_check --backend tpu --chips 8 \
  --benchmark-config "$CONFIG" --benchmark-steps 10 \
  --output "$BASE/preflight.json"
"$PY" - <<'PYCODE'
import os
from pathlib import Path
from rg_nanogpt_one_head.continuous_support import CloudPublisher
CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI']).file(Path(os.environ['RG_CONTINUOUS_BASE'])/'preflight.json','preflight.json')
PYCODE
# Dataset preparation runs on the TPU VM CPU and durable disk, not Cloud Shell.
"$PY" -u continuous8/prepare_tpu_data.py
"$PY" - <<'PY'
import json, os, time
from pathlib import Path
from rg_nanogpt_one_head.continuous_support import CloudPublisher
p=CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI'])
path=Path(os.environ['RG_CONTINUOUS_BASE'])/'preflight.json'
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
# No resilient supervisor, --resume, or automatic retry of scientific training.
"$PY" -u -m rg_nanogpt_one_head.continuous_run \
  --config "$CONFIG" --data-root "$RG_CONTINUOUS_DATA_ROOT" \
  --results-root "$BASE/results" --device tpu --seed "$RG_CONTINUOUS_SEED" \
  --deadline-unix "$RG_CONTINUOUS_DEADLINE_UNIX"

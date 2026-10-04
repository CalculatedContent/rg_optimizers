#!/usr/bin/env bash
# Four guarded updates only; caller supplies verified allocation expiry minus backup reserve.
set -euo pipefail
old=$1
deadline=$2
base=$(cd "$(dirname "$0")/.." && pwd)
source "$base/scripts/tpu_environment.sh"
python=/mnt/disks/rg-data/continuous8/venv/bin/python
export PYTHONPATH="$base/src:$base/../nanogpt_one_head/src"
"$python" -c 'import sys,time; assert float(sys.argv[1])>time.time()+60,"Too little allocation time; diagnostic not started"' "$deadline"
root="${3:-$old/diagnostic-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$root"
test ! -e "$root/config.yaml"
export RG_GPT2_SOURCE_COMMIT=$(git -C "$base" rev-parse HEAD)
export PT_XLA_DEBUG_LEVEL=2
"$python" - "$old" "$root" <<'PY'
import sys,yaml
from pathlib import Path
old,root=map(Path,sys.argv[1:])
cfg=yaml.safe_load((old/'configs/adamw.yaml').read_text())
cfg.update(run_id=root.name,validation_gradient_checks=True,validation_tensor_checks=True,
           metrics_interval=1,benchmark_sync_every_step=True)
cfg['ww']['enabled']=False
(root/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
backup() {
  rc=$?
  trap - EXIT
  sync
  "$python" "$base/scripts/backup.py" "$root" || exit 1
  exit "$rc"
}
trap backup EXIT
"$python" -u "$base/scripts/run_adamw_probe.py" "$root" "$deadline" 2>&1 | tee "$root/diagnostic.log"

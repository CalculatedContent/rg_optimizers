#!/usr/bin/env bash
set -Eeuo pipefail
root=$1
deadline=$2
base=$(cd "$(dirname "$0")/.." && pwd)
python=/mnt/disks/rg-data/continuous8/venv/bin/python
export PYTHONPATH="$base/src"
finish() {
  rc=$?
  trap - EXIT
  echo "Validation worker exit code: $rc"
  sync
  if ! "$python" "$base/gpt2small/backup.py" "$root"; then
    echo "Cloud backup FAILED; files remain on the preserved disk: $root" >&2
    exit 1
  fi
  exit "$rc"
}
trap finish EXIT
source "$base/gpt2small/tpu_environment.sh"
mountpoint -q /mnt/disks/rg-data
test -x "$python"
# Reuse installed dependencies and the verified corpus. Do not reinstall or redownload.
"$python" -c 'import sys,torch,torch_xla,weightwatcher,yaml; print("Dependencies loaded:",sys.version,torch.__version__,torch_xla.__version__,flush=True)'
"$python" - "$root" <<'PY'
import json,sys
from pathlib import Path
from rg_nanogpt_one_head.continuous_support import CloudPublisher
root=Path(sys.argv[1])
probe=root/'backup_probe.json'; probe.write_text(json.dumps({'run':root.name,'phase':'before_training'}))
sink=CloudPublisher('gs://tpu-builders-504820-ww-continuous8/gpt2small/'+root.name)
receipt=sink.file(probe,probe.name)
(root/'backup_probe_receipt.json').write_text(json.dumps(receipt,indent=2))
print('Cloud upload permission and checksum verification passed before training.',flush=True)
PY
"$python" -u "$base/gpt2small/validate.py" --root "$root" \
  --data /mnt/disks/rg-data/continuous8/data --deadline "$deadline" 2>&1 | tee "$root/validation.log"
echo 'Short validation completed. No long experiment launched.'

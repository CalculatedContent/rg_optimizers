#!/usr/bin/env bash
# One fresh process, on the current allocation and preserved corpus.
set -Eeuo pipefail
root=$1
deadline=$2
service_deadline=$3
base=$(cd "$(dirname "$0")/.." && pwd)
python=/mnt/disks/rg-data/continuous8/venv/bin/python
export PYTHONPATH="$base/src:$base/../nanogpt_one_head/src"
export RG_GPT2_SOURCE_COMMIT=$(git -C "$base" rev-parse HEAD)
export RG_GPT2_GCS_URI="gs://tpu-builders-504820-ww-continuous8/gpt2small/$(basename "$root")"
export RG_GPT2_RUN_LOG="$root/run.log"
source "$base/scripts/tpu_environment.sh"
mountpoint -q /mnt/disks/rg-data
test -x "$python"
finish() {
  rc=$?
  trap - EXIT
  echo "MuonClip worker exit code: $rc; retaining all persistent-disk evidence."
  sync
  "$python" "$base/scripts/backup.py" "$root" || exit 1
  exit "$rc"
}
trap finish EXIT
"$python" -u "$base/scripts/supervise_muonclip.py" "$root" "$deadline" "$service_deadline"

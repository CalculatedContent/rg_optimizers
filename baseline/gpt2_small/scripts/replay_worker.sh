#!/usr/bin/env bash
set -Eeuo pipefail
root=$1
deadline=$2
checkpoint=$3
base=$(cd "$(dirname "$0")/.." && pwd)
python=/mnt/disks/rg-data/continuous8/venv/bin/python
export PYTHONPATH="$base/src:$base/../nanogpt_one_head/src"
export RG_GPT2_SOURCE_COMMIT=$(git -C "$base" rev-parse HEAD)
source "$base/scripts/tpu_environment.sh"
finish() {
  rc=$?
  trap - EXIT
  sync
  "$python" "$base/scripts/backup.py" "$root" || exit 1
  exit "$rc"
}
trap finish EXIT
"$python" -u "$base/scripts/supervise_replay.py" "$root" "$deadline" "$checkpoint"

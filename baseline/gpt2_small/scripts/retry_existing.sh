#!/usr/bin/env bash
# Retry validation ONLY. Never invokes old-run cleanup or downloads data.
set -euo pipefail
old=$1
base=$(cd "$(dirname "$0")/.." && pwd)
source "$base/scripts/tpu_environment.sh"
mountpoint -q /mnt/disks/rg-data
test -f "$old/old_allocation.json"
test -f /mnt/disks/rg-data/continuous8/data/meta.json
root="${old}-retry-$(date -u +%Y%m%d-%H%M%S)"
sudo mkdir "$root"
sudo chown "$(id -u):$(id -g)" "$root"
cp "$old/old_allocation.json" "$root/old_allocation.json"
python=/mnt/disks/rg-data/continuous8/venv/bin/python
export PYTHONPATH="$base/src:$base/../nanogpt_one_head/src"
git -C "$base" rev-parse HEAD | tee "$root/commit.txt"
deadline=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["deadline_unix"])' "$root/old_allocation.json")
backup() {
  rc=$?
  trap - EXIT
  sync
  "$python" "$base/scripts/backup.py" "$root" || exit 1
  exit "$rc"
}
trap backup EXIT
"$python" -u "$base/scripts/validate.py" --root "$root" --data /mnt/disks/rg-data/continuous8/data --deadline "$deadline" 2>&1 | tee "$root/validation.log"

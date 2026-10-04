#!/usr/bin/env bash
set -euo pipefail
commit=$1
root=$2
mountpoint -q /mnt/disks/rg-data
shared=/mnt/disks/rg-data/continuous8
python="$shared/venv/bin/python"
test -x "$python"
# This script and prepare_existing.py are transferred from the pushed commit by Cloud Shell.
sudo "$python" "$(dirname "$0")/prepare_existing.py" "$root"
sudo chown -R "$(id -u):$(id -g)" "$root"
repo="$root/repo"
git clone --no-checkout https://github.com/CalculatedContent/rg_optimizers.git "$repo"
git -C "$repo" checkout --detach "$commit"
test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
git -C "$repo" rev-parse HEAD
export PYTHONPATH="$repo/baseline/gpt2_small/src:$repo/baseline/nanogpt_one_head/src"
source "$repo/baseline/gpt2_small/scripts/tpu_environment.sh"
cd "$repo/baseline/gpt2_small"
"$python" -c 'import torch, torch_xla, weightwatcher, yaml; print("Installed dependencies loaded")'
deadline=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["deadline_unix"])' "$root/old_allocation.json")
bucket=gs://tpu-builders-504820-ww-continuous8/gpt2small
backup() {
  rc=$?
  trap - EXIT
  sync
  # Keep backups separate from prior pilot archives. No remote deletion.
  if ! "$python" "$repo/baseline/gpt2_small/scripts/backup.py" "$root"; then
    echo "Cloud backup FAILED; all outputs remain on persistent disk: $root" >&2
    exit 1
  fi
  exit "$rc"
}
trap backup EXIT
"$python" -u scripts/validate.py --root "$root" --data "$shared/data" --deadline "$deadline" 2>&1 | tee "$root/validation.log"
echo 'Validation finished. No long experiment launched.'

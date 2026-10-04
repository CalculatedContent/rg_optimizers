#!/usr/bin/env bash
# Runs as root on a replacement VM. This disk already contains the verified corpus.
set -Eeuo pipefail
DEVICE=/dev/disk/by-id/google-persistent-disk-1
for attempt in $(seq 1 60); do
  [ -b "$DEVICE" ] && break
  sleep 2
done
test -b "$DEVICE"
test "$(blkid -s TYPE -o value "$DEVICE")" = ext4
mkdir -p /mnt/disks/rg-data
mountpoint -q /mnt/disks/rg-data || mount "$DEVICE" /mnt/disks/rg-data
test "$(readlink -f "$(findmnt -n -o SOURCE --target /mnt/disks/rg-data)")" = "$(readlink -f "$DEVICE")"
ROOT=__ROOT__
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/startup.log") 2>&1
test -s /mnt/disks/rg-data/continuous8/data/meta.json
test -s /mnt/disks/rg-data/continuous8/data/train.bin
# No automatic restart after reboot, including a reboot during initialization.
if ! mkdir "$ROOT/STARTED_ONCE"; then
  echo 'Already started once. Retaining all previous outputs; no automatic restart.'
  exit 0
fi
DEADLINE=$(python3 -c 'import time; print(time.time()-float(open("/proc/uptime").read().split()[0])+__HOURS__*3600-600)')
python3 - "$ROOT" "$DEADLINE" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1], 'allocation.json').write_text(json.dumps({
    'validation_deadline_unix':float(sys.argv[2]), 'max_run_hours':__HOURS__,
    'backup_reserve_seconds':600, 'fresh_validation':True, 'long_run_started':False},indent=2))
PY
if ! command -v git >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y git
fi
mkdir "$ROOT/repo"
git -C "$ROOT/repo" init
git -C "$ROOT/repo" remote add origin https://github.com/CalculatedContent/rg_optimizers.git
git -C "$ROOT/repo" fetch --depth 1 origin __COMMIT__
git -C "$ROOT/repo" checkout --detach FETCH_HEAD
test "$(git -C "$ROOT/repo" rev-parse HEAD)" = __COMMIT__
echo __COMMIT__ > "$ROOT/commit.txt"
cat > /etc/systemd/system/rg-gpt2-validation.service <<EOF
[Unit]
Description=Bounded GPT-2 validation on the preserved FineWeb disk
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
ExecStart=/bin/bash $ROOT/repo/baseline/nanogpt_one_head/gpt2small/replacement_worker.sh $ROOT $DEADLINE
Restart=no
KillSignal=SIGTERM
TimeoutStopSec=600
StandardOutput=append:$ROOT/run.log
StandardError=append:$ROOT/run.log
EOF
systemctl daemon-reload
systemctl start rg-gpt2-validation.service
echo "Validation service started; log: $ROOT/run.log"

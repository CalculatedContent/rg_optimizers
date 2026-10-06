#!/usr/bin/env bash
# Provisioner substitutes the pinned Git SHA and the dedicated GCS run URI.
set -Eeuo pipefail
# Stop training 30+ minutes before the server-enforced allocation limit.
# VM uptime includes setup; a slow setup cannot extend the requested budget.
DEADLINE=$(python3 -c 'import time; print(time.time()-float(open("/proc/uptime").read().split()[0])+__STOP_HOURS__*3600)')
DEVICE=/dev/disk/by-id/google-persistent-disk-1
for attempt in $(seq 1 60); do
  [ -b "$DEVICE" ] && break
  sleep 2
done
test -b "$DEVICE"
TYPE="$(blkid -s TYPE -o value "$DEVICE" || true)"
if [ -z "$TYPE" ]; then
  # The provisioner creates this new dedicated disk; never format another device.
  test "$(lsblk -dn -o TYPE "$DEVICE")" = disk
  mkfs.ext4 -F "$DEVICE"
elif [ "$TYPE" != ext4 ]; then
  echo 'Unexpected filesystem; refusing to format.' >&2
  exit 1
fi
mkdir -p /mnt/disks/rg-data
mountpoint -q /mnt/disks/rg-data || mount "$DEVICE" /mnt/disks/rg-data
BASE=/mnt/disks/rg-data/continuous8
mkdir -p "$BASE"
# Persistent guard prevents either setup or training from restarting after a reboot.
if ! mkdir "$BASE/STARTED_ONCE"; then
  echo 'Already started once. Automatic restart is forbidden.'
  exit 0
fi
exec > >(tee -a "$BASE/startup.log") 2>&1
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv git
mkdir -p "$BASE/repo"
git -C "$BASE/repo" init
git -C "$BASE/repo" remote add origin https://github.com/CalculatedContent/rg_optimizers.git
git -C "$BASE/repo" fetch --depth 1 origin __COMMIT__
git -C "$BASE/repo" checkout --detach FETCH_HEAD
cat > /etc/systemd/system/rg-continuous8.service <<EOF
[Unit]
Description=One continuous MuonClip experiment, no restart
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
Environment=RG_CONTINUOUS_GCS_URI=__GCS_URI__
Environment=RG_CONTINUOUS_DATA_URI=__DATA_URI__
Environment=RG_CONTINUOUS_SEED=__SEED__
Environment=RG_CONTINUOUS_DEADLINE_UNIX=$DEADLINE
ExecStart=/bin/bash $BASE/repo/baseline/nanogpt_one_head/continuous8/worker.sh
Restart=no
SuccessExitStatus=75
KillSignal=SIGTERM
TimeoutStopSec=1800
StandardOutput=append:$BASE/run.log
StandardError=append:$BASE/run.log
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
# Deliberately not enabled: reboot must not relaunch the service.
systemctl start rg-continuous8.service

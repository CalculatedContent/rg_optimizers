#!/usr/bin/env bash
set -euo pipefail
AUDIT_NODE=ww-long-20260930-232752-node
AUDIT_PROJECT=tpu-builders-504820
AUDIT_ZONE=us-west4-a
case "${1:-start}" in
  start)
    gcloud compute tpus tpu-vm ssh "$AUDIT_NODE" --project="$AUDIT_PROJECT" --zone="$AUDIT_ZONE" --command='set -eu
mkdir -p /mnt/disks/rg-data/generalization_audit
if [ -f /mnt/disks/rg-data/generalization_audit/audit.pid ] && kill -0 "$(cat /mnt/disks/rg-data/generalization_audit/audit.pid)" 2>/dev/null; then
  echo "An audit process is already running; use status."; exit 0
fi
AUDIT_REPO=/mnt/disks/rg-data/rg_optimizers_generalization
if [ ! -d "$AUDIT_REPO" ]; then
  git clone --single-branch --branch codex/generalization-audit https://github.com/CalculatedContent/rg_optimizers.git "$AUDIT_REPO"
else
  test "$(git -C "$AUDIT_REPO" branch --show-current)" = codex/generalization-audit
  git -C "$AUDIT_REPO" pull --ff-only origin codex/generalization-audit
fi
cd "$AUDIT_REPO/baseline/nanogpt_one_head/generalization_audit"
nohup bash run_on_tpu.sh >> /mnt/disks/rg-data/generalization_audit/audit.log 2>&1 < /dev/null &
echo $! > /mnt/disks/rg-data/generalization_audit/audit.pid
echo "Audit started on CPU. Log: /mnt/disks/rg-data/generalization_audit/audit.log"'
    ;;
  status)
    gcloud compute tpus tpu-vm ssh "$AUDIT_NODE" --project="$AUDIT_PROJECT" --zone="$AUDIT_ZONE" --command='tail -n 25 /mnt/disks/rg-data/generalization_audit/audit.log'
    ;;
  fetch)
    gcloud compute tpus tpu-vm scp "$AUDIT_NODE:/mnt/disks/rg-data/generalization_audit/generalization_results.tgz" "$HOME/generalization_results.tgz" --project="$AUDIT_PROJECT" --zone="$AUDIT_ZONE"
    cloudshell download "$HOME/generalization_results.tgz"
    ;;
  *) echo "Usage: bash cloudshell.sh {start|status|fetch}" >&2; exit 2 ;;
esac

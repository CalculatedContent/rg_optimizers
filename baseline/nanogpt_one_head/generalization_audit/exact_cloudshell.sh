#!/usr/bin/env bash
set -euo pipefail
NODE=ww-long-20260930-232752-node
PROJECT=tpu-builders-504820
ZONE=us-west4-a
case "${1:-start}" in
 start)
 gcloud compute tpus tpu-vm ssh "$NODE" --project="$PROJECT" --zone="$ZONE" --command='set -eu
REPO=/mnt/disks/rg-data/rg_optimizers_generalization
BASE=/mnt/disks/rg-data/generalization_audit
mkdir -p "$BASE"
if [ ! -d "$REPO/.git" ]; then
 git clone --single-branch --branch codex/generalization-audit https://github.com/CalculatedContent/rg_optimizers.git "$REPO"
fi
test "$(git -C "$REPO" branch --show-current)" = codex/generalization-audit
if [ -f "$BASE/exact_extension.pid" ] && kill -0 "$(cat "$BASE/exact_extension.pid")" 2>/dev/null; then
 echo "Exact-probe audit is already running."; exit 0
fi
git -C "$REPO" pull --ff-only origin codex/generalization-audit
cd "$REPO/baseline/nanogpt_one_head/generalization_audit"
nohup nice -n 10 python3 -u extend_audit.py >> "$BASE/exact_extension.log" 2>&1 < /dev/null &
echo $! > "$BASE/exact_extension.pid"
echo "Launched exact-probe audit. Log: $BASE/exact_extension.log"'
 ;;
 status)
 gcloud compute tpus tpu-vm ssh "$NODE" --project="$PROJECT" --zone="$ZONE" --command='tail -n 30 /mnt/disks/rg-data/generalization_audit/exact_extension.log'
 ;;
 fetch)
 gcloud compute tpus tpu-vm scp "$NODE:/mnt/disks/rg-data/generalization_audit/exact_probe_results.tgz" "$HOME/exact_probe_results.tgz" --project="$PROJECT" --zone="$ZONE"
 if ! cloudshell download "$HOME/exact_probe_results.tgz"; then
   echo "Use Cloud Shell Download for $HOME/exact_probe_results.tgz"
 fi
 ;;
 *) echo 'Usage: bash exact_cloudshell.sh {start|status|fetch}' >&2;exit 2;;
esac

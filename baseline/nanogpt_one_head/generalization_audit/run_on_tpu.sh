#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
AUDIT_BASE=/mnt/disks/rg-data/generalization_audit
AUDIT_RUN=/mnt/disks/rg-data/muonclip-extended/segments/segment_000001/muon_clip/seed_1337
AUDIT_DATA=/mnt/disks/rg-data/rg-nanogpt-one-head/data
AUDIT_OUT="$AUDIT_BASE/results"
mkdir -p "$AUDIT_BASE"
trap 'printf "Evaluation failed; inspect %s/audit.log\n" "$AUDIT_BASE" >&2' ERR
# The pinned model module imports no XLA runtime; CPU evaluation only.
python3 -c 'import torch,numpy,pandas,matplotlib,scipy,sacrebleu,tiktoken'
nice -n 10 python3 -u audit.py run \
  --run-dir "$AUDIT_RUN" --data-root "$AUDIT_DATA" --output "$AUDIT_OUT" \
  --device cpu --threads 2 --max-checkpoints 16 \
  --documents 128 --generation-documents 64 --batch-size 2 \
  --prompt-tokens 64 --new-tokens 32 --bootstrap 500
tar -czf "$AUDIT_BASE/generalization_results.tgz" -C "$AUDIT_BASE" results
printf '\nCOMPLETE: %s/generalization_results.tgz\n' "$AUDIT_BASE"

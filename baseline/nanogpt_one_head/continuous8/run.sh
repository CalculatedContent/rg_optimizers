#!/usr/bin/env bash
# Persistent terminal transcript. This does not make Cloud Shell a durable VM.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAUNCH_LOG=${CONTINUOUS8_LAUNCH_LOG:-"$PWD/continuous8-launch.log"}
echo "Log: $LAUNCH_LOG"
python3 -u "$SCRIPT_DIR/cloudshell.py" "$@" 2>&1 | tee -a "$LAUNCH_LOG"

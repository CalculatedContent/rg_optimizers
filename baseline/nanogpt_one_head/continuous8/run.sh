#!/usr/bin/env bash
# Persistent terminal transcript. This does not make Cloud Shell a durable VM.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
echo "Log: $HOME/continuous8-launch.log"
python3 -u "$SCRIPT_DIR/cloudshell.py" "$@" 2>&1 | tee -a "$HOME/continuous8-launch.log"

#!/usr/bin/env bash
# New entry point only. Never modifies an existing checkout or experiment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
export PYTHONPATH="$SCRIPT_DIR/src"
export PYTHONDONTWRITEBYTECODE=1
exec python3 -u -m rg_nanogpt_one_head.tpu_memorization_sweep "$@"

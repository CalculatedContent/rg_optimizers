#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: run this script with bash; do not source it" >&2
  return 2
fi
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_ROOT="${RG_ONE_HEAD_VENV:-$EXPERIMENT_ROOT/.venv-one-head}"
PYTHON_BIN="${RG_ONE_HEAD_PYTHON:-$VENV_ROOT/bin/python}"
ROOT="${RG_NANOGPT_ONE_HEAD_ROOT:-$HOME/rg-nanogpt-one-head}"
DATA_ROOT="${RG_NANOGPT_ONE_HEAD_DATA_ROOT:-$ROOT/data}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' "error: run scripts/setup_mac.sh first" >&2
  exit 2
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$PYTHON_BIN" -m rg_nanogpt_one_head.data \
  --config "$EXPERIMENT_ROOT/configs/reference.yaml" \
  --output-dir "$DATA_ROOT"

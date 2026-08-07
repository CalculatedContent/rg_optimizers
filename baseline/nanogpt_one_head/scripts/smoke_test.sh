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

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' "error: run scripts/setup_mac.sh first" >&2
  exit 2
fi

"$PYTHON_BIN" -m pytest -q "$EXPERIMENT_ROOT/tests"

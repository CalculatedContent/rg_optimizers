#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "error: run this script with bash; do not source it" >&2
  return 2
fi
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_ROOT="${RG_ONE_HEAD_VENV:-$EXPERIMENT_ROOT/.venv-one-head}"
KERNEL_NAME="${RG_ONE_HEAD_KERNEL:-rg-nanogpt-one-head}"

"$PYTHON_BIN" -m venv "$VENV_ROOT"
"$VENV_ROOT/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_ROOT/bin/python" -m pip install -e "$EXPERIMENT_ROOT[dev]"
"$VENV_ROOT/bin/python" -m ipykernel install --user \
  --name "$KERNEL_NAME" \
  --display-name "RG nanoGPT One-Head (MPS)"
"$VENV_ROOT/bin/python" -m pip check

PYTORCH_ENABLE_MPS_FALLBACK=1 "$VENV_ROOT/bin/python" - <<'PY'
import platform
import torch
print(f"platform: {platform.platform()}")
print(f"torch: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    print("WARNING: MPS is unavailable; the suite will fall back to CPU.")
PY

printf '\nSetup complete.\n'
printf 'Prepare data: %s/bin/python -m rg_nanogpt_one_head.data --config %s/configs/reference.yaml\n' "$VENV_ROOT" "$EXPERIMENT_ROOT"
printf 'Run suite:    %s/bin/python -m rg_nanogpt_one_head.training --config %s/configs/reference.yaml --optimizer all --device auto\n' "$VENV_ROOT" "$EXPERIMENT_ROOT"

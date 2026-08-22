#!/usr/bin/env bash

# Observable, notebook-free Jacobian analysis.  Python writes a permanent log
# itself; tee also makes every stdout/stderr line visible in the launching shell.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
RUN_ROOT="${RG_MNIST_TANGENT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-runs}"
CACHE_ROOT="${RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT:-/private/tmp/rg-mnist-mlp3-short100-checkpoints}"
OUTPUT_ROOT="${RG_MNIST_JACOBIAN_CLI_OUTPUT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-jacobians}"

mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

cd "$REPO_ROOT"
python -u \
  baseline/experiments/mnist_mlp3_tangent_rg/scripts/run_short100_jacobians_cli.py \
  --run-root "$RUN_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  "$@" \
  2>&1 | tee -a "$OUTPUT_ROOT/terminal.log"

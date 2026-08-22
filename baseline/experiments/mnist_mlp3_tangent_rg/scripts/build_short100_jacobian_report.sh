#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
ANALYSIS_ROOT="${RG_MNIST_REDUCED_JACOBIAN_OUTPUT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-jacobians-reduced}"
RUN_ROOT="${RG_MNIST_TANGENT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-runs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

cd "$REPO_ROOT"
python -u \
  baseline/experiments/mnist_mlp3_tangent_rg/scripts/build_short100_jacobian_report.py \
  --analysis-root "$ANALYSIS_ROOT" \
  --run-root "$RUN_ROOT" \
  "$@"

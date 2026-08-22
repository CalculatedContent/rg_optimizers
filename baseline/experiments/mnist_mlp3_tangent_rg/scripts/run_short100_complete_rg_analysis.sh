#!/usr/bin/env bash

# Complete notebook-free analysis: single-checkpoint Jacobians, transformed
# weight quotient representatives, finite between-checkpoint RG flow, and the
# static shareable HTML report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OUTPUT_ROOT="${RG_MNIST_REDUCED_JACOBIAN_OUTPUT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-jacobians-reduced}"

export RG_MNIST_REDUCED_JACOBIAN_OUTPUT_ROOT="${OUTPUT_ROOT}"

bash "${SCRIPT_DIR}/run_short100_jacobians_reduced.sh" "$@"

python -u "${SCRIPT_DIR}/run_short100_quotient_flow_cli.py" \
  --run-root "${RG_MNIST_TANGENT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-runs}" \
  --cache-root "${RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT:-/private/tmp/rg-mnist-mlp3-short100-checkpoints}" \
  --output-root "${OUTPUT_ROOT}"

bash "${SCRIPT_DIR}/build_short100_jacobian_report.sh"

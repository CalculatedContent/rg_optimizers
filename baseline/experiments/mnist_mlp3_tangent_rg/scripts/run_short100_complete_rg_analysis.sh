#!/usr/bin/env bash

# Complete notebook-free analysis: single-checkpoint Jacobians, transformed
# weight quotient representatives, finite between-checkpoint RG flow, and the
# static shareable HTML report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OUTPUT_ROOT="${RG_MNIST_REDUCED_JACOBIAN_OUTPUT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-jacobians-reduced}"

export RG_MNIST_REDUCED_JACOBIAN_OUTPUT_ROOT="${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}"

interrupted() {
  printf '\nINTERRUPTED: completed units are already stored atomically.\n' >&2
  printf 'Rerun this same command to resume. Use Ctrl-C, not Ctrl-Z, to stop.\n' >&2
  exit 130
}
trap interrupted INT TERM

stage() {
  local label="$1"
  shift
  printf '\n============================================================\n'
  printf 'STAGE %s\n' "${label}"
  printf 'STARTED %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf '============================================================\n'
  "$@"
  printf 'COMPLETED %s at %s\n' "${label}" "$(date '+%Y-%m-%d %H:%M:%S')"
}

stage "1/4 exact single-checkpoint Jacobians" \
  bash "${SCRIPT_DIR}/run_short100_jacobians_reduced.sh" "$@"

stage "2/4 transformed-weight quotients" \
  python -u "${SCRIPT_DIR}/run_short100_quotient_flow_cli.py" \
    --run-root "${RG_MNIST_TANGENT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-runs}" \
    --cache-root "${RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT:-/private/tmp/rg-mnist-mlp3-short100-checkpoints}" \
    --output-root "${OUTPUT_ROOT}" \
    --epoch-stride 10 \
    --skip-checkpoint-flows

stage "3/4 between-checkpoint flow and local transport" \
  python -u "${SCRIPT_DIR}/run_short100_quotient_flow_cli.py" \
    --run-root "${RG_MNIST_TANGENT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-runs}" \
    --cache-root "${RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT:-/private/tmp/rg-mnist-mlp3-short100-checkpoints}" \
    --output-root "${OUTPUT_ROOT}" \
    --epoch-stride 10 \
    --skip-state-quotients

stage "4/4 static HTML report" \
  bash "${SCRIPT_DIR}/build_short100_jacobian_report.sh"

printf '\nCOMPLETE REPORT: %s\n' "${OUTPUT_ROOT}/report/index.html"

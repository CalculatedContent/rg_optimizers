#!/usr/bin/env bash

# Fast scientific preset: the scale-quotiented radial Jacobian on every layer
# plus both ECS/Grassmann covers on fc1 and fc2. ECS deterministic shell copies are
# compressed to physical retained-core groups before the power-law fit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export RG_MNIST_JACOBIAN_CLI_OUTPUT_ROOT="${RG_MNIST_REDUCED_JACOBIAN_OUTPUT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-jacobians-reduced}"

exec bash "${SCRIPT_DIR}/run_short100_jacobians_cli.sh" \
  --methods centered_log_singular_radial_pullback \
  --ecs-layers fc1.weight,fc2.weight \
  --compress-ecs-groups \
  --top-k 0 \
  "$@"

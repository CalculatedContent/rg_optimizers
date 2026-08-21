#!/usr/bin/env bash

# Post-facto weight-only quotient and Jacobian analysis for the matched
# 100-epoch, ten-seed MuonClip-RMS/AdamW experiment. This script never trains.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
NOTEBOOK_ROOT="${REPO_ROOT}/baseline/experiments/mnist_mlp3_tangent_rg/notebooks"

RUN_ROOT="${RG_MNIST_TANGENT_ROOT:-/private/tmp/rg-mnist-mlp3-short100-runs}"
CACHE_ROOT="${RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT:-/private/tmp/rg-mnist-mlp3-short100-checkpoints}"
SUITE="mnist_mlp3_tangent_rg_v1_muonclip_short100_10seed"
SEEDS=(101 202 303 404 505 606 707 808 909 1010)
OPTIMIZERS=(muonclip_rms adamw)
KERNEL_NAME="${RG_MNIST_JUPYTER_KERNEL:-rg-muonclip-run}"

OUTPUT_ROOT="${RUN_ROOT}/${SUITE}/notebook_outputs"
EXECUTED_ROOT="${RUN_ROOT}/executed_notebooks/short100_quotients_jacobians"
PARAMETER_FILE="${EXECUTED_ROOT}/papermill_parameters.yaml"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-cache}"
export MPLCONFIGDIR

mkdir -p "$OUTPUT_ROOT" "$EXECUTED_ROOT" "$MPLCONFIGDIR"

for optimizer in "${OPTIMIZERS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    source_seed_dir="${RUN_ROOT}/${SUITE}/${optimizer}/seed_${seed}"
    cache_checkpoint_dir="${CACHE_ROOT}/${SUITE}/${optimizer}/seed_${seed}/checkpoints"
    if [[ ! -f "${source_seed_dir}/run_complete.json" ]]; then
      echo "ERROR: missing completed run: ${source_seed_dir}/run_complete.json" >&2
      exit 1
    fi
    capture_dir="${source_seed_dir}/captures"
    capture_count="0"
    if [[ -d "$capture_dir" ]]; then
      capture_count="$(find "$capture_dir" -type f -name '*.pt' | wc -l | tr -d '[:space:]')"
    fi
    if [[ "$capture_count" == "0" ]]; then
      echo "ERROR: no saved dense captures for ${optimizer} seed ${seed}: ${capture_dir}" >&2
      echo "The complete Jacobian rerun requires the captures saved during baseline training." >&2
      exit 1
    fi
    if [[ ! -d "$cache_checkpoint_dir" ]]; then
      echo "ERROR: missing checkpoint cache: $cache_checkpoint_dir" >&2
      exit 1
    fi
    checkpoint_count="$(find "$cache_checkpoint_dir" -type f -name '*.pt' | wc -l | tr -d '[:space:]')"
    if [[ "$checkpoint_count" != "100" ]]; then
      echo "ERROR: ${optimizer} seed ${seed} has ${checkpoint_count} cached checkpoints; expected 100" >&2
      exit 1
    fi
  done
done

python - "$PARAMETER_FILE" "$RUN_ROOT" "$OUTPUT_ROOT" "$CACHE_ROOT" "$SUITE" <<'PY'
from pathlib import Path
import sys
import yaml

path, run_root, output_root, cache_root, suite = sys.argv[1:]
payload = {
    "RUN_ROOT": run_root,
    "OUTPUT_ROOT": output_root,
    "CHECKPOINT_CACHE_ROOT": cache_root,
    "PROTOCOL_SLUG": suite,
    "SEEDS": [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010],
    "OPTIMIZER_SLUGS": ["muonclip_rms", "adamw"],
    "CHECKPOINT_PAYLOAD_CACHE_SIZE": 6,
    "SHOW_PLOTS": True,
    "REQUIRE_ARTIFACTS": True,
    "ALLOW_TEMPORARY_LONG_RUN": False,
}
Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY

cd "$REPO_ROOT"

run_notebook() {
  local filename="$1"
  echo "Running ${filename}"
  python -m papermill \
    "${NOTEBOOK_ROOT}/${filename}" \
    "${EXECUTED_ROOT}/${filename%.ipynb}.executed.ipynb" \
    --kernel "$KERNEL_NAME" \
    --cwd "$REPO_ROOT" \
    --parameters_file "$PARAMETER_FILE"
}

# Complete genuine-Jacobian suite used by the earlier audit. Notebooks 11, 14,
# and 17 consume saved dense captures; notebooks 13 and 16 use saved weights.
run_notebook "11_Muon_Update_Stiefel_Tangent.ipynb"
run_notebook "13_Single_Checkpoint_Map_Jacobians.ipynb"
run_notebook "14_Calibrated_Local_Training_Map.ipynb"
run_notebook "16_Additional_Weight_Only_ECS_Jacobians.ipynb"
run_notebook "17_Data_Dependent_ECS_Jacobians.ipynb"

# Five quotient candidates plus raw/midpoint/uniform controls, all measured by
# the same dual WeightWatcher raw and fix_fingers=clip_xmax path.
run_notebook "23_Short100_10Seed_Weight_Quotients.ipynb"

echo "Complete. Executed notebooks: ${EXECUTED_ROOT}"
echo "Analysis tables and figures: ${OUTPUT_ROOT}"

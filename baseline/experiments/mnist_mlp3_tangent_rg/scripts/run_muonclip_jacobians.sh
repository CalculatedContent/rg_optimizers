#!/usr/bin/env bash

# Train the preregistered 1,000-epoch MuonClip-RMS MNIST/MLP3 baseline for
# three seeds, retain the final 100 checkpoints for each run under /tmp, and
# execute every notebook that computes a genuine Jacobian.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
BASELINE_ROOT="${REPO_ROOT}/baseline"
EXPERIMENT_ROOT="${BASELINE_ROOT}/experiments/mnist_mlp3_tangent_rg"
NOTEBOOK_ROOT="${EXPERIMENT_ROOT}/notebooks"
CONFIG_PATH="${EXPERIMENT_ROOT}/configs/pilot_1000_epochs.yaml"

ENV_NAME="rg-muonclip-run"
KERNEL_NAME="rg-muonclip-run"
RUN_ROOT="${RG_MNIST_TANGENT_ROOT:-}"
CACHE_ROOT="${RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT:-/tmp/rg-mnist-mlp3-tangent-checkpoints}"
DATA_ROOT="${RG_MNIST_DATA_ROOT:-}"
DEVICE="auto"
SEEDS=(1337 2027 31415)
DO_SETUP=1
DO_TRAINING=1
DO_ANALYSIS=1
RUN_MODE="--resume"

usage() {
  cat <<'EOF'
Usage: run_muonclip_jacobians.sh [options]

Runs the 1,000-epoch MuonClip-RMS MNIST/MLP3 baseline and all genuine
Jacobian notebooks. The final 100 checkpoints for each seed are cached under
/tmp by default. Persistent run artifacts are stored outside /tmp.

Options:
  --env-name NAME       Conda environment name (default: rg-muonclip-run)
  --run-root PATH       Persistent training/artifact root
  --cache-root PATH     Final-100 checkpoint cache root beneath /tmp
  --data-root PATH      MNIST download directory
  --device DEVICE       auto, cpu, cuda, or mps (default: auto)
  --seeds CSV           Comma-separated seeds (default: 1337,2027,31415)
  --overwrite           Start each selected seed from scratch
  --skip-setup          Reuse an already prepared Conda environment
  --training-only       Train and verify checkpoints; do not run notebooks
  --analysis-only       Run notebooks from completed artifacts; do not train
  -h, --help            Show this help

Environment-variable equivalents for paths:
  RG_MNIST_TANGENT_ROOT
  RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT
  RG_MNIST_DATA_ROOT

The default --resume mode starts a new run when none exists and resumes a
compatible interrupted run. Use --overwrite only when deliberately replacing
the exact selected optimizer/seed run artifacts.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "ERROR: $1 requires a value" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      require_value "$@"
      ENV_NAME="$2"
      KERNEL_NAME="$2"
      shift 2
      ;;
    --run-root)
      require_value "$@"
      RUN_ROOT="$2"
      shift 2
      ;;
    --cache-root)
      require_value "$@"
      CACHE_ROOT="$2"
      shift 2
      ;;
    --data-root)
      require_value "$@"
      DATA_ROOT="$2"
      shift 2
      ;;
    --device)
      require_value "$@"
      DEVICE="$2"
      shift 2
      ;;
    --seeds)
      require_value "$@"
      IFS=',' read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --overwrite)
      RUN_MODE="--overwrite"
      shift
      ;;
    --skip-setup)
      DO_SETUP=0
      shift
      ;;
    --training-only)
      DO_ANALYSIS=0
      shift
      ;;
    --analysis-only)
      DO_TRAINING=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "ERROR: at least one seed is required" >&2
  exit 2
fi
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid seed: $seed" >&2
    exit 2
  fi
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required but was not found on PATH" >&2
  exit 1
fi

clean_conda() {
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 conda "$@"
}

run_in_env() {
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
    conda run --no-capture-output -n "$ENV_NAME" "$@"
}

run_from_source() {
  env -u PYTHONHOME PYTHONNOUSERSITE=1 PYTHONPATH="$BASELINE_ROOT" \
    conda run --no-capture-output -n "$ENV_NAME" "$@"
}

if [[ "$DO_SETUP" -eq 1 ]]; then
  if ! clean_conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
    clean_conda create -y -n "$ENV_NAME" python=3.11 pip
  fi

  run_in_env python -m pip install --upgrade pip setuptools wheel
  # Keep NumPy, Torch, and TorchVision in one wheel ecosystem. Mixing Conda's
  # numerical libraries with pip Torch can load two libomp runtimes on macOS.
  run_in_env python -m pip install "numpy==1.26.4" torch torchvision
  run_in_env python -m pip install --no-cache-dir -e "${BASELINE_ROOT}[experiment]"
  run_in_env python -m ipykernel install --user \
    --name "$KERNEL_NAME" --display-name "Python (${ENV_NAME})"
fi

# Exercise both import orders because duplicate OpenMP runtimes can depend on
# which compiled package is loaded first.
run_in_env python -c \
  'import numpy, torch, torchvision; print("imports A:", numpy.__version__, torch.__version__, torchvision.__version__)'
run_in_env python -c \
  'import torch, torchvision, numpy; print("imports B:", torch.__version__, torchvision.__version__, numpy.__version__)'

USER_ROOT="$(run_in_env python -c 'import os; print(os.path.expanduser("~"))')"
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="${USER_ROOT}/rg-mnist-mlp3-tangent-runs"
fi
if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="${USER_ROOT}/rg-mnist-data"
fi

mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$DATA_ROOT"
LOG_ROOT="${RUN_ROOT}/mnist_mlp3_tangent_rg_v1_pilot1000/logs/muonclip_rms"
ANALYSIS_ROOT="${RUN_ROOT}/mnist_mlp3_tangent_rg_v1_pilot1000/notebook_outputs/muonclip_only"
EXECUTED_ROOT="${RUN_ROOT}/mnist_mlp3_tangent_rg_v1_pilot1000/executed_notebooks/muonclip_only"
mkdir -p "$LOG_ROOT" "$ANALYSIS_ROOT" "$EXECUTED_ROOT"

echo "repository:       $REPO_ROOT"
echo "Conda environment: $ENV_NAME"
echo "persistent runs:   $RUN_ROOT"
echo "checkpoint cache:  $CACHE_ROOT"
echo "MNIST data:        $DATA_ROOT"
echo "device:            $DEVICE"
echo "seeds:             ${SEEDS[*]}"

if [[ "$DO_TRAINING" -eq 1 ]]; then
  for seed in "${SEEDS[@]}"; do
    echo "Training MuonClip-RMS seed $seed"
    run_from_source python -m rg_baselines.tangent_rg.cli train \
      --config "$CONFIG_PATH" \
      --optimizer muonclip_rms \
      --seed "$seed" \
      --device "$DEVICE" \
      --data-dir "$DATA_ROOT" \
      --output-root "$RUN_ROOT" \
      --tail-checkpoint-root "$CACHE_ROOT" \
      "$RUN_MODE" \
      2>&1 | tee "$LOG_ROOT/seed_${seed}.log"
  done
fi

for seed in "${SEEDS[@]}"; do
  checkpoint_dir="${CACHE_ROOT}/mnist_mlp3_tangent_rg_v1_pilot1000/muonclip_rms/seed_${seed}/checkpoints"
  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "ERROR: missing checkpoint directory for seed $seed: $checkpoint_dir" >&2
    exit 1
  fi
  checkpoint_count="$(find "$checkpoint_dir" -type f -name '*.pt' | wc -l | tr -d '[:space:]')"
  if [[ "$checkpoint_count" != "100" ]]; then
    echo "ERROR: seed $seed has $checkpoint_count cached checkpoints; expected 100" >&2
    exit 1
  fi
  echo "Verified seed $seed: 100 cached checkpoints"
done

if [[ "$DO_ANALYSIS" -eq 1 ]]; then
  seed_csv="$(IFS=,; echo "${SEEDS[*]}")"
  PARAMETER_FILE="${ANALYSIS_ROOT}/papermill_parameters.yaml"
  run_in_env python -c '
import pathlib, sys, yaml
path, run_root, output_root, cache_root, seed_csv = sys.argv[1:]
payload = {
    "RUN_ROOT": run_root,
    "OUTPUT_ROOT": output_root,
    "CHECKPOINT_CACHE_ROOT": cache_root,
    "PROFILE": "pilot_1000_epochs",
    "SEEDS": [int(value) for value in seed_csv.split(",")],
    "OPTIMIZER_SLUGS": ["muonclip_rms"],
    "SHOW_PLOTS": False,
    "REQUIRE_ARTIFACTS": True,
}
pathlib.Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
' "$PARAMETER_FILE" "$RUN_ROOT" "$ANALYSIS_ROOT" "$CACHE_ROOT" "$seed_csv"

  JACOBIAN_NOTEBOOKS=(
    11_Muon_Update_Stiefel_Tangent.ipynb
    13_Single_Checkpoint_Map_Jacobians.ipynb
    14_Calibrated_Local_Training_Map.ipynb
    16_Additional_Weight_Only_ECS_Jacobians.ipynb
    17_Data_Dependent_ECS_Jacobians.ipynb
  )

  for notebook in "${JACOBIAN_NOTEBOOKS[@]}"; do
    input_path="${NOTEBOOK_ROOT}/${notebook}"
    output_path="${EXECUTED_ROOT}/${notebook%.ipynb}.executed.ipynb"
    echo "Executing $notebook"
    run_in_env papermill "$input_path" "$output_path" \
      --parameters_file "$PARAMETER_FILE" \
      --kernel "$KERNEL_NAME" \
      --log-output
  done
fi

echo "MuonClip-RMS Jacobian experiment complete."
echo "Persistent artifacts: $RUN_ROOT"
echo "Final-100 cache:       $CACHE_ROOT"
echo "Executed notebooks:    $EXECUTED_ROOT"

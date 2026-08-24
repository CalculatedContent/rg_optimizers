#!/usr/bin/env bash
# Reproducible bootstrap for the one-head nanoGPT baseline on a Cloud TPU v5e VM.
#
# Run this *inside* an active TPU VM after cloning rg_optimizers.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USER_ROOT="$(getent passwd "$(id -u)" | cut -d: -f6)"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-4}"
STORAGE_MODE="auto"
PERSISTENT_ROOT=""
RUN_QUICK_SMOKE=0
SMOKE_OPTIMIZER="adamw"
FORCE_FRAMEWORK_REINSTALL=0

log() {
  printf '[tpu-setup] %s\n' "$*"
}

warn() {
  printf '[tpu-setup] WARNING: %s\n' "$*" >&2
}

fail() {
  printf '[tpu-setup] ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  printf '[tpu-setup] ERROR: command failed at line %s (exit=%s)\n' "${BASH_LINENO[0]}" "$exit_code" >&2
  exit "$exit_code"
}
trap on_error ERR

usage() {
  cat <<EOF_USAGE
Usage: ${SCRIPT_NAME} [options]

Install and verify the known-good PyTorch/XLA stack for the one-head nanoGPT
baseline on a Cloud TPU v5e VM.

Options:
  --ephemeral                 Use /tmp for a disposable smoke test.
  --persistent-root PATH      Use a mounted persistent volume (recommended for
                              scientific runs), for example /mnt/disks/rg-data.
  --accelerator-type TYPE     TPU provenance label (default: ${ACCELERATOR_TYPE}).
  --torch-version VERSION     Matching torch/torch_xla version (default: ${TORCH_VERSION}).
  --force-framework-reinstall Reinstall torch and torch_xla even if versions match.
  --run-quick-smoke           Prepare a small FineWeb-Edu cache and run ~49 AdamW
                              optimizer steps after setup. Non-scientific.
  --optimizer NAME            Optimizer for --run-quick-smoke (default: adamw).
  -h, --help                  Show this help.

With no storage option, the script uses /mnt/disks/rg-data when it is mounted
and writable; otherwise it explicitly enables ephemeral /tmp storage and warns.
EOF_USAGE
}

while (($#)); do
  case "$1" in
    --ephemeral)
      STORAGE_MODE="ephemeral"
      shift
      ;;
    --persistent-root)
      (($# >= 2)) || fail "--persistent-root requires a path"
      STORAGE_MODE="persistent"
      PERSISTENT_ROOT="$2"
      shift 2
      ;;
    --accelerator-type)
      (($# >= 2)) || fail "--accelerator-type requires a value"
      ACCELERATOR_TYPE="$2"
      shift 2
      ;;
    --torch-version)
      (($# >= 2)) || fail "--torch-version requires a value"
      TORCH_VERSION="$2"
      shift 2
      ;;
    --force-framework-reinstall)
      FORCE_FRAMEWORK_REINSTALL=1
      shift
      ;;
    --run-quick-smoke)
      RUN_QUICK_SMOKE=1
      shift
      ;;
    --optimizer)
      (($# >= 2)) || fail "--optimizer requires a value"
      SMOKE_OPTIMIZER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ -f "${PROJECT_DIR}/pyproject.toml" ]] || \
  fail "could not find ${PROJECT_DIR}/pyproject.toml; run this script from the cloned repository"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: ${PYTHON_BIN}"
[[ -n "$USER_ROOT" && -d "$USER_ROOT" ]] || fail "could not resolve the current user directory"

if [[ "$STORAGE_MODE" == "auto" ]]; then
  if [[ -d /mnt/disks/rg-data && -w /mnt/disks/rg-data ]]; then
    STORAGE_MODE="persistent"
    PERSISTENT_ROOT="/mnt/disks/rg-data"
  else
    STORAGE_MODE="ephemeral"
  fi
fi

if [[ "$STORAGE_MODE" == "persistent" ]]; then
  [[ -n "$PERSISTENT_ROOT" ]] || fail "persistent storage selected without a root"
  [[ -d "$PERSISTENT_ROOT" ]] || fail "persistent root does not exist: ${PERSISTENT_ROOT}"
  [[ -w "$PERSISTENT_ROOT" ]] || fail "persistent root is not writable: ${PERSISTENT_ROOT}"
else
  warn "using ephemeral /tmp storage; checkpoints and data disappear when the TPU VM is deleted"
  warn "do not use --ephemeral for a long or scientific run"
fi

log "project directory: ${PROJECT_DIR}"
log "python: $($PYTHON_BIN --version 2>&1)"
log "accelerator type: ${ACCELERATOR_TYPE}"
log "storage mode: ${STORAGE_MODE}"

log "upgrading pip, setuptools, and wheel (required to avoid UNKNOWN-0.0.0 installs)"
"$PYTHON_BIN" -m pip install --user --upgrade pip setuptools wheel

# Remove the bad artifact produced by the old TPU VM packaging toolchain, if present.
if "$PYTHON_BIN" -m pip show UNKNOWN >/dev/null 2>&1; then
  log "removing stale UNKNOWN-0.0.0 package"
  "$PYTHON_BIN" -m pip uninstall -y UNKNOWN
fi

framework_matches() {
  PJRT_DEVICE=TPU "$PYTHON_BIN" - "$TORCH_VERSION" <<'PY'
import re
import sys

expected = sys.argv[1]

def major_minor(value: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", value)
    if not match:
        raise RuntimeError(f"cannot parse version: {value!r}")
    return int(match.group(1)), int(match.group(2))

import torch
import torch_xla

if major_minor(torch.__version__) != major_minor(expected):
    raise SystemExit(1)
if major_minor(torch_xla.__version__) != major_minor(expected):
    raise SystemExit(1)
PY
}

if ((FORCE_FRAMEWORK_REINSTALL)) || ! framework_matches >/dev/null 2>&1; then
  log "installing matching torch=${TORCH_VERSION} and torch_xla=${TORCH_VERSION}"
  "$PYTHON_BIN" -m pip uninstall -y torch torch_xla torchvision || true
  "$PYTHON_BIN" -m pip install --user \
    "torch==${TORCH_VERSION}" \
    "torch_xla[tpu]==${TORCH_VERSION}" \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -f https://storage.googleapis.com/libtpu-wheels/index.html
else
  log "matching torch/torch_xla ${TORCH_VERSION%.*}.x stack is already installed"
fi

log "installing the nanoGPT package without editable mode"
"$PYTHON_BIN" -m pip install --user --upgrade "$PROJECT_DIR"

ENV_DIR="${USER_ROOT}/.config/rg_optimizers"
ENV_FILE="${ENV_DIR}/tpu_env.sh"
mkdir -p "$ENV_DIR"
{
  printf '# Generated by %s. Safe to source from the user shell profile.\n' "$SCRIPT_NAME"
  printf 'export PATH=%q:$PATH\n' "${USER_ROOT}/.local/bin"
  printf 'export PJRT_DEVICE=TPU\n'
  printf 'export TPU_ACCELERATOR_TYPE=%q\n' "$ACCELERATOR_TYPE"
  printf 'unset XLA_USE_BF16\n'
  printf 'unset XLA_DOWNCAST_BF16\n'
  if [[ "$STORAGE_MODE" == "persistent" ]]; then
    printf 'unset RG_NANOGPT_ALLOW_EPHEMERAL_TPU_STORAGE\n'
    printf 'export RG_TPU_PERSISTENT_ROOT=%q\n' "$PERSISTENT_ROOT"
  else
    printf 'unset RG_TPU_PERSISTENT_ROOT\n'
    printf 'export RG_NANOGPT_ALLOW_EPHEMERAL_TPU_STORAGE=1\n'
  fi
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"

BASHRC_FILE="${USER_ROOT}/.bashrc"
printf -v BASHRC_LINE '[ -f %q ] && source %q' "$ENV_FILE" "$ENV_FILE"
if ! grep -Fqx "$BASHRC_LINE" "$BASHRC_FILE" 2>/dev/null; then
  printf '\n%s\n' "$BASHRC_LINE" >> "$BASHRC_FILE"
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

log "verifying the PyTorch/XLA runtime"
"$PYTHON_BIN" - "$TORCH_VERSION" <<'PY'
import json
import re
import sys

expected = sys.argv[1]

def major_minor(value: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)", value)
    if not match:
        raise RuntimeError(f"cannot parse version: {value!r}")
    return int(match.group(1)), int(match.group(2))

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

if major_minor(torch.__version__) != major_minor(expected):
    raise RuntimeError(f"torch version mismatch: {torch.__version__} vs {expected}")
if major_minor(torch_xla.__version__) != major_minor(expected):
    raise RuntimeError(f"torch_xla version mismatch: {torch_xla.__version__} vs {expected}")
if str(xr.device_type()).upper() != "TPU":
    raise RuntimeError(f"PJRT device is not TPU: {xr.device_type()!r}")

devices = xm.get_xla_supported_devices("TPU")
if not devices:
    raise RuntimeError("PyTorch/XLA found no TPU devices")

print(json.dumps({
    "torch": torch.__version__,
    "torch_xla": torch_xla.__version__,
    "pjrt_device": xr.device_type(),
    "tpu_devices": devices,
}, indent=2))
PY

RG_ENV="$(command -v rg-onehead-env || true)"
[[ -n "$RG_ENV" ]] || RG_ENV="${USER_ROOT}/.local/bin/rg-onehead-env"
[[ -x "$RG_ENV" ]] || fail "rg-onehead-env was not installed"

log "verifying nanoGPT runtime and storage resolution"
"$RG_ENV" --device auto

if ((RUN_QUICK_SMOKE)); then
  case "$SMOKE_OPTIMIZER" in
    adamw|muon|sgd_momentum) ;;
    *) fail "unsupported quick-smoke optimizer: ${SMOKE_OPTIMIZER}" ;;
  esac

  QUICK_CONFIG="${TMPDIR:-/tmp}/rg_nanogpt_tpu_quick.yaml"
  log "writing non-scientific quick-smoke config: ${QUICK_CONFIG}"
  "$PYTHON_BIN" - "${PROJECT_DIR}/configs/tpu_smoke.yaml" "$QUICK_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
with source.open(encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["protocol"]["name"] = "rg_nanogpt_one_head_tpu_quick_smoke"
cfg["protocol"]["description"] = (
    "Non-scientific TPU/XLA compatibility and throughput smoke test."
)
cfg["dataset"]["train_tokens"] = 4_000_000
cfg["dataset"]["val_tokens"] = 100_000
cfg["dataset"]["test_tokens"] = 100_000
cfg["training"]["target_epochs"] = 0.10
cfg["training"]["epoch_interval"] = 0.10
cfg["training"]["eval_interval_steps"] = 25
cfg["training"]["checkpoint_interval_steps"] = 25
cfg["weightwatcher"]["enabled"] = False

with target.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

  RG_PREPARE="$(command -v rg-onehead-prepare || true)"
  RG_TRAIN="$(command -v rg-onehead-train || true)"
  [[ -n "$RG_PREPARE" ]] || RG_PREPARE="${USER_ROOT}/.local/bin/rg-onehead-prepare"
  [[ -n "$RG_TRAIN" ]] || RG_TRAIN="${USER_ROOT}/.local/bin/rg-onehead-train"

  log "preparing the small pinned FineWeb-Edu cache"
  "$RG_PREPARE" --config "$QUICK_CONFIG" --force

  log "running the quick TPU smoke test with optimizer=${SMOKE_OPTIMIZER}"
  "$RG_TRAIN" \
    --config "$QUICK_CONFIG" \
    --optimizer "$SMOKE_OPTIMIZER" \
    --device auto \
    --no-resume
fi

log "setup complete"
log "future SSH shells load ${ENV_FILE} automatically"
log "for this already-open shell, run: source ${ENV_FILE}"
if ((RUN_QUICK_SMOKE == 0)); then
  log "optional quick test: ${SCRIPT_NAME} --ephemeral --run-quick-smoke"
fi

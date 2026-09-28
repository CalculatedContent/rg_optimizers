#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="https://github.com/CalculatedContent/rg_optimizers.git"
REPO_DIR="${RG_REPO_DIR:-/tmp/rg_optimizers}"
BASE_DIR="$REPO_DIR/baseline/nanogpt_one_head"

log() { printf '[installer] %s\n' "$*"; }
fail() { printf '[installer] ERROR: %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || fail "git is not installed"
command -v python3 >/dev/null 2>&1 || fail "python3 is not installed"

if [[ -d "$REPO_DIR/.git" ]]; then
  log "Updating existing repository at $REPO_DIR"
  git -C "$REPO_DIR" fetch --prune origin
  CURRENT_BRANCH="$(git -C "$REPO_DIR" symbolic-ref --quiet --short HEAD || true)"
  if [[ -n "$CURRENT_BRANCH" ]]; then
    git -C "$REPO_DIR" pull --ff-only origin "$CURRENT_BRANCH"
  fi
else
  log "Cloning $REPO_URL -> $REPO_DIR"
  rm -rf "$REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

[[ -f "$BASE_DIR/setup_tpu_v5e.sh" ]] || fail "missing $BASE_DIR/setup_tpu_v5e.sh"
cd "$BASE_DIR"

log "Running repository-provided TPU installer (ephemeral mode)"
bash setup_tpu_v5e.sh --ephemeral

USER_ROOT="$(getent passwd "$(id -u)" | cut -d: -f6)"
ENV_FILE="${USER_ROOT}/.config/rg_optimizers/tpu_env.sh"
[[ -f "$ENV_FILE" ]] || fail "setup completed but $ENV_FILE is missing"
# shellcheck disable=SC1090
source "$ENV_FILE"

log "Verifying TPU/XLA and installed package"
rg-onehead-env --device auto | tee /tmp/rg_nanogpt_tpu_environment.json

python3 - <<'PY'
import torch
import torch_xla
import torch_xla.runtime as xr

assert str(xr.device_type()).upper() == "TPU", xr.device_type()
print("torch:", torch.__version__)
print("torch_xla:", torch_xla.__version__)
print("PJRT:", xr.device_type())
print("TPU installer verification: PASS")
PY

log "Repository commit: $(git -C "$REPO_DIR" rev-parse HEAD)"
log "INSTALL COMPLETE"
log "Run the experiment with: ./tpu_quick_muon_ww.sh start"

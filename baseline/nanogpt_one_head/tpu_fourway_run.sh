#!/usr/bin/env bash
set -Eeuo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="/tmp/rg_nanogpt_fourway_last_run.env"
DEFAULT_CONFIG="${BASE_DIR}/configs/tpu_fourway_quick.yaml"
SHARED_DATA_ROOT="${RG_TPU_FOURWAY_DATA_ROOT:-/tmp/rg-nanogpt-fourway-data/4m}"

usage() {
  cat <<'EOF'
Usage:
  ./tpu_fourway_run.sh start [CONFIG]
  ./tpu_fourway_run.sh status
  ./tpu_fourway_run.sh tail
  ./tpu_fourway_run.sh attach

The start command prepares/reuses the pinned FineWeb-Edu cache and launches one
fresh four-process TPU run in detached tmux. Existing single-device commands and
files are not modified.
EOF
}

load_state() {
  [[ -f "$STATE_FILE" ]] || {
    echo "No four-way run has been started from this TPU VM." >&2
    exit 1
  }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

worker() {
  local config="$1"
  local run_root="$2"
  local session="$3"
  local results_root="${run_root}/results"
  local log_file="${run_root}/fourway.log"

  mkdir -p "$run_root" "$results_root" "$SHARED_DATA_ROOT"
  exec > >(tee -a "$log_file") 2>&1

  local user_root env_file
  user_root="$(getent passwd "$(id -u)" | cut -d: -f6)"
  env_file="${user_root}/.config/rg_optimizers/tpu_env.sh"
  [[ -f "$env_file" ]] || {
    echo "ERROR: TPU environment is not installed. Run setup_tpu_v5e.sh first."
    exit 2
  }
  # shellcheck disable=SC1090
  source "$env_file"

  export PYTHONPATH="${BASE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
  export RG_NANOGPT_ALLOW_EPHEMERAL_TPU_STORAGE=1
  export RG_NANOGPT_CAMPAIGN_COMMAND="$0 start $config"

  echo "============================================================"
  echo "FOUR-WAY TPU NANOGPT RUN"
  echo "START:      $(date -Is)"
  echo "CONFIG:     $config"
  echo "DATA_ROOT:  $SHARED_DATA_ROOT"
  echo "RESULTS:    $results_root"
  echo "SESSION:    $session"
  echo "============================================================"

  echo
  echo "=== PREPARE / VERIFY FINEWEB-EDU ==="
  rg-onehead-prepare \
    --config "$config" \
    --output-dir "$SHARED_DATA_ROOT"

  echo
  echo "=== RUN ALL FOUR TPU DEVICES ==="
  python3 -m rg_nanogpt_one_head.tpu_distributed \
    --config "$config" \
    --optimizer muon \
    --seed 1337 \
    --data-root "$SHARED_DATA_ROOT" \
    --results-root "$results_root" \
    --world-size 4 \
    --overwrite

  local run_dir archive
  run_dir="${results_root}/muon/seed_1337"
  test -f "${run_dir}/run_complete.json"
  test -f "${run_dir}/spectral/layers.csv"

  archive="${run_root}/fourway_muon_ww_results.tgz"
  tar -C "$results_root" -czf "$archive" muon/seed_1337

  echo
  echo "============================================================"
  echo "FOUR-WAY RUN COMPLETE"
  echo "RUN_DIR:  $run_dir"
  echo "ARCHIVE:  $archive"
  echo "END:      $(date -Is)"
  echo "============================================================"
}

start_run() {
  local config="${1:-$DEFAULT_CONFIG}"
  [[ -f "$config" ]] || {
    echo "Config not found: $config" >&2
    exit 2
  }
  command -v tmux >/dev/null 2>&1 || {
    echo "tmux is required on the TPU VM." >&2
    exit 2
  }

  local run_id run_root session
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  run_root="/tmp/rg-nanogpt-fourway-${run_id}"
  session="nanogpt4-${run_id}"

  cat > "$STATE_FILE" <<EOF
SESSION=$(printf '%q' "$session")
RUN_ROOT=$(printf '%q' "$run_root")
CONFIG=$(printf '%q' "$config")
EOF

  tmux new-session -d -s "$session" \
    "bash '$0' _worker '$config' '$run_root' '$session'"

  echo "Started detached four-way TPU run."
  echo "Session: $session"
  echo "Root:    $run_root"
  echo "Status:  $0 status"
  echo "Log:     $0 tail"
}

status_run() {
  load_state
  echo "SESSION=$SESSION"
  echo "RUN_ROOT=$RUN_ROOT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "STATE=RUNNING"
  elif [[ -f "$RUN_ROOT/results/muon/seed_1337/run_complete.json" ]]; then
    echo "STATE=COMPLETE"
  else
    echo "STATE=STOPPED_OR_FAILED"
  fi
  echo
  [[ -f "$RUN_ROOT/fourway.log" ]] && tail -n 60 "$RUN_ROOT/fourway.log"
}

tail_run() {
  load_state
  touch "$RUN_ROOT/fourway.log"
  tail -f "$RUN_ROOT/fourway.log"
}

attach_run() {
  load_state
  tmux attach -t "$SESSION"
}

case "${1:-start}" in
  start)
    shift
    start_run "${1:-$DEFAULT_CONFIG}"
    ;;
  status)
    status_run
    ;;
  tail)
    tail_run
    ;;
  attach)
    attach_run
    ;;
  _worker)
    shift
    [[ $# -eq 3 ]] || exit 2
    worker "$1" "$2" "$3"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

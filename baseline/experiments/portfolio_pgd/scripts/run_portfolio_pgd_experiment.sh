#!/usr/bin/env bash

# Foreground, terminal-safe campaign runner. It never backgrounds, detaches,
# kills, or replaces the caller's shell. All output is streamed and persisted.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PREFIX="${PORTFOLIO_PGD_ENV:-/tmp/portfolio-pgd-env}"
RUN_ROOT="${PORTFOLIO_PGD_RUN_ROOT:-/tmp/portfolio-pgd-runs}"
RUN_ID="${PORTFOLIO_PGD_RUN_ID:-$(date '+%Y%m%d_%H%M%S')_$$}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG_FILE="$RUN_DIR/run.log"

mkdir -p "$RUN_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() {
  printf '%s %-8s %s\n' "$(timestamp)" "$1" "$2" | tee -a "$LOG_FILE"
}

run_stage() {
  local stage_number="$1"
  local title="$2"
  shift 2
  log INFO "========================================================================"
  log INFO "CAMPAIGN STAGE $stage_number/3 $title"
  log INFO "COMMAND $*"
  log INFO "========================================================================"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local command_status="${PIPESTATUS[0]}"
  if [[ "$command_status" -ne 0 ]]; then
    log ERROR "FAILED stage=$stage_number status=$command_status"
    return "$command_status"
  fi
  log INFO "PASSED stage=$stage_number"
  return 0
}

log INFO "START portfolio-PGD complete campaign"
log INFO "experiment_dir=$EXPERIMENT_DIR"
log INFO "environment=$ENV_PREFIX"
log INFO "run_dir=$RUN_DIR"
log INFO "log_file=$LOG_FILE"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  log ERROR "Missing environment. Run: bash $SCRIPT_DIR/setup_mac.sh"
  exit 1
fi

PYTHON=(conda run --no-capture-output -p "$ENV_PREFIX" python -u)

if ! run_stage 1 "automated unit and reference-solver tests" \
  "${PYTHON[@]}" "$EXPERIMENT_DIR/tests/run_tests.py"; then
  exit 1
fi

if ! run_stage 2 "logged quadratic, nonlinear, and realistic PGD solves" \
  "${PYTHON[@]}" "$SCRIPT_DIR/run_portfolio_pgd_experiment.py" \
  --output-dir "$RUN_DIR/metrics" \
  --seed "${PORTFOLIO_PGD_SEED:-20260822}" \
  --log-every "${PORTFOLIO_PGD_LOG_EVERY:-25}"; then
  exit 1
fi

if ! run_stage 3 "execute all demonstration notebook cells" \
  "${PYTHON[@]}" "$SCRIPT_DIR/execute_notebooks.py"; then
  exit 1
fi

log INFO "COMPLETE all stages passed"
log INFO "summary=$RUN_DIR/metrics/summary.json"
log INFO "terminal remains active; this script did not detach or kill any process"

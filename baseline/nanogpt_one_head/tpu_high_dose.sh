#!/usr/bin/env bash
# Separate high-dose launcher. No cloud resources or historical results modified.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ROOT="${RG_HIGH_DOSE_ROOT:-/mnt/disks/rg-data/fineweb-highdose-rawalpha-v1}"
DATA="${RG_HIGH_DOSE_DATA:-/mnt/disks/rg-data/rg-nanogpt-one-head/data}"
PREVIOUS="${RG_HIGH_DOSE_PREVIOUS:-/mnt/disks/rg-data/fineweb-memorization-tpu-v1}"
BLOCK="${RG_HIGH_DOSE_BLOCK:-v5e4-highdose-rawalpha-fp32}"
SESSION="ww_highdose"
USER_ROOT="$(getent passwd "$(id -u)" | cut -d: -f6)"
ENV_FILE="$USER_ROOT/.config/rg_optimizers/tpu_env.sh"
[[ -f "$ENV_FILE" ]] || { echo "STOP: activate the existing TPU environment first: $ENV_FILE"; exit 1; }
source "$ENV_FILE"
export PYTHONPATH="$SCRIPT_DIR/src"
export PYTHONDONTWRITEBYTECODE=1
MODULE="rg_nanogpt_one_head.tpu_high_dose_sweep"
ARGS=(--code "$CODE" --root "$ROOT" --data-root "$DATA" --hardware-block "$BLOCK" --previous-root "$PREVIOUS")
CONTROL="${ROOT}.control"
case "${1:-start}" in
  plan|run)
    exec python3 -u -m "$MODULE" "$1" "${ARGS[@]}"
    ;;
  status)
    exec python3 -u -m "$MODULE" status --root "$ROOT"
    ;;
  tail)
    [[ -f "$CONTROL/latest_log" ]] || { echo "No launch log yet."; exit 1; }
    exec tail -n 60 -f "$(cat "$CONTROL/latest_log")"
    ;;
  start)
    command -v tmux >/dev/null || { echo "STOP: tmux is missing."; exit 1; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "Existing $SESSION session preserved. Inspect with: bash '$0' status"
        exit 0
    fi
    # Check the mount before writing any logs or pretending boot storage is durable.
    mountpoint -q /mnt/disks/rg-data || { echo "STOP: persistent disk is not mounted."; exit 1; }
    mkdir -p "$CONTROL"
    STAMP="$(date -u +%Y%m%dT%H%M%S%N)"
    PLANLOG="$CONTROL/plan-$STAMP.log"
    if python3 -u -m "$MODULE" plan "${ARGS[@]}" > "$PLANLOG" 2>&1; then
        tail -n 1 "$PLANLOG"
    else
        rc=$?
        tail -n 30 "$PLANLOG"
        echo "STOP: preflight failed; training was NOT started. Log: $PLANLOG"
        exit "$rc"
    fi
    LOG="$CONTROL/run-$STAMP.log"
    printf '%s\n' "$LOG" > "$CONTROL/latest_log"
    printf -v RUN '%q ' python3 -u -m "$MODULE" run "${ARGS[@]}"
    printf -v QUOTED_LOG '%q' "$LOG"
    PIPE="$RUN 2>&1 | tee -a $QUOTED_LOG"
    printf -v COMMAND 'exec bash -o pipefail -c %q' "$PIPE"
    tmux new-session -d -s "$SESSION" "$COMMAND"
    echo "Started detached tmux session: $SESSION"
    echo "Supervisor log: $LOG"
    echo "Worker progress is printed every 60 seconds. Individual logs are under $ROOT/logs."
    echo "Status: bash '$0' status"
    echo "Live log: bash '$0' tail"
    echo "No TPU lifetime was changed; this program stops after five full runs."
    sleep 3
    tail -n 12 "$LOG" 2>/dev/null || true
    ;;
  *)
    echo "Usage: bash $0 {plan|start|run|status|tail}"
    exit 2
    ;;
esac

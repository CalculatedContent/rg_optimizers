#!/usr/bin/env bash
# Detached lifecycle manager for the complete short-100 RG analysis.
#
# This file intentionally avoids `set -e` because callers commonly invoke it
# from an interactive terminal.  Every failure is handled inside this process;
# no shell options are changed in the caller.

set -u

COMMAND="${1:-start}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
DEFAULT_REPO="$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)"

REPO="${REPO:-$DEFAULT_REPO}"
RUNS="${RUNS:-/private/tmp/rg-mnist-mlp3-short100-runs}"
CACHE="${CACHE:-/private/tmp/rg-mnist-mlp3-short100-checkpoints}"
OUT="${OUT:-/private/tmp/rg-mnist-mlp3-short100-jacobians-reduced}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-cache}"
PYTHON_BIN="${PYTHON_BIN:-python}"

WORKER="$REPO/baseline/experiments/mnist_mlp3_tangent_rg/scripts/run_short100_complete_rg_analysis.sh"
LOG="$OUT/complete_rg_analysis.log"
PID_FILE="$OUT/run.pid"
LAUNCH_FILE="$OUT/launch.json"

export REPO RUNS CACHE OUT MPLCONFIGDIR
export RG_MNIST_TANGENT_ROOT="$RUNS"
export RG_MNIST_TANGENT_CHECKPOINT_CACHE_ROOT="$CACHE"
export RG_MNIST_REDUCED_JACOBIAN_OUTPUT_ROOT="$OUT"
export PYTHONUNBUFFERED=1

usage() {
    printf '%s\n' \
        "Usage: bash $0 {start|status|log|stop|open|help}" \
        "" \
        "  start   launch in a new OS session and return immediately" \
        "  status  print process state, stage JSON, and recent log lines" \
        "  log     print the most recent log lines and return immediately" \
        "  stop    send SIGTERM only to the recorded analysis process group" \
        "  open    open the completed static HTML report" \
        "" \
        "Data, logs, PID state, figures, CSVs, and HTML all remain under:" \
        "  $OUT"
}

read_pid() {
    if [ ! -f "$PID_FILE" ]; then
        return 1
    fi
    local value
    value="$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null)"
    case "$value" in
        ''|*[!0-9]*) return 1 ;;
        *) printf '%s\n' "$value" ;;
    esac
}

pid_is_live() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null
}

print_json_if_present() {
    local label="$1"
    local path="$2"
    if [ -f "$path" ]; then
        printf '\n%s\n' "$label"
        "$PYTHON_BIN" -m json.tool "$path" 2>/dev/null || sed -n '1,120p' "$path"
    fi
}

start_analysis() {
    mkdir -p "$OUT" "$MPLCONFIGDIR" || {
        printf 'ERROR: cannot create output directories beneath %s\n' "$OUT" >&2
        return 1
    }
    if [ ! -f "$WORKER" ]; then
        printf 'ERROR: worker script not found: %s\n' "$WORKER" >&2
        return 1
    fi

    local old_pid
    old_pid="$(read_pid 2>/dev/null || true)"
    if [ -n "$old_pid" ] && pid_is_live "$old_pid"; then
        printf 'Analysis is already running as PID %s.\n' "$old_pid"
        printf 'Use `%s status` for a one-shot progress report.\n' "$0"
        return 0
    fi

    if [ "${RG_SKIP_ENV_PREFLIGHT:-0}" != "1" ]; then
        "$PYTHON_BIN" -c 'import torch, weightwatcher, powerlaw' >/dev/null 2>&1 || {
            printf '%s\n' \
                "ERROR: the active Python environment lacks torch, weightwatcher, or powerlaw." \
                "Activate rg-muonclip-run, then run this start command again." >&2
            return 1
        }
    fi

    local launched_pid
    launched_pid="$("$PYTHON_BIN" - "$REPO" "$WORKER" "$LOG" "$LAUNCH_FILE" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

repo, worker, log_path, launch_path = sys.argv[1:]
Path(log_path).parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "ab", buffering=0) as log_handle:
    process = subprocess.Popen(
        ["bash", worker],
        cwd=repo,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
payload = {
    "pid": process.pid,
    "session_id": process.pid,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "repository": repo,
    "worker": worker,
    "log": log_path,
    "output_root": os.environ["OUT"],
}
temporary = Path(launch_path + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(launch_path)
print(process.pid)
PY
)" || {
        printf 'ERROR: failed to launch the detached analysis.\n' >&2
        return 1
    }

    printf '%s\n' "$launched_pid" > "$PID_FILE"
    printf '%s\n' \
        "Analysis started in a detached OS session." \
        "PID: $launched_pid" \
        "Log: $LOG" \
        "Output: $OUT" \
        "Your prompt is now safe to use. Do not press Ctrl-Z; no foreground job is attached." \
        "Run this for a one-shot update:" \
        "  bash $0 status"
}

show_status() {
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    if [ -n "$pid" ] && pid_is_live "$pid"; then
        printf 'PROCESS: RUNNING\n'
        ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,state,command 2>/dev/null || \
            printf 'PID %s is live; detailed process statistics are unavailable.\n' "$pid"
    elif [ -n "$pid" ]; then
        printf 'PROCESS: NOT RUNNING (recorded PID %s)\n' "$pid"
    else
        printf 'PROCESS: NOT STARTED (no valid PID file)\n'
    fi
    print_json_if_present "STAGE 1 STATUS" "$OUT/status.json"
    print_json_if_present "STAGES 2-3 STATUS" "$OUT/quotient_flow_status.json"
    if [ -f "$OUT/report/index.html" ]; then
        printf '\nREPORT: %s\n' "$OUT/report/index.html"
    fi
    printf '\nLATEST LOG LINES\n'
    if [ -f "$LOG" ]; then
        tail -n "${LOG_LINES:-40}" "$LOG"
    else
        printf 'No log exists yet: %s\n' "$LOG"
    fi
}

show_log() {
    if [ ! -f "$LOG" ]; then
        printf 'No log exists yet: %s\n' "$LOG" >&2
        return 1
    fi
    tail -n "${LOG_LINES:-100}" "$LOG"
}

stop_analysis() {
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    if [ -z "$pid" ]; then
        printf 'No valid analysis PID is recorded in %s\n' "$PID_FILE"
        return 0
    fi
    if ! pid_is_live "$pid"; then
        printf 'Analysis PID %s is already stopped.\n' "$pid"
        return 0
    fi
    "$PYTHON_BIN" - "$pid" "$LAUNCH_FILE" "$WORKER" <<'PY'
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

pid = int(sys.argv[1])
launch_path = Path(sys.argv[2])
expected_worker = str(Path(sys.argv[3]).resolve())
managed = False
if launch_path.is_file():
    launch = json.loads(launch_path.read_text())
    managed = (
        int(launch.get("pid", -1)) == pid
        and str(Path(launch.get("worker", "")).resolve()) == expected_worker
    )

if managed:
    try:
        session_id = os.getsid(pid)
    except ProcessLookupError:
        raise SystemExit(f"Analysis PID {pid} is already stopped")
    if session_id != pid:
        raise SystemExit(
            f"Refusing to signal PID {pid}: it is not the detached session leader"
        )
    os.killpg(pid, signal.SIGTERM)
else:
    # Compatibility path for the old nohup launcher.  Signal only the recorded
    # job and its descendants--never its process group, which may contain the
    # interactive shell.  Refuse unless the process tree contains a known RG
    # analysis command.
    listing = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    processes = {}
    children = {}
    for line in listing.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 2:
            continue
        observed_pid, parent_pid = map(int, fields[:2])
        command = fields[2] if len(fields) == 3 else ""
        processes[observed_pid] = command
        children.setdefault(parent_pid, []).append(observed_pid)
    if pid not in processes:
        raise SystemExit(f"Analysis PID {pid} is already stopped")
    descendants = []
    stack = list(children.get(pid, ()))
    while stack:
        child = stack.pop()
        descendants.append(child)
        stack.extend(children.get(child, ()))
    known = (
        "run_short100_complete_rg_analysis.sh",
        "run_short100_jacobians_cli.py",
        "run_short100_quotient_flow_cli.py",
        "build_short100_jacobian_report.py",
    )
    tree = [pid, *descendants]
    if not any(any(name in processes.get(item, "") for name in known) for item in tree):
        raise SystemExit(
            f"Refusing to signal PID {pid}: its process tree is not the RG analysis"
        )
    for item in [*reversed(descendants), pid]:
        try:
            os.kill(item, signal.SIGTERM)
        except ProcessLookupError:
            pass
PY
    if [ "$?" -ne 0 ]; then
        return 1
    fi
    printf 'Sent SIGTERM to detached analysis process group %s.\n' "$pid"
    printf 'Completed checkpoint units remain on disk and the next start resumes them.\n'
}

open_report() {
    local report="$OUT/report/index.html"
    if [ ! -f "$report" ]; then
        printf 'Report is not ready: %s\n' "$report" >&2
        printf 'Use `%s status` to inspect the current stage.\n' "$0" >&2
        return 1
    fi
    /usr/bin/open "$report"
}

case "$COMMAND" in
    start) start_analysis ;;
    status) show_status ;;
    log) show_log ;;
    stop) stop_analysis ;;
    open) open_report ;;
    help|-h|--help) usage ;;
    *)
        printf 'Unknown command: %s\n\n' "$COMMAND" >&2
        usage >&2
        exit 2
        ;;
esac

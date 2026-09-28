#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${RG_REPO_DIR:-/tmp/rg_optimizers}"
BASE_DIR="$REPO_DIR/baseline/nanogpt_one_head"
STATE_FILE="/tmp/rg_nanogpt_last_run.env"
SELF="$(readlink -f "$0")"

usage() {
  cat <<'TXT'
Usage:
  tpu_quick_muon_ww.sh start   Start a new detached short Muon + WeightWatcher run
  tpu_quick_muon_ww.sh status  Show last run status and last 50 log lines
  tpu_quick_muon_ww.sh tail    Follow the last run log
  tpu_quick_muon_ww.sh attach  Attach to the tmux session for the last run
TXT
}

load_state() {
  [[ -f "$STATE_FILE" ]] || { echo "No recorded run yet: $STATE_FILE" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

write_state() {
  local session="$1" root="$2"
  cat > "$STATE_FILE" <<STATE
SESSION=$(printf '%q' "$session")
RUN_ROOT=$(printf '%q' "$root")
STATE
}

worker() {
  local run_root="$1"
  local config="$run_root/config.yaml"
  local data_root="$run_root/data"
  local results_root="$run_root/results"
  local log_file="$run_root/run.log"
  local export_dir="$run_root/export"

  mkdir -p "$run_root" "$data_root" "$results_root" "$export_dir"
  exec > >(tee -a "$log_file") 2>&1

  USER_ROOT="$(getent passwd "$(id -u)" | cut -d: -f6)"
  ENV_FILE="${USER_ROOT}/.config/rg_optimizers/tpu_env.sh"
  [[ -f "$ENV_FILE" ]] || { echo "ERROR: TPU environment missing; run setup_tpu_v5e.sh first"; exit 2; }
  # shellcheck disable=SC1090
  source "$ENV_FILE"

  cd "$BASE_DIR"
  [[ -f configs/tpu_smoke.yaml ]] || { echo "ERROR: configs/tpu_smoke.yaml missing"; exit 2; }

  echo "============================================================"
  echo "NanoGPT FineWeb-Edu / Muon / WeightWatcher short TPU run"
  echo "START: $(date -Is)"
  echo "RUN_ROOT: $run_root"
  echo "GIT_COMMIT: $(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "============================================================"

  python3 - "$config" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path("configs/tpu_smoke.yaml")
dst = Path(sys.argv[1])
with src.open(encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["protocol"]["name"] = "tpu_muon_fineweb_4m_weightwatcher_alpha"
cfg["protocol"]["description"] = (
    "Short single-seed TPU v5e Muon run on pinned FineWeb-Edu with repeated "
    "one-call WeightWatcher raw and clip_xmax alpha measurements."
)

cfg["dataset"]["train_tokens"] = 4_000_000
cfg["dataset"]["val_tokens"] = 100_000
cfg["dataset"]["test_tokens"] = 100_000

cfg["training"]["seeds"] = [1337]
cfg["training"]["target_epochs"] = 0.20
cfg["training"]["epoch_interval"] = 0.02
cfg["training"]["eval_interval_steps"] = 25
cfg["training"]["eval_batches"] = 8
cfg["training"]["checkpoint_interval_steps"] = 25

for profile in cfg["optimizer_profiles"].values():
    if "lr_schedule_epochs" in profile:
        profile["lr_schedule_epochs"] = min(
            float(profile["lr_schedule_epochs"]),
            float(cfg["training"]["target_epochs"]),
        )

ww = cfg["weightwatcher"]
ww["enabled"] = True
ww["ERG"] = True
ww["randomize"] = True
ww["strict"] = True
ww["min_evals"] = 20
ww["fix_fingers"] = "clip_xmax"
ww["max_fingers"] = 10
ww["require_raw_alpha"] = True

cfg.setdefault("runtime", {})["matmul_precision"] = "highest"
cfg["runtime"]["allow_tf32"] = False

with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(dst)
PY

  echo
  echo "=== REPOSITORY CONFIG VALIDATION ==="
  python3 - "$config" <<'PY'
import sys
from rg_nanogpt_one_head.config import load_config
cfg = load_config(sys.argv[1])
assert cfg["weightwatcher"]["randomize"] is True
assert cfg["weightwatcher"]["ERG"] is True
assert cfg["weightwatcher"]["fix_fingers"] == "clip_xmax"
assert cfg["training"]["seeds"] == [1337]
print("CONFIG VALIDATION PASSED")
print("train_tokens:", cfg["dataset"]["train_tokens"])
print("target_epochs:", cfg["training"]["target_epochs"])
print("epoch_interval:", cfg["training"]["epoch_interval"])
print("WeightWatcher randomize:", cfg["weightwatcher"]["randomize"])
print("WeightWatcher fix_fingers:", cfg["weightwatcher"]["fix_fingers"])
PY

  echo
  echo "=== TPU ENVIRONMENT ==="
  rg-onehead-env --device auto

  echo
  echo "=== PREPARE PINNED FINEWEB-EDU ==="
  prep_start=$(date +%s)
  rg-onehead-prepare --config "$config" --output-dir "$data_root"
  prep_end=$(date +%s)
  echo "PREP_WALL_SECONDS=$((prep_end-prep_start))"

  echo
  echo "=== TRAIN MUON ==="
  train_start=$(date +%s)
  rg-onehead-train \
    --config "$config" \
    --optimizer muon \
    --seeds 1337 \
    --data-root "$data_root" \
    --results-root "$results_root" \
    --device auto
  train_end=$(date +%s)
  echo "TRAIN_WALL_SECONDS=$((train_end-train_start))"
  echo "TOTAL_WALL_SECONDS=$((train_end-prep_start))"

  run_dir="$results_root/muon/seed_1337"
  layers="$run_dir/spectral/layers.csv"
  complete="$run_dir/run_complete.json"
  [[ -f "$complete" ]] || { echo "ERROR: run_complete.json missing"; exit 3; }
  [[ -f "$layers" ]] || { echo "ERROR: spectral/layers.csv missing"; exit 3; }

  echo
  echo "=== ALPHA SUMMARY ==="
  python3 - "$layers" "$export_dir/alpha_summary.csv" <<'PY'
import sys
import pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
wanted = [
    "step", "tokens_seen", "epoch", "matrix_name",
    "alpha_raw", "alpha_clip_xmax", "alpha",
    "ERG_gap", "num_traps", "rand_distance", "num_fingers",
]
cols = [c for c in wanted if c in df.columns]
out = df[cols].copy()
out.to_csv(dst, index=False)
print(out.to_string(index=False))
print("\nWROTE:", dst)
PY

  cp "$config" "$export_dir/config.yaml"
  cp "$log_file" "$export_dir/run.log" || true
  cp "$layers" "$export_dir/layers.csv"
  for f in manifest.json metrics.csv epoch_metrics.csv test_results.json run_complete.json; do
    [[ -f "$run_dir/$f" ]] && cp "$run_dir/$f" "$export_dir/$f"
  done
  git -C "$REPO_DIR" rev-parse HEAD > "$export_dir/git_commit.txt"

  archive="$run_root/muon_ww_results.tgz"
  tar -C "$export_dir" -czf "$archive" .

  echo
  echo "============================================================"
  echo "RUN COMPLETE"
  echo "RESULTS: $run_dir"
  echo "ALPHAS:  $export_dir/alpha_summary.csv"
  echo "ARCHIVE: $archive"
  echo "END: $(date -Is)"
  echo "============================================================"
}

start() {
  [[ -d "$REPO_DIR/.git" ]] || { echo "Repository missing; run installer first." >&2; exit 2; }
  [[ -x "$HOME/.local/bin/rg-onehead-train" || -n "$(command -v rg-onehead-train || true)" ]] || {
    echo "rg-onehead-train missing; run installer first." >&2; exit 2;
  }

  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  run_root="/tmp/rg-nanogpt-muon-ww-$run_id"
  session="nanogpt-$run_id"
  mkdir -p "$run_root"
  write_state "$session" "$run_root"

  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "$session" "bash '$SELF' _worker '$run_root'"
    echo "Started detached tmux session: $session"
  else
    nohup bash "$SELF" _worker "$run_root" >/dev/null 2>&1 &
    pid=$!
    echo "PID=$pid" >> "$STATE_FILE"
    echo "tmux not found; started detached with nohup, PID $pid"
  fi

  echo "Run root: $run_root"
  echo "Log:      $run_root/run.log"
  echo "Monitor:  $SELF status"
  echo "Follow:   $SELF tail"
}

status() {
  load_state
  echo "SESSION=$SESSION"
  echo "RUN_ROOT=$RUN_ROOT"
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "STATE=RUNNING (tmux)"
  elif [[ -f "$RUN_ROOT/results/muon/seed_1337/run_complete.json" ]]; then
    echo "STATE=COMPLETE"
  else
    echo "STATE=NOT RUNNING / INCOMPLETE"
  fi
  echo
  [[ -f "$RUN_ROOT/run.log" ]] && tail -n 50 "$RUN_ROOT/run.log" || true
}

tail_log() {
  load_state
  touch "$RUN_ROOT/run.log"
  tail -f "$RUN_ROOT/run.log"
}

attach() {
  load_state
  command -v tmux >/dev/null 2>&1 || { echo "tmux unavailable" >&2; exit 2; }
  tmux attach -t "$SESSION"
}

case "${1:-start}" in
  start) start ;;
  status) status ;;
  tail) tail_log ;;
  attach) attach ;;
  _worker) [[ $# -eq 2 ]] || exit 2; worker "$2" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac

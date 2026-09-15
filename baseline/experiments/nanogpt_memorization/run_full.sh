#!/usr/bin/env bash
# Run all configured smoke checks, then the complete five-seed campaign.
# Usage: bash run_full.sh [mps|cuda|cpu] [/tmp/dedicated-root] [repository|shared_aux_decay] [--plan]
set -euo pipefail

DEVICE="${1:-mps}"
ROOT="${2:-/tmp/rg-nanogpt-memorization-20260914}"
RECIPE="${3:-repository}"
MODE="${4:-run}"
PYTHON="${PYTHON:-python}"
case "$DEVICE" in mps|cuda|cpu) ;; *) echo "Invalid device: $DEVICE" >&2; exit 2;; esac
case "$RECIPE" in repository|shared_aux_decay) ;; *) echo "Invalid recipe: $RECIPE" >&2; exit 2;; esac
case "$MODE" in run|--plan) ;; *) echo "Fourth argument must be --plan when provided" >&2; exit 2;; esac
if [ "$#" -gt 4 ]; then echo "Too many arguments" >&2; exit 2; fi
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Resolve Python before the runner redirects HOME and other cache locations.
PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
ROOT="$("$PYTHON" - "$ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).expanduser().resolve()
if not any(root.is_relative_to(Path(base)) and root != Path(base)
           for base in ('/tmp', '/private/tmp')):
    raise SystemExit('Use a dedicated absolute output directory beneath /tmp.')
print(root)
PY
)"

make_plan() {
  "$PYTHON" - "$DEVICE" "$ROOT" "$RECIPE" <<'PY'
import json
from pathlib import Path
import shlex
import sys

device, root, recipe = sys.argv[1:]
cfg = json.loads(Path('configs/suite.json').read_text())
full_count = len(cfg['conditions']) * len(cfg['optimizers']) * len(cfg['seeds'])
print('#!/usr/bin/env bash\nset -euo pipefail')
print(shlex.join([sys.executable, '-m', 'pytest', '-q', 'tests']))
print('echo ' + shlex.quote(f'Full campaign: {full_count} runs, '
      f"{cfg['stages']['full']['steps']:,} updates/run; recipe={recipe}; device={device}."))
for stage, seeds in [('smoke', cfg['seeds'][:1]), ('full', cfg['seeds'])]:
    print('echo ' + shlex.quote(f'Starting {stage}; an error stops the campaign.'))
    for condition in cfg['conditions']:
        for seed in seeds:
            for optimizer in cfg['optimizers']:
                label = f'{stage}/{recipe}/{condition}/{optimizer}/seed_{seed}'
                print('echo ' + shlex.quote('RUN ' + label))
                print(shlex.join([sys.executable, '-u', 'run.py', 'run',
                      '--stage', stage, '--condition', condition,
                      '--optimizer', optimizer, '--seed', str(seed),
                      '--device', device, '--root', root, '--recipe', recipe, '--resume']))
print('echo ' + shlex.quote(f'All {full_count} full runs returned successfully.'))
PY
}

if [ "$MODE" = --plan ]; then
  make_plan
  exit 0
fi
mkdir -p "$ROOT/logs"
# Each invocation gets its own plan and append-only log; rerunning resumes results.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
PLAN="$ROOT/logs/full_${RECIPE}_${DEVICE}_${STAMP}.sh"
LOG="$ROOT/logs/full_${RECIPE}_${DEVICE}_${STAMP}.log"
make_plan > "$PLAN"
printf 'Plan: %s\nLog: %s\nResults: %s/full/%s\n' "$PLAN" "$LOG" "$ROOT" "$RECIPE"
echo 'Keep this process in the foreground. Re-run the same command to resume.'
# Pipe failure is not hidden by tee; never continue to the next run after an error.
bash "$PLAN" 2>&1 | tee -a "$LOG"

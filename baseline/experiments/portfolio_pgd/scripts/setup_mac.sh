#!/usr/bin/env bash

# Create the complete disposable Mac environment under literal /tmp.
# Run as: bash baseline/experiments/portfolio_pgd/scripts/setup_mac.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PREFIX="${PORTFOLIO_PGD_ENV:-/tmp/portfolio-pgd-env}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s %-8s %s\n' "$(timestamp)" "$1" "$2"; }

log INFO "START portfolio-PGD Mac setup"
log INFO "experiment_dir=$EXPERIMENT_DIR"
log INFO "environment=$ENV_PREFIX"

if ! command -v conda >/dev/null 2>&1; then
  log ERROR "conda was not found on PATH"
  exit 1
fi

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  log INFO "REUSE existing environment"
else
  log INFO "CREATE conda environment with Python 3.11"
  if ! conda create -p "$ENV_PREFIX" python=3.11 -y; then
    log ERROR "conda environment creation failed"
    exit 1
  fi
fi

log INFO "INSTALL editable experiment and notebook dependencies"
if ! conda run --no-capture-output -p "$ENV_PREFIX" \
  python -m pip install -e "$EXPERIMENT_DIR[notebook]"; then
  log ERROR "dependency installation failed"
  exit 1
fi

log INFO "VERIFY imports"
if ! conda run --no-capture-output -p "$ENV_PREFIX" python - <<'PY'
import numpy
import scipy
import portfolio_pgd
print(f"portfolio_pgd={portfolio_pgd.__version__} numpy={numpy.__version__} scipy={scipy.__version__}")
PY
then
  log ERROR "import verification failed"
  exit 1
fi

log INFO "COMPLETE setup succeeded"
log INFO "Next: bash baseline/experiments/portfolio_pgd/scripts/run_portfolio_pgd_experiment.sh"

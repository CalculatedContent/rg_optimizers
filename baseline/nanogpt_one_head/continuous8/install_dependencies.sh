#!/usr/bin/env bash
# Sourced by worker.sh; PY, BASE and the working directory are already set.
# pip 25.0.1 connection retries do not recover a mid-download read timeout,
# so retry the install command as well. Successful downloads remain cached.
export PIP_CACHE_DIR="$BASE/pip-cache"
export PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=8 PIP_PROGRESS_BAR=off
export PIP_DISABLE_PIP_VERSION_CHECK=1
pip_install() {
  local attempt result=1
  for attempt in 1 2 3; do
    if [ -n "${RG_CONTINUOUS_DEADLINE_UNIX:-}" ] &&
       [ "$(date +%s)" -ge "${RG_CONTINUOUS_DEADLINE_UNIX%%.*}" ]; then
      echo 'Original allocation deadline passed; refusing further installation.' >&2
      return 1
    fi
    echo "Dependency install attempt $attempt/3: $*"
    if "$PY" -m pip install "$@"; then
      return 0
    else
      result=$?
    fi
    if [ "$attempt" -lt 3 ]; then
      echo "Dependency install exited $result; retrying in 5 seconds." >&2
      sleep 5
    fi
  done
  return "$result"
}
pip_install --upgrade 'pip==25.0.1' 'setuptools==75.8.2' 'wheel==0.45.1'
# XLA supplies the TPU backend. Avoid the unnecessary CUDA dependency downloads.
pip_install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
pip_install 'torch_xla[tpu]==2.6.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
pip_install -r continuous8/requirements.txt
pip_install --no-deps --no-build-isolation -e .

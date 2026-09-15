#!/usr/bin/env bash
# Compatibility entry point: defaults to the TWO verbatim runs, not 80.
# Examples: bash run_full.sh --optimizer adamw
#           bash run_full.sh --device mps --root /tmp/my_study
python "$(dirname "$0")/study.py" run "$@"

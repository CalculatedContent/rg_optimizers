#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
required=(am_data.py am_metrics.py am_runtime.py am_spectral.py am_train.py am_report_fast.py am_posthoc_ww.py run_study.py protocol_four_head_muon_qkclip.json)
for f in "${required[@]}"; do
  if [ ! -s "$f" ]; then
    echo "ERROR: missing or empty $HERE/$f"
    exit 1
  fi
done
python -m py_compile am_data.py am_metrics.py am_runtime.py am_spectral.py am_train.py am_report_fast.py am_posthoc_ww.py run_study.py || exit 1
python -m json.tool protocol_four_head_muon_qkclip.json >/dev/null || exit 1
python - <<'PY' || exit 1
import am_data, am_metrics, am_runtime, am_spectral, am_train
print("Muon + QK-Clip package OK")
PY
exec caffeinate -dimsu python run_study.py run --protocol protocol_four_head_muon_qkclip.json

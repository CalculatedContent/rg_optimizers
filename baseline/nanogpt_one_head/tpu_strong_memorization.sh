#!/usr/bin/env bash
# Explicit second campaign: preserve the first study and change tracked doses only.
set -Eeuo pipefail
trap 'printf "[strong-memorization] FAILED at line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PERSIST="/mnt/disks/rg-data"
DATA="$PERSIST/rg-nanogpt-one-head/data"
ROOT="$PERSIST/fineweb-memorization-tpu-v2-strong-repeat"
BLOCK="v5e4-fineweb-strong-repetition-fp32"
DOSES="0,64,256,1024,4096"

mountpoint -q "$PERSIST"
test -r "$PERSIST/FIRST_EXPERIMENT_FINAL_20260930T140303Z.zip"
test -r "$DATA/meta.json"
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)"

# This runs on a fresh TPU VM; the original reference environment is recreated.
bash "$SCRIPT_DIR/setup_tpu_v5e.sh" --persistent-root "$PERSIST" --torch-version 2.6.0
source "$HOME/.config/rg_optimizers/tpu_env.sh"
export PYTHONPATH="$SCRIPT_DIR/src"
export PYTHONDONTWRITEBYTECODE=1

python3 "$SCRIPT_DIR/tests/test_tpu_strong_doses.py"

# Verify actual injection counts and protected zero-exposure controls before training.
python3 - "$CODE" <<'PY'
from collections import Counter
from pathlib import Path
import sys
import tempfile
import torch
from rg_nanogpt_one_head.tpu_memorization_sweep import (
    EXPECTED_STEPS, load_configs, with_canary_doses,
)
from rg_nanogpt_one_head.random_canaries import RandomCanaryExperiment

configs = with_canary_doses(load_configs(Path(sys.argv[1])), [0,64,256,1024,4096])
cfg = configs["10pct"]
with tempfile.TemporaryDirectory(prefix="strong-canary-check-") as folder:
    experiment = RandomCanaryExperiment(cfg, seed=1337, total_steps=EXPECTED_STEPS,
                                        run_dir=Path(folder))
    counts = Counter(c["id"] for c in experiment.schedule.values())
    for c in experiment.canaries:
        assert counts[c["id"]] == c["dose"], (c["id"], counts[c["id"]])
    assert sum(counts[c["id"]] for c in experiment.canaries) == 43520
    end = round(EXPECTED_STEPS * cfg["memorization"]["acquisition_fraction"])
    assert all(0 <= step < end for step, _, _ in experiment.schedule)
    key = next(k for k, c in experiment.schedule.items() if c["id"] == "dose4096_canary0")
    step, micro, row = key
    x = torch.zeros((4, 256), dtype=torch.long)
    y = torch.zeros_like(x)
    xi, yi = experiment.inject(x, y, completed_step=step, micro_index=micro)
    tokens = experiment.schedule[key]["tokens"]
    assert torch.equal(xi[row], tokens[:-1]) and torch.equal(yi[row], tokens[1:])
print("Verified actual injection: 43,520 tracked presentations; zero controls never injected.")
PY

LAUNCH="$SCRIPT_DIR/tpu_full_memorization.sh"
bash "$LAUNCH" prepare --code "$CODE" --data-root "$DATA"
bash "$LAUNCH" plan --code "$CODE" --data-root "$DATA" --root "$ROOT" \
  --hardware-block "$BLOCK" --canary-doses "$DOSES"
bash "$LAUNCH" run --code "$CODE" --data-root "$DATA" --root "$ROOT" \
  --hardware-block "$BLOCK" --canary-doses "$DOSES"

# Produce a portable diagnostics archive only after the full sweep succeeds.
python3 - "$ROOT" "$PERSIST" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

root, persist = map(Path, sys.argv[1:])
receipts = list((root / "receipts").glob("task_*.json"))
assert len(receipts) == 25, f"Expected 25 receipts, found {len(receipts)}"
for path in receipts:
    completion = json.loads(path.read_text())["completion"]
    assert completion["completed"] and completion["optimizer_steps"] == 39063
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
archive = persist / f"SECOND_EXPERIMENT_FINAL_{stamp}.zip"
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".csv", ".json", ".jsonl", ".yaml"} and not ({"cache", "recovery"} & set(path.relative_to(root).parts)):
            z.write(path, path.relative_to(root))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(f"All 25 runs completed. Verified archive: {archive}", flush=True)
PY

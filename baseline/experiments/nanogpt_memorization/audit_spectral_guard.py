#!/usr/bin/env python3
"""Run the existing read-only checkpoint audit on spectral-guard runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import audit_checkpoints as base

HERE = Path(__file__).resolve().parent
GUARD = HERE / "spectral_guard_train.py"
PARENT = HERE / "run.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def guard_inventory(run_dir):
    run_dir = Path(run_dir)
    manifest = base.load_json(run_dir / "manifest.json")
    if manifest.get("condition") != "verbatim":
        raise ValueError("spectral-guard audit accepts only verbatim runs")
    if manifest.get("controller", {}).get("type") != "spectral_guard_v1":
        raise ValueError("run is not a spectral-guard run")
    family = manifest.get("profile", {}).get("family")
    if family not in {"muon", "sgd"}:
        raise ValueError(f"unsupported guard optimizer family: {family}")
    horizon = int(manifest["suite"]["stages"][manifest["stage"]]["steps"])
    done = base.load_json(run_dir / "complete.json")
    if (done.get("status") != "complete" or done.get("steps") != horizon
            or done.get("fingerprint") != manifest.get("fingerprint")
            or not done.get("accepted_trajectory_gated")):
        raise ValueError("a complete accepted spectral-guard trajectory is required")
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in rows]
    if not rows or steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != horizon:
        raise ValueError("invalid guard metric trajectory")
    found = {}
    for path in sorted(run_dir.glob("model_step_*.pt")):
        match = re.fullmatch(r"model_step_(\d+)\.pt", path.name)
        if match and int(match[1]) in steps:
            found[int(match[1])] = path
    if horizon not in found and (run_dir / "checkpoint_latest.pt").is_file():
        found[horizon] = run_dir / "checkpoint_latest.pt"
    if 0 not in found or horizon not in found or len(found) < 2:
        raise ValueError("initial, final, and intermediate saved weights are required")
    return manifest, rows, sorted(found.items())


def verify_guard_source(manifest):
    if sha256(GUARD) != manifest.get("runner_sha256"):
        raise ValueError("spectral_guard_train.py differs from the training manifest")
    if sha256(PARENT) != manifest.get("base_runner_sha256"):
        raise ValueError("run.py differs from the training manifest")
    model_path = base.REPO / base.MODEL_PATH
    raw = model_path.read_bytes()
    git_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    expected = manifest["suite"]["source_blobs"][base.MODEL_PATH]
    if git_sha != expected:
        raise ValueError("model source differs from the source used for training")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--background-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--audit-seed", type=int, default=20260915)
    parser.add_argument("--output", required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    base.inventory = guard_inventory
    base.verify_source = verify_guard_source
    try:
        base.run_audit(args)
        if not args.list:
            report = Path(args.output).expanduser().resolve() / "report.md"
            text = report.read_text().replace(
                "# Existing AdamW checkpoint audit",
                "# Existing adaptive spectral-guard checkpoint audit",
                1,
            )
            report.write_text(text)
    except KeyboardInterrupt:
        print("\nAudit stopped; source checkpoints were not overwritten.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

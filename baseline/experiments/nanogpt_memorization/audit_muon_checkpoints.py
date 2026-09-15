#!/usr/bin/env python3
"""Run the existing checkpoint audit on a completed Muon verbatim run.

The underlying audit implementation remains read-only. This wrapper only widens
its inventory check from AdamW to Muon and relabels the generated report title.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import audit_checkpoints as base


def muon_inventory(run_dir):
    run_dir = Path(run_dir)
    manifest = base.load_json(run_dir / "manifest.json")
    if manifest.get("condition") != "verbatim" or manifest.get("profile", {}).get("family") != "muon":
        raise ValueError("This wrapper accepts only completed Muon verbatim runs.")
    horizon = int(manifest["suite"]["stages"][manifest["stage"]]["steps"])
    done = base.load_json(run_dir / "complete.json")
    if (done.get("status") != "complete" or done.get("steps") != horizon
            or done.get("fingerprint") != manifest.get("fingerprint")):
        raise ValueError("A matching completion marker is required.")
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in rows]
    if not rows or steps != sorted(set(steps)) or steps[-1] != horizon:
        raise ValueError("Invalid or incomplete recorded metric trajectory.")
    found = {}
    for path in sorted(run_dir.glob("model_step_*.pt")):
        match = re.fullmatch(r"model_step_(\d+)\.pt", path.name)
        if match and int(match[1]) in steps:
            found[int(match[1])] = path
    if horizon not in found and (run_dir / "checkpoint_latest.pt").is_file():
        found[horizon] = run_dir / "checkpoint_latest.pt"
    if 0 not in found or horizon not in found or len(found) < 2:
        raise ValueError("Initialization, final, and at least two saved model snapshots are required.")
    return manifest, rows, sorted(found.items())


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
    base.inventory = muon_inventory
    try:
        base.run_audit(args)
        if not args.list:
            report = Path(args.output).expanduser().resolve() / "report.md"
            text = report.read_text()
            text = text.replace("# Existing AdamW checkpoint audit", "# Existing Muon checkpoint audit", 1)
            report.write_text(text)
    except KeyboardInterrupt:
        print("\nAudit stopped; original Muon checkpoints were not overwritten.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run from any working directory with Python 3.12 and requirements.txt installed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from audit_measurements import build_summary

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
        raise SystemExit("Run with assertions enabled (unset PYTHONOPTIMIZE and omit -O).")
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Input checksum mismatch: {name}")
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "check_results.py")], check=True)
    summary = build_summary()
    if summary != json.loads((ROOT / "results.json").read_text()):
        raise SystemExit("Numerical output differs from results.json")
    (build / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    subprocess.run([sys.executable, str(ROOT / "make_optimizer_figures.py"), "--out", str(build)],
                   env={**os.environ, "MPLBACKEND":"Agg"}, check=True)
    print(f"PASS: archived inputs, numerical checks, and four figure pages. Outputs: {build}")

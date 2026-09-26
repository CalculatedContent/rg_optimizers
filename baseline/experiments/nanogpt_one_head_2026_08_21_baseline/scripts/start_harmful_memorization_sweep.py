#!/usr/bin/env python3
"""Run the one-seed FineWeb harmful-memorization dose-finding sweep.

Each load gets an isolated experiment root and protocol fingerprint. The
underlying FineWeb corpus may be copied from a previously verified experiment
root. No prior results are modified.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
BASE_CONFIG = EXPERIMENT / "configs" / "harmful_memorization_load_sweep.yaml"
RUNNER = HERE / "run_experiment.py"


def run(cmd, *, env):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=EXPERIMENT, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-root", type=Path, required=True,
                    help="existing experiment root containing verified data/")
    ap.add_argument("--output-parent", type=Path, default=Path("/private/tmp"))
    ap.add_argument("--loads", default="0,0.001,0.005,0.02,0.10")
    ap.add_argument("--optimizers", default="adamw,muon_clip")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    source_data = args.reuse_root / "data"
    if not source_data.is_dir():
        raise SystemExit(f"verified corpus directory not found: {source_data}")

    loads = [float(x) for x in args.loads.split(",")]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    campaign = args.output_parent / f"rg-harmful-memorization-sweep-{stamp}"
    campaign.mkdir(parents=True, exist_ok=False)
    print("CAMPAIGN ROOT:", campaign)

    base = yaml.safe_load(BASE_CONFIG.read_text())
    for load in loads:
        pct = 100.0 * load
        tag = ("%.3f" % pct).rstrip("0").rstrip(".").replace(".", "p")
        root = campaign / f"load_{tag}pct"
        root.mkdir()
        config_path = EXPERIMENT / "configs" / f"harmful_memorization_{tag}pct.yaml"
        if not config_path.is_file():
            raise SystemExit(f"tracked config missing for load {load}: {config_path}")

        # Copy only the already prepared corpus. Results always start empty.
        shutil.copytree(source_data, root / "data")

        env = os.environ.copy()
        env["RG_NANOGPT_EXPERIMENT_ROOT"] = str(root)
        cache = root / "cache"
        tmp = root / "tmp"
        for p in [
            cache / "home", cache / "pip", cache / "xdg/cache",
            cache / "xdg/config", cache / "xdg/data", cache / "xdg/state",
            cache / "matplotlib", tmp,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        env.update({
            "HOME": str(cache / "home"),
            "PIP_CACHE_DIR": str(cache / "pip"),
            "XDG_CACHE_HOME": str(cache / "xdg/cache"),
            "XDG_CONFIG_HOME": str(cache / "xdg/config"),
            "XDG_DATA_HOME": str(cache / "xdg/data"),
            "XDG_STATE_HOME": str(cache / "xdg/state"),
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "TMPDIR": str(tmp),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        })

        print("\n" + "=" * 78)
        print(f"LOAD {pct:g}%  root={root}")
        print("=" * 78)
        run([sys.executable, RUNNER, "doctor", "--config", config_path,
             "--device", args.device], env=env)
        run([sys.executable, RUNNER, "prepare", "--config", config_path], env=env)
        run([sys.executable, "-u", RUNNER, "run", "--config", config_path,
             "--optimizers", args.optimizers, "--seeds", str(args.seed),
             "--device", args.device], env=env)

    print("\nSWEEP COMPLETE:", campaign)
    print("Loads:", loads)
    print("Analyze each load against load_0pct as the paired clean-load control.")


if __name__ == "__main__":
    main()

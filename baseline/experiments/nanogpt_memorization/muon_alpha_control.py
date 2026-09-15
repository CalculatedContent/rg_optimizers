#!/usr/bin/env python3
"""Select and run a Muon control whose saved spectra stay above an alpha gate.

Selection is performed only on a development seed (default 2027). The full
comparison then uses the locked scale on seed 1337. This script reuses the
existing run.py training implementation and only scales Muon's hidden-matrix
learning-rate pair; the auxiliary AdamW recipe and all other settings are left
unchanged. The selected profile is recorded in run.py's manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "run.py"
CONFIG_PATH = HERE / "configs" / "suite.json"
DEFAULT_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
DEFAULT_DEV_SEED = 2027
DEFAULT_EVAL_SEED = 1337


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def import_runner():
    spec = importlib.util.spec_from_file_location("_muon_alpha_control_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scale_tag(scale: float) -> str:
    text = f"{scale:.5f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def recipe_name(scale: float) -> str:
    return f"muon_alpha_s{scale_tag(scale)}"


def scaled_profile(base_resolver, scale: float):
    """Return a resolver that changes only Muon's hidden-matrix LR pair."""
    if not (0 < scale <= 1.0):
        raise ValueError("matrix LR scale must be in (0, 1]")

    def resolve(source: dict, optimizer: str, recipe: str) -> dict:
        if optimizer != "muon":
            raise ValueError("alpha-control runner is Muon-only")
        profile = base_resolver(source, "muon", "repository")
        profile["matrix_learning_rate"] = float(profile["matrix_learning_rate"]) * scale
        profile["matrix_min_learning_rate"] = float(profile["matrix_min_learning_rate"]) * scale
        profile["control_matrix_lr_scale"] = float(scale)
        profile["control_base_recipe"] = "repository"
        return profile

    return resolve


def run_dir(root: Path, stage: str, scale: float, seed: int) -> Path:
    return root / stage / recipe_name(scale) / "verbatim" / "muon" / f"seed_{seed}"


def alpha_gate(path: Path, threshold: float = 2.0) -> dict:
    """Require both clipped and raw alpha to remain at/above threshold."""
    path = Path(path)
    complete = path / "complete.json"
    if not complete.is_file():
        raise ValueError(f"run is not complete: {path}")
    spectral = sorted((path / "spectral").glob("step_*.csv"))
    if not spectral:
        raise ValueError(f"no spectral checkpoints: {path}")
    rows = []
    support_values = []
    for csv_path in spectral:
        table = pd.read_csv(csv_path)
        required = {"step", "matrix_name", "alpha_clip_xmax", "alpha_raw"}
        if not required.issubset(table.columns):
            raise ValueError(f"missing alpha columns: {csv_path}")
        if len(table) != 6 or table["matrix_name"].nunique() != 6:
            raise ValueError(f"expected six unique transformer matrices: {csv_path}")
        for _, r in table.iterrows():
            clip = float(r["alpha_clip_xmax"])
            raw = float(r["alpha_raw"])
            if not math.isfinite(clip) or not math.isfinite(raw):
                raise ValueError(f"nonfinite alpha in {csv_path}")
            rows.append((int(r["step"]), str(r["matrix_name"]), clip, raw, min(clip, raw)))
        if "tail_support_at_least_20" in table.columns:
            for value in table["tail_support_at_least_20"]:
                support_values.append(str(value).strip().lower() in {"true", "1"})
    worst = min(rows, key=lambda x: x[4])
    violations = [r for r in rows if r[4] < threshold]
    return {
        "passed": not violations,
        "threshold": float(threshold),
        "min_alpha_both_fits": float(worst[4]),
        "min_step": int(worst[0]),
        "min_matrix": worst[1],
        "min_alpha_clip_xmax": float(worst[2]),
        "min_alpha_raw": float(worst[3]),
        "spectral_checkpoints": len(spectral),
        "spectral_rows": len(rows),
        "violations": len(violations),
        "tail_support_fraction": (float(np.mean(support_values)) if support_values else None),
        "run_dir": str(path.resolve()),
    }


def launch(stage: str, seed: int, scale: float, device: str, root: Path) -> Path:
    runner = import_runner()
    cfg = json.loads(CONFIG_PATH.read_text())
    original = runner.resolve_profile
    runner.resolve_profile = scaled_profile(original, scale)
    args = SimpleNamespace(
        stage=stage,
        condition="verbatim",
        optimizer="muon",
        seed=seed,
        device=device,
        root=str(root),
        recipe=recipe_name(scale),
        resume=False,
    )
    try:
        runner.run(args, cfg)
    finally:
        runner.resolve_profile = original
    path = run_dir(root, stage, scale, seed)
    manifest = json.loads((path / "manifest.json").read_text())
    observed = float(manifest["profile"].get("control_matrix_lr_scale", -1))
    if not math.isclose(observed, scale, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("manifest did not record the requested Muon matrix-LR scale")
    return path


def pilot(args) -> int:
    if args.dev_seed == args.eval_seed:
        raise ValueError("development and evaluation seeds must be different")
    scales = tuple(float(x) for x in args.scales.split(","))
    if not scales or any(not (0 < x <= 1) for x in scales):
        raise ValueError("invalid scale list")
    if tuple(sorted(scales, reverse=True)) != scales or len(set(scales)) != len(scales):
        raise ValueError("scales must be unique and listed from largest to smallest")
    root = Path(args.root).expanduser().resolve() if args.root else Path("/private/tmp") / (
        "nanogpt_muon_alpha_dev_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    results = []
    selected = None
    for scale in scales:
        candidate = {"scale": scale, "recipe": recipe_name(scale)}
        print(f"\nMuon development pilot: scale={scale:g}, seed={args.dev_seed}", flush=True)
        try:
            path = launch("pilot", args.dev_seed, scale, args.device, root / f"candidate_{scale_tag(scale)}")
            gate = alpha_gate(path, args.alpha_threshold)
            candidate.update(status="complete", gate=gate)
            print(f"alpha gate: min={gate['min_alpha_both_fits']:.4f}, passed={gate['passed']}", flush=True)
            if gate["passed"]:
                selected = scale
                results.append(candidate)
                break
        except Exception as exc:
            candidate.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"candidate failed: {exc}", flush=True)
        results.append(candidate)
    lock = {
        "version": 1,
        "purpose": "development-seed selection of a Muon high-alpha control",
        "selection_rule": "largest pre-registered matrix-LR scale whose complete pilot has both clipped and raw alpha >= threshold at every saved spectral checkpoint",
        "condition": "verbatim",
        "pilot_stage": "pilot",
        "dev_seed": int(args.dev_seed),
        "evaluation_seed": int(args.eval_seed),
        "alpha_threshold": float(args.alpha_threshold),
        "candidate_scales": list(scales),
        "selected_scale": selected,
        "results": results,
        "runner_sha256": sha256(RUNNER_PATH),
        "control_script_sha256": sha256(Path(__file__)),
    }
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "muon_alpha_lock.json"
    atomic_json(lock_path, lock)
    print(f"\nLock file: {lock_path}")
    if selected is None:
        print("No pre-registered Muon candidate passed the alpha gate. Do not run the seed-1337 comparison.")
        return 2
    print(f"Selected matrix-LR scale: {selected:g}")
    print("The selected scale is now locked; do not tune it on seed 1337.")
    return 0


def full(args) -> int:
    lock_path = Path(args.lock).expanduser().resolve()
    lock = json.loads(lock_path.read_text())
    scale = lock.get("selected_scale")
    if scale is None:
        raise ValueError("lock has no qualified scale")
    if int(lock["dev_seed"]) == int(lock["evaluation_seed"]):
        raise ValueError("invalid lock: development and evaluation seeds coincide")
    if sha256(RUNNER_PATH) != lock.get("runner_sha256"):
        raise ValueError("run.py changed after the Muon scale was locked")
    scale = float(scale)
    root = Path(args.root).expanduser().resolve() if args.root else Path("/private/tmp") / (
        "nanogpt_muon_high_alpha_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    seed = int(lock["evaluation_seed"])
    print(f"Running locked Muon full comparison: scale={scale:g}, seed={seed}", flush=True)
    path = launch("full", seed, scale, args.device, root)
    gate = alpha_gate(path, float(lock["alpha_threshold"]))
    gate.update(
        locked_scale=scale,
        dev_seed=int(lock["dev_seed"]),
        evaluation_seed=seed,
        source_lock=str(lock_path),
        control_script_sha256=sha256(Path(__file__)),
    )
    atomic_json(root / "full_alpha_gate.json", gate)
    print(f"\nFull run: {path}")
    print(f"Minimum clipped/raw alpha: {gate['min_alpha_both_fits']:.4f}")
    if not gate["passed"]:
        print("FULL CONTROL FAILED THE ALPHA>=2 GATE. Do not label this run a high-alpha control.")
        return 3
    print("FULL CONTROL QUALIFIED: every saved raw and clip_xmax alpha is at/above the locked threshold.")
    return 0


def check(args) -> int:
    gate = alpha_gate(Path(args.run_dir), args.alpha_threshold)
    print(json.dumps(gate, indent=2))
    return 0 if gate["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pilot", help="development-seed scale selection")
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    p.add_argument("--root")
    p.add_argument("--dev-seed", type=int, default=DEFAULT_DEV_SEED)
    p.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    p.add_argument("--alpha-threshold", type=float, default=2.0)
    p.add_argument("--scales", default=",".join(str(x) for x in DEFAULT_SCALES))

    f = sub.add_parser("full", help="run the locked scale on the evaluation seed")
    f.add_argument("--lock", required=True)
    f.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    f.add_argument("--root")

    c = sub.add_parser("check", help="check alpha gate for a completed Muon run")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--alpha-threshold", type=float, default=2.0)

    args = parser.parse_args()
    try:
        if args.command == "pilot":
            return pilot(args)
        if args.command == "full":
            return full(args)
        return check(args)
    except KeyboardInterrupt:
        print("\nStopped. Existing completed runs were not modified.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

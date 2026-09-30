"""High-dose MuonClip study using the qualified independent-chip TPU executor.

No historical training/optimizer/evaluator code is patched. The old executor's
worker consumes an immutable plan, so it can execute this separately validated
protocol without going through the old load-sweep configuration builder.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading

from . import tpu_memorization_sweep as base

SEEDS = [1337, 2027, 4099, 31415, 271828]
CHIPS = [0, 1, 2, 3]
DOSES = [0, 64, 128, 256, 512, 1024]
VERSION = "high-dose-raw-alpha-v1"
CONFIG_DIR = Path("baseline/experiments/nanogpt_one_head_2026_08_21_baseline/configs")
CONFIG_REL = CONFIG_DIR / "high_dose_memorization_raw_alpha.yaml"
DEFAULT_PREVIOUS = Path("/mnt/disks/rg-data/fineweb-memorization-tpu-v1")


def validate_study(cfg: dict, reference: dict) -> None:
    """Allow only the explicitly agreed scientific changes from the prior study."""
    expected = deepcopy(reference)
    expected["protocol"] = deepcopy(cfg["protocol"])
    expected["training"]["epoch_interval"] = 0.125
    expected["memorization"]["doses"] = DOSES.copy()
    # The optional ordinary-Adam arm is not present in this MuonClip-only YAML.
    expected["optimizer_profiles"].pop("adam", None)
    if cfg != expected:
        bad = sorted(k for k in set(cfg) | set(expected) if cfg.get(k) != expected.get(k))
        raise ValueError("unregistered high-dose protocol change in: " + ", ".join(bad))
    if cfg["protocol"]["name"] != "fineweb_high_dose_memorization_raw_alpha":
        raise ValueError("wrong protocol name")
    if cfg["protocol"]["version"] != 1 or cfg["training"]["seeds"] != SEEDS:
        raise ValueError("wrong protocol version or seed inventory")


def load_study(code: Path) -> dict:
    # Install before loading YAML: the historical validator otherwise does not
    # register MuonClip. Worker processes independently install the same extension.
    from .muonclip import install_muonclip_extension
    install_muonclip_extension()
    from .config import load_config
    cfg = load_config(code / CONFIG_REL)
    reference = load_config(code / CONFIG_DIR / "harmful_memorization_0pct.yaml")
    validate_study(cfg, reference)
    return cfg


def tasks() -> list:
    # Do not call base.build_tasks(..., loads=["highdose"]): that public helper
    # deliberately accepts only the five historical additional-bank load names.
    return [base.Task(i, "highdose", seed, "muon_clip", CHIPS[i % 4])
            for i, seed in enumerate(SEEDS)]


def require_previous_complete(root: Path) -> None:
    """Read-only guard: all 25 previous tasks must have matching completion receipts."""
    plan = base.json_read(root / "tpu_sweep_plan.json")
    previous_tasks = plan["tasks"]
    if len(previous_tasks) != 25:
        raise ValueError("--previous-root is not the 25-run campaign")
    count = 0
    for task in previous_tasks:
        receipt = root / "receipts" / f"task_{task['index']:03d}.json"
        completion = (root / f"load_{task['load']}" / "results" / task["optimizer"]
                      / f"seed_{task['seed']}" / "run_complete.json")
        if not receipt.is_file() or not completion.is_file():
            continue
        record, done = base.json_read(receipt), base.json_read(completion)
        saved = record.get("completion", {})
        if (record.get("task") == task and done.get("completed") is True
                and done.get("optimizer_steps") == 39063
                and done.get("seed") == task["seed"]
                and done.get("optimizer") == task["optimizer"]
                and done.get("fingerprint")
                and saved == done):
            count += 1
    if count != 25:
        raise RuntimeError(f"previous campaign has {count}/25 matching completions; wait, then retry. Nothing stopped.")


def make(args) -> dict:
    code, data, root = args.code.resolve(), args.data_root.resolve(), args.root.resolve()
    source = base.source_info(code)
    if Path(__file__).resolve() != (code / "baseline/nanogpt_one_head/src/rg_nanogpt_one_head/tpu_high_dose_sweep.py").resolve():
        raise ValueError("this launcher is not running from --code")
    previous = args.previous_root.resolve()
    if root == previous or root in previous.parents or previous in root.parents:
        raise ValueError("new output root must be separate from the previous campaign")
    if (data == code or code in data.parents or root == data
            or root in data.parents):
        raise ValueError("data/output/source paths overlap unsafely")
    require_previous_complete(previous)
    cfg = load_study(code)
    from .data import validate_prepared_data
    metadata = validate_prepared_data(data, cfg)
    base.check_root(root, code, False)
    return dict(schema_version=1, executor=VERSION, root=str(root), code=str(code),
        data_root=str(data), source=source, hardware_block=args.hardware_block,
        chips=CHIPS, seeds=SEEDS, loads=["highdose"], optimizers=["muon_clip"],
        canary_batch_size=48, allow_ephemeral=False, corpus=metadata,
        configs={"highdose": cfg}, tasks=[asdict(t) for t in tasks()],
        extension_sha256=base.sha256(Path(__file__)),
        previous_root=str(previous),
        scientific_hypothesis={"primary_spectral_variable": "alpha_raw", "threshold": 2.0,
            "primary_summary": "minimum valid raw alpha across six hidden matrices; report argmin matrix",
            "doses": DOSES, "canaries_per_dose": 8, "total_tracked_canaries": 48,
            "acquisition_fraction": 0.5, "extra_random_bank_fraction": 0.0,
            "clipped_alpha": "secondary diagnostic only; never substitute for raw",
            "replication_unit": "training seed; doses share the same model at each step",
            "interpretation": "within-run dose-response and stronger-recall pilot, not a between-model dose intervention"})


def validate_plan_source(plan: dict) -> None:
    code = Path(plan["code"])
    if base.source_info(code) != plan["source"]:
        raise ValueError("source changed; use the original checkout")
    path = code / "baseline/nanogpt_one_head/src/rg_nanogpt_one_head/tpu_high_dose_sweep.py"
    if base.sha256(path) != plan["extension_sha256"]:
        raise ValueError("high-dose launcher changed")


def run(plan: dict, retries: int) -> int:
    root = Path(plan["root"])
    root.mkdir(parents=True, exist_ok=True)
    with base.exclusive_lock(root / ".sweep.lock"):
        validate_plan_source(plan)
        pf = root / "tpu_sweep_plan.json"
        base.freeze_plan(pf, plan)
        done = threading.Event()

        def heartbeat():
            while not done.wait(60):
                print("\n[high-dose] " + datetime.now(timezone.utc).isoformat(), flush=True)
                try:
                    base.show_status(root)
                except (OSError, ValueError, KeyError) as exc:
                    print(f"[high-dose] status unavailable on this poll: {exc}", flush=True)
                sys.stdout.flush()

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        rc = 1
        try:
            print("[high-dose] START: 5 seeds; chips 0,1,2,3; 39063 steps/run; RAW alpha primary", flush=True)
            # Qualified worker restores MuonClip and the batched evaluator from
            # the frozen plan; it does NOT call the historical load validator.
            rc = base.run_sweep(pf, retries)
            return rc
        finally:
            done.set()
            thread.join(timeout=2)
            base.atomic_json(root / "highdose_exit.json", {"exit_code": rc, "at": base.utc()})
            print(f"[high-dose] EXIT CODE: {rc}; results={root}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("plan", "run"):
        p = sub.add_parser(cmd)
        for name in ("code", "data-root", "root"):
            p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--hardware-block", required=True)
        p.add_argument("--previous-root", type=Path, default=DEFAULT_PREVIOUS)
        p.add_argument("--retries", type=int, default=2)
    p = sub.add_parser("status")
    p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "status":
            return base.show_status(args.root.resolve())
        if args.retries < 0:
            raise ValueError("retries must be nonnegative")
        plan = make(args)
        if args.cmd == "plan":
            print(json.dumps({k: v for k, v in plan.items() if k != "configs"}, indent=2))
            print("HIGH-DOSE PLAN OK: 5 runs x 39063 steps; doses 0,64,128,256,512,1024; RAW alpha primary; no training started")
            return 0
        return run(plan, args.retries)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"[high-dose] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

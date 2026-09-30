"""Five-seed high-dose canary experiment on four isolated TPU chips.

Additive launcher built on the already-qualified TPU sweep supervisor. It creates
one condition ("highdose"), assigns the five seeds round-robin to chips 0..3,
and therefore runs four seeds concurrently followed by the fifth on chip 0.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from copy import deepcopy

from .config import load_config
from .data import validate_prepared_data
from . import tpu_memorization_sweep as base

SEEDS = [1337, 2027, 4099, 31415, 271828]
CHIPS = [0, 1, 2, 3]
CONFIG_REL = Path("baseline/experiments/nanogpt_one_head_2026_08_21_baseline/configs/high_dose_memorization_raw_alpha.yaml")

def make(args):
    code = args.code.resolve()
    data = args.data_root.resolve()
    root = args.root.resolve()
    cfg = load_config(code / CONFIG_REL)
    if list(map(int, cfg["training"]["seeds"])) != SEEDS:
        raise ValueError("registered seed set changed")
    mem = cfg["memorization"]
    if list(map(int, mem["doses"])) != [0,64,128,256,512,1024]:
        raise ValueError("registered dose set changed")
    if float(mem.get("harmful_load_fraction", -1)) != 0:
        raise ValueError("high-dose study must not contain an additional random bank")
    if float(cfg["training"]["epoch_interval"]) != 0.125:
        raise ValueError("spectral checkpoint interval must be 0.125 epoch")
    if cfg["weightwatcher"].get("require_raw_alpha") is not True:
        raise ValueError("raw alpha is required")
    metadata = validate_prepared_data(data, cfg)
    base.check_root(root, code, False)
    tasks = base.build_tasks(SEEDS, CHIPS, ["highdose"], ["muon_clip"])
    source = base.source_info(code)
    plan = dict(
        schema_version=1, executor="high-dose-raw-alpha-v1",
        root=str(root), code=str(code), data_root=str(data), source=source,
        hardware_block=args.hardware_block, chips=CHIPS, seeds=SEEDS,
        loads=["highdose"], optimizers=["muon_clip"], canary_batch_size=48,
        allow_ephemeral=False, corpus=metadata,
        configs={"highdose": deepcopy(cfg)},
        tasks=[base.asdict(t) for t in tasks],
        scientific_hypothesis={
            "primary_spectral_variable":"alpha_raw",
            "threshold":2.0,
            "independent_variable":"tracked_canary_exposure_dose",
            "doses":[0,64,128,256,512,1024],
            "acquisition_fraction":0.5,
            "note":"alpha_clip_xmax is retained only as a secondary diagnostic",
        },
    )
    return plan

def main(argv=None):
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    for cmd in ("plan","run"):
        q=sub.add_parser(cmd)
        q.add_argument("--code",type=Path,required=True)
        q.add_argument("--data-root",type=Path,required=True)
        q.add_argument("--root",type=Path,required=True)
        q.add_argument("--hardware-block",required=True)
        q.add_argument("--retries",type=int,default=2)
    q=sub.add_parser("status"); q.add_argument("--root",type=Path,required=True)
    args=p.parse_args(argv)
    if args.cmd=="status":
        return base.show_status(args.root.resolve())
    plan=make(args)
    if args.cmd=="plan":
        print(json.dumps({k:v for k,v in plan.items() if k!="configs"},indent=2))
        print("HIGH-DOSE STUDY: 5 runs x 39063 steps; doses 0,64,128,256,512,1024; RAW alpha primary; no training started")
        return 0
    root=Path(plan["root"]); root.mkdir(parents=True,exist_ok=True)
    with base.exclusive_lock(root/".sweep.lock"):
        pf=root/"tpu_sweep_plan.json"
        base.freeze_plan(pf,plan)
        return base.run_sweep(pf,args.retries)

if __name__=="__main__":
    raise SystemExit(main())

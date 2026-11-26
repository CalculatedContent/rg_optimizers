"""Prepared future study. Default is dry run; no workflow or automatic dispatch."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAP = 20_000

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-training", action="store_true", help="Only after separate user authorization")
    parser.add_argument("--rate-index", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    plan = json.loads((HERE / "future_plan.json").read_text())
    rates, seeds = plan["dropout_probabilities"], plan["fresh_seeds"]
    t = plan["training"]
    if (t["development_updates"] != CAP or t["total_updates"] != CAP
            or any(not 0 < step <= CAP for step in t["checkpoint_updates"])
            or plan["maximum_updates_per_run"] != CAP or plan["planned_runs"] != len(rates)*len(seeds)):
        raise ValueError("Invalid capped study plan")
    if not args.execute_training:
        print(json.dumps({"status":"dry_run_no_training", "runs":[{"rate_index":i,"dropout":p,"seed":s,"updates":CAP}
              for i,p in enumerate(rates) for s in seeds]}, indent=2))
        return
    if args.rate_index is None or not 0 <= args.rate_index < len(rates) or args.seed not in seeds:
        parser.error("An explicit valid rate-index and seed are required for one authorized run")
    cfg = {**plan["optimizer"], "dropout": rates[args.rate_index]}
    if cfg["optimizer"] != "sgdm":
        raise ValueError("Only the frozen SGDM experiment is allowed")
    output = HERE / "future_results" / f"p{args.rate_index:02d}" / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError("Refusing to restart, resume or overwrite an existing run")
    data_dir = ROOT / plan["data_dir"]
    if not (data_dir / "manifest.json").is_file() or not (data_dir / "prepared.npz").is_file():
        raise FileNotFoundError("Restore immutable prepared data; do not regenerate it")
    sys.path.insert(0, str(ROOT))
    from california_housing_sgdm_search.long_train import train_long
    record = train_long(t, cfg, data_dir, output, args.seed, phase="confirmation",
                        parent_checkpoint=None, stop_step=CAP, resume=False)
    if record["status"] != "complete" or record["completed_updates"] != CAP or record["start_step"] != 0:
        raise RuntimeError("Incomplete or inconsistent fresh 20k run")
    print(json.dumps({"seed":args.seed,"dropout":cfg["dropout"],"completed_updates":CAP}))
if __name__ == "__main__":
    main()

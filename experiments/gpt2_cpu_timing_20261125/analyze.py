"""Project a fixed token budget from measured full-model CPU timings."""
import argparse
import json
from pathlib import Path
import statistics


def project(record):
    protocol = record["protocol"]
    rows = []
    for item in record["microbatches"]:
        for optimizer, timing in record["optimizers"].items():
            fb = item["forward_backward"]["median_seconds"]
            ev = item["validation_forward"]["median_seconds"]
            update = timing["clip_and_step"]["median_seconds"] + timing["zero_grad_seconds"]
            passes = protocol["training_tokens"] // item["tokens"]
            evaluations = (protocol["validation_events_excluding_initial"] *
                           protocol["validation_tokens_per_event"] // item["tokens"])
            train_seconds = passes * fb + protocol["updates"] * update
            evaluation_seconds = evaluations * ev
            lower = (passes * item["forward_backward"]["min_seconds"] +
                     protocol["updates"] * (timing["clip_and_step"]["min_seconds"] + timing["zero_grad_seconds"]) +
                     evaluations * item["validation_forward"]["min_seconds"])
            upper = (passes * item["forward_backward"]["max_seconds"] +
                     protocol["updates"] * (timing["clip_and_step"]["max_seconds"] + timing["zero_grad_seconds"]) +
                     evaluations * item["validation_forward"]["max_seconds"])
            rows.append({"replica": record["replica"], "batch_sequences": item["batch_sequences"],
                         "optimizer": optimizer, "training_tokens_per_second": item["tokens"] / fb,
                         "median_fb_seconds": fb, "median_eval_seconds": ev,
                         "clip_update_zero_seconds": update,
                         "training_days": train_seconds / 86400,
                         "validation_days": evaluation_seconds / 86400,
                         "total_compute_days": (train_seconds + evaluation_seconds) / 86400,
                         "sample_min_projection_days": lower / 86400,
                         "sample_max_projection_days": upper / 86400,
                         "mean_full_update_minutes": (item["accumulations_per_update"] * fb + update) / 60})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    records = [json.loads(f.read_text()) for f in args.files]
    rows = [row for record in records for row in project(record)]
    result = {"method": "Measured medians, full 19,560-update budget and 79 validation events; no convergence prediction",
              "warning": "Observed sample extrema are not a statistical confidence interval or guaranteed bound",
              "rows": rows}
    best = []
    for replica in sorted({r["replica"] for r in rows}):
        for optimizer in ("adamw", "sgdm", "adamw_sgdm"):
            best.append(min((r for r in rows if r["replica"] == replica and r["optimizer"] == optimizer),
                            key=lambda r: r["total_compute_days"]))
    result["best_measured_microbatch_per_runner_and_optimizer"] = best
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

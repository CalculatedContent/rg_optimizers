"""Export real run summaries and plots; never fabricate missing layer results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoint import write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def build_report(run_dir):
    run_dir = Path(run_dir)
    status = read_json(run_dir / "TRAINING_STATUS.json")
    tracking_path = run_dir / "TRACKING_STATUS.json"
    tracking = read_json(tracking_path) if tracking_path.exists() else {"complete": False}
    results = [read_json(path) for path in sorted((run_dir / "tracking/measurements").glob("*.json"))]
    results.sort(key=lambda row: row["step"])
    if len({row["step"] for row in results}) != len(results):
        raise ValueError("Duplicate measurement steps")
    for row in results:
        if (row["run_id"] != status["run_id"] or
                row["manifest"].get("config_sha256") != status["config_sha256"]):
            raise ValueError("Measurement provenance differs from this run")
    final = next((row for row in reversed(results) if row["step"] == status["step"]
                  and row["tokens_seen"] == status["tokens_seen"]), None)
    report = {"schema_version": 1, "run_id": status["run_id"], "mode": status["mode"],
              "training_status": status["status"], "step": status["step"], "tokens_seen": status["tokens_seen"],
              "target_met": status["target_met"], "full_training_recipe_completed": status["full_training_recipe_completed"],
              "target_ever_met": status.get("target_ever_met", False),
              "first_target_validation": status.get("first_target_validation"),
              "best_validation": status.get("best_validation"),
              "validation": status["validation"], "tracking": tracking,
              "final_layer_summary": final["summary"] if final else None,
              "final_layer_coverage": final["coverage"] if final else None,
              "layer_summary_matches_final_training_step": final is not None,
              "interpretation": "Observed run; this report alone does not establish a search winner or mathematical optimum.",
              "confidence_note": "KS/tail weights are fit-quality heuristics, not confidence intervals or probabilities."}
    output = run_dir / "reports"
    output.mkdir(exist_ok=True)
    write_json(output / "summary.json", report)
    if final:
        write_json(output / "final_layers.json", final["layers"])
        write_json(output / "final_groups.json", final["groups"])
    else:
        (output / "final_layers.json").unlink(missing_ok=True)
        (output / "final_groups.json").unlink(missing_ok=True)
    return report, results


def make_plots(run_dir, results, *, axis="tokens"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not results:
        raise ValueError("No committed spectral measurements to plot")
    xkey = "tokens_seen" if axis == "tokens" else "step"
    x = [row[xkey] for row in results]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(x, [row["validation"].get("val_nll") for row in results], marker=".", label="NLL at spectral snapshots")
    official = all(row["validation"].get("full_benchmark_evaluation") for row in results)
    if official:
        axes[0, 0].axhline(3.28, color="black", linestyle="--", label="Benchmark target 3.28")
    axes[0, 0].set_ylabel("NLL (nats/token)")
    axes[0, 0].legend()
    axes[0, 1].plot(x, [row["validation"].get("val_perplexity") for row in results], marker=".")
    axes[0, 1].set_ylabel("Validation perplexity")
    for fit, linestyle in (("raw", "-"), ("clip_xmax", "--")):
        for suffix, color, label in (("mean", "#2471A3", "Mean alpha"),
                                     ("confidence_tail_weighted_mean", "#B9770E", "Confidence/tail-weighted alpha")):
            key = f"alpha_{fit}_{suffix}"
            axes[1, 0].plot(x, [row["summary"].get(key) for row in results],
                            linestyle=linestyle, color=color, label=f"{label} ({fit})")
        axes[1, 1].plot(x, [row["summary"].get(f"ERG_gap_{fit}_mean") for row in results],
                        linestyle=linestyle, label=f"Mean ERG gap ({fit})")
    axes[1, 0].set_ylabel("Alpha across transformer matrices")
    axes[1, 1].set_ylabel("Native WeightWatcher ERG gap")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Processed training tokens" if axis == "tokens" else "Optimizer update")
        ax.grid(alpha=0.2)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    config = read_json(Path(run_dir) / "config.json")
    mode = "SYNTHETIC SMOKE — NOT A BENCHMARK" if config["mode"] == "smoke" else "Stock GPT-2 / FineWeb / SGDM"
    warnings = sum(row["coverage"]["status"] != "complete" for row in results)
    fig.suptitle(f"{mode}\n{config['candidate_id']} | {warnings} snapshot(s) with fit-coverage warnings")
    output = Path(run_dir) / "reports"
    for extension in ("png", "pdf"):
        fig.savefig(output / f"training_and_layer_metrics.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--axis", choices=("tokens", "step"), default="tokens")
    args = parser.parse_args()
    report, results = build_report(args.run_dir)
    if args.plots:
        make_plots(args.run_dir, results, axis=args.axis)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

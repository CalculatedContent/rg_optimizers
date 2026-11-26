#!/usr/bin/env python3
"""Reproduce the completed five-seed California Housing alpha trajectories.

Usage:
  python make_previous_alpha_plot.py --input-dir ../housing_independent_audit
  python make_previous_alpha_plot.py --input-dir /path/to/published_mlp_results

Inputs must be the completed GitHub cohort, never exploratory or new-grid runs.
The supplied local audit copies omit only eigenvalue arrays and verbose WW logs.
The script uses their retained raw ordinary-alpha fits, not alpha_weighted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
from scipy.stats import t

SEEDS = list(range(2340, 2345))
ARMS = ("sgdm", "adamw", "modified_adamw")
LABELS = {"sgdm": "SGD + momentum", "adamw": "Native AdamW", "modified_adamw": "Modified AdamW"}
COLORS = {"sgdm": "#16825D", "adamw": "#2766B3", "modified_adamw": "#D66A22"}
STEPS = list(range(0, 20001, 500))
RUN_ID = "45753214421"
LAUNCH = "67a0bf203ab0f6b34565e37b4f510f123f669876"
ARCHIVE = "84b0219dc87c7caaefdc7245e455e53c300b8dfc"
REPO_PREFIX = "california_housing_benchmark/results/2026-11-26_published_mlp"
URL_PREFIX = f"https://github.com/CalculatedContent/rg_optimizers/blob/{ARCHIVE}/{REPO_PREFIX}"


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ordinary_confidence_tail_alpha(spectrum):
    assert spectrum["complete"] and spectrum["n_eligible"] == spectrum["n_valid"] == 1
    assert spectrum["eligible_matrices"] == ["backbone.blocks.1.0.weight"]
    assert spectrum["layer_alpha_sd"] is None
    assert spectrum["model_average_equals_single_matrix_fit"] is True
    matrices = spectrum["matrices"]
    assert len(matrices) == 1 and matrices[0]["shape"] == [784, 784]
    numerator = denominator = 0.0
    for row in matrices:
        fit = row["fit"]
        alpha, distance, spikes = float(fit["alpha"]), float(fit["D"]), int(fit["num_pl_spikes"])
        assert all(math.isfinite(x) for x in (alpha, distance)) and distance >= 0 and spikes > 0
        assert math.isclose(alpha, row["alpha"], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(distance, row["D"], rel_tol=0, abs_tol=1e-12)
        assert spikes == row["num_pl_spikes"]
        weight = math.sqrt(spikes) / (distance + 0.02)
        assert math.isclose(weight, row["weight"], rel_tol=1e-12, abs_tol=1e-12)
        numerator += alpha * weight
        denominator += weight
    value = numerator / denominator
    assert math.isclose(value, spectrum["confidence_tail_alpha"], rel_tol=0, abs_tol=1e-12)
    return value


def make_figure(statistics, out, zoom=False):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    fig.subplots_adjust(left=0.095, right=0.97, bottom=0.24, top=0.80)
    fig.suptitle("California Housing: average alpha during training", x=0.095,
                 y=0.965, ha="left", fontsize=17, fontweight="bold", color="#172B3A")
    fig.text(0.095, 0.914, "Five seeds per optimizer  |  Shading: pointwise 95% Student-t confidence intervals",
             fontsize=11, color="#465563")
    for arm in ARMS:
        rows = [r for r in statistics if r["optimizer"] == arm and (not zoom or r["step"] >= 2000)]
        x = np.asarray([r["step"] for r in rows])
        y = np.asarray([r["mean"] for r in rows])
        low = np.asarray([r["ci95_low"] for r in rows])
        high = np.asarray([r["ci95_high"] for r in rows])
        ax.fill_between(x, low, high, color=COLORS[arm], alpha=0.17, linewidth=0)
        ax.plot(x, y, color=COLORS[arm], label=LABELS[arm], linewidth=2.1,
                marker="o", markersize=2.3, markeredgewidth=0)
    ax.axhline(2, color="#63727B", linestyle=(0, (4, 3)), linewidth=1.25, zorder=0)
    ax.text(0.982, 2.025, r"$\alpha=2$ reference", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", color="#596870", fontsize=10)
    if zoom:
        ax.set_xlim(1800, 20300)
        ax.set_ylim(1.45, 4.65)
        ax.yaxis.set_major_locator(FixedLocator([1.5, 2, 2.5, 3, 3.5, 4, 4.5]))
        ax.set_ylabel(r"Confidence/tail-adjusted average $\bar{\alpha}$")
        ax.set_xticks([2000, 5000, 10000, 15000, 20000])
    else:
        ax.set_yscale("log")
        ax.set_ylim(1.43, 29)
        ax.yaxis.set_major_locator(FixedLocator([1.5, 2, 3, 5, 10, 20]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(-100, 20300)
        ax.set_xticks([0, 2000, 5000, 10000, 15000, 20000])
        ax.set_ylabel(r"Confidence/tail-adjusted average $\bar{\alpha}$  (log scale)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Training updates", labelpad=8)
    ax.grid(axis="y", alpha=0.17, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.008, 1.015), ncol=3, frameon=False,
              fontsize=11, handlelength=2.6, columnspacing=2.3, borderaxespad=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#8E9AA3")
        ax.spines[spine].set_linewidth(0.7)
    fig.text(0.095, 0.10,
             "The model has one eligible 784 × 784 weight matrix; its alpha therefore equals the model average.",
             fontsize=10, color="#465563")
    fig.text(0.095, 0.065,
             "Uncertainty reflects variation across seeds 2340–2344. Points are measurements every 500 updates.",
             fontsize=10, color="#465563")
    fig.text(0.095, 0.030,
             "Completed GitHub cohort 45753214421" + ("  |  Detail: 2,000–20,000 updates" if zoom else
             "  |  Log scale preserves the full initial-to-final range."), fontsize=9, color="#65737E")
    stem = "california_housing_previous_alpha_2000_20000" if zoom else "california_housing_previous_alpha"
    fig.savefig(out / f"{stem}.png", dpi=220)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent.parent / "housing_independent_audit")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, out = args.input_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    official_path = root / "official_summary.json"
    if not official_path.exists():
        official_path = root / "analysis/summary.json"
    official = load(official_path)
    assert official["complete_five_seeds"] and official["n_completed"] == 5
    assert official["n_completed_runs"] == 15 and not official["audit_errors"]
    official_rows = {(r["optimizer"], r["step"]): r for r in official["statistics"]
                     if r["endpoint"] == "fixed_updates" and r["metric"] == "confidence_tail_alpha"}
    official_hashes = {r["path"]: r["sha256"] for r in official["source_files"]}
    tree_path = root / "repository_tree.json"
    tree = {x["path"]: x["sha"] for x in load(tree_path)["tree"]} if tree_path.exists() else {}
    records, source_audit, values = {}, [], []
    for arm in ARMS:
        for seed in SEEDS:
            relative = f"runs/seed_{seed}/{arm}/result.json"
            path = root / relative
            if not path.exists():
                path = root / f"seed_{seed}_{arm}_result.json"
            record = load(path)
            assert record["seed"] == seed and record["optimizer"] == arm
            assert record["status"] == "complete" and record["phase"] == "confirmation"
            assert record["completed_updates"] >= 20000
            execution = record["execution"]["github"]
            assert execution["GITHUB_RUN_ID"] == RUN_ID and execution["GITHUB_SHA"] == LAUNCH
            history = {int(row["step"]): row for row in record["history"] if int(row["step"]) <= 20000}
            assert sorted(history) == STEPS and len(history) == 41
            assert sum(row["step"] <= 20000 for row in record["history"]) == 41
            archive_path = f"{REPO_PREFIX}/{relative}"
            provenance = record.get("_independent_audit_source")
            if provenance is not None:
                assert provenance["omitted_fields"] == ["spectral.matrices[*].eigenvalues", "spectral.matrices[*].ww_output"]
                assert tree[archive_path] == provenance["git_blob_sha"]
                blob_sha = provenance["git_blob_sha"]
            else:
                assert sha256(path) == official_hashes[relative]
                content = path.read_bytes()
                blob_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            source_audit.append({"seed":seed, "optimizer":arm, "source_url":f"{URL_PREFIX}/{relative}",
                                 "repository_path":archive_path, "original_git_blob_sha":blob_sha,
                                 "original_result_sha256":official_hashes[relative],
                                 "local_input_sha256":sha256(path),
                                 "input_is_audited_projection":provenance is not None,
                                 "omitted_fields":[] if provenance is None else provenance["omitted_fields"]})
            records[(arm, seed)] = (record, history)
            for step in STEPS:
                row = history[step]
                value = ordinary_confidence_tail_alpha(row["spectral"])
                values.append({"optimizer":arm, "seed":seed, "step":step,
                               "confidence_tail_alpha":value, "n_eligible_matrices":1,
                               "original_git_blob_sha":blob_sha})
    for seed in SEEDS:
        assert len({records[(arm, seed)][0]["initialization_sha256"] for arm in ARMS}) == 1
        for step in STEPS:
            assert len({records[(arm, seed)][1][step]["minibatch_sha256"] for arm in ARMS}) == 1
    statistics, differences = [], []
    critical = float(t.ppf(0.975, 4))
    for arm in ARMS:
        for step in STEPS:
            rows = [r for r in values if r["optimizer"] == arm and r["step"] == step]
            assert sorted(r["seed"] for r in rows) == SEEDS
            x = np.asarray([r["confidence_tail_alpha"] for r in rows])
            mean, sd = float(x.mean()), float(x.std(ddof=1))
            error = sd / math.sqrt(5)
            half = critical * error
            row = {"optimizer":arm,"step":step,"n":5,"mean":mean,"sample_sd":sd,
                   "standard_error":error,"ci95_half_width":half,"ci95_low":mean-half,
                   "ci95_high":mean+half,"t_critical":critical,"degrees_of_freedom":4,
                   "uncertainty_unit":"seed", "eligible_matrices_per_model":1}
            ref = official_rows[(arm, step)]
            assert ref["n"] == 5 and ref["complete_five_seeds"]
            for key in ("mean","sample_sd","standard_error","ci95_half_width","ci95_low","ci95_high"):
                difference = abs(row[key] - ref[key])
                assert difference < 2e-12, (arm, step, key, difference)
                differences.append(difference)
            statistics.append(row)
    csv_write(out / "alpha_trajectory_statistics.csv", statistics)
    csv_write(out / "alpha_trajectory_seed_values.csv", values)
    audit = {"workflow_run_id":RUN_ID,"launch_commit":LAUNCH,"archive_commit":ARCHIVE,
             "n_runs":15,"seeds":SEEDS,"optimizers":list(ARMS),"steps":STEPS,
             "n_seed_step_values_recalculated":len(values),"n_statistics_rows_checked":len(statistics),
             "n_numeric_comparisons":len(differences),"maximum_absolute_difference":max(differences),
             "official_summary_input_sha256":sha256(official_path),"complete":True,"errors":[],
             "aggregation":"ordinary fitted alpha; weights sqrt(num_pl_spikes)/(D+0.02); no extra size weight",
             "scope":"one eligible 784x784 matrix; model mean equals its alpha; no layer dispersion",
             "uncertainty":"equal-weight five-seed means, sample SD, pointwise 95% Student-t (df=4); no simultaneous-band claim",
             "plotting":"all measured 500-update points 0..20000; straight connecting segments; no smoothing, synthesis, extrapolation, or best-checkpoint substitution",
             "sources":source_audit}
    (out / "source_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    make_figure(statistics, out)
    make_figure(statistics, out, zoom=True)
    summary = ["# Completed California Housing alpha trajectories", "",
               f"Source: [GitHub workflow {RUN_ID}](https://github.com/CalculatedContent/rg_optimizers/actions/runs/{RUN_ID}), immutable archive `{ARCHIVE}`.", "",
               "All 15 runs use seeds 2340–2344. Every curve uses all five seed measurements at each of 41 fixed budgets (0, 500, ..., 20,000). The shaded regions are pointwise 95% Student-t intervals across seeds, not layer dispersion or simultaneous confidence bands.", "",
               "Ordinary fitted alpha is aggregated with weights sqrt(num_pl_spikes)/(D+0.02), without a matrix-size factor. Only the 784×784 hidden matrix meets the size threshold, so the model average equals that matrix's alpha. The initial broad spectrum fit does not establish a power-law regime.", "",
               "| Optimizer | 2,000 updates: mean ± 95% CI half-width | 20,000 updates: mean ± 95% CI half-width |", "|---|---:|---:|"]
    for arm in ARMS:
        endpoints = [next(r for r in statistics if r["optimizer"]==arm and r["step"]==step) for step in (2000,20000)]
        summary.append(f"| {LABELS[arm]} | {endpoints[0]['mean']:.6f} ± {endpoints[0]['ci95_half_width']:.6f} | {endpoints[1]['mean']:.6f} ± {endpoints[1]['ci95_half_width']:.6f} |")
    summary += ["", "The main figure uses a logarithmic alpha axis to retain the high initialization fits while showing later differences. A supplementary linear detail includes only the measured 2,000–20,000 interval.", "",
                "The audit recomputed all 615 seed/step averages from retained raw fit fields and checked all 123 mean/SD/CI summaries against the committed official analysis. Local input projections omit only eigenvalue arrays and verbose fitting logs; their original Git blob IDs and original SHA-256 hashes are preserved in source_audit.json. Raw original result JSONs are available at the pinned source URLs. No new training was run.", ""]
    (out / "SOURCE_AUDIT.md").write_text("\n".join(summary))
    print(json.dumps({"complete":True,"measurements":len(values),"summaries":len(statistics),
                      "maximum_absolute_difference":max(differences),
                      "endpoints":[r for r in statistics if r["step"] in (2000,20000)]},indent=2))


if __name__ == "__main__":
    main()

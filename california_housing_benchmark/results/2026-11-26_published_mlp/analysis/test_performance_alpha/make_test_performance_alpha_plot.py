#!/usr/bin/env python3
"""Plot actual paired test RMSE and ordinary confidence/tail alpha.

Run with the adjacent independently audited JSON; no notebook is needed.
The audit contains all 615 paired measurements and original Git blob identities.
Each curve follows training-update order, never sorted-alpha order.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
from scipy.stats import t

ARMS = ("sgdm", "adamw", "modified_adamw")
LABELS = {"sgdm": "SGD + momentum", "adamw": "Native AdamW", "modified_adamw": "Modified AdamW"}
COLORS = {"sgdm": "#16825D", "adamw": "#2766B3", "modified_adamw": "#D66A22"}
SEEDS = list(range(2340, 2345))
STEPS = list(range(0, 20001, 500))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_and_summarize(audit):
    assert audit["complete"] and audit["errors"] == []
    assert audit["cohort"] == "45753214421" and audit["seeds"] == SEEDS
    assert audit["n_seed_steps"] == 615 and audit["n_runs"] == 15
    values = audit["measurements"]
    assert len(values) == 615
    assert len({(r["optimizer"], r["seed"], r["step"]) for r in values}) == 615
    source_stats = {(r["optimizer"], r["step"]): r for r in audit["statistics"]}
    critical = float(t.ppf(.975, 4))
    summary, maximum_difference = [], 0.0
    for arm in ARMS:
        for step in STEPS:
            rows = sorted((r for r in values if r["optimizer"] == arm and r["step"] == step), key=lambda r:r["seed"])
            assert [r["seed"] for r in rows] == SEEDS
            for row in rows:
                assert row["same_history_row"] is True
                assert math.isclose(row["test_rmse"], math.sqrt(row["test_mse"]), rel_tol=0, abs_tol=1e-12)
            result = {"optimizer": arm, "step": step, "n_seeds": 5}
            for key in ("confidence_tail_alpha", "test_rmse"):
                samples = np.asarray([r[key] for r in rows], dtype=float)
                assert np.isfinite(samples).all()
                mean, sd = float(samples.mean()), float(samples.std(ddof=1))
                half = critical * sd / math.sqrt(5)
                computed = {"mean": mean, "sample_sd": sd, "ci95_half_width": half,
                            "ci95_low": mean-half, "ci95_high": mean+half}
                for field, value in computed.items():
                    difference = abs(value - source_stats[(arm, step)][key][field])
                    assert difference < 2e-12
                    maximum_difference = max(maximum_difference, difference)
                    result[f"{key}_{field}"] = value
            summary.append(result)
    return values, summary, maximum_difference


def draw_panel(ax, values, summary, detail=False):
    minimum_step = 2000 if detail else 0
    for arm in ARMS:
        color = COLORS[arm]
        points = [r for r in values if r["optimizer"] == arm and r["step"] >= minimum_step]
        if detail:
            ax.scatter([r["confidence_tail_alpha"] for r in points], [r["test_rmse"] for r in points],
                       s=10, color=color, alpha=.17, linewidths=0, zorder=1)
        rows = [r for r in summary if r["optimizer"] == arm and r["step"] >= minimum_step]
        rows.sort(key=lambda r:r["step"])
        x = [r["confidence_tail_alpha_mean"] for r in rows]
        y = [r["test_rmse_mean"] for r in rows]
        ax.plot(x, y, color=color, linewidth=1.9, marker="o", markersize=2.5,
                markeredgewidth=0, zorder=3)
        for step, marker, size in ((2000, "D", 6), (20000, "o", 8)):
            row = next(r for r in rows if r["step"] == step)
            if detail:
                ax.errorbar(row["confidence_tail_alpha_mean"], row["test_rmse_mean"],
                            xerr=row["confidence_tail_alpha_ci95_half_width"],
                            yerr=row["test_rmse_ci95_half_width"], fmt="none",
                            ecolor=color, elinewidth=1.1, capsize=2.5, alpha=.8, zorder=4)
            ax.plot(row["confidence_tail_alpha_mean"], row["test_rmse_mean"],
                    marker=marker, markersize=size, color=color, markeredgecolor="white",
                    markeredgewidth=.8, linestyle="none", zorder=5)
    ax.axvline(2, color="#697982", linestyle=(0,(4,3)), linewidth=1.1, zorder=0)
    ax.text(2.025, .985, r"$\alpha=2$", transform=ax.get_xaxis_transform(), color="#596870",
            va="top", ha="left", fontsize=10)
    ax.set_xlabel(r"Confidence/tail-adjusted average $\bar{\alpha}$", labelpad=8)
    ax.grid(alpha=.15, linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#929DA5")
        ax.spines[spine].set_linewidth(.7)
    if detail:
        ax.set_xlim(1.48, 4.62)
        ax.set_ylim(.472, .586)
        ax.set_xticks([1.5,2,2.5,3,3.5,4,4.5])
        ax.set_yticks([.48,.50,.52,.54,.56,.58])
        ax.set_title("Training detail: 2,000–20,000 updates", loc="left", fontsize=12, pad=12)
        positions = {"sgdm": (2.65,.500), "adamw": (2.1,.4778), "modified_adamw": (1.78,.4915)}
        for arm, xytext in positions.items():
            row = next(r for r in summary if r["optimizer"] == arm and r["step"] == 20000)
            ax.annotate("20k", xy=(row["confidence_tail_alpha_mean"], row["test_rmse_mean"]),
                        xytext=xytext, color=COLORS[arm], fontsize=10, fontweight="bold",
                        arrowprops={"arrowstyle":"-", "color":COLORS[arm], "lw":.9}, va="center")
    else:
        ax.set_xscale("log")
        ax.set_xlim(1.43, 29)
        ax.set_ylim(.45, 1.19)
        ax.xaxis.set_major_locator(FixedLocator([1.5,2,3,5,10,20]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f"{value:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks([.5,.6,.8,1,1.15])
        ax.set_title("All checkpoints: 0–20,000 updates", loc="left", fontsize=12, pad=12)
        ax.text(.98,.025,"Logarithmic α axis", transform=ax.transAxes, ha="right", color="#596870",fontsize=10)
        ax.annotate("Initialization", xy=(19.06729275288,1.1323649233844888),
                    xytext=(7.0,1.155), fontsize=10, color="#465563",
                    arrowprops={"arrowstyle":"-", "color":"#72818B", "lw":.8})


def make_plot(values, summary, output):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                         "axes.labelsize":11, "xtick.labelsize":10, "ytick.labelsize":10,
                         "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1,2,figsize=(11.5,6.5),gridspec_kw={"width_ratios":[1,1.4]},facecolor="white")
    fig.subplots_adjust(left=.073,right=.98,bottom=.25,top=.77,wspace=.25)
    fig.text(.073,.947,"California Housing: test performance versus average α",fontsize=18,fontweight="bold",color="#172B3A")
    fig.text(.073,.900,"Test RMSE ↓  |  Five seeds per optimizer  |  Each pair uses the same recorded model",fontsize=11,color="#465563")
    handles = [Line2D([0],[0],color=COLORS[arm],lw=2.7,label=LABELS[arm]) for arm in ARMS]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.061,.868),ncol=3,frameon=False,
               fontsize=11,handlelength=2.3,columnspacing=2.5)
    draw_panel(axes[0], values, summary, False)
    draw_panel(axes[1], values, summary, True)
    axes[0].set_ylabel("Test RMSE ($100,000; lower is better)",labelpad=8)
    axes[1].set_ylabel("Test RMSE",labelpad=6)
    fig.text(.073,.153,"Lines: five-seed means connected in training order. Faint points: individual seed checkpoints in the detail.",fontsize=9.6,color="#465563")
    fig.text(.073,.115,"Diamonds: 2,000 updates. Large circles: 20,000. Crossbars: separate 95% Student-t intervals across seeds.",fontsize=9.6,color="#465563")
    fig.text(.073,.077,"Only one 784 × 784 matrix is eligible; its fitted α equals this model’s confidence/tail-adjusted average.",fontsize=9.6,color="#465563")
    fig.text(.073,.039,"Completed GitHub cohort 45753214421 • Measurements every 500 updates • RMSE 0.50 means $50,000",fontsize=9.2,color="#65737E")
    for extension in ("png","pdf"):
        fig.savefig(output/f"california_housing_test_rmse_vs_alpha.{extension}",dpi=220,facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,default=Path(__file__).with_name("independent_performance_alpha_audit.json"))
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    values, summary, difference = check_and_summarize(audit)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/"test_performance_alpha_seed_values.csv", values)
    write_csv(args.output_dir/"test_performance_alpha_statistics.csv", summary)
    make_plot(values,summary,args.output_dir)
    result = {"complete":True,"seed_measurements":615,"summary_rows":123,
              "maximum_summary_difference":difference,"input_audit_sha256":hashlib.sha256(args.audit.read_bytes()).hexdigest(),
              "plot":"Full0–20k means and2k–20k detail, update-order connections, independent seed points, marginal95%CIs at2k/20k.",
              "metric":"Test RMSE in $100,000 units; lower is better; no classification accuracy is inferred.",
              "uncertainty":"Seed variability conditional on one fixed split. Crossbars are marginal pointwise intervals, not joint95% regions.",
              "same_model_evidence":audit["model_binding_evidence"]}
    (args.output_dir/"plot_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

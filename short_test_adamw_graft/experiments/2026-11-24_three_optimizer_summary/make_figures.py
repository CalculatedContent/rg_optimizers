#!/usr/bin/env python3
"""Rebuild the Shakespeare GPT figures from audited, bundled CSV measurements.

No fitting, synthetic observations, or training is performed. The audit script
computes seed statistics; these figures use its recorded pointwise Student-t
95% confidence intervals. Run: python make_figures.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
ARMS = ["sgdm", "native", "modified"]
COLORS = {"sgdm": "#2467A0", "native": "#CB571D", "modified": "#7960A8"}
NAMES = {"sgdm": "SGD + momentum", "native": "Native AdamW", "modified": "Modified AdamW"}
MARKERS = {"sgdm": "o", "native": "s", "modified": "^"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8E9BA6", "axes.linewidth": .6,
    "xtick.color": "#394B57", "ytick.color": "#394B57",
    "text.color": "#263D4B", "axes.labelcolor": "#263D4B",
    "axes.titlecolor": "#173E56", "savefig.facecolor": "white",
    "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 3, "ytick.major.size": 3,
})

def save(fig, name):
    # Fixed physical page size keeps type sizes consistent in the manuscript.
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)

def legend_handles(markers=False):
    return [Line2D([0], [0], color=COLORS[a], lw=1.6,
                   marker=MARKERS[a] if markers else None,
                   markersize=3.5, label=NAMES[a]) for a in ARMS]

def grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#DDE4E8", linewidth=.5)

def trajectory_figure():
    df = pd.read_csv(DATA / "trajectory.csv")
    assert set(df.arm) == set(ARMS)
    assert (df.n_seeds == 5).all()
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 2.53))
    fig.subplots_adjust(left=.075, right=.985, top=.76, bottom=.19, wspace=.27)
    for ax, metric in zip(axes, ["alpha", "validation_ppl"]):
        for arm in ARMS:
            r = df[(df.arm == arm) & (df.step >= 500)].sort_values("step")
            assert len(r) == 40
            x = r.step.to_numpy()
            ax.fill_between(x, r[f"{metric}_ci95_low"].to_numpy(),
                            r[f"{metric}_ci95_high"].to_numpy(),
                            color=COLORS[arm], alpha=.13, linewidth=0)
            ax.plot(x, r[f"{metric}_mean"].to_numpy(),
                    color=COLORS[arm], linewidth=1.55, zorder=3)
            endpoint = r[r.step.isin([2000, 20000])]
            ax.scatter(endpoint.step, endpoint[f"{metric}_mean"],
                       color=COLORS[arm], marker=MARKERS[arm],
                       s=13, linewidths=.3, edgecolors="white", zorder=4,
                       clip_on=False)
        ax.axvline(2000, color="#87939A", linewidth=.65, ls=(0,(2,3)), zorder=0)
        ax.set_xlim(500, 20000)
        ax.set_xticks([2000, 5000, 10000, 15000, 20000], ["2k", "5k", "10k", "15k", "20k"])
        ax.set_xlabel("Optimizer updates", labelpad=4)
        grid(ax)
    axes[0].axhline(2, color="#384551", linewidth=.85, ls=(0,(3,2)), zorder=1)
    axes[0].set_ylim(1.4, 3.9)
    axes[0].set_yticks([1.5, 2, 2.5, 3, 3.5])
    axes[0].set_ylabel(r"Model average $\bar\alpha$", labelpad=4)
    axes[0].set_title("a  Collective spectral trajectory", loc="left", pad=7)
    axes[1].set_ylim(4.0, 11.75)
    axes[1].set_yticks([4, 6, 8, 10])
    axes[1].set_ylabel("Validation character perplexity", labelpad=4)
    axes[1].set_title("b  Prediction improves with training", loc="left", pad=7)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(.54,1.005), handlelength=2,
               columnspacing=2.2)
    save(fig, "trajectories")

def family_figure():
    df = pd.read_csv(DATA / "family_summary.csv")
    families = ["Token embedding", "Position embedding", "QKV",
                "Attention output", "MLP expansion", "MLP projection"]
    labels = ["Token / output (1)", "Position (1)", "QKV (4)", "Attn output (4)",
              "MLP expansion (4)", "MLP projection (4)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.14,2.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=.171, right=.985, top=.78, bottom=.18, wspace=.16)
    offsets = {"sgdm": .21, "native": 0, "modified": -.21}
    ypos = np.arange(len(families))[::-1]
    lows, highs = [], []
    for ax, step in zip(axes, [2000,20000]):
        for y in ypos:
            ax.axhline(y, color="#E6EAEE", linewidth=.55, zorder=0)
        for arm in ARMS:
            r = df[(df.arm == arm) & (df.step == step)].set_index("family").loc[families]
            assert (r.alpha_n == 5).all()
            lows.extend(r.alpha_ci95_low); highs.extend(r.alpha_ci95_high)
            ax.errorbar(r.alpha_mean, ypos + offsets[arm],
                        xerr=r.alpha_ci95_half, color=COLORS[arm],
                        marker=MARKERS[arm], markersize=3.4,
                        linestyle="none", elinewidth=.85, capsize=1.6,
                        markeredgewidth=.55, zorder=3)
        ax.axvline(2, color="#384551", linewidth=.85, ls=(0,(3,2)), zorder=1)
        ax.set_title(f"{'a' if step == 2000 else 'b'}  {step:,} updates", loc="left", pad=7)
        ax.set_xlabel(r"Within-family $\bar\alpha$", labelpad=4)
        ax.set_yticks(ypos, labels)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.spines["left"].set_visible(False)
    lo, hi = min(lows), max(highs)
    axes[0].set_xlim(min(.75,lo-.15),max(6.4,hi+.2))
    axes[0].set_xticks([1,2,3,4,5,6])
    axes[0].set_ylim(-.55,5.55)
    fig.legend(handles=legend_handles(markers=True), loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(.56,1.005), handlelength=1.5,
               columnspacing=2)
    save(fig,"family_comparison")

def sgdm_matrix_figure():
    df = pd.read_csv(DATA / "matrix_fits.csv")
    df = df[(df.arm == "sgdm") & (df.step == 20000)]
    matrices = ["transformer.wte.weight", "transformer.wpe.weight"]
    labels = ["Token / output", "Position"]
    for block in range(4):
        for suffix, label in [("attn.c_attn.weight","QKV"),
                              ("attn.c_proj.weight","attn out"),
                              ("mlp.c_fc.weight","MLP in"),
                              ("mlp.c_proj.weight","MLP out")]:
            matrices.append(f"transformer.h.{block}.{suffix}")
            labels.append(f"B{block+1} {label}")
    assert len(df) == 90 and df.matrix.nunique() == 18
    assert (df.alpha > 0).all()
    fig, ax = plt.subplots(figsize=(3.45,4.1))
    fig.subplots_adjust(left=.305, right=.985, top=.86, bottom=.15)
    for start,end in [(2,5),(10,13)]:
        ax.axhspan(17-end-.48,17-start+.48,facecolor="#F1F5F8",zorder=0)
    jitter = np.linspace(-.19,.19,5)
    means=[]
    for i, name in enumerate(matrices):
        r = df[df.matrix == name].sort_values("seed")
        assert len(r) == 5
        y=17-i
        ax.scatter(r.alpha, y+jitter, s=12, color="#81A9C9", alpha=.85,
                   linewidths=.25,edgecolors="white",zorder=3)
        means.append(r.alpha.mean())
        ax.scatter(r.alpha.mean(),y,s=13,marker="D",color=COLORS["sgdm"],
                   linewidths=.3,edgecolors="white",zorder=4)
    ax.axvline(2, color="#384551", linewidth=.85, ls=(0,(3,2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(1.15,15.5)
    ax.set_ylim(-.7,17.7)
    ax.xaxis.set_major_locator(FixedLocator([1.5,2,3,5,10,15]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1.5","2","3","5","10","15"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(np.arange(18)[::-1],labels,fontsize=6.8)
    ax.tick_params(axis="y",length=0,pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"Matrix exponent $\alpha_\ell$ (log scale)",labelpad=4)
    grid(ax,axis="x")
    ax.set_title("SGDM: all 18 matrices at 20,000 updates",fontsize=8.3,
                 loc="left",pad=23,x=-.43)
    handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor="#81A9C9",
                    markeredgecolor="white",markersize=4,label="Individual seed"),
             Line2D([0],[0],marker="D",color="none",markerfacecolor=COLORS["sgdm"],
                    markeredgecolor="white",markersize=4,label="Mean of five seeds")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.948),ncol=2,
               frameon=False,fontsize=6.9,handletextpad=.25,columnspacing=.8)
    save(fig,"sgdm_matrices")

CAPTIONS = {
"trajectories": (
    "Training trajectories across five seeds. Lines show seed means; shading shows "
    "pointwise 95% Student-t confidence intervals for those means. The horizontal "
    "reference marks alpha=2; the vertical guide marks 2,000 updates. Both panels "
    "start at500 updates; the shared initialization (mean alpha6.255, character "
    "perplexity65.436) is omitted to resolve the trained regime. Character perplexity "
    "is averaged after exponentiating each seed's full-validation cross-entropy. "
    "Lower perplexity is better. All18unique matrices enter each model mean. "
    "The bands quantify seed uncertainty, not layer dispersion or simultaneous coverage."
),
"family_comparison": (
    "Layer families at matched budgets. Each marker averages five within-seed "
    "family estimates; bars show95% Student-t confidence intervals across seeds. "
    "Each family estimate uses the same confidence/tail weights as the model mean, "
    "renormalized within that family. Transformer families contain four matrices "
    "(one per block); each embedding family contains one unique matrix. The tied "
    "token/output matrix appears once. These family estimates are not equally "
    "weighted components of the model mean. The vertical reference marks alpha=2."
),
"sgdm_matrices": (
    "SGDM layer variation at20,000 updates. Pale circles show the five individual "
    "seed fits for each unique matrix; dark diamonds show their arithmetic mean. "
    "The horizontal axis is logarithmic and displays the complete observed range. "
    "B1--B4 identify decoder blocks. MLP in/out are expansion/projection matrices. "
    "The vertical reference marks alpha=2; the tied token/output matrix appears once. "
    "No across-layer error bar is substituted for uncertainty across seeds."
)}

if __name__ == "__main__":
    trajectory_figure()
    family_figure()
    sgdm_matrix_figure()
    (OUT/"captions.txt").write_text("\n\n".join(f"{k}\n{v}" for k,v in CAPTIONS.items())+"\n")
    (OUT/"figure_metadata.json").write_text(json.dumps({
        "colors": COLORS,
        "seed_uncertainty": "pointwise two-sided95% Student-t interval; n=5, df=4",
        "dimensions_inches": {"trajectories":[7.14,2.53],
                              "family_comparison":[7.14,2.75],
                              "sgdm_matrices":[3.45,4.1]},
        "captions": CAPTIONS},indent=2)+"\n")
    print("Wrote three vector PDFs, three PNGs, captions, and figure metadata.")

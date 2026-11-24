"""Rebuild the single v9 section figure from recorded layer-fit CSV data.

Run: python make_section_figure.py
Dependencies: numpy, pandas, matplotlib. No training or network access.
The full provenance and raw-JSON hash checks live in make_figures.py; running
that script first regenerates the input CSVs from the archived measurements.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA, FIGURES = ROOT / "data", ROOT / "figures"
BLUE, PALE, INK = "#2467a0", "#acc5df", "#173e56"
GRID, SPINE, REF = "#dde4e8", "#8e9ba6", "#384551"
SEEDS = list(range(1340, 1345))


def main():
    fits = pd.read_csv(DATA / "sgdm_endpoint_layer_fits.csv")
    fits = fits[(fits.arm == "SGDM") & (fits.step == 20000)].copy()
    summary = pd.read_csv(DATA / "sgdm_layer_endpoint_summary.csv")
    assert len(fits) == 90 and len(summary) == 18
    assert fits.matrix.nunique() == 18 and sorted(fits.seed.unique()) == SEEDS
    assert not fits.duplicated(["matrix", "seed"]).any()
    assert fits.fit_status.eq("success").all()
    assert np.isfinite(fits.alpha).all()
    assert summary.matrix.is_unique and summary.n.eq(5).all()

    matrices = summary.matrix.tolist()
    expected = ["transformer.wte.weight", "transformer.wpe.weight"]
    for block in range(4):
        expected.extend(f"transformer.h.{block}.{name}.weight" for name in
                        ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"))
    assert matrices == expected
    values = np.array([fits[fits.matrix == matrix].sort_values("seed").alpha.to_numpy()
                       for matrix in matrices])
    means, sds = values.mean(axis=1), values.std(axis=1, ddof=1)
    assert np.allclose(means, summary.alpha_mean, atol=1e-12)
    assert np.allclose(sds, summary.alpha_sample_sd, atol=1e-12)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": SPINE, "axes.axisbelow": True,
        "axes.labelsize": 9, "xtick.labelsize": 7.3, "ytick.labelsize": 8,
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(6.9, 2.55))
    fig.subplots_adjust(left=.072, right=.985, bottom=.29, top=.86)
    positions = np.array([0, 1] + [2.6 + b * 4.6 + k for b in range(4) for k in range(4)])
    labels = ["Token/\nhead", "Pos."] + ["QKV", "Attn\nout", "MLP\nin", "MLP\nout"] * 4
    jitter = np.linspace(-.17, .17, 5)
    for j, x in enumerate(positions):
        ax.scatter(x + jitter, values[j], s=13, color=PALE, alpha=.85,
                   edgecolor="white", linewidth=.25, zorder=3)
    ax.errorbar(positions, means, yerr=sds, fmt="o", color=BLUE, ms=3.4,
                markeredgecolor="white", markeredgewidth=.3, capsize=2.4,
                elinewidth=.95, zorder=4)
    ax.axhline(2, color=REF, ls=":", lw=.9, zorder=2)
    ax.text(positions[-1] + .42, 2.24, r"$\alpha=2$", color=REF, fontsize=8,
            ha="right", va="bottom")
    for boundary in [1.8, 6.4, 11., 15.6]:
        ax.axvline(boundary, color=GRID, lw=.7, zorder=0)
    ax.set_xlim(-.65, positions[-1] + .65)
    ymax = max(float(values.max()), float((means + sds).max()))
    ax.set_ylim(.6, max(15.2, ymax + .8))
    ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
    ax.set_ylabel(r"Layer exponent $\alpha$")
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=2.5, pad=2)
    ax.grid(axis="y", color=GRID, lw=.55)
    groups = [(positions[:2].mean(), "Embeddings")]
    groups += [(positions[2 + b * 4:6 + b * 4].mean(), f"Block {b + 1}") for b in range(4)]
    for center, label in groups:
        ax.text(center, -.30, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.2, color=INK, weight="bold")
    ax.set_title("SGDM at 20,000 updates", loc="left", fontsize=9.3,
                 fontweight="bold", color=INK, pad=9)
    legend = [
        Line2D([], [], color=BLUE, marker="o", markersize=3.5, lw=.95,
               label="Mean ± sample SD (5 seeds)"),
        Line2D([], [], color=PALE, marker="o", markersize=3.5, lw=0,
               label="Individual seed"),
    ]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1, 1.20),
              ncol=2, frameon=False, fontsize=7.8, handlelength=1.3,
              columnspacing=1.2, borderaxespad=0)
    FIGURES.mkdir(exist_ok=True)
    fixed = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fig.savefig(FIGURES / "sgdm_layer_variation.pdf",
                metadata={"CreationDate": fixed, "ModDate": fixed})
    fig.savefig(FIGURES / "sgdm_layer_variation.png", dpi=300)
    plt.close(fig)
    print("Verified and plotted 90 successful fits: 18 matrices × 5 seeds; all points shown.")


if __name__ == "__main__":
    main()

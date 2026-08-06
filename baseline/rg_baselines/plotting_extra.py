"""Supplemental effective-rank, gradient, norm, and timing plots."""
from pathlib import Path
import matplotlib.pyplot as plt

def _save(fig,path):
    if path: Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160,bbox_inches="tight")
def plot_4_effective_rank_and_energy(r,*,output_dir=None,show=True):
    f=r.spectral_metrics[r.spectral_metrics.status=="ok"].sort_values(["layer","epoch"])
    items=[("stable_rank","Stable rank"),("participation_ratio","Participation-ratio rank"),("midpoint_energy_fraction","Midpoint energy fraction"),("boundary_overlap_ratio","PL/detX overlap")]
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for ax,(metric,title) in zip(axes.ravel(),items):
        for layer,g in f.groupby("layer"): ax.plot(g.epoch,g[metric],marker="o",label=str(layer).upper())
        ax.set(ylabel=metric,title=title); ax.grid(True,alpha=.3); ax.legend()
    fig.suptitle(f"Additional spectral diagnostics — {r.config.optimizer_label}"); fig.tight_layout(); _save(fig,Path(output_dir)/"4_effective_rank_and_energy.png" if output_dir else None)
    if show: plt.show()
    return fig
def plot_5_optimizer_diagnostics(r,*,output_dir=None,show=True):
    f=r.performance.sort_values("epoch"); fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    axes[0].plot(f.epoch,f.mean_gradient_norm_before_clip,marker="o",label="mean"); axes[0].plot(f.epoch,f.max_gradient_norm_before_clip,marker="s",label="max")
    axes[0].set(ylabel="Gradient norm",title="Gradient norm before clipping"); axes[0].legend()
    axes[1].plot(f.epoch,f.parameter_l2_norm,marker="o"); axes[1].set(ylabel="Parameter L2 norm",title="Whole-model parameter norm")
    for metric,label,marker in [("train_time_sec","training","o"),("evaluation_time_sec","evaluation","s"),("weightwatcher_time_sec","WeightWatcher","^")]: axes[2].plot(f.epoch,f[metric],marker=marker,label=label)
    axes[2].set(xlabel="Epoch",ylabel="Seconds",title="Epoch timing"); axes[2].legend()
    for a in axes: a.grid(True,alpha=.3)
    fig.suptitle(r.config.optimizer_label); fig.tight_layout(); _save(fig,Path(output_dir)/"5_optimizer_diagnostics.png" if output_dir else None)
    if show: plt.show()
    return fig

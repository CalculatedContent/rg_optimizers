"""The four plots explicitly required for every baseline."""
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from .results import BaselineResult

def _save(fig,path):
    if path: Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160,bbox_inches="tight")
def _valid(r): return r.spectral_metrics[r.spectral_metrics.status=="ok"].sort_values(["layer","epoch"])
def plot_0_loss_and_accuracy(r:BaselineResult,*,output_dir:Optional[str|Path]=None,show=True):
    f=r.performance.sort_values("epoch"); fig,ax=plt.subplots(1,2,figsize=(13,5))
    for metric,label,marker in [("train_loss","train","o"),("test_loss","test","s")]: ax[0].plot(f.epoch,f[metric],marker=marker,label=label)
    for metric,label,marker in [("train_accuracy","train","o"),("test_accuracy","test","s")]: ax[1].plot(f.epoch,f[metric],marker=marker,label=label)
    ax[0].set(xlabel="Epoch",ylabel="Cross-entropy loss",title="Train and test loss")
    ax[1].set(xlabel="Epoch",ylabel="Accuracy",title="Train and test accuracy")
    for a in ax: a.grid(True,alpha=.3); a.legend()
    fig.suptitle(r.config.optimizer_label); fig.tight_layout(); _save(fig,Path(output_dir)/"0_train_test_loss_accuracy.png" if output_dir else None)
    if show: plt.show()
    return fig
def plot_1_layerwise_alpha(r,*,output_dir=None,show=True):
    f=_valid(r); fig,ax=plt.subplots(figsize=(10,5.5))
    for layer,g in f.groupby("layer"): ax.plot(g.epoch,g.alpha,marker="o",label=str(layer).upper())
    ax.axhline(2,linestyle="--",label="alpha = 2"); ax.set(xlabel="Epoch",ylabel="WeightWatcher alpha",title=f"Layerwise alpha — {r.config.optimizer_label}")
    ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); _save(fig,Path(output_dir)/"1_layerwise_weightwatcher_alpha.png" if output_dir else None)
    if show: plt.show()
    return fig
def plot_2_original_erg_boundaries(r,*,output_dir=None,show=True):
    f=_valid(r); items=[("detX_num","WeightWatcher detX_num"),("num_pl_spikes","WeightWatcher num_pl_spikes"),("ERG_gap","Full-M ERG_gap")]
    fig,axes=plt.subplots(3,1,figsize=(11,12),sharex=True)
    for ax,(metric,title) in zip(axes,items):
        for layer,g in f.groupby("layer"): ax.plot(g.epoch,g[metric],marker="o",label=str(layer).upper())
        if metric=="ERG_gap": ax.axhline(0,linestyle="--")
        ax.set(ylabel=metric,title=title); ax.grid(True,alpha=.3); ax.legend()
    axes[-1].set_xlabel("Epoch"); fig.suptitle(r.config.optimizer_label); fig.tight_layout(); _save(fig,Path(output_dir)/"2_original_weightwatcher_erg_boundaries.png" if output_dir else None)
    if show: plt.show()
    return fig
def plot_3_midpoint_and_trace_log(r,*,output_dir=None,show=True):
    f=_valid(r); fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for layer,g in f.groupby("layer"):
        axes[0].plot(g.epoch,g.m_midpoint,marker="o",label=str(layer).upper())
        axes[1].plot(g.epoch,g.trace_log_midpoint_per_eval,marker="o",label=str(layer).upper())
    axes[0].set(ylabel="m_midpoint",title="Original midpoint retained rank")
    axes[1].axhline(0,linestyle="--"); axes[1].set(xlabel="Epoch",ylabel="Mean log rescaled eigenvalue",title="Midpoint trace-log per retained eigenvalue")
    for a in axes: a.grid(True,alpha=.3); a.legend()
    fig.suptitle(r.config.optimizer_label); fig.tight_layout(); _save(fig,Path(output_dir)/"3_midpoint_rank_and_trace_log.png" if output_dir else None)
    if show: plt.show()
    return fig

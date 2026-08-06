"""Public plotting API."""
from .plotting_required import *
from .plotting_extra import *

def plot_all(result,*,output_dir=None,show=True):
    funcs=[plot_0_loss_and_accuracy,plot_1_layerwise_alpha,plot_2_original_erg_boundaries,
           plot_3_midpoint_and_trace_log,plot_4_effective_rank_and_energy,plot_5_optimizer_diagnostics]
    return [f(result,output_dir=output_dir,show=show) for f in funcs]

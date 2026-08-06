"""Public plotting API for single runs and independent-seed ensembles."""

from .plotting_required import *
from .plotting_extra import *
from .plotting_replicates import *


def plot_all(result, *, output_dir=None, show=True):
    """Create the legacy single-run plots.

    The baseline notebooks use :func:`plot_all_replicates` so every required
    curve has independent-seed 95% confidence intervals.  This function remains
    available for quick single-run debugging.
    """

    functions = [
        plot_0_loss_and_accuracy,
        plot_1_layerwise_alpha,
        plot_2_original_erg_boundaries,
        plot_3_midpoint_and_trace_log,
        plot_4_effective_rank_and_energy,
        plot_5_optimizer_diagnostics,
    ]
    return [
        function(result, output_dir=output_dir, show=show)
        for function in functions
    ]

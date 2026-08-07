"""Public plotting API for single runs and independent-seed ensembles."""

from .plotting_required import *
from .plotting_extra import *
from .plotting_replicates import *
from .plotting_replicates import plot_all_replicates as _plot_all_replicates_base
from .plotting_traps import *


def plot_all(result, *, output_dir=None, show=True):
    """Create all legacy single-run plots, including correlation traps."""

    functions = [
        plot_0_loss_and_accuracy,
        plot_1_layerwise_alpha,
        plot_2_original_erg_boundaries,
        plot_3_midpoint_and_trace_log,
        plot_4_effective_rank_and_energy,
        plot_5_optimizer_diagnostics,
        plot_layerwise_num_traps,
    ]
    return [function(result, output_dir=output_dir, show=show) for function in functions]


def plot_all_replicates(result, *, output_dir=None, show=True):
    """Create every replicated baseline plot, including ``num_traps``."""

    figures = _plot_all_replicates_base(result, output_dir=output_dir, show=show)
    figures.append(
        plot_7_layerwise_num_traps_with_ci(
            result, output_dir=output_dir, show=show
        )
    )
    return figures

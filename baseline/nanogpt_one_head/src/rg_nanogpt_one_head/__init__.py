"""One-block, one-head nanoGPT optimizer baselines on pinned FineWeb-Edu."""

from .analysis import (
    MATRIX_COLORS,
    OPTIMIZER_COLORS,
    OPTIMIZER_LABELS,
    final_test_summary,
    load_epoch_metrics,
    load_layer_metrics,
    load_metrics,
    load_spectral_summary,
    load_test_results,
    plot_epoch_metric,
    plot_layer_metric,
    plot_spectral_optimizer_summary,
    run_status_table,
)
from .config import (
    SUPPORTED_OPTIMIZERS,
    canonical_seeds,
    load_config,
    roots,
)
from .data import prepare_fineweb_edu
from .model import GPT, GPTConfig
from .runtime import choose_device
from .training import (
    run_all_replicates,
    run_one,
    run_optimizer_replicates,
)

__all__ = [
    "GPT",
    "GPTConfig",
    "MATRIX_COLORS",
    "OPTIMIZER_COLORS",
    "OPTIMIZER_LABELS",
    "SUPPORTED_OPTIMIZERS",
    "canonical_seeds",
    "choose_device",
    "final_test_summary",
    "load_config",
    "load_epoch_metrics",
    "load_layer_metrics",
    "load_metrics",
    "load_spectral_summary",
    "load_test_results",
    "plot_epoch_metric",
    "plot_layer_metric",
    "plot_spectral_optimizer_summary",
    "prepare_fineweb_edu",
    "roots",
    "run_all_replicates",
    "run_one",
    "run_optimizer_replicates",
    "run_status_table",
]

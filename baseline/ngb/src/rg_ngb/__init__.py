"""NGB v4 tuned nanoGPT baselines."""

from .analysis import (
    MATRIX_COLORS,
    OPTIMIZER_COLORS,
    OPTIMIZER_LABELS,
    completed_seed_sets,
    diagnostic_summary,
    discover_matched_seeds,
    final_test_summary,
    load_epoch_metrics,
    load_layer_metrics,
    load_metrics,
    load_spectral_summary,
    load_test_results,
    paired_optimizer_differences,
    plot_epoch_metric,
    plot_layer_metric,
    run_diagnostics,
    run_status_table,
)
from .config import (
    DEFAULT_ROOT,
    SUPPORTED_OPTIMIZERS,
    canonical_seeds,
    load_config,
    roots,
)
from .data import prepare_fineweb_edu
from .model import GPT, GPTConfig, transformer_matrix_items
from .training import run_all_replicates, run_optimizer_replicates

__all__ = [
    "DEFAULT_ROOT",
    "GPT",
    "GPTConfig",
    "MATRIX_COLORS",
    "OPTIMIZER_COLORS",
    "OPTIMIZER_LABELS",
    "SUPPORTED_OPTIMIZERS",
    "canonical_seeds",
    "completed_seed_sets",
    "diagnostic_summary",
    "discover_matched_seeds",
    "final_test_summary",
    "load_config",
    "load_epoch_metrics",
    "load_layer_metrics",
    "load_metrics",
    "load_spectral_summary",
    "load_test_results",
    "paired_optimizer_differences",
    "plot_epoch_metric",
    "plot_layer_metric",
    "prepare_fineweb_edu",
    "roots",
    "run_all_replicates",
    "run_diagnostics",
    "run_optimizer_replicates",
    "run_status_table",
    "transformer_matrix_items",
]

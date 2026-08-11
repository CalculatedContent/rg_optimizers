"""Shared nanoGPT optimizer-baseline runtime for v3 and NGB v4."""

from .analysis import (
    MATRIX_COLORS,
    OPTIMIZER_COLORS,
    OPTIMIZER_LABELS,
    discover_complete_seeds,
    discover_matched_complete_seeds,
    final_test_summary,
    load_epoch_metrics,
    load_layer_metrics,
    load_metrics,
    load_spectral_summary,
    load_test_results,
    mean_ci95,
    paired_test_differences,
    plot_epoch_metric,
    plot_layer_metric,
    plot_spectral_optimizer_summary,
    run_diagnostics_table,
    run_status_table,
    summarize_run_diagnostics,
)
from .config import (
    SUPPORTED_OPTIMIZERS,
    canonical_seeds,
    expected_transformer_matrix_count,
    load_config,
    roots,
    run_slug,
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
    "discover_complete_seeds",
    "discover_matched_complete_seeds",
    "expected_transformer_matrix_count",
    "final_test_summary",
    "load_config",
    "load_epoch_metrics",
    "load_layer_metrics",
    "load_metrics",
    "load_spectral_summary",
    "load_test_results",
    "mean_ci95",
    "paired_test_differences",
    "plot_epoch_metric",
    "plot_layer_metric",
    "plot_spectral_optimizer_summary",
    "prepare_fineweb_edu",
    "roots",
    "run_all_replicates",
    "run_diagnostics_table",
    "run_one",
    "run_optimizer_replicates",
    "run_slug",
    "run_status_table",
    "summarize_run_diagnostics",
]

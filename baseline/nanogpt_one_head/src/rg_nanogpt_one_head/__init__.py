"""One-block, one-head nanoGPT optimizer baselines on pinned FineWeb-Edu."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Callable

_ANALYSIS_CALLABLE_EXPORTS = (
    "final_test_summary",
    "load_epoch_metrics",
    "load_layer_metrics",
    "load_metrics",
    "load_spectral_summary",
    "load_test_results",
    "plot_epoch_metric",
    "plot_layer_metric",
    "plot_spectral_optimizer_summary",
    "run_status_table",
)


def _analysis_unavailable(error: ImportError):
    def unavailable(*args: Any, **kwargs: Any):
        del args, kwargs
        raise RuntimeError(
            "rg_nanogpt_one_head analysis utilities are unavailable. "
            "Training and checkpoint recovery remain usable; reinstall the "
            "experiment dependencies to restore plotting and aggregation."
        ) from error

    return unavailable


def _analysis_fallback(error: ImportError) -> ModuleType:
    """Return the minimal analysis surface required by training launchers."""

    module_name = f"{__name__}.analysis"
    module = ModuleType(
        module_name,
        "Training-safe fallback used only when the optional analysis module "
        "cannot be imported.",
    )
    module.__package__ = __name__
    module.IMPORT_ERROR = repr(error)
    module.OPTIMIZER_LABELS = {
        "sgd_momentum": "SGD + Nesterov",
        "adamw": "AdamW",
        "muon": "Muon + auxiliary AdamW",
    }
    module.OPTIMIZER_COLORS = {
        "sgd_momentum": "#0072B2",
        "adamw": "#D55E00",
        "muon": "#009E73",
    }
    module.MATRIX_COLORS = {
        "W_Q": "#0072B2",
        "W_K": "#E69F00",
        "W_V": "#009E73",
        "W_O": "#D55E00",
        "W_MLP_IN": "#CC79A7",
        "W_MLP_OUT": "#56B4E9",
    }
    unavailable = _analysis_unavailable(error)
    for name in _ANALYSIS_CALLABLE_EXPORTS:
        setattr(module, name, unavailable)
    return module


def _load_analysis_module(
    importer: Callable[[str], ModuleType] = importlib.import_module,
    *,
    register: bool = True,
) -> ModuleType:
    """Load analysis when available without making it a training dependency."""

    module_name = f"{__name__}.analysis"
    try:
        module = importer(module_name)
    except ImportError as error:
        module = _analysis_fallback(error)
    if register:
        sys.modules[module_name] = module
    return module


# MuonClip updates these label dictionaries while installing its opt-in
# extension. A plotting/import failure must not prevent a fresh MPS worker from
# resuming a valid training checkpoint.
analysis = _load_analysis_module()
MATRIX_COLORS = analysis.MATRIX_COLORS
OPTIMIZER_COLORS = analysis.OPTIMIZER_COLORS
OPTIMIZER_LABELS = analysis.OPTIMIZER_LABELS
final_test_summary = analysis.final_test_summary
load_epoch_metrics = analysis.load_epoch_metrics
load_layer_metrics = analysis.load_layer_metrics
load_metrics = analysis.load_metrics
load_spectral_summary = analysis.load_spectral_summary
load_test_results = analysis.load_test_results
plot_epoch_metric = analysis.plot_epoch_metric
plot_layer_metric = analysis.plot_layer_metric
plot_spectral_optimizer_summary = analysis.plot_spectral_optimizer_summary
run_status_table = analysis.run_status_table

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
    "analysis",
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

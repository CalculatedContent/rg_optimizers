"""Clean optimizer/model baselines for RG optimizer experiments.

Plotting is imported lazily so the numerical package and its tests do not
require Matplotlib until a notebook actually requests a figure.
"""

from .config import (
    MNIST_REFERENCE_INITIALIZATION,
    MNIST_REFERENCE_RECIPE_VERSION,
    MNIST_REFERENCE_SUITE_SLUG,
    BaselineConfig,
)
from .diagnostics import (
    SpectralCheckpoint,
    measure_weightwatcher_checkpoint,
    spectral_metrics_from_esd,
)
from .mnist_runtime import run_baseline
from .model import MLP3
from .muon import (
    MuonWithAuxAdamW,
    SGDMomentumMuon,
    zeropower_via_newton_schulz_5,
)
from .optimizers import (
    build_optimizer,
    optimizer_group_rows,
    scheduled_learning_rates,
    set_scheduled_learning_rates,
    warmup_cosine_learning_rate,
)
from .replicates import (
    DEFAULT_BASELINE_SEEDS,
    BaselineReplicateResult,
    run_baseline_replicates,
    validate_replicate_result,
)
from .results import BaselineResult, validate_result
from .statistics import student_t_critical_95, summarize_numeric_metrics


def plot_all(*args, **kwargs):
    from .plotting import plot_all as implementation

    return implementation(*args, **kwargs)


def plot_all_replicates(*args, **kwargs):
    from .plotting import plot_all_replicates as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "BaselineConfig",
    "BaselineReplicateResult",
    "BaselineResult",
    "DEFAULT_BASELINE_SEEDS",
    "MLP3",
    "MNIST_REFERENCE_INITIALIZATION",
    "MNIST_REFERENCE_RECIPE_VERSION",
    "MNIST_REFERENCE_SUITE_SLUG",
    "MuonWithAuxAdamW",
    "SGDMomentumMuon",
    "SpectralCheckpoint",
    "build_optimizer",
    "measure_weightwatcher_checkpoint",
    "optimizer_group_rows",
    "plot_all",
    "plot_all_replicates",
    "run_baseline",
    "run_baseline_replicates",
    "scheduled_learning_rates",
    "set_scheduled_learning_rates",
    "spectral_metrics_from_esd",
    "student_t_critical_95",
    "summarize_numeric_metrics",
    "validate_replicate_result",
    "validate_result",
    "warmup_cosine_learning_rate",
    "zeropower_via_newton_schulz_5",
]

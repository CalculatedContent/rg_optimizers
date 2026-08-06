"""Clean MLP3/MNIST optimizer baselines."""

from .config import BaselineConfig
from .diagnostics import (
    SpectralCheckpoint,
    measure_weightwatcher_checkpoint,
    spectral_metrics_from_esd,
)
from .model import MLP3
from .muon import SGDMomentumMuon, zeropower_via_newton_schulz_5
from .optimizers import build_optimizer, optimizer_group_rows
from .plotting import plot_all, plot_all_replicates
from .replicates import (
    DEFAULT_BASELINE_SEEDS,
    BaselineReplicateResult,
    run_baseline_replicates,
    validate_replicate_result,
)
from .results import BaselineResult, validate_result
from .runner import run_baseline
from .statistics import student_t_critical_95, summarize_numeric_metrics

__all__ = [
    "BaselineConfig",
    "BaselineReplicateResult",
    "BaselineResult",
    "DEFAULT_BASELINE_SEEDS",
    "MLP3",
    "SGDMomentumMuon",
    "SpectralCheckpoint",
    "build_optimizer",
    "measure_weightwatcher_checkpoint",
    "optimizer_group_rows",
    "plot_all",
    "plot_all_replicates",
    "run_baseline",
    "run_baseline_replicates",
    "spectral_metrics_from_esd",
    "student_t_critical_95",
    "summarize_numeric_metrics",
    "validate_replicate_result",
    "validate_result",
    "zeropower_via_newton_schulz_5",
]

"""RG trace-log optimizer extension and WeightWatcher diagnostics."""

from .geometry import (
    CorrectionResult,
    TraceLogGeometry,
    correct_trace_log_component,
    trace_log_geometry,
)
from .mnist_experiment import (
    MLP3,
    MNISTExperimentConfig,
    MNISTExperimentResult,
    run_mnist_comparison,
)
from .spectral import shell_balance_metrics
from .weightwatcher import WeightWatcherCheckpoint, analyze_weightwatcher_checkpoint
from .wrapper import TraceLogConfig, TraceLogRGWrapper

__all__ = [
    "CorrectionResult",
    "MLP3",
    "MNISTExperimentConfig",
    "MNISTExperimentResult",
    "TraceLogConfig",
    "TraceLogGeometry",
    "TraceLogRGWrapper",
    "WeightWatcherCheckpoint",
    "analyze_weightwatcher_checkpoint",
    "correct_trace_log_component",
    "run_mnist_comparison",
    "shell_balance_metrics",
    "trace_log_geometry",
]

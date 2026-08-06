"""Self-consistent ECS trace-log optimizer extension."""

from .ecs import (
    AdaptiveSupportState,
    SelfConsistentECS,
    bulk_effective_normalization_dimension,
    effective_contributor_count,
    self_consistent_candidate_scan,
    solve_self_consistent_ecs,
)
from .geometry import (
    AdaptiveTraceLogGeometry,
    CorrectionResult,
    adaptive_trace_log_geometry,
    correct_trace_log_component,
)
from .mnist_experiment import (
    MLP3,
    MNISTExperimentConfig,
    MNISTExperimentResult,
    run_mnist_comparison,
)
from .weightwatcher import (
    SelfConsistentWeightWatcherCheckpoint,
    analyze_weightwatcher_checkpoint,
)
from .wrapper import (
    SelfConsistentTraceLogConfig,
    SelfConsistentTraceLogRGWrapper,
)

__all__ = [
    "AdaptiveSupportState",
    "AdaptiveTraceLogGeometry",
    "CorrectionResult",
    "MLP3",
    "MNISTExperimentConfig",
    "MNISTExperimentResult",
    "SelfConsistentECS",
    "SelfConsistentTraceLogConfig",
    "SelfConsistentTraceLogRGWrapper",
    "SelfConsistentWeightWatcherCheckpoint",
    "adaptive_trace_log_geometry",
    "analyze_weightwatcher_checkpoint",
    "bulk_effective_normalization_dimension",
    "correct_trace_log_component",
    "effective_contributor_count",
    "run_mnist_comparison",
    "self_consistent_candidate_scan",
    "solve_self_consistent_ecs",
]

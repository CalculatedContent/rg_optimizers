"""ECS probe-loss TraceWall optimizer experiments."""

from .config import BaseOptimizerConfig, ExperimentConfig, TraceWallConfig
from .ecs import (
    ECSSelection,
    ECSSVDState,
    compute_ecs_svd,
    participation_ratio,
    project_gradient_to_ecs,
    select_self_consistent_ecs,
)
from .experiment import (
    MLP3,
    PairedExperimentResult,
    WarmupCosineSchedule,
    build_base_optimizer,
    choose_device,
    evaluate,
    load_mnist,
    run_paired_experiment,
    set_seed,
    state_dict_checksum,
)
from .optimizer import (
    CorrectionRecord,
    ECSProbeLossTraceWall,
    LayerCorrectionRecord,
)
from .plotting import plot_all
from .sampler import ProbeDraw, RotatingSubsetSampler, materialize_probe_batches

__all__ = [
    "BaseOptimizerConfig",
    "CorrectionRecord",
    "ECSProbeLossTraceWall",
    "ECSSelection",
    "ECSSVDState",
    "ExperimentConfig",
    "LayerCorrectionRecord",
    "MLP3",
    "PairedExperimentResult",
    "ProbeDraw",
    "RotatingSubsetSampler",
    "TraceWallConfig",
    "WarmupCosineSchedule",
    "build_base_optimizer",
    "choose_device",
    "compute_ecs_svd",
    "evaluate",
    "load_mnist",
    "materialize_probe_batches",
    "participation_ratio",
    "plot_all",
    "project_gradient_to_ecs",
    "run_paired_experiment",
    "select_self_consistent_ecs",
    "set_seed",
    "state_dict_checksum",
]

__version__ = "0.1.0"

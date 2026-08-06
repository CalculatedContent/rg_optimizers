"""Local-delta ECS WW-PGD optimizer extension."""

from .config import LocalDeltaECSConfig, MNISTRunConfig
from .ecs import (
    ECSScanResult,
    LocalDeltaCorrectionResult,
    LocalECSGeometry,
    damp_delta_outside_ecs,
    local_ecs_geometry,
    select_self_consistent_ecs,
    split_delta_by_ecs,
)
from .optimizer import LocalDeltaECSOptimizer

__all__ = [
    "ECSScanResult",
    "LocalDeltaCorrectionResult",
    "LocalDeltaECSConfig",
    "LocalDeltaECSOptimizer",
    "LocalECSGeometry",
    "MNISTRunConfig",
    "damp_delta_outside_ecs",
    "local_ecs_geometry",
    "select_self_consistent_ecs",
    "split_delta_by_ecs",
]

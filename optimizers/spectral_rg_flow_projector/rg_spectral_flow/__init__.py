"""Spectral RG-flow projection against the trivial/collapse branch."""

from .ecs import (
    AdaptiveSupportState,
    SelfConsistentECS,
    solve_self_consistent_ecs,
)
from .flow import (
    SpectralFlowCorrection,
    SpectralFlowGeometry,
    collapse_potential_from_shape,
    remove_trivial_branch_component,
    trivial_branch_flow_vector,
)
from .weightwatcher import SpectralFlowCheckpoint, analyze_weightwatcher_checkpoint
from .wrapper import SpectralRGFlowConfig, SpectralRGFlowProjector

__all__ = [
    "AdaptiveSupportState",
    "SelfConsistentECS",
    "SpectralFlowCheckpoint",
    "SpectralFlowCorrection",
    "SpectralFlowGeometry",
    "SpectralRGFlowConfig",
    "SpectralRGFlowProjector",
    "analyze_weightwatcher_checkpoint",
    "collapse_potential_from_shape",
    "remove_trivial_branch_component",
    "solve_self_consistent_ecs",
    "trivial_branch_flow_vector",
]

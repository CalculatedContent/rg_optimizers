"""AdaptiveSpectralGuard optimizer package."""

from .config import (
    ControllerConfig,
    GuardConfig,
    LayerPolicy,
    default_layer_policies,
    preset_policies,
    stabilized_layer_policies,
)
from .controller import AdaptiveSpectralController, LayerState
from .experiment import (
    MNISTGuardExperimentConfig,
    MNISTGuardExperimentResult,
    run_mnist_guard_comparison,
)
from .optimizer import AdaptiveSpectralGuard
from .stabilized_v2 import (
    STABILIZED_V2_API,
    StabilizedV2Configuration,
    assert_stabilized_v2_controller_frame,
    build_stabilized_v2_configuration,
    install_stabilized_v2_live_report,
    run_stabilized_v2_mnist,
    run_stabilized_v2_preflight,
    stabilized_v2_policy_table,
    validate_stabilized_v2_configuration,
)

__version__ = "0.3.0"

__all__ = [
    "AdaptiveSpectralController",
    "AdaptiveSpectralGuard",
    "ControllerConfig",
    "GuardConfig",
    "LayerPolicy",
    "LayerState",
    "MNISTGuardExperimentConfig",
    "MNISTGuardExperimentResult",
    "STABILIZED_V2_API",
    "StabilizedV2Configuration",
    "assert_stabilized_v2_controller_frame",
    "build_stabilized_v2_configuration",
    "default_layer_policies",
    "install_stabilized_v2_live_report",
    "preset_policies",
    "run_mnist_comparison",
    "run_mnist_guard_comparison",
    "run_stabilized_v2_mnist",
    "run_stabilized_v2_preflight",
    "stabilized_layer_policies",
    "stabilized_v2_policy_table",
    "validate_stabilized_v2_configuration",
]

# Backward-compatible alias retained for callers that imported the old name.
run_mnist_comparison = run_mnist_guard_comparison

"""Full matrix-log RG optimizer."""

from .config import (
    CorrectionMode,
    FullMatrixLogConfig,
    MomentumProjectionMode,
    NormalizationMode,
)
from .cone import (
    ActiveSetQPSolution,
    ConeProjectionResult,
    project_matrix_log_cone,
    solve_active_set_nonnegative_qp,
)
from .geometry import (
    MatrixLogGeometry,
    ProjectionResult,
    full_matrix_log_geometry,
    mode_drifts,
    remove_inward_matrix_log_flow,
)
from .support import (
    MatrixLogSupport,
    SupportCheckpoint,
    analyze_supports,
    build_support,
    self_consistent_dimension_from_eigenvalues,
)
from .wrapper import FullMatrixLogRG, project_sgd_momentum_buffer

__all__ = [
    "ActiveSetQPSolution",
    "ConeProjectionResult",
    "CorrectionMode",
    "FullMatrixLogConfig",
    "FullMatrixLogRG",
    "MatrixLogGeometry",
    "MatrixLogSupport",
    "MomentumProjectionMode",
    "NormalizationMode",
    "ProjectionResult",
    "SupportCheckpoint",
    "analyze_supports",
    "build_support",
    "full_matrix_log_geometry",
    "mode_drifts",
    "project_matrix_log_cone",
    "project_sgd_momentum_buffer",
    "remove_inward_matrix_log_flow",
    "self_consistent_dimension_from_eigenvalues",
    "solve_active_set_nonnegative_qp",
]

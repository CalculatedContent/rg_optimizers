"""Full matrix-log RG optimizer."""

from .config import (
    CorrectionMode,
    EffectiveRankMethod,
    FullMatrixLogConfig,
    MomentumProjectionMode,
    NormalizationMode,
)
from .cone import (
    ActiveSetQPSolution,
    project_matrix_log_cone,
    solve_active_set_nonnegative_qp,
)
from .geometry import (
    MatrixLogGeometry,
    ProjectionResult,
    full_matrix_log_geometry,
    matrix_log_mode_drifts,
    matrix_log_mode_gram,
    remove_inward_matrix_log_flow,
)
from .support import MatrixLogSupport, SupportCheckpoint, analyze_supports
from .wrapper import FullMatrixLogProjectedSGD, FullMatrixLogRG

__all__ = [
    "ActiveSetQPSolution",
    "CorrectionMode",
    "EffectiveRankMethod",
    "FullMatrixLogConfig",
    "FullMatrixLogProjectedSGD",
    "FullMatrixLogRG",
    "MatrixLogGeometry",
    "MatrixLogSupport",
    "MomentumProjectionMode",
    "NormalizationMode",
    "ProjectionResult",
    "SupportCheckpoint",
    "analyze_supports",
    "full_matrix_log_geometry",
    "matrix_log_mode_drifts",
    "matrix_log_mode_gram",
    "project_matrix_log_cone",
    "remove_inward_matrix_log_flow",
    "solve_active_set_nonnegative_qp",
]

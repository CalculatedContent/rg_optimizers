"""Public spectral RG-flow API.

The implementation is split across ``shape``, ``geometry``, and ``correction``
to keep the individual experimental components independently testable.
"""

from .correction import SpectralFlowCorrection, remove_trivial_branch_component
from .geometry import SpectralFlowGeometry, spectral_flow_geometry
from .shape import (
    CollapsePotential,
    centered_log_eigenvalue_shape,
    collapse_potential_from_shape,
    effective_rank_from_shape,
    rank_alpha_proxy_from_shape,
    spectral_probabilities_from_shape,
    trivial_branch_flow_vector,
)

__all__ = [
    "CollapsePotential",
    "SpectralFlowCorrection",
    "SpectralFlowGeometry",
    "centered_log_eigenvalue_shape",
    "collapse_potential_from_shape",
    "effective_rank_from_shape",
    "rank_alpha_proxy_from_shape",
    "remove_trivial_branch_component",
    "spectral_flow_geometry",
    "spectral_probabilities_from_shape",
    "trivial_branch_flow_vector",
]

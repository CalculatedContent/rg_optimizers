"""Configuration for the spectral RG-flow projector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .shape import CollapsePotential


@dataclass(frozen=True)
class SpectralRGFlowConfig:
    """Configuration for the finite spectral-flow projection experiment."""

    collapse_potential: CollapsePotential = "participation_ratio"
    projection_strength: float = 1.0
    min_alignment_cosine: float = 0.0
    max_abs_log_eigenvalue_correction: Optional[float] = 0.20
    max_correction_ratio: Optional[float] = 0.10
    preserve_frobenius_norm: bool = True
    min_retained: int = 20
    apply_every_steps: int = 25
    warmup_steps: int = 0
    eps: float = 1e-12

    def validate(self) -> None:
        if self.collapse_potential not in {"participation_ratio", "entropy"}:
            raise ValueError(
                f"Unknown collapse_potential: {self.collapse_potential!r}"
            )
        if not 0.0 <= float(self.projection_strength) <= 1.0:
            raise ValueError("projection_strength must lie in [0, 1].")
        if not -1.0 <= float(self.min_alignment_cosine) <= 1.0:
            raise ValueError("min_alignment_cosine must lie in [-1, 1].")
        if (
            self.max_abs_log_eigenvalue_correction is not None
            and self.max_abs_log_eigenvalue_correction < 0.0
        ):
            raise ValueError(
                "max_abs_log_eigenvalue_correction must be non-negative or None."
            )
        if self.max_correction_ratio is not None and self.max_correction_ratio < 0.0:
            raise ValueError("max_correction_ratio must be non-negative or None.")
        if int(self.min_retained) < 2:
            raise ValueError("min_retained must be at least two.")
        if int(self.apply_every_steps) < 1:
            raise ValueError("apply_every_steps must be positive.")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

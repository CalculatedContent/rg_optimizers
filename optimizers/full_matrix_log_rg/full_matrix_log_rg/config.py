from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FullMatrixLogConfig:
    projection_strength: float = 1.0
    max_correction_ratio: float | None = 0.10
    apply_every_steps: int = 1
    min_retained_rank: int = 3
    ridge_relative: float = 1e-6
    eps: float = 1e-12
    normalization_dimension: float | None = None

    def validate(self) -> None:
        if not 0.0 <= float(self.projection_strength) <= 1.0:
            raise ValueError('projection_strength must lie in [0, 1]')
        if self.max_correction_ratio is not None and self.max_correction_ratio < 0.0:
            raise ValueError('max_correction_ratio must be non-negative or None')
        if int(self.apply_every_steps) < 1:
            raise ValueError('apply_every_steps must be >= 1')
        if int(self.min_retained_rank) < 1:
            raise ValueError('min_retained_rank must be >= 1')
        if float(self.ridge_relative) < 0.0:
            raise ValueError('ridge_relative must be non-negative')

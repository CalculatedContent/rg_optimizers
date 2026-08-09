"""Configuration for the full matrix-log RG optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CorrectionMode = Literal["radial", "modewise", "cone"]
MomentumProjectionMode = Literal["post_step", "projected_state"]
NormalizationMode = Literal["full_m", "self_consistent"]
EffectiveRankMethod = Literal["participation_ratio", "entropy", "stable_rank"]


@dataclass(frozen=True)
class FullMatrixLogConfig:
    """Controls one-sided projection away from the trivial matrix manifold.

    ``cone`` + ``projected_state`` is the primary implementation. ``radial`` is
    the conservative potential-flow control. ``modewise`` and ``post_step`` are
    retained as legacy ablations.
    """

    mode: CorrectionMode = "cone"
    momentum_projection: MomentumProjectionMode = "projected_state"
    normalization: NormalizationMode = "self_consistent"
    effective_rank_method: EffectiveRankMethod = "participation_ratio"
    normalization_gamma: float = 0.0

    projection_strength: float = 1.0
    max_correction_ratio: float | None = 0.10
    apply_every_steps: int = 100
    warmup_steps: int = 0
    min_retained_rank: int = 3

    ridge_relative: float = 1e-6
    gram_ridge_relative: float = 1e-6
    cone_tolerance: float = 1e-6
    cone_max_iterations: int = 128
    log_deadband: float = 1e-6
    eps: float = 1e-12

    parameter_names: tuple[str, ...] | None = None
    require_support: bool = True

    def validate(self) -> None:
        if self.mode not in {"radial", "modewise", "cone"}:
            raise ValueError(f"Unknown correction mode: {self.mode!r}")
        if self.momentum_projection not in {"post_step", "projected_state"}:
            raise ValueError(
                f"Unknown momentum projection mode: {self.momentum_projection!r}"
            )
        if self.normalization not in {"full_m", "self_consistent"}:
            raise ValueError(f"Unknown normalization mode: {self.normalization!r}")
        if self.effective_rank_method not in {
            "participation_ratio",
            "entropy",
            "stable_rank",
        }:
            raise ValueError(
                f"Unknown effective-rank method: {self.effective_rank_method!r}"
            )
        if not 0.0 <= float(self.normalization_gamma) <= 1.0:
            raise ValueError("normalization_gamma must lie in [0, 1]")
        if not 0.0 <= float(self.projection_strength) <= 1.0:
            raise ValueError("projection_strength must lie in [0, 1]")
        if self.max_correction_ratio is not None and self.max_correction_ratio < 0.0:
            raise ValueError("max_correction_ratio must be non-negative or None")
        if int(self.apply_every_steps) < 1:
            raise ValueError("apply_every_steps must be >= 1")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        if int(self.min_retained_rank) < 1:
            raise ValueError("min_retained_rank must be >= 1")
        if float(self.ridge_relative) < 0.0:
            raise ValueError("ridge_relative must be non-negative")
        if float(self.gram_ridge_relative) < 0.0:
            raise ValueError("gram_ridge_relative must be non-negative")
        if float(self.cone_tolerance) <= 0.0:
            raise ValueError("cone_tolerance must be positive")
        if int(self.cone_max_iterations) < 1:
            raise ValueError("cone_max_iterations must be >= 1")
        if float(self.log_deadband) < 0.0:
            raise ValueError("log_deadband must be non-negative")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if self.parameter_names is not None and not self.parameter_names:
            raise ValueError("parameter_names must be non-empty or None")

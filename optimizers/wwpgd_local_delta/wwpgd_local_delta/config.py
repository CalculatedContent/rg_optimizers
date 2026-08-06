"""Configuration objects for local-delta ECS WW-PGD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ECSReference = Literal["epoch_start", "epoch_end"]
OptimizerKind = Literal["adamw", "sgd_momentum"]


@dataclass(frozen=True)
class LocalDeltaECSConfig:
    """Configuration for epoch-boundary local-delta ECS damping.

    The completed epoch displacement is decomposed in the oriented layer
    coordinates used by the trace-log/ECS construction. If ``Delta_perp`` is
    the component outside the retained ECS, the applied displacement is

        Delta_new = Delta - correction_fraction * Delta_perp.

    ``reference='epoch_end'`` is the default because the requested operation is
    to project the completed optimizer displacement into the new/current ECS of
    the proposed endpoint. ``epoch_start`` remains available as an ablation.
    """

    correction_fraction: float = 0.25
    apply_every_epochs: int = 1
    warmup_epochs: int = 0
    min_retained: int = 3
    max_retained: Optional[int] = None
    normalization_gamma: float = 0.0
    reference: ECSReference = "epoch_end"
    parameter_name_filter: Optional[tuple[str, ...]] = None
    eps: float = 1e-12

    def validate(self) -> None:
        if not 0.0 <= float(self.correction_fraction) <= 1.0:
            raise ValueError("correction_fraction must lie in [0, 1].")
        if int(self.apply_every_epochs) < 1:
            raise ValueError("apply_every_epochs must be positive.")
        if int(self.warmup_epochs) < 0:
            raise ValueError("warmup_epochs must be non-negative.")
        if int(self.min_retained) < 1:
            raise ValueError("min_retained must be positive.")
        if self.max_retained is not None and int(self.max_retained) < int(self.min_retained):
            raise ValueError("max_retained must be >= min_retained when supplied.")
        if not 0.0 <= float(self.normalization_gamma) <= 1.0:
            raise ValueError("normalization_gamma must lie in [0, 1].")
        if self.reference not in {"epoch_start", "epoch_end"}:
            raise ValueError("reference must be 'epoch_start' or 'epoch_end'.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")


@dataclass(frozen=True)
class MNISTRunConfig:
    """MLP3-MNIST experiment settings used by the notebooks."""

    optimizer_kind: OptimizerKind = "adamw"
    epochs: int = 10
    batch_size: int = 128
    test_batch_size: int = 512
    hidden_width: int = 512
    adamw_lr: float = 1e-3
    adamw_weight_decay: float = 1e-4
    sgd_lr: float = 5e-2
    sgd_momentum: float = 0.9
    sgd_weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    correction_fraction: float = 0.25
    apply_every_epochs: int = 1
    warmup_epochs: int = 0
    normalization_gamma: float = 0.0
    ecs_reference: ECSReference = "epoch_end"
    corrected_parameters: Optional[tuple[str, ...]] = None
    train_limit: Optional[int] = None
    test_limit: Optional[int] = None
    ww_enabled: bool = True
    ww_required: bool = True
    ww_min_evals: int = 8
    ww_svd_method: str = "accurate"
    data_dir: str = "./data"
    output_dir: str = "./runs_local_delta_ecs"
    seeds: tuple[int, ...] = field(
        default_factory=lambda: (1337, 2027, 4099, 7919, 104729)
    )

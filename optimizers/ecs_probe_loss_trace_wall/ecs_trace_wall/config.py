"""Configuration objects for the ECS probe-loss TraceWall experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Optional

OptimizerName = Literal["adamw", "sgd_momentum"]
ProjectionMode = Literal["core", "rank_m_tangent"]
SVDDevice = Literal["cpu", "model"]


@dataclass(frozen=True)
class BaseOptimizerConfig:
    """Repository-standard MLP3/MNIST optimizer and schedule settings.

    The peak learning rates and optimizer coefficients match the clean
    baselines already present in ``baseline/``.  Both the baseline and the
    TraceWall arm use the same linear-warmup/cosine-decay schedule.
    """

    name: OptimizerName
    peak_learning_rate: float
    weight_decay: float
    warmup_epochs: float = 1.0
    minimum_learning_rate_ratio: float = 0.05

    # SGD-only settings.
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False

    # AdamW-only settings.
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False

    @classmethod
    def adamw_baseline(cls) -> "BaseOptimizerConfig":
        return cls(
            name="adamw",
            peak_learning_rate=1e-3,
            weight_decay=1e-2,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            amsgrad=False,
        )

    @classmethod
    def sgd_momentum_baseline(cls) -> "BaseOptimizerConfig":
        return cls(
            name="sgd_momentum",
            peak_learning_rate=5e-2,
            weight_decay=1e-4,
            momentum=0.9,
            dampening=0.0,
            nesterov=False,
        )

    def validate(self) -> None:
        if self.name not in {"adamw", "sgd_momentum"}:
            raise ValueError(f"unknown optimizer {self.name!r}")
        if self.peak_learning_rate <= 0.0:
            raise ValueError("peak_learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_epochs < 0.0:
            raise ValueError("warmup_epochs must be non-negative")
        if not 0.0 < self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must lie in (0, 1]")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if self.dampening < 0.0:
            raise ValueError("dampening must be non-negative")
        if self.nesterov and (self.momentum <= 0.0 or self.dampening != 0.0):
            raise ValueError("Nesterov requires positive momentum and zero dampening")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must lie in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class TraceWallConfig:
    """ECS probe-loss correction settings.

    A correction is a direct post-optimizer matrix displacement.  Its direction
    is the negative gradient of the loss measured on a rotating training probe
    subset while all selected matrices are replaced by their current ECS-
    truncated SVDs.  The gradient is projected back into the same ECS before it
    is added to the completed base-optimizer displacement.
    """

    parameter_names: tuple[str, ...] = (
        "fc1.weight",
        "fc2.weight",
        "fc3.weight",
    )
    correction_interval_steps: int = 1
    correction_start_step: int = 1
    probe_batch_size: int = 256
    probe_batches_per_correction: int = 2
    probe_seed_offset: int = 910_003

    # The unscaled correction has this Frobenius norm relative to the completed
    # base step, with a small weight-relative floor so the channel remains live
    # when the base optimizer nearly converges.
    correction_to_base_step_ratio: float = 0.25
    minimum_weight_fraction: float = 1e-5
    maximum_weight_fraction: float = 2.5e-3

    projection_mode: ProjectionMode = "core"
    min_ecs_rank: int = 2
    normalization_gamma: float = 0.0
    svd_device: SVDDevice = "cpu"

    use_backtracking: bool = True
    backtracking_factor: float = 0.5
    maximum_backtracking_steps: int = 7
    armijo_coefficient: float = 1e-4
    loss_tolerance: float = 1e-10
    numerical_epsilon: float = 1e-12
    strict: bool = True

    def validate(self) -> None:
        if not self.parameter_names:
            raise ValueError("parameter_names must not be empty")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter_names contains duplicates")
        if self.correction_interval_steps < 1:
            raise ValueError("correction_interval_steps must be positive")
        if self.correction_start_step < 1:
            raise ValueError("correction_start_step must be positive")
        if self.probe_batch_size < 1 or self.probe_batches_per_correction < 1:
            raise ValueError("probe batch counts must be positive")
        if self.correction_to_base_step_ratio < 0.0:
            raise ValueError("correction_to_base_step_ratio must be non-negative")
        if self.minimum_weight_fraction < 0.0:
            raise ValueError("minimum_weight_fraction must be non-negative")
        if self.maximum_weight_fraction <= 0.0:
            raise ValueError("maximum_weight_fraction must be positive")
        if self.minimum_weight_fraction > self.maximum_weight_fraction:
            raise ValueError(
                "minimum_weight_fraction cannot exceed maximum_weight_fraction"
            )
        if self.projection_mode not in {"core", "rank_m_tangent"}:
            raise ValueError(f"unknown projection_mode {self.projection_mode!r}")
        if self.min_ecs_rank < 1:
            raise ValueError("min_ecs_rank must be positive")
        if not 0.0 <= self.normalization_gamma <= 1.0:
            raise ValueError("normalization_gamma must lie in [0, 1]")
        if self.svd_device not in {"cpu", "model"}:
            raise ValueError(f"unknown svd_device {self.svd_device!r}")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must lie in (0, 1)")
        if self.maximum_backtracking_steps < 0:
            raise ValueError("maximum_backtracking_steps must be non-negative")
        if not 0.0 < self.armijo_coefficient < 1.0:
            raise ValueError("armijo_coefficient must lie in (0, 1)")
        if self.loss_tolerance < 0.0 or self.numerical_epsilon <= 0.0:
            raise ValueError("invalid numerical tolerances")

    def with_runtime_cadence(
        self,
        *,
        interval_steps: int,
        start_step: int,
    ) -> "TraceWallConfig":
        return replace(
            self,
            correction_interval_steps=int(interval_steps),
            correction_start_step=int(start_step),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete paired MLP3/MNIST experiment configuration."""

    optimizer: BaseOptimizerConfig
    trace_wall: TraceWallConfig = field(default_factory=TraceWallConfig)
    seeds: tuple[int, ...] = (1337, 2027, 31415)
    epochs: int = 20
    batch_size: int = 128
    num_workers: int = 0
    gradient_clip_norm: float = 1.0
    corrections_per_epoch: int = 1
    train_eval_max_batches: Optional[int] = None
    measure_weightwatcher: bool = True
    require_weightwatcher: bool = True
    weightwatcher_min_evals: int = 8
    weightwatcher_svd_method: str = "accurate"
    save_epoch_checkpoints: bool = True

    def validate(self) -> None:
        self.optimizer.validate()
        self.trace_wall.validate()
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.corrections_per_epoch < 1:
            raise ValueError("corrections_per_epoch must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty tuple of unique values")
        if self.weightwatcher_min_evals < 2:
            raise ValueError("weightwatcher_min_evals must be at least two")
        probe_examples = (
            self.trace_wall.probe_batch_size
            * self.trace_wall.probe_batches_per_correction
        )
        if probe_examples < 1:
            raise ValueError("the probe subset must contain at least one example")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

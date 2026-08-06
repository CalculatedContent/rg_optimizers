"""Configuration and presets for AdaptiveSpectralGuard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Mapping, Optional


@dataclass(frozen=True)
class LayerPolicy:
    """Per-layer intervention policy.

    The slow WeightWatcher controller decides whether the layer is off, weak,
    or strong. These fields determine how much of each correction channel may
    be used once the layer is active.
    """

    enabled: bool = True
    cadence: int = 5

    weak_gain: float = 0.25
    strong_gain: float = 0.75

    volume_scale: float = 1.0
    volume_max_ratio: Optional[float] = 0.10

    shape_scale: float = 0.20
    shape_max_ratio: Optional[float] = 0.04

    combined_max_ratio: Optional[float] = 0.15

    min_retained: int = 5
    min_shape_retained: int = 20
    n_shape_shells: int = 5
    min_shape_decades: float = 0.50

    loss_neutral: bool = True
    allowed_task_conflict_ratio: float = 0.0

    def validate(self) -> None:
        if self.cadence < 1:
            raise ValueError("cadence must be positive")
        if self.weak_gain < 0.0 or self.strong_gain < 0.0:
            raise ValueError("gains must be non-negative")
        if self.weak_gain > self.strong_gain:
            raise ValueError("weak_gain cannot exceed strong_gain")
        for name, value in (
            ("volume_scale", self.volume_scale),
            ("shape_scale", self.shape_scale),
            ("allowed_task_conflict_ratio", self.allowed_task_conflict_ratio),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("volume_max_ratio", self.volume_max_ratio),
            ("shape_max_ratio", self.shape_max_ratio),
            ("combined_max_ratio", self.combined_max_ratio),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative or None")
        if self.min_retained < 1 or self.min_shape_retained < 3:
            raise ValueError("retained-rank thresholds are invalid")
        if self.n_shape_shells < 2:
            raise ValueError("n_shape_shells must be at least two")
        if self.min_shape_decades < 0.0:
            raise ValueError("min_shape_decades must be non-negative")


@dataclass(frozen=True)
class ControllerConfig:
    """Slow per-epoch state machine driven by WeightWatcher diagnostics."""

    alpha_on: float = 2.08
    alpha_strong: float = 1.98
    alpha_off: float = 2.18
    alpha_trend_on: float = -0.04
    trend_ceiling: float = 2.30
    off_patience: int = 2

    min_confidence: float = 0.20
    support_change_scale: float = 0.20
    erg_gap_ratio_scale: float = 0.30

    beta_on: float = 0.05
    shape_alpha_on: float = 2.05

    task_conflict_ema_decay: float = 0.80
    task_conflict_penalty: float = 2.0
    minimum_task_throttle: float = 0.10

    def validate(self) -> None:
        if not self.alpha_strong <= self.alpha_on < self.alpha_off:
            raise ValueError("Require alpha_strong <= alpha_on < alpha_off")
        if self.off_patience < 1:
            raise ValueError("off_patience must be positive")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1]")
        if self.support_change_scale <= 0.0 or self.erg_gap_ratio_scale <= 0.0:
            raise ValueError("confidence scales must be positive")
        if not 0.0 <= self.task_conflict_ema_decay < 1.0:
            raise ValueError("task_conflict_ema_decay must lie in [0, 1)")
        if self.task_conflict_penalty < 0.0:
            raise ValueError("task_conflict_penalty must be non-negative")
        if not 0.0 <= self.minimum_task_throttle <= 1.0:
            raise ValueError("minimum_task_throttle must lie in [0, 1]")


@dataclass(frozen=True)
class GuardConfig:
    """Top-level optimizer configuration."""

    controller: ControllerConfig = field(default_factory=ControllerConfig)
    policies: Mapping[str, LayerPolicy] = field(default_factory=dict)

    ridge_relative: float = 1e-6
    eps: float = 1e-12

    def validate(self) -> None:
        self.controller.validate()
        if self.ridge_relative < 0.0:
            raise ValueError("ridge_relative must be non-negative")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        for policy in self.policies.values():
            policy.validate()

    def policy_for(self, parameter_name: str) -> LayerPolicy:
        """Resolve exact, module, or suffix policy names."""
        candidates = [parameter_name]
        if parameter_name.endswith(".weight"):
            candidates.append(parameter_name[: -len(".weight")])
        for key in candidates:
            if key in self.policies:
                return self.policies[key]
        suffixes = [
            policy
            for key, policy in self.policies.items()
            if parameter_name.endswith(key)
            or parameter_name.endswith(f"{key}.weight")
        ]
        if len(suffixes) == 1:
            return suffixes[0]
        return LayerPolicy(enabled=False)


def default_layer_policies() -> dict[str, LayerPolicy]:
    """Conservative defaults motivated by the first TraceLogRG experiment.

    FC1 keeps frequent branch protection and a small beta-shape channel.
    FC2 is much weaker and less frequent because every-step projection slowed
    its convergence. FC3 is disabled because it has only ten singular values.
    """

    return {
        "fc1.weight": LayerPolicy(
            enabled=True,
            cadence=2,
            weak_gain=0.45,
            strong_gain=0.90,
            volume_scale=1.0,
            volume_max_ratio=0.15,
            shape_scale=0.25,
            shape_max_ratio=0.05,
            combined_max_ratio=0.18,
            min_retained=5,
            min_shape_retained=20,
            n_shape_shells=5,
            min_shape_decades=0.50,
            loss_neutral=True,
        ),
        "fc2.weight": LayerPolicy(
            enabled=True,
            cadence=10,
            weak_gain=0.12,
            strong_gain=0.30,
            volume_scale=0.70,
            volume_max_ratio=0.05,
            shape_scale=0.10,
            shape_max_ratio=0.02,
            combined_max_ratio=0.06,
            min_retained=5,
            min_shape_retained=20,
            n_shape_shells=5,
            min_shape_decades=0.50,
            loss_neutral=True,
        ),
        "fc3.weight": LayerPolicy(enabled=False),
    }


def preset_policies(name: str) -> dict[str, LayerPolicy]:
    """Return an explicit ablation preset."""
    policies = default_layer_policies()
    key = name.strip().lower()
    if key in {"adaptive", "fc1_fc2", "full"}:
        return policies
    if key == "fc1_only":
        policies["fc2.weight"] = replace(policies["fc2.weight"], enabled=False)
        policies["fc3.weight"] = replace(policies["fc3.weight"], enabled=False)
        return policies
    if key == "fc2_only":
        policies["fc1.weight"] = replace(policies["fc1.weight"], enabled=False)
        policies["fc3.weight"] = replace(policies["fc3.weight"], enabled=False)
        return policies
    if key in {"off", "baseline"}:
        return {
            parameter: replace(policy, enabled=False)
            for parameter, policy in policies.items()
        }
    raise ValueError(
        f"Unknown preset {name!r}; expected adaptive, fc1_only, fc2_only, or off"
    )


def config_to_dict(config: GuardConfig) -> dict:
    return {
        "controller": asdict(config.controller),
        "policies": {
            name: asdict(policy) for name, policy in config.policies.items()
        },
        "ridge_relative": config.ridge_relative,
        "eps": config.eps,
    }

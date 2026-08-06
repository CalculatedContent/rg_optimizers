"""PyTorch optimizer wrapper using a self-consistent adaptive ECS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import torch

from .ecs import AdaptiveSupportState, EffectiveRankMethod, SupportPolicy
from .geometry import (
    AdaptiveTraceLogGeometry,
    CorrectionMode,
    NormalizationResponse,
    adaptive_trace_log_geometry,
    correct_trace_log_component,
)


@dataclass(frozen=True)
class SelfConsistentTraceLogConfig:
    """Configuration for the adaptive trace-log RG wrapper.

    Defaults preserve the semantics of the original trace-log experiment:
    WeightWatcher supplies an outer-loop ECS state, and the wrapper removes
    only contracting flow along the corresponding local trace-log normal.
    """

    mode: CorrectionMode = "one_sided"
    tracking_gamma: float = 0.10
    effective_rank_method: EffectiveRankMethod = "participation_ratio"
    normalization_gamma: float = 0.0
    normalization_response: NormalizationResponse = "frozen"
    support_policy: SupportPolicy = "midpoint"
    min_ecs_size: int = 2
    min_retained: int = 3
    require_sign_change: bool = False
    ridge_relative: float = 1e-6
    positive_eigenvalue_floor: float = 0.0
    correction_scale: float = 1.0
    max_correction_ratio: Optional[float] = 0.25
    apply_every_steps: int = 1
    # 0 means: keep the WeightWatcher outer-loop ECS until the next checkpoint.
    refresh_ecs_every_steps: int = 0
    warmup_steps: int = 0
    # False is the conservative/original architecture.  True enables a live
    # bootstrap when no WeightWatcher support has yet been installed.
    bootstrap_without_weightwatcher: bool = False
    max_abs_trace_log_per_eval: Optional[float] = None
    eps: float = 1e-12

    def validate(self) -> None:
        if self.mode not in {"tangent", "one_sided", "tracking"}:
            raise ValueError(f"Unknown correction mode: {self.mode!r}")
        if not 0.0 <= float(self.tracking_gamma) <= 1.0:
            raise ValueError("tracking_gamma must lie in [0, 1].")
        if self.effective_rank_method not in {
            "participation_ratio",
            "entropy",
            "stable_rank",
        }:
            raise ValueError(
                f"Unknown effective_rank_method: {self.effective_rank_method!r}"
            )
        if not 0.0 <= float(self.normalization_gamma) <= 1.0:
            raise ValueError("normalization_gamma must lie in [0, 1].")
        if self.normalization_response not in {"frozen", "differentiated"}:
            raise ValueError(
                "normalization_response must be 'frozen' or 'differentiated'."
            )
        if self.support_policy not in {"ecs", "midpoint", "power_law"}:
            raise ValueError(f"Unknown support_policy: {self.support_policy!r}")
        if int(self.min_ecs_size) < 1:
            raise ValueError("min_ecs_size must be positive.")
        if int(self.min_retained) < 1:
            raise ValueError("min_retained must be positive.")
        if self.ridge_relative < 0.0:
            raise ValueError("ridge_relative must be non-negative.")
        if self.positive_eigenvalue_floor < 0.0:
            raise ValueError("positive_eigenvalue_floor must be non-negative.")
        if not 0.0 <= float(self.correction_scale) <= 1.0:
            raise ValueError("correction_scale must lie in [0, 1].")
        if self.max_correction_ratio is not None and self.max_correction_ratio < 0.0:
            raise ValueError("max_correction_ratio must be non-negative or None.")
        if int(self.apply_every_steps) < 1:
            raise ValueError("apply_every_steps must be positive.")
        if int(self.refresh_ecs_every_steps) < 0:
            raise ValueError("refresh_ecs_every_steps must be non-negative.")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if (
            self.max_abs_trace_log_per_eval is not None
            and self.max_abs_trace_log_per_eval < 0.0
        ):
            raise ValueError(
                "max_abs_trace_log_per_eval must be non-negative or None."
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")


class SelfConsistentTraceLogRGWrapper:
    """Filter completed optimizer displacements along an adaptive trace-log normal.

    The base optimizer owns momentum, adaptive preconditioning, clipping,
    weight decay, and learning rate.  This wrapper observes the completed
    displacement and then applies a local spectral correction.  The default
    ``one_sided`` mode cancels only first-order contraction of retained
    log-volume, protecting against redundant return flow toward the weak-tail
    trivial branch without forcing an intermediate checkpoint onto the final
    trace-log gauge surface.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        config: Optional[SelfConsistentTraceLogConfig] = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.config = config or SelfConsistentTraceLogConfig()
        self.config.validate()

        optimizer_parameter_ids = {
            id(parameter)
            for group in base_optimizer.param_groups
            for parameter in group["params"]
        }
        self.named_parameters: dict[str, torch.nn.Parameter] = {
            name: parameter
            for name, parameter in named_parameters
            if id(parameter) in optimizer_parameter_ids
        }
        self.support_states: dict[str, AdaptiveSupportState] = {}
        self.global_step = 0
        self._last_step_stats: list[dict[str, Any]] = []

    @property
    def param_groups(self) -> list[MutableMapping[str, Any]]:
        return self.base_optimizer.param_groups

    @property
    def state(self) -> MutableMapping[torch.Tensor, Any]:
        return self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _resolve_parameter_name(self, supplied_name: str) -> Optional[str]:
        candidates = [supplied_name]
        if not supplied_name.endswith(".weight"):
            candidates.append(f"{supplied_name}.weight")
        target = next((name for name in candidates if name in self.named_parameters), None)
        if target is not None:
            return target
        suffixes = [
            name
            for name in self.named_parameters
            if any(name.endswith(candidate) for candidate in candidates)
        ]
        return suffixes[0] if len(suffixes) == 1 else None

    def set_support_states(
        self,
        supports: Mapping[str, AdaptiveSupportState | Mapping[str, Any] | int],
        *,
        replace: bool = False,
    ) -> None:
        """Install outer-loop self-consistent ECS states.

        Names may be module names (``fc1``) or parameter names
        (``fc1.weight``).  Unknown/non-matrix parameters are ignored.
        """
        resolved: dict[str, AdaptiveSupportState] = {}
        for supplied_name, value in supports.items():
            target = self._resolve_parameter_name(str(supplied_name))
            if target is None:
                continue
            parameter = self.named_parameters[target]
            if parameter.ndim != 2:
                continue
            state = AdaptiveSupportState.from_value(value)
            max_rank = min(parameter.shape)
            state.ecs_rank = int(np.clip(state.ecs_rank, 1, max_rank))
            state.normalization_dimension = float(
                np.clip(state.normalization_dimension, state.ecs_rank, max_rank)
            )
            if state.pl_rank is not None:
                state.pl_rank = int(np.clip(state.pl_rank, 1, max_rank))
            if state.working_rank is not None:
                state.working_rank = int(np.clip(state.working_rank, 1, max_rank))
            resolved[target] = state
        if replace:
            self.support_states = resolved
        else:
            self.support_states.update(resolved)

    def set_supports(self, supports: Mapping[str, int]) -> None:
        """Backward-compatible fixed-rank support setter."""
        self.set_support_states(supports)

    def get_support_states(self) -> dict[str, AdaptiveSupportState]:
        return {
            name: AdaptiveSupportState.from_value(state)
            for name, state in self.support_states.items()
        }

    def get_supports(self) -> dict[str, int]:
        return {name: int(state.ecs_rank) for name, state in self.support_states.items()}

    def pop_step_stats(self) -> list[dict[str, Any]]:
        stats = self._last_step_stats
        self._last_step_stats = []
        return stats

    def _candidate_parameter_names(self) -> list[str]:
        if self.config.bootstrap_without_weightwatcher:
            return [
                name
                for name, parameter in self.named_parameters.items()
                if parameter.ndim == 2 and parameter.requires_grad
            ]
        return [
            name
            for name in self.support_states
            if name in self.named_parameters
            and self.named_parameters[name].ndim == 2
            and self.named_parameters[name].requires_grad
        ]

    def _refresh_due(self, next_step: int, prior: Optional[AdaptiveSupportState]) -> bool:
        cadence = int(self.config.refresh_ecs_every_steps)
        if prior is None:
            return bool(self.config.bootstrap_without_weightwatcher)
        if cadence == 0:
            return False
        return next_step % cadence == 0

    @torch.no_grad()
    def _prepare_geometries(
        self,
        next_step: int,
    ) -> dict[str, tuple[torch.Tensor, AdaptiveTraceLogGeometry, AdaptiveSupportState]]:
        prepared: dict[
            str,
            tuple[torch.Tensor, AdaptiveTraceLogGeometry, AdaptiveSupportState],
        ] = {}
        for name in self._candidate_parameter_names():
            parameter = self.named_parameters[name]
            prior = self.support_states.get(name)
            refresh = self._refresh_due(next_step, prior)

            fixed_rank = None if refresh else (prior.ecs_rank if prior else None)
            fixed_dimension = (
                None
                if refresh or prior is None
                else prior.normalization_dimension
            )
            reference_rank = prior.ecs_rank if prior is not None else None
            pl_rank = prior.pl_rank if prior is not None else None

            before = parameter.detach().clone()
            try:
                geometry = adaptive_trace_log_geometry(
                    before,
                    fixed_ecs_rank=fixed_rank,
                    fixed_normalization_dimension=fixed_dimension,
                    reference_ecs_rank=reference_rank,
                    pl_rank=pl_rank,
                    support_policy=self.config.support_policy,
                    effective_rank_method=self.config.effective_rank_method,
                    normalization_gamma=self.config.normalization_gamma,
                    normalization_response=self.config.normalization_response,
                    min_ecs_size=self.config.min_ecs_size,
                    min_retained=self.config.min_retained,
                    require_sign_change=self.config.require_sign_change,
                    ridge_relative=self.config.ridge_relative,
                    positive_eigenvalue_floor=self.config.positive_eigenvalue_floor,
                    eps=self.config.eps,
                )
            except (RuntimeError, ValueError) as exc:
                self._last_step_stats.append(
                    {
                        "global_step": next_step,
                        "parameter": name,
                        "status": "geometry_failed",
                        "reason": str(exc),
                    }
                )
                continue

            if (
                self.config.max_abs_trace_log_per_eval is not None
                and abs(float(geometry.residual.detach().float().cpu()))
                > float(self.config.max_abs_trace_log_per_eval)
            ):
                self._last_step_stats.append(
                    {
                        "global_step": next_step,
                        "parameter": name,
                        "status": "geometry_skipped",
                        "reason": "trace-log residual exceeds configured limit",
                        "ecs_rank": geometry.ecs.ecs_rank,
                        "coordinate_residual": float(
                            geometry.residual.detach().float().cpu()
                        ),
                    }
                )
                continue

            new_state = AdaptiveSupportState(
                ecs_rank=int(geometry.ecs.ecs_rank),
                normalization_dimension=float(geometry.ecs.normalization_dimension),
                bulk_effective_count=float(geometry.ecs.bulk_effective_count),
                trace_log_per_eval=float(geometry.ecs.trace_log_per_eval),
                status=str(geometry.ecs.status),
                method=self.config.effective_rank_method,
                normalization_gamma=float(self.config.normalization_gamma),
                pl_rank=pl_rank,
                working_rank=int(geometry.working_rank),
                alpha=(prior.alpha if prior is not None else float("nan")),
                erg_gap_sc=(
                    float(geometry.ecs.ecs_rank - pl_rank)
                    if pl_rank is not None
                    else float("nan")
                ),
                source_epoch=(prior.source_epoch if prior is not None else None),
                source_global_step=(
                    prior.source_global_step if prior is not None else None
                ),
            )
            prepared[name] = (before, geometry, new_state)
        return prepared

    def step(self, closure: Optional[Any] = None) -> Any:
        self._last_step_stats = []
        next_step = self.global_step + 1
        correction_due = (
            next_step > int(self.config.warmup_steps)
            and next_step % int(self.config.apply_every_steps) == 0
        )
        prepared = self._prepare_geometries(next_step) if correction_due else {}

        loss = self.base_optimizer.step(closure)
        self.global_step = next_step

        with torch.no_grad():
            for name, (before, geometry, new_state) in prepared.items():
                parameter = self.named_parameters[name]
                delta_base = parameter.detach() - before
                result = correct_trace_log_component(
                    delta_base,
                    geometry,
                    mode=self.config.mode,
                    tracking_gamma=self.config.tracking_gamma,
                    correction_scale=self.config.correction_scale,
                    max_correction_ratio=self.config.max_correction_ratio,
                    eps=self.config.eps,
                )
                parameter.copy_(before + result.corrected_delta)
                self.support_states[name] = new_state

                self._last_step_stats.append(
                    {
                        "global_step": self.global_step,
                        "parameter": name,
                        "status": "ok" if result.applied else "skipped",
                        "reason": result.reason,
                        "mode": self.config.mode,
                        "support_policy": self.config.support_policy,
                        "effective_rank_method": self.config.effective_rank_method,
                        "normalization_gamma": self.config.normalization_gamma,
                        "normalization_response": self.config.normalization_response,
                        "ecs_rank": geometry.ecs.ecs_rank,
                        "working_rank": geometry.working_rank,
                        "pl_rank": geometry.pl_rank,
                        "erg_gap_sc": (
                            geometry.ecs.ecs_rank - geometry.pl_rank
                            if geometry.pl_rank is not None
                            else np.nan
                        ),
                        "normalization_dimension": geometry.ecs.normalization_dimension,
                        "normalization_dimension_is_cached": (
                            geometry.normalization_dimension_is_cached
                        ),
                        "bulk_effective_count": geometry.ecs.bulk_effective_count,
                        "bulk_effective_fraction": geometry.ecs.bulk_effective_fraction,
                        "solver_status": geometry.ecs.status,
                        "solver_trace_log_per_eval": geometry.ecs.trace_log_per_eval,
                        "coordinate_residual_before": float(
                            geometry.residual.detach().float().cpu()
                        ),
                        "base_trace_log_drift": result.base_drift,
                        "corrected_trace_log_drift": result.corrected_drift,
                        "predicted_trace_log_after": result.predicted_residual_after,
                        "coefficient": result.coefficient,
                        "correction_ratio": result.correction_ratio,
                        "correction_capped": result.capped,
                        "gradient_norm_sq": geometry.gradient_norm_sq,
                        "gradient_radial_inner_product": geometry.radial_inner_product,
                        "retained_gradient_norm_sq": geometry.retained_gradient_norm_sq,
                        "radial_gradient_norm_sq": geometry.radial_gradient_norm_sq,
                        "normalization_gradient_norm_sq": (
                            geometry.normalization_gradient_norm_sq
                        ),
                        "smallest_retained_singular_value": (
                            geometry.smallest_retained_singular_value
                        ),
                        "largest_retained_singular_value": (
                            geometry.largest_retained_singular_value
                        ),
                    }
                )
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "support_states": {
                name: state.to_dict() for name, state in self.support_states.items()
            },
            "global_step": int(self.global_step),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.support_states = {}
        self.set_support_states(state_dict.get("support_states", {}), replace=True)
        self.global_step = int(state_dict.get("global_step", 0))

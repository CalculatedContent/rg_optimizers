"""Optimizer wrapper that filters completed matrix-valued parameter steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import torch

from .geometry import (
    CorrectionMode,
    NormalizationMode,
    TraceLogGeometry,
    correct_trace_log_component,
    trace_log_geometry,
)


@dataclass(frozen=True)
class TraceLogConfig:
    """Configuration for the trace-log RG wrapper."""

    mode: CorrectionMode = "one_sided"
    gamma: float = 0.10
    normalization: NormalizationMode = "weightwatcher"
    ridge_relative: float = 1e-6
    min_retained: int = 3
    correction_scale: float = 1.0
    max_correction_ratio: Optional[float] = 0.25
    apply_every_steps: int = 1
    warmup_steps: int = 0
    eps: float = 1e-12

    def validate(self) -> None:
        if self.mode not in {"tangent", "one_sided", "tracking"}:
            raise ValueError(f"Unknown correction mode: {self.mode!r}")
        if self.normalization not in {"weightwatcher", "raw"}:
            raise ValueError(f"Unknown normalization: {self.normalization!r}")
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        if self.ridge_relative < 0.0:
            raise ValueError("ridge_relative must be non-negative.")
        if int(self.min_retained) < 1:
            raise ValueError("min_retained must be positive.")
        if not 0.0 <= float(self.correction_scale) <= 1.0:
            raise ValueError("correction_scale must lie in [0, 1].")
        if self.max_correction_ratio is not None and self.max_correction_ratio < 0.0:
            raise ValueError("max_correction_ratio must be non-negative or None.")
        if int(self.apply_every_steps) < 1:
            raise ValueError("apply_every_steps must be positive.")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")


class TraceLogRGWrapper:
    """Post-process completed steps of a base PyTorch optimizer.

    The base optimizer owns momentum and adaptive state. The wrapper applies
    an outer correction to the actual displacement. Retained ranks are
    supplied by ``set_supports`` and may be refreshed by a slower
    WeightWatcher outer loop.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        config: Optional[TraceLogConfig] = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.config = config or TraceLogConfig()
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
        self.supports: dict[str, int] = {}
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

    def set_supports(self, supports: Mapping[str, int]) -> None:
        """Set cached retained ranks from parameter or module names."""
        resolved: dict[str, int] = {}
        for supplied_name, rank in supports.items():
            candidates = [supplied_name]
            if not supplied_name.endswith(".weight"):
                candidates.append(f"{supplied_name}.weight")

            target = next((name for name in candidates if name in self.named_parameters), None)
            if target is None:
                suffixes = [
                    name
                    for name in self.named_parameters
                    if any(name.endswith(candidate) for candidate in candidates)
                ]
                if len(suffixes) == 1:
                    target = suffixes[0]

            if target is None:
                continue
            parameter = self.named_parameters[target]
            if parameter.ndim != 2:
                continue
            resolved[target] = int(max(1, min(int(rank), min(parameter.shape))))

        self.supports = resolved

    def get_supports(self) -> dict[str, int]:
        return dict(self.supports)

    def pop_step_stats(self) -> list[dict[str, Any]]:
        stats = self._last_step_stats
        self._last_step_stats = []
        return stats

    @torch.no_grad()
    def _prepare_geometries(self) -> dict[str, tuple[torch.Tensor, TraceLogGeometry]]:
        prepared: dict[str, tuple[torch.Tensor, TraceLogGeometry]] = {}
        for name, retained_rank in self.supports.items():
            parameter = self.named_parameters.get(name)
            if parameter is None or parameter.ndim != 2 or not parameter.requires_grad:
                continue
            if int(retained_rank) < int(self.config.min_retained):
                continue

            before = parameter.detach().clone()
            try:
                geometry = trace_log_geometry(
                    before,
                    retained_rank,
                    normalization=self.config.normalization,
                    ridge_relative=self.config.ridge_relative,
                    eps=self.config.eps,
                )
            except (RuntimeError, ValueError) as exc:
                self._last_step_stats.append({
                    "global_step": self.global_step + 1,
                    "parameter": name,
                    "status": "geometry_failed",
                    "reason": str(exc),
                })
                continue
            prepared[name] = (before, geometry)
        return prepared

    def step(self, closure: Optional[Any] = None) -> Any:
        self._last_step_stats = []
        next_step = self.global_step + 1
        correction_due = (
            next_step > int(self.config.warmup_steps)
            and next_step % int(self.config.apply_every_steps) == 0
        )
        prepared = self._prepare_geometries() if correction_due else {}

        loss = self.base_optimizer.step(closure)
        self.global_step += 1

        with torch.no_grad():
            for name, (before, geometry) in prepared.items():
                parameter = self.named_parameters[name]
                delta_base = parameter.detach() - before
                result = correct_trace_log_component(
                    delta_base,
                    geometry,
                    mode=self.config.mode,
                    gamma=self.config.gamma,
                    correction_scale=self.config.correction_scale,
                    max_correction_ratio=self.config.max_correction_ratio,
                    eps=self.config.eps,
                )
                parameter.copy_(before + result.corrected_delta)
                self._last_step_stats.append({
                    "global_step": self.global_step,
                    "parameter": name,
                    "status": "ok" if result.applied else "skipped",
                    "reason": result.reason,
                    "mode": self.config.mode,
                    "normalization": self.config.normalization,
                    "retained_rank": geometry.retained_rank,
                    "trace_log_residual_before": float(geometry.residual.detach().float().cpu()),
                    "base_trace_log_drift": result.base_drift,
                    "corrected_trace_log_drift": result.corrected_drift,
                    "predicted_trace_log_after": result.predicted_residual_after,
                    "coefficient": result.coefficient,
                    "correction_ratio": result.correction_ratio,
                    "correction_capped": result.capped,
                    "gradient_norm_sq": geometry.gradient_norm_sq,
                    "gradient_radial_inner_product": geometry.radial_inner_product,
                    "smallest_retained_singular_value": geometry.smallest_retained_singular_value,
                    "largest_retained_singular_value": geometry.largest_retained_singular_value,
                })
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "supports": dict(self.supports),
            "global_step": int(self.global_step),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.set_supports(state_dict.get("supports", {}))
        self.global_step = int(state_dict.get("global_step", 0))

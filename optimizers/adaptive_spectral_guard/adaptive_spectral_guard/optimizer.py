"""Completed-step adaptive spectral optimizer wrapper."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import math
import torch

from .config import GuardConfig
from .controller import AdaptiveSpectralController
from .geometry import (
    SpectralGeometry,
    loss_neutralize,
    relative_cap,
    spectral_geometry,
)


class AdaptiveSpectralGuard:
    """Wrap a PyTorch optimizer with adaptive volume and beta-E corrections.

    The base optimizer owns momentum and adaptive state. This wrapper inspects
    the completed matrix displacement and then adds two controlled channels:

    1. one-sided trace-log branch protection;
    2. a trace-log-orthogonal shell-beta correction toward beta_E = 0.

    Both channels are layer-specific, WeightWatcher-gated, norm-capped, and
    optionally projected to be first-order loss-neutral.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        config: GuardConfig,
    ) -> None:
        config.validate()
        self.base_optimizer = base_optimizer
        self.config = config
        self.controller = AdaptiveSpectralController(config)

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

    def update_from_weightwatcher(self, metrics):
        return self.controller.update_from_weightwatcher(metrics)

    def observe_task_feedback(self, stats):
        return self.controller.observe_task_feedback(stats)

    def controller_frame(self):
        return self.controller.frame()

    def pop_step_stats(self) -> list[dict[str, Any]]:
        rows = self._last_step_stats
        self._last_step_stats = []
        return rows

    def _prepare(self, next_step: int) -> dict[
        str, tuple[torch.Tensor, torch.Tensor, SpectralGeometry]
    ]:
        prepared = {}
        for name, parameter in self.named_parameters.items():
            if parameter.ndim != 2 or not parameter.requires_grad:
                continue
            policy = self.config.policy_for(name)
            state = self.controller.get_state(name)
            if (
                not policy.enabled
                or state.regime == "off"
                or state.effective_gain <= 0.0
                or next_step % policy.cadence != 0
            ):
                continue
            if state.midpoint_rank < policy.min_retained:
                self._last_step_stats.append(
                    {
                        "global_step": next_step,
                        "parameter": name,
                        "status": "skipped",
                        "reason": "retained rank below policy minimum",
                        "regime": state.regime,
                    }
                )
                continue
            if parameter.grad is None:
                self._last_step_stats.append(
                    {
                        "global_step": next_step,
                        "parameter": name,
                        "status": "skipped",
                        "reason": "missing task gradient",
                        "regime": state.regime,
                    }
                )
                continue

            before = parameter.detach().clone()
            task_gradient = parameter.grad.detach().clone()
            try:
                geometry = spectral_geometry(
                    before,
                    volume_rank=state.midpoint_rank,
                    shape_rank=max(state.num_pl_spikes, policy.min_retained),
                    n_shells=policy.n_shape_shells,
                    min_shape_retained=policy.min_shape_retained,
                    min_shape_decades=policy.min_shape_decades,
                    ridge_relative=self.config.ridge_relative,
                    eps=self.config.eps,
                )
            except (RuntimeError, ValueError) as exc:
                self._last_step_stats.append(
                    {
                        "global_step": next_step,
                        "parameter": name,
                        "status": "geometry_failed",
                        "reason": str(exc),
                        "regime": state.regime,
                    }
                )
                continue
            prepared[name] = (before, task_gradient, geometry)
        return prepared

    @staticmethod
    def _zero_like(delta: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(delta)

    def _volume_correction(
        self,
        delta: torch.Tensor,
        geometry: SpectralGeometry,
        *,
        gain: float,
        scale: float,
        max_ratio: Optional[float],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        grad = geometry.trace_gradient.to(
            device=delta.device,
            dtype=delta.dtype,
        )
        grad_norm_sq = torch.sum(grad.float().square())
        drift = torch.sum(grad.float() * delta.float())
        drift_value = float(drift.detach().cpu())
        norm_value = float(grad_norm_sq.detach().cpu())

        if norm_value <= self.config.eps or not math.isfinite(norm_value):
            return self._zero_like(delta), {
                "volume_applied": False,
                "volume_reason": "singular trace gradient",
                "base_trace_log_drift": drift_value,
                "volume_correction_ratio": 0.0,
                "volume_capped": False,
            }

        coefficient = min(drift_value / norm_value, 0.0)
        correction = -float(gain) * float(scale) * coefficient * grad
        correction, ratio, capped = relative_cap(
            correction,
            delta,
            max_ratio,
            eps=self.config.eps,
        )
        return correction, {
            "volume_applied": bool(
                torch.linalg.vector_norm(correction.float()).item()
                > self.config.eps
            ),
            "volume_reason": (
                "applied" if coefficient < 0.0 else "no contracting component"
            ),
            "base_trace_log_drift": drift_value,
            "volume_coefficient": coefficient,
            "volume_correction_ratio": ratio,
            "volume_capped": capped,
        }

    def _shape_correction(
        self,
        delta: torch.Tensor,
        geometry: SpectralGeometry,
        *,
        active: bool,
        gain: float,
        scale: float,
        max_ratio: Optional[float],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        beta = float(geometry.beta_E.detach().float().cpu())
        grad = geometry.beta_gradient.to(
            device=delta.device,
            dtype=delta.dtype,
        )
        grad_norm_sq = torch.sum(grad.float().square())
        norm_value = float(grad_norm_sq.detach().cpu())

        if not active:
            return self._zero_like(delta), {
                "shape_applied": False,
                "shape_reason": "shape gate off",
                "beta_E_local": beta,
                "shape_correction_ratio": 0.0,
                "shape_capped": False,
            }
        if not geometry.beta_reliable:
            return self._zero_like(delta), {
                "shape_applied": False,
                "shape_reason": "beta geometry unreliable",
                "beta_E_local": beta,
                "shape_correction_ratio": 0.0,
                "shape_capped": False,
            }
        if beta <= 0.0:
            return self._zero_like(delta), {
                "shape_applied": False,
                "shape_reason": "beta_E is not on the alpha<2 side",
                "beta_E_local": beta,
                "shape_correction_ratio": 0.0,
                "shape_capped": False,
            }
        if norm_value <= self.config.eps or not math.isfinite(norm_value):
            return self._zero_like(delta), {
                "shape_applied": False,
                "shape_reason": "singular beta gradient",
                "beta_E_local": beta,
                "shape_correction_ratio": 0.0,
                "shape_capped": False,
            }

        coefficient = beta / norm_value
        correction = -float(gain) * float(scale) * coefficient * grad
        correction, ratio, capped = relative_cap(
            correction,
            delta,
            max_ratio,
            eps=self.config.eps,
        )
        return correction, {
            "shape_applied": bool(
                torch.linalg.vector_norm(correction.float()).item()
                > self.config.eps
            ),
            "shape_reason": "applied",
            "beta_E_local": beta,
            "shape_coefficient": coefficient,
            "shape_correction_ratio": ratio,
            "shape_capped": capped,
        }

    def step(self, closure: Optional[Any] = None) -> Any:
        self._last_step_stats = []
        next_step = self.global_step + 1
        prepared = self._prepare(next_step)

        loss = self.base_optimizer.step(closure)
        self.global_step += 1

        with torch.no_grad():
            for name, (before, task_gradient, geometry) in prepared.items():
                parameter = self.named_parameters[name]
                delta_base = parameter.detach() - before
                policy = self.config.policy_for(name)
                state = self.controller.get_state(name)

                volume, volume_stats = self._volume_correction(
                    delta_base,
                    geometry,
                    gain=state.effective_gain,
                    scale=policy.volume_scale,
                    max_ratio=policy.volume_max_ratio,
                )
                shape, shape_stats = self._shape_correction(
                    delta_base,
                    geometry,
                    active=state.shape_active,
                    gain=state.effective_gain,
                    scale=policy.shape_scale,
                    max_ratio=policy.shape_max_ratio,
                )
                attempted = volume + shape

                if policy.loss_neutral:
                    corrected, task_stats = loss_neutralize(
                        attempted,
                        task_gradient,
                        delta_base,
                        allowed_conflict_ratio=(
                            policy.allowed_task_conflict_ratio
                        ),
                        eps=self.config.eps,
                    )
                else:
                    corrected = attempted
                    _, task_stats = loss_neutralize(
                        attempted,
                        task_gradient,
                        delta_base,
                        allowed_conflict_ratio=float("inf"),
                        eps=self.config.eps,
                    )
                    task_stats["loss_neutral_applied"] = False
                    task_stats["loss_neutral_removed_fraction"] = 0.0

                corrected, combined_ratio, combined_capped = relative_cap(
                    corrected,
                    delta_base,
                    policy.combined_max_ratio,
                    eps=self.config.eps,
                )
                final_delta = delta_base + corrected
                parameter.copy_(before + final_delta)

                trace_grad = geometry.trace_gradient.to(
                    device=delta_base.device,
                    dtype=delta_base.dtype,
                )
                beta_grad = geometry.beta_gradient.to(
                    device=delta_base.device,
                    dtype=delta_base.dtype,
                )
                final_trace_drift = float(
                    torch.sum(trace_grad.float() * final_delta.float())
                    .detach()
                    .cpu()
                )
                final_beta_drift = float(
                    torch.sum(beta_grad.float() * final_delta.float())
                    .detach()
                    .cpu()
                )
                correction_norm = float(
                    torch.linalg.vector_norm(corrected.float()).detach().cpu()
                )
                applied = correction_norm > self.config.eps

                row = {
                    "global_step": self.global_step,
                    "parameter": name,
                    "status": "ok" if applied else "skipped",
                    "reason": "applied" if applied else "no final correction",
                    "regime": state.regime,
                    "state_reason": state.reason,
                    "alpha": state.alpha,
                    "alpha_trend": state.alpha_trend,
                    "ERG_gap": state.erg_gap,
                    "confidence": state.confidence,
                    "task_throttle": state.task_throttle,
                    "base_gain": state.base_gain,
                    "effective_gain": state.effective_gain,
                    "shape_active": state.shape_active,
                    "volume_rank": geometry.volume_rank,
                    "shape_rank": geometry.shape_rank,
                    "shells_used": geometry.shells_used,
                    "dynamic_range_decades": (
                        geometry.dynamic_range_decades
                    ),
                    "trace_log_residual_before": float(
                        geometry.trace_residual.detach().float().cpu()
                    ),
                    "beta_E_local": float(
                        geometry.beta_E.detach().float().cpu()
                    ),
                    "beta_reliable": geometry.beta_reliable,
                    "trace_beta_inner_product_before": (
                        geometry.trace_beta_inner_product_before
                    ),
                    "trace_beta_inner_product_after": (
                        geometry.trace_beta_inner_product_after
                    ),
                    "final_trace_log_drift": final_trace_drift,
                    "final_beta_drift": final_beta_drift,
                    "combined_correction_ratio": combined_ratio,
                    "combined_capped": combined_capped,
                    "correction_norm": correction_norm,
                    **volume_stats,
                    **shape_stats,
                    **task_stats,
                }
                self._last_step_stats.append(row)
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "global_step": int(self.global_step),
            "controller": self.controller.state_dict(),
            "config": {
                "controller": asdict(self.config.controller),
                "policies": {
                    key: asdict(value)
                    for key, value in self.config.policies.items()
                },
                "ridge_relative": self.config.ridge_relative,
                "eps": self.config.eps,
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.global_step = int(state_dict.get("global_step", 0))
        self.controller.load_state_dict(state_dict.get("controller", {}))

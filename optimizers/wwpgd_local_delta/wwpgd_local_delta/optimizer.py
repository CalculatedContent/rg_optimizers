"""Optimizer wrapper for epoch-boundary local-delta ECS damping."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import torch

from .config import LocalDeltaECSConfig
from .ecs import damp_delta_outside_ecs


class LocalDeltaECSOptimizer:
    """Wrap AdamW, SGD+momentum, or another PyTorch optimizer.

    The base optimizer performs ordinary minibatch steps. At an epoch boundary,
    ``apply_epoch_delta_correction`` replaces each selected matrix parameter by

        W_start + Delta_ECS + (1-eta) Delta_perp.

    The optimizer's internal first/second-moment state is deliberately left
    unchanged in this first experiment, so the causal intervention is isolated
    to the realized weight displacement. This is logged explicitly.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        config: Optional[LocalDeltaECSConfig] = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.config = config or LocalDeltaECSConfig()
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
        self.global_step = 0
        self.epoch_index = 0
        self._epoch_start: dict[str, torch.Tensor] = {}
        self._last_epoch_stats: list[dict[str, Any]] = []

    @property
    def param_groups(self) -> list[MutableMapping[str, Any]]:
        return self.base_optimizer.param_groups

    @property
    def state(self) -> MutableMapping[torch.Tensor, Any]:
        return self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Any] = None) -> Any:
        loss = self.base_optimizer.step(closure)
        self.global_step += 1
        return loss

    def _selected_matrix_parameters(self) -> dict[str, torch.nn.Parameter]:
        selected: dict[str, torch.nn.Parameter] = {}
        filters = self.config.parameter_name_filter
        for name, parameter in self.named_parameters.items():
            if not parameter.requires_grad or parameter.ndim != 2:
                continue
            if filters is not None and name not in filters and not any(
                name.endswith(token) for token in filters
            ):
                continue
            selected[name] = parameter
        return selected

    @torch.no_grad()
    def begin_epoch(self) -> None:
        """Snapshot selected matrix parameters before an epoch starts."""
        self._epoch_start = {
            name: parameter.detach().clone()
            for name, parameter in self._selected_matrix_parameters().items()
        }

    @torch.no_grad()
    def apply_epoch_delta_correction(self, *, epoch: Optional[int] = None) -> list[dict[str, Any]]:
        """Dampen outside-ECS displacement for the just-completed epoch."""
        current_epoch = self.epoch_index if epoch is None else int(epoch)
        self._last_epoch_stats = []
        due = (
            current_epoch >= int(self.config.warmup_epochs)
            and (current_epoch + 1) % int(self.config.apply_every_epochs) == 0
        )
        if not due:
            self._epoch_start = {}
            self.epoch_index = current_epoch + 1
            return []
        if not self._epoch_start:
            self._last_epoch_stats.append(
                {
                    "epoch": current_epoch,
                    "global_step": self.global_step,
                    "status": "skipped",
                    "reason": "missing epoch-start snapshot",
                }
            )
            self.epoch_index = current_epoch + 1
            return list(self._last_epoch_stats)

        selected = self._selected_matrix_parameters()
        for name, before in self._epoch_start.items():
            parameter = selected.get(name)
            if parameter is None:
                continue
            after = parameter.detach().clone()
            delta = after - before
            reference = before if self.config.reference == "epoch_start" else after
            try:
                result = damp_delta_outside_ecs(
                    delta,
                    reference,
                    correction_fraction=self.config.correction_fraction,
                    min_retained=self.config.min_retained,
                    max_retained=self.config.max_retained,
                    normalization_gamma=self.config.normalization_gamma,
                    eps=self.config.eps,
                )
            except (RuntimeError, ValueError) as exc:
                self._last_epoch_stats.append(
                    {
                        "epoch": current_epoch,
                        "global_step": self.global_step,
                        "parameter": name,
                        "status": "geometry_failed",
                        "reason": str(exc),
                        "reference": self.config.reference,
                    }
                )
                continue

            parameter.copy_(before + result.corrected_delta.to(parameter.device, parameter.dtype))
            self._last_epoch_stats.append(
                {
                    "epoch": current_epoch,
                    "global_step": self.global_step,
                    "parameter": name,
                    "status": "ok",
                    "reason": "applied",
                    "correction_fraction": result.correction_fraction,
                    "ecs_rank": result.ecs_rank,
                    "ecs_fraction": result.ecs_fraction,
                    "normalization_dimension": result.normalization_dimension,
                    "bulk_effective_count": result.bulk_effective_count,
                    "trace_log_per_eval": result.trace_log_per_eval,
                    "solver_status": result.status,
                    "transposed": result.transposed,
                    "projection_side": result.projection_side,
                    "base_delta_norm": result.base_delta_norm,
                    "ecs_delta_norm": result.ecs_delta_norm,
                    "orthogonal_delta_norm": result.orthogonal_delta_norm,
                    "post_orthogonal_delta_norm": result.post_orthogonal_delta_norm,
                    "removed_delta_norm": result.removed_delta_norm,
                    "orthogonal_fraction": result.orthogonal_fraction,
                    "post_orthogonal_fraction": result.post_orthogonal_fraction,
                    "removed_fraction_of_base": result.removed_fraction_of_base,
                    "observed_orthogonal_damping": result.observed_orthogonal_damping,
                    "expected_orthogonal_damping": result.expected_orthogonal_damping,
                    "damping_error": result.damping_error,
                    "pythagorean_error": result.pythagorean_error,
                    "correction_identity_error": result.correction_identity_error,
                    "reference": self.config.reference,
                    "optimizer_state_adjusted": False,
                }
            )

        self._epoch_start = {}
        self.epoch_index = current_epoch + 1
        return list(self._last_epoch_stats)

    def pop_epoch_stats(self) -> list[dict[str, Any]]:
        stats = self._last_epoch_stats
        self._last_epoch_stats = []
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "global_step": int(self.global_step),
            "epoch_index": int(self.epoch_index),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.global_step = int(state_dict.get("global_step", 0))
        self.epoch_index = int(state_dict.get("epoch_index", 0))

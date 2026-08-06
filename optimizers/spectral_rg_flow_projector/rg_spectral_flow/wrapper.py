"""PyTorch wrapper that removes only spectral flow toward the F0 surrogate."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import torch

from .ecs import AdaptiveSupportState
from .flow import remove_trivial_branch_component


from .config import SpectralRGFlowConfig
class SpectralRGFlowProjector:
    """Post-process completed matrix steps in centered log-spectrum space.

    The base optimizer owns the supervised trajectory and all of its internal
    state.  This wrapper observes the completed matrix displacement.  On the
    cached adaptive ECS, it subtracts only the positive projection of the
    centered log-spectrum displacement onto the local participation-ratio
    collapse vector.  It does not project onto the trace-log normal and does
    not directly target alpha.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        config: Optional[SpectralRGFlowConfig] = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.config = config or SpectralRGFlowConfig()
        self.config.validate()

        optimizer_ids = {
            id(parameter)
            for group in base_optimizer.param_groups
            for parameter in group["params"]
        }
        self.named_parameters = {
            name: parameter
            for name, parameter in named_parameters
            if id(parameter) in optimizer_ids
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

    def _resolve_name(self, supplied_name: str) -> Optional[str]:
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
        resolved: dict[str, AdaptiveSupportState] = {}
        for supplied_name, value in supports.items():
            target = self._resolve_name(str(supplied_name))
            if target is None:
                continue
            parameter = self.named_parameters[target]
            if parameter.ndim != 2:
                continue
            state = AdaptiveSupportState.from_value(value)
            max_rank = min(parameter.shape)
            state.ecs_rank = int(np.clip(state.ecs_rank, 1, max_rank))
            if state.working_rank is None:
                state.working_rank = state.ecs_rank
            state.working_rank = int(np.clip(state.working_rank, 1, max_rank))
            if state.pl_rank is not None:
                state.pl_rank = int(np.clip(state.pl_rank, 1, max_rank))
            resolved[target] = state
        if replace:
            self.support_states = resolved
        else:
            self.support_states.update(resolved)

    def get_support_states(self) -> dict[str, AdaptiveSupportState]:
        return {
            name: AdaptiveSupportState.from_value(state)
            for name, state in self.support_states.items()
        }

    def pop_step_stats(self) -> list[dict[str, Any]]:
        rows = self._last_step_stats
        self._last_step_stats = []
        return rows

    @torch.no_grad()
    def _prepare_before(self) -> dict[str, torch.Tensor]:
        prepared: dict[str, torch.Tensor] = {}
        for name, state in self.support_states.items():
            parameter = self.named_parameters.get(name)
            if parameter is None or parameter.ndim != 2 or not parameter.requires_grad:
                continue
            rank = state.working_rank or state.ecs_rank
            if int(rank) < int(self.config.min_retained):
                continue
            prepared[name] = parameter.detach().clone()
        return prepared

    def step(self, closure: Optional[Any] = None) -> Any:
        self._last_step_stats = []
        next_step = self.global_step + 1
        correction_due = (
            next_step > int(self.config.warmup_steps)
            and next_step % int(self.config.apply_every_steps) == 0
        )
        before = self._prepare_before() if correction_due else {}

        loss = self.base_optimizer.step(closure)
        self.global_step = next_step

        with torch.no_grad():
            for name, weight_before in before.items():
                parameter = self.named_parameters[name]
                state = self.support_states[name]
                rank = int(state.working_rank or state.ecs_rank)
                try:
                    result = remove_trivial_branch_component(
                        weight_before,
                        parameter.detach(),
                        rank,
                        potential=self.config.collapse_potential,
                        projection_strength=self.config.projection_strength,
                        min_alignment_cosine=self.config.min_alignment_cosine,
                        max_abs_log_eigenvalue_correction=(
                            self.config.max_abs_log_eigenvalue_correction
                        ),
                        max_correction_ratio=self.config.max_correction_ratio,
                        preserve_frobenius_norm=self.config.preserve_frobenius_norm,
                        eps=self.config.eps,
                    )
                    parameter.copy_(result.corrected_weight)
                    geometry = result.geometry
                    self._last_step_stats.append(
                        {
                            "global_step": self.global_step,
                            "parameter": name,
                            "status": "ok" if result.applied else "skipped",
                            "reason": result.reason,
                            "collapse_potential": self.config.collapse_potential,
                            "working_rank": geometry.working_rank,
                            "ecs_rank": state.ecs_rank,
                            "pl_rank": state.pl_rank,
                            "alpha_checkpoint": state.alpha,
                            "erg_gap_sc_checkpoint": state.erg_gap_sc,
                            "base_flow_component": geometry.base_flow_component,
                            "corrected_flow_component": result.corrected_flow_component,
                            "base_projection_coefficient": (
                                geometry.base_projection_coefficient
                            ),
                            "applied_projection_coefficient": (
                                result.projection_coefficient
                            ),
                            "base_alignment_cosine": geometry.base_alignment_cosine,
                            "flow_vector_norm_sq": geometry.flow_vector_norm_sq,
                            "spectral_displacement_norm": float(
                                torch.linalg.vector_norm(
                                    geometry.spectral_displacement
                                )
                                .detach()
                                .cpu()
                            ),
                            "spectral_component_removed_norm": (
                                result.spectral_component_removed_norm
                            ),
                            "collapse_potential_before": (
                                geometry.collapse_potential_before
                            ),
                            "collapse_potential_base": geometry.collapse_potential_base,
                            "collapse_potential_corrected": (
                                result.collapse_potential_corrected
                            ),
                            "effective_rank_before": geometry.effective_rank_before,
                            "effective_rank_base": geometry.effective_rank_base,
                            "effective_rank_corrected": (
                                result.effective_rank_corrected
                            ),
                            "rank_alpha_proxy_before": (
                                geometry.rank_alpha_proxy_before
                            ),
                            "rank_alpha_proxy_base": geometry.rank_alpha_proxy_base,
                            "rank_alpha_proxy_corrected": (
                                result.rank_alpha_proxy_corrected
                            ),
                            "correction_ratio": result.correction_ratio,
                            "correction_capped": result.capped,
                        }
                    )
                except (RuntimeError, ValueError) as exc:
                    self._last_step_stats.append(
                        {
                            "global_step": self.global_step,
                            "parameter": name,
                            "status": "failed",
                            "reason": str(exc),
                            "working_rank": rank,
                            "ecs_rank": state.ecs_rank,
                            "pl_rank": state.pl_rank,
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

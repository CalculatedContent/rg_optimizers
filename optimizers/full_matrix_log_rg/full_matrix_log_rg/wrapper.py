"""Optimizer wrapper for full matrix-log RG corrections."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, MutableMapping

import torch

from .config import FullMatrixLogConfig
from .cone import project_matrix_log_cone
from .geometry import full_matrix_log_geometry, remove_inward_matrix_log_flow
from .support import MatrixLogSupport


def project_sgd_momentum_buffer(
    base_optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
    correction: torch.Tensor,
    parameter_group: Mapping[str, Any],
    *,
    eps: float = 1e-12,
) -> float:
    """Shift the hidden SGD buffer so it generates the accepted correction."""
    if not isinstance(base_optimizer, torch.optim.SGD):
        raise TypeError("Projected momentum state requires torch.optim.SGD")
    if bool(parameter_group.get("maximize", False)):
        raise ValueError("maximize=True is not supported")
    momentum = float(parameter_group.get("momentum", 0.0))
    learning_rate = float(parameter_group.get("lr", 0.0))
    if momentum <= 0.0:
        raise RuntimeError("Projected state requires nonzero SGD momentum")
    if learning_rate <= eps:
        raise RuntimeError("Projected state requires a positive learning rate")
    buffer = base_optimizer.state.get(parameter, {}).get("momentum_buffer")
    if buffer is None:
        raise RuntimeError("SGD momentum buffer is missing after optimizer.step()")
    factor = momentum if bool(parameter_group.get("nesterov", False)) else 1.0
    buffer_correction = -correction.to(buffer) / (learning_rate * factor)
    buffer.add_(buffer_correction)
    return float(torch.linalg.vector_norm(buffer_correction.float()).cpu())


class FullMatrixLogRG:
    """Apply cone/radial/legacy corrections to completed optimizer steps."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        config: FullMatrixLogConfig | None = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.config = config or FullMatrixLogConfig()
        self.config.validate()
        optimizer_ids = {
            id(parameter)
            for group in base_optimizer.param_groups
            for parameter in group["params"]
        }
        allowed = set(self.config.parameter_names) if self.config.parameter_names else None
        self.named_parameters = {
            name: parameter
            for name, parameter in named_parameters
            if id(parameter) in optimizer_ids
            and parameter.ndim == 2
            and (allowed is None or name in allowed)
        }
        if allowed is not None:
            missing = sorted(allowed.difference(self.named_parameters))
            if missing:
                raise ValueError(
                    "Configured parameters are absent from the base optimizer: "
                    + ", ".join(missing)
                )
        self._groups: dict[int, MutableMapping[str, Any]] = {}
        for group in base_optimizer.param_groups:
            for parameter in group["params"]:
                self._groups[id(parameter)] = group
        if self.config.momentum_projection == "projected_state":
            if not isinstance(base_optimizer, torch.optim.SGD):
                raise TypeError("projected_state requires torch.optim.SGD")
            for name, parameter in self.named_parameters.items():
                if float(self._groups[id(parameter)].get("momentum", 0.0)) <= 0.0:
                    raise ValueError(f"{name!r} is not in an SGD momentum group")
        self.supports: dict[str, MatrixLogSupport] = {}
        self.global_step = 0
        self._last_step_stats: list[dict[str, Any]] = []

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @property
    def state(self):
        return self.base_optimizer.state

    @property
    def defaults(self):
        return self.base_optimizer.defaults

    def zero_grad(self, *args, **kwargs):
        return self.base_optimizer.zero_grad(*args, **kwargs)

    def set_supports(
        self,
        supports: Mapping[str, MatrixLogSupport],
        *,
        replace: bool = True,
    ) -> None:
        resolved: dict[str, MatrixLogSupport] = {}
        for supplied_name, support in supports.items():
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
            if not isinstance(support, MatrixLogSupport):
                raise TypeError("supports must contain MatrixLogSupport values")
            parameter = self.named_parameters[target]
            rank = int(max(1, min(support.retained_rank, min(parameter.shape))))
            resolved[target] = MatrixLogSupport(
                retained_rank=rank,
                normalization_dimension=float(support.normalization_dimension),
                full_dimension=support.full_dimension,
                self_consistent_dimension=support.self_consistent_dimension,
                bulk_effective_rank=float(support.bulk_effective_rank),
                right_basis=support.right_basis[:, :rank].detach().cpu().float(),
                transposed=bool(support.transposed),
                checkpoint_epoch=int(support.checkpoint_epoch),
            )
        if replace:
            self.supports = resolved
        else:
            self.supports.update(resolved)

    def get_supports(self) -> dict[str, MatrixLogSupport]:
        return dict(self.supports)

    def pop_step_stats(self) -> list[dict[str, Any]]:
        stats = self._last_step_stats
        self._last_step_stats = []
        return stats

    @torch.no_grad()
    def _prepare(self) -> dict[str, tuple[torch.Tensor, Any]]:
        prepared = {}
        for name, parameter in self.named_parameters.items():
            support = self.supports.get(name)
            if support is None:
                if self.config.require_support:
                    raise RuntimeError(
                        f"No support for {name!r}; run analyze_supports first"
                    )
                continue
            if support.retained_rank < self.config.min_retained_rank:
                continue
            before = parameter.detach().clone()
            geometry = full_matrix_log_geometry(
                before,
                support=support,
                normalization=self.config.normalization,
                ridge_relative=self.config.ridge_relative,
                eps=self.config.eps,
            )
            prepared[name] = (before, geometry)
        return prepared

    def step(self, closure: Any = None) -> Any:
        self._last_step_stats = []
        next_step = self.global_step + 1
        due = (
            next_step > self.config.warmup_steps
            and next_step % self.config.apply_every_steps == 0
        )
        prepared = self._prepare() if due else {}
        loss = self.base_optimizer.step(closure)
        self.global_step += 1

        with torch.no_grad():
            for name, (before, geometry) in prepared.items():
                parameter = self.named_parameters[name]
                delta = parameter.detach() - before
                if self.config.mode == "cone":
                    result = project_matrix_log_cone(
                        delta,
                        geometry,
                        projection_strength=self.config.projection_strength,
                        max_correction_ratio=self.config.max_correction_ratio,
                        gram_ridge_relative=self.config.gram_ridge_relative,
                        tolerance=self.config.cone_tolerance,
                        max_iterations=self.config.cone_max_iterations,
                        log_deadband=self.config.log_deadband,
                        eps=self.config.eps,
                    )
                    corrected = result.corrected_delta
                    correction = result.correction
                    common = {
                        "base_potential_drift": result.base_potential_drift,
                        "corrected_potential_drift": result.corrected_potential_drift,
                        "correction_ratio": result.correction_ratio,
                        "correction_capped": result.capped,
                        "initial_inward_mode_count": result.initial_inward_mode_count,
                        "eligible_mode_count": result.eligible_mode_count,
                        "active_set_size": result.active_set_size,
                        "active_set_iterations": result.iterations,
                        "active_set_converged": result.converged,
                        "active_set_kkt_residual": result.kkt_residual,
                        "max_signed_violation_before": result.max_signed_violation_before,
                        "max_signed_violation_after": result.max_signed_violation_after,
                    }
                    applied = result.applied
                else:
                    result = remove_inward_matrix_log_flow(
                        delta,
                        geometry,
                        mode=self.config.mode,
                        projection_strength=self.config.projection_strength,
                        max_correction_ratio=self.config.max_correction_ratio,
                        gram_ridge_relative=self.config.gram_ridge_relative,
                        eps=self.config.eps,
                    )
                    corrected = result.corrected_delta
                    correction = result.correction
                    common = {
                        "base_potential_drift": result.base_drift,
                        "corrected_potential_drift": result.corrected_drift,
                        "correction_ratio": result.correction_ratio,
                        "correction_capped": result.capped,
                        "initial_inward_mode_count": result.inward_mode_count,
                        "eligible_mode_count": geometry.retained_rank,
                        "active_set_size": 0,
                        "active_set_iterations": 0,
                        "active_set_converged": True,
                        "active_set_kkt_residual": 0.0,
                        "max_signed_violation_before": float("nan"),
                        "max_signed_violation_after": float("nan"),
                    }
                    applied = result.applied
                parameter.copy_(before + corrected)

                buffer_norm = 0.0
                momentum_projected = False
                if applied and self.config.momentum_projection == "projected_state":
                    buffer_norm = project_sgd_momentum_buffer(
                        self.base_optimizer,
                        parameter,
                        correction,
                        self._groups[id(parameter)],
                        eps=self.config.eps,
                    )
                    momentum_projected = True

                support = self.supports[name]
                self._last_step_stats.append(
                    {
                        "global_step": self.global_step,
                        "parameter": name,
                        "status": "ok" if applied else "skipped",
                        "mode": self.config.mode,
                        "momentum_projection": self.config.momentum_projection,
                        "momentum_state_projected": momentum_projected,
                        "momentum_buffer_correction_norm": buffer_norm,
                        "normalization": self.config.normalization,
                        "normalization_dimension": geometry.normalization_dimension,
                        "D_full_M": support.dimension("full_m"),
                        "D_self_consistent": support.dimension("self_consistent"),
                        "retained_rank": geometry.retained_rank,
                        "support_epoch": support.checkpoint_epoch,
                        "matrix_log_potential_before": float(geometry.potential.cpu()),
                        "mean_abs_log_eigenvalue": float(
                            torch.mean(torch.abs(geometry.log_eigenvalues.float())).cpu()
                        ),
                        "max_abs_log_eigenvalue": float(
                            torch.max(torch.abs(geometry.log_eigenvalues.float())).cpu()
                        ),
                        **common,
                    }
                )
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "supports": {
                name: support.state_dict() for name, support in self.supports.items()
            },
            "global_step": int(self.global_step),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        saved_config = state_dict.get("config")
        if saved_config is not None:
            current = asdict(self.config)
            for key, value in dict(saved_config).items():
                if key in current and value != current[key]:
                    raise RuntimeError(
                        f"Checkpoint config mismatch at {key!r}: "
                        f"{value!r} != {current[key]!r}"
                    )
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        restored = {
            name: MatrixLogSupport.from_state_dict(payload)
            for name, payload in state_dict.get("supports", {}).items()
        }
        self.set_supports(restored)
        self.global_step = int(state_dict.get("global_step", 0))


__all__ = ["FullMatrixLogRG", "project_sgd_momentum_buffer"]

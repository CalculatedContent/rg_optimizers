"""Full matrix-log RG optimizer wrapper and projected-state SGD path."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, MutableMapping

import torch

from .config import FullMatrixLogConfig
from .cone import project_matrix_log_cone
from .geometry import (
    MatrixLogGeometry,
    ProjectionResult,
    full_matrix_log_geometry,
    remove_inward_matrix_log_flow,
)
from .support import MatrixLogSupport


class FullMatrixLogRG:
    """Filter SGD flow using cached full matrix-log geometry.

    ``momentum_projection='projected_state'`` executes the SGD/Nesterov step,
    projects the actual proposed weight displacement, and rewrites the momentum
    buffer so rejected flow does not persist. ``post_step`` retains the original
    generic wrapper as a legacy ablation.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        config: FullMatrixLogConfig | None = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.config = config or FullMatrixLogConfig()
        self.config.validate()
        if (
            self.config.momentum_projection == "projected_state"
            and not isinstance(base_optimizer, torch.optim.SGD)
        ):
            raise TypeError(
                "projected_state requires a torch.optim.SGD base optimizer; "
                "use post_step for generic optimizers"
            )

        optimizer_parameter_ids = {
            id(parameter)
            for group in base_optimizer.param_groups
            for parameter in group["params"]
        }
        allowed = (
            set(self.config.parameter_names)
            if self.config.parameter_names is not None
            else None
        )
        self.named_parameters: dict[str, torch.nn.Parameter] = {
            name: parameter
            for name, parameter in named_parameters
            if id(parameter) in optimizer_parameter_ids
            and parameter.ndim == 2
            and (allowed is None or name in allowed)
        }
        self._parameter_names_by_id = {
            id(parameter): name for name, parameter in self.named_parameters.items()
        }
        if allowed is not None:
            missing = sorted(allowed.difference(self.named_parameters))
            if missing:
                raise ValueError(
                    "Configured matrix parameters are absent from the base optimizer: "
                    + ", ".join(missing)
                )

        self.supports: dict[str, MatrixLogSupport] = {}
        self.global_step = 0
        self._last_step_stats: list[dict[str, Any]] = []

    @property
    def param_groups(self) -> list[MutableMapping[str, Any]]:
        return self.base_optimizer.param_groups

    @property
    def state(self) -> MutableMapping[torch.Tensor, Any]:
        return self.base_optimizer.state

    @property
    def defaults(self) -> Mapping[str, Any]:
        return self.base_optimizer.defaults

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.base_optimizer.zero_grad(*args, **kwargs)

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
            target = next(
                (name for name in candidates if name in self.named_parameters),
                None,
            )
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
                raise TypeError(
                    f"Support for {supplied_name!r} must be MatrixLogSupport"
                )
            parameter = self.named_parameters[target]
            rank = int(max(1, min(support.retained_rank, min(parameter.shape))))
            basis = support.right_basis[:, :rank].detach().to(
                device="cpu", dtype=torch.float32
            )
            spectrum = (
                None
                if support.eigenvalues_ascending is None
                else torch.as_tensor(support.eigenvalues_ascending)
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            resolved[target] = MatrixLogSupport(
                retained_rank=rank,
                normalization_dimension=float(support.normalization_dimension),
                right_basis=basis,
                transposed=bool(support.transposed),
                checkpoint_epoch=int(support.checkpoint_epoch),
                eigenvalues_ascending=spectrum,
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

    def _normalization_dimension(self, support: MatrixLogSupport) -> float:
        return support.normalization_dimension_for(
            self.config.normalization,
            method=self.config.effective_rank_method,
            gamma=self.config.normalization_gamma,
        )

    @torch.no_grad()
    def _prepare_geometries(self) -> dict[str, tuple[torch.Tensor, MatrixLogGeometry]]:
        prepared: dict[str, tuple[torch.Tensor, MatrixLogGeometry]] = {}
        for name, parameter in self.named_parameters.items():
            support = self.supports.get(name)
            if support is None:
                if self.config.require_support:
                    raise RuntimeError(
                        f"No cached retained support for {name!r}. "
                        "Run analyze_supports and call set_supports before training."
                    )
                continue
            if int(support.retained_rank) < int(self.config.min_retained_rank):
                continue
            before = parameter.detach().clone()
            try:
                geometry = full_matrix_log_geometry(
                    before,
                    support=support,
                    normalization_dimension=self._normalization_dimension(support),
                    ridge_relative=self.config.ridge_relative,
                    eps=self.config.eps,
                )
            except (RuntimeError, ValueError) as exc:
                self._last_step_stats.append(
                    {
                        "global_step": self.global_step + 1,
                        "parameter": name,
                        "status": "geometry_failed",
                        "reason": str(exc),
                    }
                )
                continue
            prepared[name] = (before, geometry)
        return prepared

    @torch.no_grad()
    def _project_delta(
        self,
        delta_base: torch.Tensor,
        geometry: MatrixLogGeometry,
    ) -> ProjectionResult:
        if self.config.mode == "cone":
            return project_matrix_log_cone(
                delta_base,
                geometry,
                projection_strength=self.config.projection_strength,
                max_correction_ratio=self.config.max_correction_ratio,
                gram_ridge_relative=self.config.gram_ridge_relative,
                tolerance=self.config.cone_tolerance,
                max_iterations=self.config.cone_max_iterations,
                log_deadband=self.config.log_deadband,
                eps=self.config.eps,
            )
        return remove_inward_matrix_log_flow(
            delta_base,
            geometry,
            mode=self.config.mode,
            projection_strength=self.config.projection_strength,
            max_correction_ratio=self.config.max_correction_ratio,
            gram_ridge_relative=self.config.gram_ridge_relative,
            eps=self.config.eps,
        )

    def _stat_row(
        self,
        *,
        name: str,
        geometry: MatrixLogGeometry,
        result: ProjectionResult,
        momentum_buffer_before: torch.Tensor | None = None,
        momentum_buffer_after: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        support = self.supports[name]
        buffer_correction_norm = 0.0
        if momentum_buffer_before is not None and momentum_buffer_after is not None:
            buffer_correction_norm = float(
                torch.linalg.vector_norm(
                    (momentum_buffer_after - momentum_buffer_before).float()
                ).detach().cpu()
            )
        return {
            "global_step": self.global_step,
            "parameter": name,
            "status": "ok" if result.applied else "skipped",
            "mode": result.mode,
            "momentum_projection": self.config.momentum_projection,
            "normalization": self.config.normalization,
            "normalization_dimension": float(geometry.normalization_dimension),
            "full_m_dimension": float(support.matrix_dimension()),
            "bulk_effective_count": support.bulk_effective_count(
                method=self.config.effective_rank_method
            ),
            "retained_rank": geometry.retained_rank,
            "support_epoch": support.checkpoint_epoch,
            "matrix_log_potential_before": float(
                geometry.potential.detach().float().cpu()
            ),
            "mean_abs_log_eigenvalue": float(
                torch.mean(torch.abs(geometry.log_eigenvalues.float())).detach().cpu()
            ),
            "max_abs_log_eigenvalue": float(
                torch.max(torch.abs(geometry.log_eigenvalues.float())).detach().cpu()
            ),
            "base_potential_drift": result.base_drift,
            "corrected_potential_drift": result.corrected_drift,
            "inward_mode_count": result.inward_mode_count,
            "base_inward_mode_norm": result.base_inward_mode_norm,
            "corrected_inward_mode_norm": result.corrected_inward_mode_norm,
            "max_signed_violation_before": result.max_signed_violation_before,
            "max_signed_violation_after": result.max_signed_violation_after,
            "cone_active_set_size": result.cone_active_set_size,
            "cone_iterations": result.cone_iterations,
            "cone_converged": result.cone_converged,
            "coefficient": result.coefficient,
            "correction_ratio": result.correction_ratio,
            "correction_capped": result.capped,
            "momentum_buffer_correction_norm": buffer_correction_norm,
            "gradient_norm_sq": geometry.gradient_norm_sq,
            "smallest_retained_singular_value": (
                geometry.min_retained_singular_value
            ),
            "largest_retained_singular_value": (
                geometry.max_retained_singular_value
            ),
        }

    def _correction_due(self) -> bool:
        next_step = self.global_step + 1
        return (
            next_step > int(self.config.warmup_steps)
            and next_step % int(self.config.apply_every_steps) == 0
        )

    def _step_post_step(self, closure: Any = None) -> Any:
        prepared = self._prepare_geometries() if self._correction_due() else {}
        loss = self.base_optimizer.step(closure)
        self.global_step += 1
        with torch.no_grad():
            for name, (before, geometry) in prepared.items():
                parameter = self.named_parameters[name]
                delta_base = parameter.detach() - before
                result = self._project_delta(delta_base, geometry)
                parameter.copy_(before + result.corrected_delta)
                self._last_step_stats.append(
                    self._stat_row(name=name, geometry=geometry, result=result)
                )
        return loss

    def _step_projected_state(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        prepared = self._prepare_geometries() if self._correction_due() else {}

        with torch.no_grad():
            for group in self.base_optimizer.param_groups:
                if bool(group.get("differentiable", False)):
                    raise RuntimeError("projected_state does not support differentiable=True")
                learning_rate = float(group["lr"])
                momentum = float(group.get("momentum", 0.0))
                dampening = float(group.get("dampening", 0.0))
                weight_decay = float(group.get("weight_decay", 0.0))
                nesterov = bool(group.get("nesterov", False))
                maximize = bool(group.get("maximize", False))
                if nesterov and momentum <= 0.0:
                    raise RuntimeError("Nesterov requires positive momentum")
                if nesterov and dampening != 0.0:
                    raise RuntimeError("Nesterov requires zero dampening")
                if learning_rate <= 0.0:
                    raise RuntimeError("projected_state requires a positive learning rate")

                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if gradient.is_sparse:
                        raise RuntimeError("projected_state does not support sparse gradients")
                    direction = gradient.detach().clone()
                    if maximize:
                        direction.neg_()
                    if weight_decay != 0.0:
                        direction.add_(parameter, alpha=weight_decay)

                    momentum_buffer: torch.Tensor | None = None
                    if momentum != 0.0:
                        state = self.base_optimizer.state[parameter]
                        momentum_buffer = state.get("momentum_buffer")
                        if momentum_buffer is None:
                            momentum_buffer = direction.detach().clone()
                            state["momentum_buffer"] = momentum_buffer
                        else:
                            momentum_buffer.mul_(momentum).add_(
                                direction, alpha=1.0 - dampening
                            )
                        step_direction = (
                            direction + momentum * momentum_buffer
                            if nesterov
                            else momentum_buffer
                        )
                    else:
                        step_direction = direction

                    delta_base = -learning_rate * step_direction
                    name = self._parameter_names_by_id.get(id(parameter))
                    if name is None or name not in prepared:
                        parameter.add_(delta_base)
                        continue

                    before, geometry = prepared[name]
                    if not torch.equal(parameter.detach(), before):
                        raise RuntimeError(
                            f"Parameter {name!r} changed before its projected SGD step"
                        )
                    buffer_before = (
                        momentum_buffer.detach().clone()
                        if momentum_buffer is not None
                        else None
                    )
                    result = self._project_delta(delta_base, geometry)
                    parameter.add_(result.corrected_delta)

                    if momentum_buffer is not None and result.applied:
                        if nesterov:
                            momentum_buffer.add_(
                                result.correction,
                                alpha=-1.0 / (learning_rate * momentum),
                            )
                        else:
                            momentum_buffer.add_(
                                result.correction,
                                alpha=-1.0 / learning_rate,
                            )
                    buffer_after = (
                        momentum_buffer.detach().clone()
                        if momentum_buffer is not None
                        else None
                    )
                    self._last_step_stats.append(
                        self._stat_row(
                            name=name,
                            geometry=geometry,
                            result=result,
                            momentum_buffer_before=buffer_before,
                            momentum_buffer_after=buffer_after,
                        )
                    )

        self.global_step += 1
        for row in self._last_step_stats:
            row["global_step"] = self.global_step
        return loss

    def step(self, closure: Any = None) -> Any:
        self._last_step_stats = []
        if self.config.momentum_projection == "projected_state":
            return self._step_projected_state(closure)
        return self._step_post_step(closure)

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "supports": {
                name: support.state_dict()
                for name, support in self.supports.items()
            },
            "global_step": int(self.global_step),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        saved_config = state_dict.get("config")
        if saved_config is not None:
            normalized_saved = dict(saved_config)
            if isinstance(normalized_saved.get("parameter_names"), list):
                normalized_saved["parameter_names"] = tuple(
                    normalized_saved["parameter_names"]
                )
            if normalized_saved != asdict(self.config):
                raise RuntimeError(
                    "FullMatrixLogRG checkpoint configuration does not match the current wrapper"
                )
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        restored = {
            name: MatrixLogSupport.from_state_dict(payload)
            for name, payload in state_dict.get("supports", {}).items()
        }
        self.set_supports(restored, replace=True)
        self.global_step = int(state_dict.get("global_step", 0))


class FullMatrixLogProjectedSGD(FullMatrixLogRG):
    """Preferred active-set/projected-momentum SGD interface."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        config: FullMatrixLogConfig | None = None,
    ) -> None:
        selected = config or FullMatrixLogConfig()
        if selected.momentum_projection != "projected_state":
            raise ValueError(
                "FullMatrixLogProjectedSGD requires momentum_projection='projected_state'"
            )
        super().__init__(base_optimizer, named_parameters, selected)

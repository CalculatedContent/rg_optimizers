"""Post-step wrapper for full matrix-log RG corrections."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, MutableMapping

import torch

from .config import FullMatrixLogConfig
from .geometry import full_matrix_log_geometry, remove_inward_matrix_log_flow
from .support import MatrixLogSupport


class FullMatrixLogRG:
    """Filter completed optimizer steps using cached full matrix-log geometry."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        config: FullMatrixLogConfig | None = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.config = config or FullMatrixLogConfig()
        self.config.validate()

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
            resolved[target] = MatrixLogSupport(
                retained_rank=rank,
                normalization_dimension=float(support.normalization_dimension),
                right_basis=basis,
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
    def _prepare_geometries(self) -> dict[str, tuple[torch.Tensor, Any]]:
        prepared: dict[str, tuple[torch.Tensor, Any]] = {}
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

    def step(self, closure: Any = None) -> Any:
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
                result = remove_inward_matrix_log_flow(
                    delta_base,
                    geometry,
                    mode=self.config.mode,
                    projection_strength=self.config.projection_strength,
                    max_correction_ratio=self.config.max_correction_ratio,
                    gram_ridge_relative=self.config.gram_ridge_relative,
                    eps=self.config.eps,
                )
                parameter.copy_(before + result.corrected_delta)
                self._last_step_stats.append(
                    {
                        "global_step": self.global_step,
                        "parameter": name,
                        "status": "ok" if result.applied else "skipped",
                        "mode": result.mode,
                        "retained_rank": geometry.retained_rank,
                        "support_epoch": self.supports[name].checkpoint_epoch,
                        "matrix_log_potential_before": float(
                            geometry.potential.detach().float().cpu()
                        ),
                        "mean_abs_log_eigenvalue": float(
                            torch.mean(
                                torch.abs(geometry.log_eigenvalues.float())
                            ).detach().cpu()
                        ),
                        "max_abs_log_eigenvalue": float(
                            torch.max(
                                torch.abs(geometry.log_eigenvalues.float())
                            ).detach().cpu()
                        ),
                        "base_potential_drift": result.base_drift,
                        "corrected_potential_drift": result.corrected_drift,
                        "inward_mode_count": result.inward_mode_count,
                        "base_inward_mode_norm": result.base_inward_mode_norm,
                        "corrected_inward_mode_norm": result.corrected_inward_mode_norm,
                        "coefficient": result.coefficient,
                        "correction_ratio": result.correction_ratio,
                        "correction_capped": result.capped,
                        "gradient_norm_sq": geometry.gradient_norm_sq,
                        "smallest_retained_singular_value": (
                            geometry.min_retained_singular_value
                        ),
                        "largest_retained_singular_value": (
                            geometry.max_retained_singular_value
                        ),
                    }
                )
        return loss

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
        if saved_config is not None and dict(saved_config) != asdict(self.config):
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

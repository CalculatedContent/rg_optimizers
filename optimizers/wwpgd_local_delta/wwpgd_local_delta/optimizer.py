"""Optimizer wrapper for epoch-boundary local-delta ECS damping."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping, MutableMapping, Optional
import torch

from .config import LocalDeltaECSConfig
from .ecs import LocalECSGeometry, damp_delta_with_geometry, local_ecs_geometry, split_delta_by_ecs


class LocalDeltaECSOptimizer:
    """Apply a fractional outside-ECS correction to a completed epoch update."""

    def __init__(self, base_optimizer: torch.optim.Optimizer,
                 named_parameters: Iterable[tuple[str, torch.nn.Parameter]], *,
                 config: Optional[LocalDeltaECSConfig] = None) -> None:
        self.base_optimizer = base_optimizer
        self.config = config or LocalDeltaECSConfig()
        self.config.validate()
        ids = {id(p) for g in base_optimizer.param_groups for p in g["params"]}
        self.named_parameters = {n: p for n, p in named_parameters if id(p) in ids}
        self._selected_parameter_names = self._resolve_filters(self.config.parameter_name_filter)
        self.global_step = 0
        self.epoch_index = 0
        self._epoch_start: dict[str, torch.Tensor] = {}
        self._epoch_active = False
        self._last_epoch_stats: list[dict[str, Any]] = []
        self._previous_ecs_ranks: dict[str, int] = {}

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

    def _resolve_filters(self, filters: Optional[tuple[str, ...]]) -> Optional[set[str]]:
        names = [n for n, p in self.named_parameters.items() if p.ndim == 2]
        if filters is None:
            return None
        resolved: set[str] = set()
        for raw in filters:
            token = str(raw)
            candidates = [token] if token.endswith(".weight") else [token, f"{token}.weight"]
            matches = [n for n in names if n in candidates]
            if len(matches) != 1:
                matches = [n for n in names if any(n.endswith(f".{c}") for c in candidates)]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous parameter filter {token!r}; matches {matches}.")
            if not matches:
                raise ValueError(f"Unknown matrix parameter filter: {token!r}.")
            resolved.add(matches[0])
        return resolved

    def _selected_matrix_parameters(self) -> dict[str, torch.nn.Parameter]:
        return {
            n: p for n, p in self.named_parameters.items()
            if p.requires_grad and p.ndim == 2 and (
                self._selected_parameter_names is None or n in self._selected_parameter_names
            )
        }

    @torch.no_grad()
    def begin_epoch(self) -> None:
        if self._epoch_active and self.config.strict_epoch_lifecycle:
            raise RuntimeError("begin_epoch() called while a previous epoch snapshot is still active.")
        self._epoch_start = {n: p.detach().clone() for n, p in self._selected_matrix_parameters().items()}
        self._epoch_active = True

    def _validate_result(self, name: str, result: Any) -> None:
        if not self.config.strict_numerics:
            return
        checks = (
            ("damping_error", result.damping_error, self.config.max_damping_error),
            ("pythagorean_error", result.pythagorean_error, self.config.max_pythagorean_error),
            ("correction_identity_error", result.correction_identity_error, self.config.max_identity_error),
        )
        failures = [f"{key}={value:.3e}" for key, value, limit in checks if value > float(limit)]
        if failures:
            raise RuntimeError(f"Local-delta numerical validation failed for {name}: " + ", ".join(failures))

    @torch.no_grad()
    def _synchronize_state(self, parameter: torch.nn.Parameter,
                           geometry: LocalECSGeometry) -> tuple[list[str], dict[str, float]]:
        if not self.config.synchronize_optimizer_state:
            return [], {}
        adjusted: list[str] = []
        fractions: dict[str, float] = {}
        for key in ("momentum_buffer", "exp_avg"):
            tensor = self.base_optimizer.state.get(parameter, {}).get(key)
            if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape:
                continue
            retained, orthogonal = split_delta_by_ecs(tensor, geometry)
            original_norm = float(torch.linalg.vector_norm(tensor.float()).detach().cpu())
            orth_norm = float(torch.linalg.vector_norm(orthogonal.float()).detach().cpu())
            tensor.copy_(retained + (1.0 - float(self.config.correction_fraction)) * orthogonal)
            adjusted.append(key)
            fractions[key] = orth_norm / max(original_norm, self.config.eps)
        return adjusted, fractions

    def _result_stats(self, *, epoch: int, name: str, previous_rank: Optional[int],
                      result: Any, adjusted: list[str], state_fractions: dict[str, float]) -> dict[str, Any]:
        row = {
            "epoch": epoch, "global_step": self.global_step, "parameter": name,
            "status": "ok", "reason": "applied", "reference": self.config.reference,
            "previous_ecs_rank": previous_rank,
            "ecs_rank_change": result.ecs_rank - previous_rank if previous_rank is not None else None,
            "optimizer_state_adjusted": bool(adjusted),
            "optimizer_state_adjusted_keys": adjusted,
            "optimizer_state_orthogonal_fractions": state_fractions,
            # Provenance / dose identity (logging only; defaults unchanged).
            "actuator_id": "wwpgd_local_delta",
            "ecs_backend": "self_consistent_local_geometry",
            "dose_definition": "removed_fraction_of_base_epoch_delta_outside_ecs",
            "dose_value": float(result.removed_fraction_of_base),
            "is_first_apply": int(epoch) == int(self.config.warmup_epochs),
        }
        for key in (
            "correction_fraction", "ecs_rank", "ecs_fraction", "normalization_dimension",
            "bulk_effective_count", "trace_log_per_eval", "transposed", "projection_side",
            "base_delta_norm", "ecs_delta_norm", "orthogonal_delta_norm",
            "post_orthogonal_delta_norm", "removed_delta_norm", "orthogonal_fraction",
            "post_orthogonal_fraction", "removed_fraction_of_base",
            "observed_orthogonal_damping", "expected_orthogonal_damping", "damping_error",
            "pythagorean_error", "correction_identity_error",
        ):
            row[key] = getattr(result, key)
        row["solver_status"] = result.status
        return row

    @torch.no_grad()
    def apply_epoch_delta_correction(self, *, epoch: Optional[int] = None) -> list[dict[str, Any]]:
        current_epoch = self.epoch_index if epoch is None else int(epoch)
        self._last_epoch_stats = []
        if not self._epoch_active:
            message = "apply_epoch_delta_correction() called without begin_epoch()."
            if self.config.strict_epoch_lifecycle:
                raise RuntimeError(message)
            return [{"epoch": current_epoch, "global_step": self.global_step,
                     "status": "skipped", "reason": message}]

        due = current_epoch >= int(self.config.warmup_epochs) and (
            (current_epoch + 1) % int(self.config.apply_every_epochs) == 0
        )
        if not due:
            self._epoch_start, self._epoch_active, self.epoch_index = {}, False, current_epoch + 1
            return []

        selected = self._selected_matrix_parameters()
        try:
            for name, before in self._epoch_start.items():
                parameter = selected.get(name)
                if parameter is None:
                    continue
                after = parameter.detach().clone()
                reference = before if self.config.reference == "epoch_start" else after
                previous_rank = self._previous_ecs_ranks.get(name) if self.config.use_previous_rank_as_reference else None
                try:
                    geometry = local_ecs_geometry(
                        reference, min_retained=self.config.min_retained,
                        max_retained=self.config.max_retained,
                        normalization_gamma=self.config.normalization_gamma,
                        reference_rank=previous_rank, eps=self.config.eps,
                    )
                    result = damp_delta_with_geometry(
                        after - before, geometry,
                        correction_fraction=self.config.correction_fraction, eps=self.config.eps,
                    )
                    self._validate_result(name, result)
                except (RuntimeError, ValueError) as exc:
                    if self.config.strict_numerics:
                        raise
                    self._last_epoch_stats.append({
                        "epoch": current_epoch, "global_step": self.global_step,
                        "parameter": name, "status": "geometry_failed",
                        "reason": str(exc), "reference": self.config.reference,
                    })
                    continue
                candidate = before + result.corrected_delta.to(parameter.device, parameter.dtype)
                if not torch.isfinite(candidate).all():
                    raise RuntimeError(f"Local-delta correction produced non-finite weights for {name}.")
                parameter.copy_(candidate)
                adjusted, state_fractions = self._synchronize_state(parameter, geometry)
                self._previous_ecs_ranks[name] = int(result.ecs_rank)
                self._last_epoch_stats.append(self._result_stats(
                    epoch=current_epoch, name=name, previous_rank=previous_rank,
                    result=result, adjusted=adjusted, state_fractions=state_fractions,
                ))
        finally:
            self._epoch_start, self._epoch_active, self.epoch_index = {}, False, current_epoch + 1
        return list(self._last_epoch_stats)

    def pop_epoch_stats(self) -> list[dict[str, Any]]:
        stats, self._last_epoch_stats = self._last_epoch_stats, []
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "global_step": int(self.global_step), "epoch_index": int(self.epoch_index),
            "config": asdict(self.config), "epoch_active": bool(self._epoch_active),
            "epoch_start": {n: t.detach().cpu().clone() for n, t in self._epoch_start.items()},
            "previous_ecs_ranks": dict(self._previous_ecs_ranks),
            "last_epoch_stats": list(self._last_epoch_stats),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.global_step = int(state_dict.get("global_step", 0))
        self.epoch_index = int(state_dict.get("epoch_index", 0))
        self._epoch_active = bool(state_dict.get("epoch_active", False))
        self._epoch_start = {
            n: t.to(device=self.named_parameters[n].device, dtype=self.named_parameters[n].dtype).clone()
            for n, t in state_dict.get("epoch_start", {}).items()
            if n in self.named_parameters and isinstance(t, torch.Tensor)
        }
        self._previous_ecs_ranks = {
            str(n): int(r) for n, r in state_dict.get("previous_ecs_ranks", {}).items()
            if str(n) in self.named_parameters
        }
        self._last_epoch_stats = list(state_dict.get("last_epoch_stats", []))

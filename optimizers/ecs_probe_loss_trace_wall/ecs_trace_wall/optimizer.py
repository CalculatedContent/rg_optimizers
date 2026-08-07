"""The ECS probe-loss TraceWall post-step optimizer wrapper."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .config import TraceWallConfig
from .ecs import ECSSVDState, compute_ecs_svd, project_gradient_to_ecs

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ProbeBatch = tuple[torch.Tensor, torch.Tensor]


@dataclass
class LayerCorrectionRecord:
    parameter_name: str
    ecs_rank: int
    ecs_fractional_rank: float
    ecs_positive_count: int
    ecs_normalization_dimension: float
    ecs_bulk_effective_count: float
    ecs_trace_log: float
    ecs_trace_log_per_eval: float
    ecs_status: str
    base_step_norm: float
    weight_norm: float
    projected_gradient_norm: float
    proposed_correction_norm: float
    accepted_correction_norm: float
    correction_to_base_step_ratio: float
    correction_to_weight_ratio: float
    projection_identity_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectionRecord:
    global_step: int
    attempted: bool
    applied: bool
    reason: str
    probe_examples: int
    probe_loss_before: float
    probe_loss_after: float
    probe_loss_change: float
    directional_derivative: float
    line_search_scale: float
    line_search_iterations: int
    layers: list[LayerCorrectionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layers"] = [layer.to_dict() for layer in self.layers]
        return result


class ECSProbeLossTraceWall:
    """Add an ECS-restricted rotating-probe loss descent component.

    The base optimizer is executed first.  At the configured cadence the
    wrapper then:

    1. computes a self-consistent ECS SVD for every selected matrix;
    2. temporarily replaces all selected matrices by their ECS truncations;
    3. measures the mean loss on the supplied rotating training probe subset;
    4. differentiates that loss with respect to the truncated matrices;
    5. projects each gradient into the same ECS;
    6. adds a norm-controlled negative-gradient component to the completed base
       optimizer displacement, accepting it only when the truncated probe loss
       passes an Armijo backtracking check.

    The official test set is never consumed by this class.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        config: TraceWallConfig,
    ) -> None:
        config.validate()
        self.model = model
        self.base_optimizer = base_optimizer
        self.config = config
        self.global_step = 0
        self.previous_ranks: dict[str, int] = {}
        self.last_record: Optional[CorrectionRecord] = None

        named_parameters = dict(model.named_parameters())
        missing = [name for name in config.parameter_names if name not in named_parameters]
        if missing:
            raise ValueError(f"unknown TraceWall parameters: {missing}")
        self.parameters: dict[str, nn.Parameter] = {}
        for name in config.parameter_names:
            parameter = named_parameters[name]
            if parameter.ndim != 2:
                raise ValueError(f"{name} is not a matrix parameter")
            self.parameters[name] = parameter

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def correction_due(self, step: Optional[int] = None) -> bool:
        candidate = self.global_step + 1 if step is None else int(step)
        if candidate < self.config.correction_start_step:
            return False
        return (
            candidate - self.config.correction_start_step
        ) % self.config.correction_interval_steps == 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "global_step": int(self.global_step),
            "previous_ranks": dict(self.previous_ranks),
            "last_record": None
            if self.last_record is None
            else self.last_record.to_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.base_optimizer.load_state_dict(dict(state["base_optimizer"]))
        self.global_step = int(state.get("global_step", 0))
        self.previous_ranks = {
            str(name): int(rank)
            for name, rank in dict(state.get("previous_ranks", {})).items()
        }
        self.last_record = None

    @contextmanager
    def _temporary_weights(
        self,
        replacements: Mapping[str, torch.Tensor],
        originals: Mapping[str, torch.Tensor],
    ) -> Iterable[None]:
        with torch.no_grad():
            for name, value in replacements.items():
                self.parameters[name].copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in originals.items():
                    self.parameters[name].copy_(value)

    def _mean_probe_loss(
        self,
        probe_batches: Sequence[ProbeBatch],
        loss_function: LossFunction,
    ) -> float:
        if not probe_batches:
            raise ValueError("probe_batches must not be empty")
        total_examples = 0
        weighted_loss = 0.0
        with torch.no_grad():
            for inputs, targets in probe_batches:
                batch_examples = int(targets.shape[0])
                loss = loss_function(self.model(inputs), targets)
                weighted_loss += float(loss.detach().cpu()) * batch_examples
                total_examples += batch_examples
        if total_examples < 1:
            raise ValueError("probe subset contains no examples")
        return float(weighted_loss / total_examples)

    def _mean_probe_gradients(
        self,
        probe_batches: Sequence[ProbeBatch],
        loss_function: LossFunction,
    ) -> tuple[float, dict[str, torch.Tensor]]:
        parameters = list(self.parameters.values())
        gradients = [torch.zeros_like(parameter) for parameter in parameters]
        total_examples = sum(int(targets.shape[0]) for _, targets in probe_batches)
        if total_examples < 1:
            raise ValueError("probe subset contains no examples")

        weighted_loss = 0.0
        for inputs, targets in probe_batches:
            batch_examples = int(targets.shape[0])
            loss = loss_function(self.model(inputs), targets)
            batch_gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            weight = batch_examples / total_examples
            weighted_loss += float(loss.detach().cpu()) * weight
            for accumulator, gradient in zip(gradients, batch_gradients):
                accumulator.add_(gradient.detach(), alpha=float(weight))
        return float(weighted_loss), dict(zip(self.parameters, gradients))

    def _build_states(
        self,
        full_weights: Mapping[str, torch.Tensor],
    ) -> dict[str, ECSSVDState]:
        states: dict[str, ECSSVDState] = {}
        for name, weight in full_weights.items():
            state = compute_ecs_svd(
                weight,
                min_rank=self.config.min_ecs_rank,
                normalization_gamma=self.config.normalization_gamma,
                reference_rank=self.previous_ranks.get(name),
                svd_device=self.config.svd_device,
                numeric_epsilon=self.config.numerical_epsilon,
            )
            states[name] = state
        return states

    def _propose_corrections(
        self,
        states: Mapping[str, ECSSVDState],
        gradients: Mapping[str, torch.Tensor],
        before_base_step: Mapping[str, torch.Tensor],
        after_base_step: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], list[LayerCorrectionRecord], float]:
        corrections: dict[str, torch.Tensor] = {}
        records: list[LayerCorrectionRecord] = []
        directional_derivative = 0.0
        epsilon = self.config.numerical_epsilon

        for name, state in states.items():
            gradient = gradients[name]
            projected = project_gradient_to_ecs(
                gradient,
                state,
                mode=self.config.projection_mode,
            )
            projected_norm = float(torch.linalg.vector_norm(projected).detach().cpu())
            base_displacement = after_base_step[name] - before_base_step[name]
            base_norm = float(
                torch.linalg.vector_norm(base_displacement).detach().cpu()
            )
            weight_norm = float(
                torch.linalg.vector_norm(after_base_step[name]).detach().cpu()
            )
            target_norm = (
                self.config.correction_to_base_step_ratio * base_norm
                + self.config.minimum_weight_fraction * weight_norm
            )
            target_norm = min(
                target_norm,
                self.config.maximum_weight_fraction * max(weight_norm, epsilon),
            )
            if projected_norm <= epsilon or target_norm <= epsilon:
                correction = torch.zeros_like(projected)
            else:
                correction = projected.mul(-target_norm / projected_norm)

            directional_derivative += float(
                torch.sum(gradient * correction).detach().cpu()
            )
            correction_norm = float(
                torch.linalg.vector_norm(correction).detach().cpu()
            )

            # Orthogonal projection audit: P(P(G)) == P(G).
            projected_twice = project_gradient_to_ecs(
                projected,
                state,
                mode=self.config.projection_mode,
            )
            identity_error = float(
                torch.linalg.vector_norm(projected_twice - projected).detach().cpu()
                / max(projected_norm, epsilon)
            )
            corrections[name] = correction
            records.append(
                LayerCorrectionRecord(
                    parameter_name=name,
                    ecs_rank=state.selection.rank,
                    ecs_fractional_rank=state.selection.fractional_rank,
                    ecs_positive_count=state.selection.positive_count,
                    ecs_normalization_dimension=(
                        state.selection.normalization_dimension
                    ),
                    ecs_bulk_effective_count=(
                        state.selection.bulk_effective_count
                    ),
                    ecs_trace_log=state.selection.trace_log,
                    ecs_trace_log_per_eval=state.selection.trace_log_per_eval,
                    ecs_status=state.selection.status,
                    base_step_norm=base_norm,
                    weight_norm=weight_norm,
                    projected_gradient_norm=projected_norm,
                    proposed_correction_norm=correction_norm,
                    accepted_correction_norm=0.0,
                    correction_to_base_step_ratio=(
                        correction_norm / base_norm if base_norm > epsilon else 0.0
                    ),
                    correction_to_weight_ratio=(
                        correction_norm / weight_norm if weight_norm > epsilon else 0.0
                    ),
                    projection_identity_error=identity_error,
                )
            )
        return corrections, records, float(directional_derivative)

    def step(
        self,
        *,
        probe_batches: Optional[Sequence[ProbeBatch]],
        loss_function: LossFunction,
    ) -> CorrectionRecord:
        """Execute one base step and, when due, one TraceWall correction."""

        before_base_step = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }
        self.base_optimizer.step()
        self.global_step += 1

        if not self.correction_due(self.global_step):
            record = CorrectionRecord(
                global_step=self.global_step,
                attempted=False,
                applied=False,
                reason="cadence",
                probe_examples=0,
                probe_loss_before=float("nan"),
                probe_loss_after=float("nan"),
                probe_loss_change=float("nan"),
                directional_derivative=float("nan"),
                line_search_scale=0.0,
                line_search_iterations=0,
                layers=[],
            )
            self.last_record = record
            return record
        if probe_batches is None or len(probe_batches) == 0:
            raise ValueError("a correction is due but no probe batches were supplied")

        original_training_mode = self.model.training
        self.model.eval()
        after_base_step = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }
        states = self._build_states(after_base_step)
        truncated = {name: state.truncated_weight for name, state in states.items()}

        try:
            with self._temporary_weights(truncated, after_base_step):
                probe_loss_before, gradients = self._mean_probe_gradients(
                    probe_batches,
                    loss_function,
                )

            corrections, layer_records, directional_derivative = (
                self._propose_corrections(
                    states,
                    gradients,
                    before_base_step,
                    after_base_step,
                )
            )
            if directional_derivative >= -self.config.numerical_epsilon:
                reason = "no_descent_direction"
                if self.config.strict and directional_derivative > self.config.loss_tolerance:
                    raise RuntimeError(
                        "projected probe correction is not a descent direction"
                    )
                scale = 0.0
                iterations = 0
                accepted_loss = probe_loss_before
            else:
                scale = 1.0
                iterations = 0
                accepted_loss = float("inf")
                maximum_trials = (
                    self.config.maximum_backtracking_steps + 1
                    if self.config.use_backtracking
                    else 1
                )
                for trial in range(maximum_trials):
                    candidate_truncated = {
                        name: truncated[name] + scale * corrections[name]
                        for name in truncated
                    }
                    with self._temporary_weights(
                        candidate_truncated,
                        after_base_step,
                    ):
                        candidate_loss = self._mean_probe_loss(
                            probe_batches,
                            loss_function,
                        )
                    armijo_bound = (
                        probe_loss_before
                        + self.config.armijo_coefficient
                        * scale
                        * directional_derivative
                    )
                    if candidate_loss <= armijo_bound + self.config.loss_tolerance:
                        accepted_loss = candidate_loss
                        iterations = trial
                        break
                    scale *= self.config.backtracking_factor
                else:
                    scale = 0.0
                    iterations = maximum_trials
                    accepted_loss = probe_loss_before
                reason = "accepted" if scale > 0.0 else "line_search_rejected"

            if scale > 0.0:
                with torch.no_grad():
                    for name, correction in corrections.items():
                        self.parameters[name].add_(correction, alpha=float(scale))
                applied = True
            else:
                applied = False

            for layer_record in layer_records:
                layer_record.accepted_correction_norm = (
                    float(scale) * layer_record.proposed_correction_norm
                )
                layer_record.correction_to_base_step_ratio *= float(scale)
                layer_record.correction_to_weight_ratio *= float(scale)

            for name, state in states.items():
                self.previous_ranks[name] = int(state.selection.rank)

            probe_examples = sum(int(targets.shape[0]) for _, targets in probe_batches)
            record = CorrectionRecord(
                global_step=self.global_step,
                attempted=True,
                applied=applied,
                reason=reason,
                probe_examples=probe_examples,
                probe_loss_before=float(probe_loss_before),
                probe_loss_after=float(accepted_loss),
                probe_loss_change=float(accepted_loss - probe_loss_before),
                directional_derivative=float(directional_derivative),
                line_search_scale=float(scale),
                line_search_iterations=int(iterations),
                layers=layer_records,
            )
            if self.config.strict and applied:
                if not record.probe_loss_after <= (
                    record.probe_loss_before + self.config.loss_tolerance
                ):
                    raise RuntimeError("accepted TraceWall correction increased probe loss")
                if any(
                    layer.projection_identity_error > 5e-5
                    for layer in layer_records
                ):
                    raise RuntimeError("ECS projection idempotence audit failed")
                for parameter in self.parameters.values():
                    if not torch.isfinite(parameter).all():
                        raise RuntimeError("TraceWall produced non-finite weights")
            self.last_record = record
            return record
        finally:
            self.model.train(original_training_mode)

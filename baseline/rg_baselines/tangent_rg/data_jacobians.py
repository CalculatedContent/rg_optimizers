"""Data-dependent ECS quotient Jacobians for the captured MNIST MLP state.

Unlike weight-only maps, these operators require a model, a batch, targets,
and (for the one-step map) the complete captured optimizer state.  The module
keeps that dependency explicit and returns the eigenvalues of ``J*J`` for each
declared Jacobian or, where the full quotient is too large, for an explicitly
named orthonormal restricted domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class GrassmannTangentBasis:
    left: torch.Tensor
    sigma_hat: torch.Tensor
    retained_right: torch.Tensor
    shell_right: torch.Tensor
    retained_rank: int
    outer_rank: int
    shell_rank: int
    coordinate_dimension: int
    numerical_rank: int
    rank_tolerance: float
    retained_boundary_gap: float
    retained_boundary_relative_gap: float
    sigma_hat_policy: str


@dataclass(frozen=True)
class DataJacobianSpectrumRecord:
    singular_amplitudes: np.ndarray
    jt_j_nonzero_eigenvalues: np.ndarray
    derivative_rank: int
    input_dimension: int
    output_dimension: int
    operator_kind: str
    map_definition: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class GrassmannLogitJacobianRecord:
    logits: torch.Tensor
    jacobian: torch.Tensor
    spectrum: DataJacobianSpectrumRecord
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class PerExampleLossJacobianRecord:
    quotient_gradients: torch.Tensor
    empirical_fisher: torch.Tensor
    spectrum: DataJacobianSpectrumRecord
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class QuotientGGNRecord:
    weighted_logit_jacobian: torch.Tensor
    ggn_nonzero_eigenvalues: np.ndarray
    spectrum: DataJacobianSpectrumRecord
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class QuotientObservableRecord:
    combined: np.ndarray
    centered_log_singular: np.ndarray
    row_projector: np.ndarray
    retained_rank: int
    outer_rank: int
    retained_boundary_gap: float
    retained_boundary_relative_gap: float
    operator_kind: str
    map_definition: str


@dataclass(frozen=True)
class StepQuotientSketchRecord:
    combined_response: np.ndarray
    radial_response: np.ndarray
    projector_response: np.ndarray
    combined_spectrum: DataJacobianSpectrumRecord
    radial_spectrum: DataJacobianSpectrumRecord
    projector_spectrum: DataJacobianSpectrumRecord
    epsilon: float
    probe_count: int
    operator_kind: str
    map_definition: str


def _positive(values: np.ndarray) -> np.ndarray:
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample) & (sample > 0.0)]
    if sample.size == 0:
        return sample
    tolerance = (
        max(1, sample.size)
        * np.finfo(np.float64).eps
        * float(np.max(sample))
    )
    return np.sort(sample[sample > tolerance])[::-1]


def _record_from_matrix(
    matrix: torch.Tensor | np.ndarray,
    *,
    input_dimension: int,
    output_dimension: int,
    operator_kind: str,
    map_definition: str,
    parameters: Mapping[str, Any],
) -> DataJacobianSpectrumRecord:
    array = (
        matrix.detach().cpu().double().numpy()
        if torch.is_tensor(matrix)
        else np.asarray(matrix, dtype=np.float64)
    )
    amplitudes = _positive(np.linalg.svd(array, compute_uv=False))
    return DataJacobianSpectrumRecord(
        singular_amplitudes=amplitudes,
        jt_j_nonzero_eigenvalues=amplitudes**2,
        derivative_rank=int(amplitudes.size),
        input_dimension=int(input_dimension),
        output_dimension=int(output_dimension),
        operator_kind=str(operator_kind),
        map_definition=str(map_definition),
        parameters=dict(parameters),
    )


def grassmann_tangent_basis(
    weight: torch.Tensor,
    *,
    retained_rank: int,
    outer_rank: int,
    rcond: float = 1.0e-9,
    sigma_hat_policy: str = "checkpoint_core_singular_values",
) -> GrassmannTangentBasis:
    """Build ``B(K)=U_k Sigma_hat K^T V_c^T`` at one checkpoint."""

    if weight.ndim != 2 or weight.shape[0] >= weight.shape[1]:
        raise ValueError("Grassmann tangent basis requires a wide weight matrix")
    if not torch.isfinite(weight).all():
        raise ValueError("weight contains non-finite values")
    if not np.isfinite(rcond) or rcond < 0.0:
        raise ValueError("rcond must be finite and non-negative")
    left, singular_values, right_h = torch.linalg.svd(
        weight, full_matrices=True
    )
    tolerance = float(rcond) * float(singular_values[0].detach().cpu())
    numerical_rank = int(
        torch.count_nonzero(singular_values > tolerance).detach().cpu()
    )
    k, q = int(retained_rank), int(outer_rank)
    if not (1 <= k < q <= numerical_rank):
        raise ValueError("ranks must satisfy 1 <= k < q <= numerical rank")
    squared = singular_values**2
    boundary_gap = float((squared[k - 1] - squared[k]).detach().cpu())
    squared_scale = float(squared[0].detach().cpu())
    if boundary_gap <= float(rcond) * squared_scale:
        raise np.linalg.LinAlgError(
            "retained Grassmann subspace has no numerically differentiable gap"
        )
    policy = str(sigma_hat_policy)
    retained = singular_values[:k]
    if policy == "checkpoint_core_singular_values":
        sigma_hat = retained
    elif policy == "unit_geometric_mean_core":
        sigma_hat = retained / torch.exp(torch.mean(torch.log(retained)))
    elif policy == "unit_core":
        sigma_hat = torch.ones_like(retained)
    else:
        raise ValueError(
            "sigma_hat_policy must be checkpoint_core_singular_values, "
            "unit_geometric_mean_core, or unit_core"
        )
    right = right_h.mT
    shell_rank = q - k
    return GrassmannTangentBasis(
        left=left[:, :k],
        sigma_hat=sigma_hat,
        retained_right=right[:, :k],
        shell_right=right[:, k:q],
        retained_rank=k,
        outer_rank=q,
        shell_rank=shell_rank,
        coordinate_dimension=int(k * shell_rank),
        numerical_rank=numerical_rank,
        rank_tolerance=tolerance,
        retained_boundary_gap=boundary_gap,
        retained_boundary_relative_gap=boundary_gap / squared_scale,
        sigma_hat_policy=policy,
    )


def embed_grassmann_coordinate(
    basis: GrassmannTangentBasis,
    coordinate: torch.Tensor,
) -> torch.Tensor:
    expected = (basis.shell_rank, basis.retained_rank)
    if tuple(coordinate.shape) != expected:
        raise ValueError(f"coordinate shape must be {expected}")
    core = basis.left * basis.sigma_hat.unsqueeze(0)
    return core @ coordinate.mT @ basis.shell_right.mT


def pullback_grassmann_direction(
    basis: GrassmannTangentBasis,
    direction: torch.Tensor,
) -> torch.Tensor:
    expected = (basis.left.shape[0], basis.shell_right.shape[0])
    if tuple(direction.shape) != expected:
        raise ValueError(f"direction shape must be {expected}")
    core = basis.left * basis.sigma_hat.unsqueeze(0)
    return basis.shell_right.mT @ direction.mT @ core


def input_output_jacobian_spectrum(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    maximum_examples: int | None = None,
) -> DataJacobianSpectrumRecord:
    """Exact block-diagonal batch input-output Jacobian for the MLP."""

    count = int(inputs.shape[0])
    if maximum_examples is not None:
        count = min(count, int(maximum_examples))
    if count < 1:
        raise ValueError("at least one input example is required")
    blocks: list[np.ndarray] = []
    output_count = None
    for index in range(count):
        sample = inputs[index : index + 1].detach().clone().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            lambda value: model(value).squeeze(0),
            sample,
            vectorize=True,
            create_graph=False,
        )
        output_count = int(jacobian.shape[0])
        block = jacobian.reshape(output_count, -1)
        blocks.append(torch.linalg.svdvals(block).detach().cpu().double().numpy())
    amplitudes = _positive(np.concatenate(blocks))
    input_dimension = int(count * inputs[0].numel())
    output_dimension = int(count * int(output_count))
    return DataJacobianSpectrumRecord(
        singular_amplitudes=amplitudes,
        jt_j_nonzero_eigenvalues=amplitudes**2,
        derivative_rank=int(amplitudes.size),
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        operator_kind="exact_batch_block_diagonal_input_output_jacobian",
        map_definition=(
            "J_x=d f(W;X)/dX for the captured examples; the MLP has no "
            "cross-example layers, so the batch Jacobian is block diagonal"
        ),
        parameters={"examples": count, "classes": int(output_count)},
    )


def grassmann_parameter_output_jacobian(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    parameter_name: str,
    basis: GrassmannTangentBasis,
    maximum_examples: int | None = None,
    chunk_size: int | None = 8,
) -> GrassmannLogitJacobianRecord:
    """Form the exact captured-batch ``D_W f o B`` Jacobian."""

    from torch.func import functional_call, jacrev

    count = int(inputs.shape[0])
    if maximum_examples is not None:
        count = min(count, int(maximum_examples))
    if count < 1:
        raise ValueError("at least one input example is required")
    selected_inputs = inputs[:count]
    parameters = {name: value.detach() for name, value in model.named_parameters()}
    buffers = {name: value.detach() for name, value in model.named_buffers()}
    if parameter_name not in parameters:
        raise KeyError(f"model parameter {parameter_name!r} was not found")
    base_weight = parameters[parameter_name]
    if tuple(base_weight.shape) != (
        basis.left.shape[0], basis.shell_right.shape[0]
    ):
        raise ValueError("basis and selected model parameter shapes disagree")

    def logits_from_coordinate(coordinate_flat: torch.Tensor) -> torch.Tensor:
        coordinate = coordinate_flat.reshape(
            basis.shell_rank, basis.retained_rank
        )
        local_parameters = dict(parameters)
        local_parameters[parameter_name] = base_weight + embed_grassmann_coordinate(
            basis, coordinate
        )
        return functional_call(
            model, (local_parameters, buffers), (selected_inputs,)
        )

    zero = torch.zeros(
        basis.coordinate_dimension,
        dtype=base_weight.dtype,
        device=base_weight.device,
    )
    jacobian_function = jacrev(logits_from_coordinate, chunk_size=chunk_size)
    jacobian = jacobian_function(zero)
    logits = logits_from_coordinate(zero).detach()
    batch, classes = int(jacobian.shape[0]), int(jacobian.shape[1])
    flattened = jacobian.reshape(batch * classes, basis.coordinate_dimension)
    definition = (
        "J_f,Gr=D_W f(W;X) o B, B(K)=U_k Sigma_hat K^T V_c^T; "
        "captured model and inputs fixed"
    )
    spectrum = _record_from_matrix(
        flattened,
        input_dimension=basis.coordinate_dimension,
        output_dimension=batch * classes,
        operator_kind="exact_grassmann_parameter_output_jacobian",
        map_definition=definition,
        parameters={
            "examples": batch,
            "classes": classes,
            "retained_rank": basis.retained_rank,
            "outer_rank": basis.outer_rank,
            "sigma_hat_policy": basis.sigma_hat_policy,
        },
    )
    return GrassmannLogitJacobianRecord(
        logits=logits,
        jacobian=jacobian.detach(),
        spectrum=spectrum,
        operator_kind=spectrum.operator_kind,
        map_definition=definition,
    )


def per_example_quotient_loss_jacobian(
    logit_record: GrassmannLogitJacobianRecord,
    targets: torch.Tensor,
) -> PerExampleLossJacobianRecord:
    """Stack ``B* grad_W ell_b`` and return the quotient empirical Fisher."""

    logits = logit_record.logits
    count, classes = int(logits.shape[0]), int(logits.shape[1])
    selected_targets = targets[:count].to(device=logits.device)
    probabilities = torch.softmax(logits, dim=-1)
    one_hot = torch.nn.functional.one_hot(
        selected_targets, num_classes=classes
    ).to(dtype=logits.dtype)
    loss_logit_gradient = probabilities - one_hot
    jacobian = logit_record.jacobian.reshape(count, classes, -1)
    quotient_gradients = torch.einsum(
        "bc,bcd->bd", loss_logit_gradient, jacobian
    )
    empirical_fisher = quotient_gradients.mT @ quotient_gradients
    definition = (
        "rows are g_b^Gr=B* grad_W ell_b=(grad_logits ell_b)^T J_f,Gr; "
        "J_loss,Gr*J_loss,Gr is the quotient empirical Fisher"
    )
    spectrum = _record_from_matrix(
        quotient_gradients,
        input_dimension=int(quotient_gradients.shape[1]),
        output_dimension=count,
        operator_kind="exact_per_example_quotient_loss_jacobian",
        map_definition=definition,
        parameters={"examples": count, "classes": classes},
    )
    return PerExampleLossJacobianRecord(
        quotient_gradients=quotient_gradients.detach(),
        empirical_fisher=empirical_fisher.detach(),
        spectrum=spectrum,
        operator_kind=spectrum.operator_kind,
        map_definition=definition,
    )


def quotient_generalized_gauss_newton(
    logit_record: GrassmannLogitJacobianRecord,
    *,
    loss_reduction: str = "mean",
) -> QuotientGGNRecord:
    """Exact quotient GGN and its ``J*J`` energy spectrum.

    If ``A=H_loss^(1/2) J_f,Gr``, then ``GGN=A* A``.  Its nonzero
    eigenvalues are ``sv(A)^2``.  Treating the GGN itself as the requested
    quotient-gradient Jacobian gives energy eigenvalues ``sv(A)^4``.
    """

    logits = logit_record.logits
    jacobian = logit_record.jacobian.reshape(logits.shape[0], logits.shape[1], -1)
    probabilities = torch.softmax(logits, dim=-1)
    weighted_blocks = []
    for example in range(int(logits.shape[0])):
        probability = probabilities[example]
        hessian = torch.diag(probability) - torch.outer(probability, probability)
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        root = (
            eigenvectors
            * torch.sqrt(torch.clamp(eigenvalues, min=0.0)).unsqueeze(0)
        ) @ eigenvectors.mT
        weighted_blocks.append(root @ jacobian[example])
    weighted = torch.cat(weighted_blocks, dim=0)
    reduction = str(loss_reduction)
    if reduction == "mean":
        weighted = weighted / np.sqrt(float(logits.shape[0]))
    elif reduction != "sum":
        raise ValueError("loss_reduction must be 'mean' or 'sum'")
    factor_amplitudes = _positive(
        torch.linalg.svdvals(weighted).detach().cpu().double().numpy()
    )
    ggn_eigenvalues = factor_amplitudes**2
    definition = (
        "G_GN,Gr=J_f,Gr^T H_loss J_f,Gr=A^T A on the captured batch; "
        f"cross-entropy reduction={reduction}; the reported J*J energies for "
        "J=G_GN,Gr are lambda_GGN^2"
    )
    spectrum = DataJacobianSpectrumRecord(
        singular_amplitudes=ggn_eigenvalues,
        jt_j_nonzero_eigenvalues=ggn_eigenvalues**2,
        derivative_rank=int(ggn_eigenvalues.size),
        input_dimension=int(jacobian.shape[-1]),
        output_dimension=int(jacobian.shape[-1]),
        operator_kind="exact_quotient_generalized_gauss_newton_jacobian",
        map_definition=definition,
        parameters={
            "examples": int(logits.shape[0]),
            "classes": int(logits.shape[1]),
            "factor_rank": int(factor_amplitudes.size),
            "loss_reduction": reduction,
        },
    )
    return QuotientGGNRecord(
        weighted_logit_jacobian=weighted.detach(),
        ggn_nonzero_eigenvalues=ggn_eigenvalues,
        spectrum=spectrum,
        operator_kind=spectrum.operator_kind,
        map_definition=definition,
    )


def quotient_observable(
    weight: np.ndarray,
    *,
    retained_rank: int,
    outer_rank: int,
    rcond: float = 1.0e-9,
) -> QuotientObservableRecord:
    """Evaluate ``Q(W)=(centered log sigma_1:q, P_k(W))``."""

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] >= matrix.shape[1]:
        raise ValueError("quotient observable requires a wide matrix")
    _, singular_values, right_h = np.linalg.svd(matrix, full_matrices=True)
    tolerance = float(rcond) * float(singular_values[0])
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    k, q = int(retained_rank), int(outer_rank)
    if not (1 <= k < q <= numerical_rank):
        raise ValueError("ranks must satisfy 1 <= k < q <= numerical rank")
    squared = singular_values**2
    boundary_gap = float(squared[k - 1] - squared[k])
    squared_scale = float(squared[0])
    if boundary_gap <= float(rcond) * squared_scale:
        raise np.linalg.LinAlgError(
            "row-projector quotient has no numerically differentiable top-k gap"
        )
    radial = np.log(singular_values[:q])
    radial -= float(np.mean(radial))
    right = right_h.T
    projector = right[:, :k] @ right[:, :k].T
    combined = np.concatenate([radial, projector.reshape(-1)])
    return QuotientObservableRecord(
        combined=combined,
        centered_log_singular=radial,
        row_projector=projector,
        retained_rank=k,
        outer_rank=q,
        retained_boundary_gap=boundary_gap,
        retained_boundary_relative_gap=boundary_gap / squared_scale,
        operator_kind="centered_log_singular_plus_row_projector_quotient_map",
        map_definition=(
            "Q(W)=(log sigma_1:q-mean(log sigma_1:q), P_k(W)) with Euclidean "
            "radial metric and full-projector Frobenius metric"
        ),
    )


def _response_spectrum(
    response: np.ndarray,
    *,
    input_dimension: int,
    operator_kind: str,
    map_definition: str,
    parameters: Mapping[str, Any],
) -> DataJacobianSpectrumRecord:
    return _record_from_matrix(
        response,
        input_dimension=input_dimension,
        output_dimension=int(response.shape[0]),
        operator_kind=operator_kind,
        map_definition=map_definition,
        parameters=parameters,
    )


def step_quotient_jacobian_sketch(
    step_map: Callable[[np.ndarray], np.ndarray],
    coordinate_directions: np.ndarray,
    *,
    retained_rank: int,
    outer_rank: int,
    epsilon: float,
    rcond: float = 1.0e-9,
    map_definition: str,
) -> StepQuotientSketchRecord:
    """Central-difference an actual one-step map on an orthonormal K-domain."""

    directions = np.asarray(coordinate_directions, dtype=np.float64)
    if directions.ndim != 3:
        raise ValueError("coordinate_directions must have shape (probes,shell,k)")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    probes = int(directions.shape[0])
    if probes < 1:
        raise ValueError("at least one coordinate direction is required")
    flattened = directions.reshape(probes, -1)
    gram = flattened @ flattened.T
    if not np.allclose(gram, np.eye(probes), rtol=2.0e-10, atol=2.0e-10):
        raise ValueError("coordinate directions must be orthonormal")
    combined_columns = []
    radial_columns = []
    projector_columns = []
    base = quotient_observable(
        np.asarray(step_map(np.zeros_like(directions[0])), dtype=np.float64),
        retained_rank=retained_rank,
        outer_rank=outer_rank,
        rcond=rcond,
    )
    for direction in directions:
        plus_weight = np.asarray(step_map(float(epsilon) * direction), dtype=np.float64)
        minus_weight = np.asarray(step_map(-float(epsilon) * direction), dtype=np.float64)
        plus = quotient_observable(
            plus_weight,
            retained_rank=retained_rank,
            outer_rank=outer_rank,
            rcond=rcond,
        )
        minus = quotient_observable(
            minus_weight,
            retained_rank=retained_rank,
            outer_rank=outer_rank,
            rcond=rcond,
        )
        combined_columns.append((plus.combined - minus.combined) / (2.0 * epsilon))
        radial_columns.append(
            (plus.centered_log_singular - minus.centered_log_singular)
            / (2.0 * epsilon)
        )
        projector_columns.append(
            ((plus.row_projector - minus.row_projector) / (2.0 * epsilon)).reshape(-1)
        )
    combined = np.column_stack(combined_columns)
    radial = np.column_stack(radial_columns)
    projector = np.column_stack(projector_columns)
    full_coordinate_dimension = int(flattened.shape[1])
    restricted_input_dimension = probes
    common = {
        "epsilon": float(epsilon),
        "probe_count": probes,
        "restricted_domain": True,
        "full_coordinate_dimension": full_coordinate_dimension,
        "restricted_input_dimension": restricted_input_dimension,
        "retained_rank": int(retained_rank),
        "outer_rank": int(outer_rank),
        "domain_metric": "canonical Euclidean K metric",
        "post_step_retained_boundary_gap": base.retained_boundary_gap,
        "post_step_retained_boundary_relative_gap": (
            base.retained_boundary_relative_gap
        ),
    }
    combined_definition = (
        str(map_definition)
        + "; D_K[Q(Phi_X(W+B(K)))] restricted to the recorded orthonormal K probes"
    )
    radial_definition = combined_definition + "; centered-log-singular output component"
    projector_definition = combined_definition + "; full row-projector Frobenius output component"
    return StepQuotientSketchRecord(
        combined_response=combined,
        radial_response=radial,
        projector_response=projector,
        combined_spectrum=_response_spectrum(
            combined,
            input_dimension=restricted_input_dimension,
            operator_kind="full_one_step_quotient_stability_jacobian_sketch",
            map_definition=combined_definition,
            parameters=common,
        ),
        radial_spectrum=_response_spectrum(
            radial,
            input_dimension=restricted_input_dimension,
            operator_kind="one_step_centered_log_singular_jacobian_sketch",
            map_definition=radial_definition,
            parameters=common,
        ),
        projector_spectrum=_response_spectrum(
            projector,
            input_dimension=restricted_input_dimension,
            operator_kind="one_step_grassmann_projector_jacobian_sketch",
            map_definition=projector_definition,
            parameters=common,
        ),
        epsilon=float(epsilon),
        probe_count=probes,
        operator_kind="full_one_step_quotient_stability_jacobian_sketch",
        map_definition=combined_definition,
    )


__all__ = [
    "DataJacobianSpectrumRecord",
    "GrassmannLogitJacobianRecord",
    "GrassmannTangentBasis",
    "PerExampleLossJacobianRecord",
    "QuotientGGNRecord",
    "QuotientObservableRecord",
    "StepQuotientSketchRecord",
    "embed_grassmann_coordinate",
    "grassmann_parameter_output_jacobian",
    "grassmann_tangent_basis",
    "input_output_jacobian_spectrum",
    "per_example_quotient_loss_jacobian",
    "pullback_grassmann_direction",
    "quotient_generalized_gauss_newton",
    "quotient_observable",
    "step_quotient_jacobian_sketch",
]

"""Active-set projection onto the full matrix-log no-return cone."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import (
    MatrixLogGeometry,
    ProjectionResult,
    apply_correction_cap,
    matrix_log_correction_from_coefficients,
    matrix_log_mode_drifts,
    matrix_log_mode_gram,
    solve_with_cpu_fallback,
)


@dataclass
class ActiveSetQPSolution:
    multipliers: torch.Tensor
    active_indices: torch.Tensor
    iterations: int
    converged: bool
    kkt_residual: float


def _kkt_residual(
    multipliers: torch.Tensor,
    gradient: torch.Tensor,
) -> float:
    primal_dual = torch.maximum(
        torch.clamp(-multipliers, min=0.0),
        torch.clamp(-gradient, min=0.0),
    )
    complementarity = torch.abs(multipliers * gradient)
    return float(
        torch.max(
            torch.cat([primal_dual.reshape(-1), complementarity.reshape(-1)])
        ).detach().cpu()
    )


@torch.no_grad()
def solve_active_set_nonnegative_qp(
    gram: torch.Tensor,
    linear: torch.Tensor,
    *,
    ridge_relative: float = 1e-8,
    tolerance: float = 1e-8,
    max_iterations: int = 128,
) -> ActiveSetQPSolution:
    """Solve ``min 1/2 l^T G l + q^T l`` subject to ``l >= 0``.

    The solver uses a primal-feasible active set in the dual variables. It adds
    the most violated inactive KKT condition, solves the equality-constrained
    subproblem, and removes multipliers that would cross zero.
    """

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    if linear.ndim != 1 or linear.numel() != gram.shape[0]:
        raise ValueError("linear must match gram")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    n = int(linear.numel())
    if n == 0:
        empty = torch.zeros_like(linear)
        return ActiveSetQPSolution(
            multipliers=empty,
            active_indices=torch.empty(0, dtype=torch.long, device=linear.device),
            iterations=0,
            converged=True,
            kkt_residual=0.0,
        )

    symmetric = 0.5 * (gram + gram.transpose(0, 1))
    diagonal_scale = torch.max(torch.abs(torch.diagonal(symmetric))).clamp_min(
        torch.finfo(symmetric.dtype).eps
    )
    ridge = float(ridge_relative) * diagonal_scale
    regularized = symmetric + ridge * torch.eye(
        n, device=symmetric.device, dtype=symmetric.dtype
    )

    multipliers = torch.zeros_like(linear)
    active: list[int] = []
    converged = False
    iterations = 0

    for iterations in range(1, int(max_iterations) + 1):
        gradient = regularized @ multipliers + linear
        inactive = [index for index in range(n) if index not in active]
        if inactive:
            inactive_tensor = torch.tensor(
                inactive, device=linear.device, dtype=torch.long
            )
            inactive_gradient = gradient[inactive_tensor]
            minimum_value, minimum_offset = torch.min(inactive_gradient, dim=0)
            if float(minimum_value.detach().cpu()) < -float(tolerance):
                active.append(inactive[int(minimum_offset.detach().cpu())])
            else:
                active_gradient = (
                    gradient[
                        torch.tensor(active, device=linear.device, dtype=torch.long)
                    ]
                    if active
                    else torch.empty(0, device=linear.device, dtype=linear.dtype)
                )
                if not active or float(torch.max(torch.abs(active_gradient)).detach().cpu()) <= 10.0 * float(tolerance):
                    converged = True
                    break
        else:
            active_gradient = gradient[
                torch.tensor(active, device=linear.device, dtype=torch.long)
            ]
            if float(torch.max(torch.abs(active_gradient)).detach().cpu()) <= 10.0 * float(tolerance):
                converged = True
                break

        while active:
            active_tensor = torch.tensor(
                active, device=linear.device, dtype=torch.long
            )
            sub_gram = regularized.index_select(0, active_tensor).index_select(
                1, active_tensor
            )
            candidate_active = solve_with_cpu_fallback(
                sub_gram,
                -linear[active_tensor],
            )
            candidate = torch.zeros_like(multipliers)
            candidate[active_tensor] = candidate_active

            if bool(torch.all(candidate_active > float(tolerance)).item()):
                multipliers = candidate
                break

            direction = candidate - multipliers
            crossing: list[tuple[float, int]] = []
            for index in active:
                direction_value = float(direction[index].detach().cpu())
                if direction_value < -float(tolerance):
                    current_value = float(multipliers[index].detach().cpu())
                    crossing.append((-current_value / direction_value, index))

            if not crossing:
                active = [
                    index
                    for index in active
                    if float(candidate[index].detach().cpu()) > float(tolerance)
                ]
                multipliers.zero_()
                if active:
                    kept = torch.tensor(active, device=linear.device, dtype=torch.long)
                    multipliers[kept] = candidate[kept]
                continue

            step, _ = min(crossing, key=lambda item: item[0])
            step = min(1.0, max(0.0, float(step)))
            multipliers = multipliers + step * direction
            multipliers = torch.clamp(multipliers, min=0.0)
            active = [
                index
                for index in active
                if float(multipliers[index].detach().cpu()) > float(tolerance)
            ]

        if not active:
            multipliers.zero_()

    gradient = regularized @ multipliers + linear
    residual = _kkt_residual(multipliers, gradient)
    if residual <= 20.0 * float(tolerance):
        converged = True
    active_indices = torch.nonzero(
        multipliers > float(tolerance), as_tuple=False
    ).reshape(-1)
    return ActiveSetQPSolution(
        multipliers=multipliers,
        active_indices=active_indices,
        iterations=int(iterations),
        converged=bool(converged),
        kkt_residual=float(residual),
    )


@torch.no_grad()
def project_matrix_log_cone(
    delta_base: torch.Tensor,
    geometry: MatrixLogGeometry,
    *,
    projection_strength: float = 1.0,
    max_correction_ratio: float | None = 0.10,
    gram_ridge_relative: float = 1e-8,
    tolerance: float = 1e-8,
    max_iterations: int = 128,
    log_deadband: float = 1e-6,
    eps: float = 1e-12,
) -> ProjectionResult:
    """Minimum-norm active-set projection onto the no-return cone.

    For every retained mode outside the log deadband, the accepted full-strength
    direction satisfies

        sign(log(lambda_i)) * d log(lambda_i) >= 0.

    A strength below one or a correction-norm cap deliberately relaxes exact
    feasibility and is reported through the residual diagnostics.
    """

    if not 0.0 <= float(projection_strength) <= 1.0:
        raise ValueError("projection_strength must lie in [0, 1]")
    if log_deadband < 0.0:
        raise ValueError("log_deadband must be non-negative")

    base_mode_drifts = matrix_log_mode_drifts(delta_base, geometry)
    log_values = geometry.log_eigenvalues.to(
        device=base_mode_drifts.device,
        dtype=base_mode_drifts.dtype,
    )
    eligible = torch.abs(log_values) > float(log_deadband)
    signed_all = torch.sign(log_values) * base_mode_drifts
    inward_all = eligible & (signed_all < -float(tolerance))
    inward_count = int(torch.count_nonzero(inward_all).detach().cpu())
    base_inward_norm = (
        float(torch.linalg.vector_norm(base_mode_drifts[inward_all]).detach().cpu())
        if inward_count
        else 0.0
    )
    base_drift = float(
        torch.sum(
            geometry.gradient.float() * delta_base.float()
        ).detach().cpu()
    )
    max_before = (
        float(torch.max(torch.clamp(-signed_all[eligible], min=0.0)).detach().cpu())
        if bool(torch.any(eligible).item())
        else 0.0
    )

    if inward_count == 0:
        zero = torch.zeros_like(delta_base)
        return ProjectionResult(
            corrected_delta=delta_base,
            correction=zero,
            mode="cone",
            base_drift=base_drift,
            corrected_drift=base_drift,
            coefficient=0.0,
            correction_ratio=0.0,
            applied=False,
            capped=False,
            inward_mode_count=0,
            base_inward_mode_norm=0.0,
            corrected_inward_mode_norm=0.0,
            max_signed_violation_before=max_before,
            max_signed_violation_after=max_before,
        )

    eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    signs = torch.sign(log_values[eligible_indices])
    linear = signs * base_mode_drifts[eligible_indices]
    mode_gram = matrix_log_mode_gram(geometry)
    sub_gram = mode_gram.index_select(0, eligible_indices).index_select(
        1, eligible_indices
    )
    signed_gram = signs.unsqueeze(1) * sub_gram * signs.unsqueeze(0)

    solution = solve_active_set_nonnegative_qp(
        signed_gram,
        linear,
        ridge_relative=gram_ridge_relative,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    coefficients = torch.zeros_like(base_mode_drifts)
    coefficients[eligible_indices] = solution.multipliers * signs
    full_correction = matrix_log_correction_from_coefficients(
        coefficients,
        geometry,
        output_like=delta_base,
    )
    correction = float(projection_strength) * full_correction
    correction, ratio, capped = apply_correction_cap(
        correction,
        delta_base,
        max_correction_ratio=max_correction_ratio,
        eps=eps,
    )
    corrected = delta_base + correction
    corrected_mode_drifts = matrix_log_mode_drifts(corrected, geometry)
    signed_after = torch.sign(log_values) * corrected_mode_drifts
    corrected_inward_norm = (
        float(
            torch.linalg.vector_norm(
                corrected_mode_drifts[inward_all]
            ).detach().cpu()
        )
        if inward_count
        else 0.0
    )
    max_after = (
        float(torch.max(torch.clamp(-signed_after[eligible], min=0.0)).detach().cpu())
        if bool(torch.any(eligible).item())
        else 0.0
    )
    corrected_drift = float(
        torch.sum(
            geometry.gradient.float() * corrected.float()
        ).detach().cpu()
    )
    applied = bool(
        float(torch.linalg.vector_norm(correction.float()).detach().cpu()) > float(eps)
    )

    return ProjectionResult(
        corrected_delta=corrected,
        correction=correction,
        mode="cone",
        base_drift=base_drift,
        corrected_drift=corrected_drift,
        coefficient=float(
            torch.linalg.vector_norm(solution.multipliers).detach().cpu()
        ),
        correction_ratio=ratio,
        applied=applied,
        capped=capped,
        inward_mode_count=inward_count,
        base_inward_mode_norm=base_inward_norm,
        corrected_inward_mode_norm=corrected_inward_norm,
        cone_active_set_size=int(solution.active_indices.numel()),
        cone_iterations=int(solution.iterations),
        cone_converged=bool(solution.converged),
        max_signed_violation_before=max_before,
        max_signed_violation_after=max_after,
    )

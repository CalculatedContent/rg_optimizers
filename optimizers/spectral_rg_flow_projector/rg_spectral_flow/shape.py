"""Centered log-spectrum coordinates and a local trivial-branch surrogate."""

from __future__ import annotations

from typing import Literal

import torch

CollapsePotential = Literal["participation_ratio", "entropy"]


def _clip_rank(value: int, lower: int, upper: int) -> int:
    return max(int(lower), min(int(value), int(upper)))


def centered_log_eigenvalue_shape(
    singular_values_descending: torch.Tensor,
    retained_rank: int,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Centered log eigenvalues for the largest retained singular values."""
    if singular_values_descending.ndim != 1:
        raise ValueError("singular_values_descending must be one-dimensional.")
    rank = _clip_rank(retained_rank, 1, singular_values_descending.numel())
    s = singular_values_descending[:rank].clamp_min(float(eps))
    log_eigenvalues = 2.0 * torch.log(s)
    return log_eigenvalues - torch.mean(log_eigenvalues)


def spectral_probabilities_from_shape(shape: torch.Tensor) -> torch.Tensor:
    """Convert centered log eigenvalues to normalized retained energy weights."""
    return torch.softmax(shape, dim=0)


def effective_rank_from_shape(
    shape: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    p = spectral_probabilities_from_shape(shape)
    h2 = torch.sum(p.square()).clamp_min(float(eps))
    return 1.0 / h2


def rank_alpha_proxy_from_shape(
    shape: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """OLS rank-order alpha proxy on the same retained window.

    For ``lambda_k ~ k^-q``, ``alpha = 1 + 1/q``.  This is a local
    diagnostic only; WeightWatcher remains the authoritative alpha estimator.
    """
    if shape.numel() < 3:
        return float("nan")
    x = torch.log(
        torch.arange(
            1,
            shape.numel() + 1,
            device=shape.device,
            dtype=shape.dtype,
        )
    )
    x_centered = x - torch.mean(x)
    y_centered = shape - torch.mean(shape)
    denominator = torch.sum(x_centered.square()).clamp_min(float(eps))
    slope = torch.sum(x_centered * y_centered) / denominator
    q = -float(slope.detach().cpu())
    return 1.0 + 1.0 / q if q > float(eps) else float("nan")


def collapse_potential_from_shape(
    shape: torch.Tensor,
    *,
    potential: CollapsePotential = "participation_ratio",
    eps: float = 1e-12,
) -> torch.Tensor:
    """A continuous potential increasing toward spectral rank collapse."""
    p = spectral_probabilities_from_shape(shape)
    if potential == "participation_ratio":
        return torch.log(torch.sum(p.square()).clamp_min(float(eps)))
    if potential == "entropy":
        p_safe = p.clamp_min(float(eps))
        return torch.sum(p * torch.log(p_safe))
    raise ValueError(f"Unknown collapse potential: {potential!r}")


def trivial_branch_flow_vector(
    shape: torch.Tensor,
    *,
    potential: CollapsePotential = "participation_ratio",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Gradient of the collapse potential in centered log-spectrum space."""
    p = spectral_probabilities_from_shape(shape)
    if potential == "participation_ratio":
        h2 = torch.sum(p.square()).clamp_min(float(eps))
        vector = 2.0 * (p.square() / h2 - p)
    elif potential == "entropy":
        p_safe = p.clamp_min(float(eps))
        c = torch.sum(p * torch.log(p_safe))
        vector = p * (torch.log(p_safe) - c)
    else:
        raise ValueError(f"Unknown collapse potential: {potential!r}")
    # Remove roundoff in the scale direction explicitly.
    return vector - torch.mean(vector)

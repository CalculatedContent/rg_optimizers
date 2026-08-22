"""Deterministic synthetic data helpers for examples and tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def factor_covariance(
    n_assets: int,
    n_factors: int = 4,
    *,
    seed: int = 7,
    specific_risk: float = 0.08,
) -> tuple[FloatArray, FloatArray]:
    """Create a positive-definite covariance and its asset-factor loadings."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(scale=0.35, size=(n_assets, n_factors))
    raw = rng.normal(size=(n_factors, n_factors))
    factor_cov = raw @ raw.T / n_factors + 0.2 * np.eye(n_factors)
    idiosyncratic = specific_risk * (0.5 + rng.random(n_assets))
    covariance = loadings @ factor_cov @ loadings.T + np.diag(idiosyncratic)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, loadings


def capped_long_only_portfolio(n_assets: int, *, cap: float, seed: int = 11) -> FloatArray:
    """Generate a fully invested long-only portfolio below a per-name cap."""
    if cap * n_assets < 1.0:
        raise ValueError("cap is too small for a fully invested portfolio")
    rng = np.random.default_rng(seed)
    weights = np.full(n_assets, 1.0 / n_assets)
    perturbation = rng.normal(scale=0.15 / n_assets, size=n_assets)
    perturbation -= np.mean(perturbation)
    weights = np.clip(weights + perturbation, 0.0, cap)
    for _ in range(100):
        deficit = 1.0 - float(np.sum(weights))
        if abs(deficit) < 1.0e-14:
            break
        if deficit > 0.0:
            room = cap - weights
            active = room > 1.0e-14
            weights[active] += deficit * room[active] / np.sum(room[active])
        else:
            active = weights > 1.0e-14
            weights[active] += deficit * weights[active] / np.sum(weights[active])
        weights = np.clip(weights, 0.0, cap)
    return weights / np.sum(weights)


def sector_membership(n_assets: int, n_sectors: int) -> FloatArray:
    """Return a sector-by-asset binary membership matrix."""
    sectors = np.zeros((n_sectors, n_assets), dtype=float)
    for asset in range(n_assets):
        sectors[asset % n_sectors, asset] = 1.0
    return sectors

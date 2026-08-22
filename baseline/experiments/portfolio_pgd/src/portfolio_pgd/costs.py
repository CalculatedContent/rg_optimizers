"""Differentiable convex transaction-cost models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class TransactionCost(ABC):
    """Interface for an additive convex cost applied to the trade vector."""

    @abstractmethod
    def value(self, trades: FloatArray) -> float:
        """Return the total transaction cost."""

    @abstractmethod
    def gradient(self, trades: FloatArray) -> FloatArray:
        """Return the gradient with respect to trades."""

    def hessian_diag(self, trades: FloatArray) -> FloatArray:
        """Return a diagonal Hessian approximation, used only for step initialization."""
        return np.zeros_like(trades, dtype=float)


def _broadcast_parameter(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array))
    if array.shape != (size,):
        raise ValueError(f"{name} must be scalar or have shape ({size},)")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


@dataclass(frozen=True)
class PowerLawCost(TransactionCost):
    r"""Separable cost :math:`\sum_i \eta_i |t_i|^p` with optional smoothing.

    With ``epsilon > 0`` the implementation uses
    ``eta * ((t**2 + epsilon**2)**(p/2) - epsilon**p)``.  This keeps the
    objective smooth and convex for every ``p > 1`` while preserving zero cost
    at zero trade.
    """

    eta: ArrayLike
    p: float = 1.5
    epsilon: float = 0.0

    def _eta(self, size: int) -> FloatArray:
        eta = _broadcast_parameter(self.eta, size, "eta")
        if np.any(eta < 0.0):
            raise ValueError("eta must be nonnegative")
        if self.p <= 1.0:
            raise ValueError("p must be greater than one for a differentiable convex cost")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative")
        return eta

    def value(self, trades: FloatArray) -> float:
        trades = np.asarray(trades, dtype=float)
        eta = self._eta(trades.size)
        if self.epsilon == 0.0:
            return float(np.sum(eta * np.abs(trades) ** self.p))
        radius2 = trades * trades + self.epsilon * self.epsilon
        return float(np.sum(eta * (radius2 ** (0.5 * self.p) - self.epsilon**self.p)))

    def gradient(self, trades: FloatArray) -> FloatArray:
        trades = np.asarray(trades, dtype=float)
        eta = self._eta(trades.size)
        if self.epsilon == 0.0:
            return self.p * eta * np.abs(trades) ** (self.p - 1.0) * np.sign(trades)
        radius2 = trades * trades + self.epsilon * self.epsilon
        return self.p * eta * trades * radius2 ** (0.5 * self.p - 1.0)

    def hessian_diag(self, trades: FloatArray) -> FloatArray:
        trades = np.asarray(trades, dtype=float)
        eta = self._eta(trades.size)
        if self.epsilon == 0.0:
            magnitude = np.abs(trades)
            with np.errstate(divide="ignore", invalid="ignore"):
                diagonal = self.p * (self.p - 1.0) * eta * magnitude ** (self.p - 2.0)
            return np.nan_to_num(diagonal, nan=0.0, posinf=np.finfo(float).max ** 0.25)
        radius2 = trades * trades + self.epsilon * self.epsilon
        return (
            self.p
            * eta
            * radius2 ** (0.5 * self.p - 2.0)
            * (self.epsilon * self.epsilon + (self.p - 1.0) * trades * trades)
        )


@dataclass(frozen=True)
class SmoothAbsoluteCost(TransactionCost):
    r"""Smooth bid-ask cost ``rate * (sqrt(t**2 + epsilon**2) - epsilon)``."""

    rate: ArrayLike
    epsilon: float = 1.0e-4

    def _rate(self, size: int) -> FloatArray:
        rate = _broadcast_parameter(self.rate, size, "rate")
        if np.any(rate < 0.0):
            raise ValueError("rate must be nonnegative")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be strictly positive")
        return rate

    def value(self, trades: FloatArray) -> float:
        trades = np.asarray(trades, dtype=float)
        rate = self._rate(trades.size)
        return float(np.sum(rate * (np.sqrt(trades * trades + self.epsilon**2) - self.epsilon)))

    def gradient(self, trades: FloatArray) -> FloatArray:
        trades = np.asarray(trades, dtype=float)
        rate = self._rate(trades.size)
        return rate * trades / np.sqrt(trades * trades + self.epsilon**2)

    def hessian_diag(self, trades: FloatArray) -> FloatArray:
        trades = np.asarray(trades, dtype=float)
        rate = self._rate(trades.size)
        return rate * self.epsilon**2 / (trades * trades + self.epsilon**2) ** 1.5

"""Polar-projection Jacobian spectra for single matrix checkpoints."""
from __future__ import annotations
import numpy as np


def polar_factor(weight: np.ndarray) -> np.ndarray:
    w = np.asarray(weight, dtype=np.float64)
    u, _, vh = np.linalg.svd(w, full_matrices=False)
    return u @ vh


def frechet_action(weight: np.ndarray, perturbation: np.ndarray) -> np.ndarray:
    """Closed-form D Pi(W)[E] for Pi(W)=U V^T at full-rank W."""
    w = np.asarray(weight, dtype=np.float64)
    e = np.asarray(perturbation, dtype=np.float64)
    u, s, vh = np.linalg.svd(w, full_matrices=False)
    v = vh.T
    a = u.T @ e @ v
    omega = (a - a.T) / (s[:, None] + s[None, :])
    np.fill_diagonal(omega, 0.0)
    out = u @ omega @ v.T
    m, n = w.shape
    if m > n:
        out += (e - u @ (u.T @ e)) @ v @ np.diag(1.0 / s) @ v.T
    elif n > m:
        out += u @ np.diag(1.0 / s) @ u.T @ e @ (np.eye(n) - v @ v.T)
    return out


def analytic_gram_spectrum(weight: np.ndarray) -> tuple[np.ndarray, int]:
    """Exact positive spectrum of (D Pi)^* D Pi and zero-mode count."""
    w = np.asarray(weight, dtype=np.float64)
    m, n = w.shape
    s = np.linalg.svd(w, compute_uv=False)
    r = min(m, n)
    rot = np.fromiter(
        (4.0 / (s[i] + s[j]) ** 2 for i in range(r) for j in range(i + 1, r)),
        dtype=np.float64,
        count=r * (r - 1) // 2,
    )
    trans = np.repeat(1.0 / (s * s), abs(m - n)) if m != n else np.empty(0)
    positive = np.sort(np.concatenate([rot, trans]))
    return positive, int(m * n - positive.size)


def numerical_probe_gram_spectrum(
    weight: np.ndarray,
    *,
    probes: int = 128,
    eps_rel: float = 1e-5,
    seed: int = 918273,
) -> tuple[np.ndarray, dict[str, float]]:
    """Finite-difference single-checkpoint response Gram X^T X spectrum."""
    w = np.asarray(weight, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    step = float(eps_rel) * max(1.0, float(np.linalg.norm(w, "fro")))
    x = np.empty((w.size, int(probes)), dtype=np.float64)
    for k in range(int(probes)):
        e = rng.normal(size=w.shape)
        e /= np.linalg.norm(e, "fro")
        response = (polar_factor(w + step * e) - polar_factor(w - step * e)) / (2.0 * step)
        x[:, k] = response.reshape(-1)
    gram = x.T @ x
    evals = np.linalg.eigvalsh(gram)
    tol = np.finfo(float).eps * max(gram.shape) * max(float(evals[-1]), 1.0)
    return np.sort(evals[evals > tol]), {
        "probes": int(probes),
        "epsilon": float(step),
        "epsilon_relative": float(eps_rel),
    }


def weightwatcher_pl_fit(evals: np.ndarray) -> dict[str, float]:
    """Use WeightWatcher's own PL fitter on raw positive eigenvalues."""
    from weightwatcher.WW_powerlaw import pl_fit
    values = np.asarray(evals, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    fit = pl_fit(data=values, xmin=None, xmax=None, verbose=False)
    return {
        "alpha": float(fit.alpha),
        "D": float(fit.D),
        "xmin": float(fit.xmin),
        "xmax": float(np.max(values)),
        "tail_evals": int(np.count_nonzero(values >= fit.xmin)),
    }


def finite_difference_error(weight: np.ndarray, perturbation: np.ndarray, eps_rel: float = 1e-6) -> float:
    w = np.asarray(weight, dtype=np.float64)
    e = np.asarray(perturbation, dtype=np.float64)
    step = float(eps_rel) * max(1.0, float(np.linalg.norm(w, "fro")))
    fd = (polar_factor(w + step * e) - polar_factor(w - step * e)) / (2.0 * step)
    exact = frechet_action(w, e)
    return float(np.linalg.norm(fd - exact) / max(np.linalg.norm(fd), np.finfo(float).tiny))


def probe_convergence(weight: np.ndarray, counts=(16, 32, 64, 128, 256), *, eps_rel=1e-5, seed=918273):
    """Return analytic and finite-probe WeightWatcher fits for one checkpoint."""
    analytic, _ = analytic_gram_spectrum(weight)
    reference = weightwatcher_pl_fit(analytic)
    rows = [{"method": "analytic", "probes": np.nan, **reference}]
    for count in counts:
        sampled, _ = numerical_probe_gram_spectrum(weight, probes=int(count), eps_rel=eps_rel, seed=seed)
        rows.append({"method": "numerical_finite_difference", "probes": int(count), **weightwatcher_pl_fit(sampled)})
    for row in rows:
        row["alpha_analytic"] = reference["alpha"]
        row["alpha_difference"] = row["alpha"] - reference["alpha"]
    return rows

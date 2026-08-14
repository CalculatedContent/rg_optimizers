from __future__ import annotations

import numpy as np

from rg_nanogpt_one_head.angular_three_checkpoint import (
    _monte_carlo_ks_variable,
    _monte_carlo_tail_ks,
    projective_continuous,
)


def test_projective_continuous_excludes_upper_endpoint_atom():
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    x, endpoint_atoms, zero_atoms = projective_continuous(
        values,
        4.0,
        endpoint_tol=1e-10,
    )

    # lambda=4 is a discrete endpoint atom, not a manufactured x~1e12 tail
    # observation. lambda=0 is also not a positive tail observation.
    np.testing.assert_allclose(x, [1.0 / 3.0, 1.0, 3.0])
    assert endpoint_atoms == 1
    assert zero_atoms == 1
    assert np.all(np.isfinite(x))
    assert float(np.max(x)) == 3.0


def test_projective_continuous_removes_near_endpoint_numerical_atom():
    values = np.asarray([2.0, 4.0 * (1.0 - 5e-11)])
    x, endpoint_atoms, zero_atoms = projective_continuous(
        values,
        4.0,
        endpoint_tol=1e-10,
    )
    np.testing.assert_allclose(x, [1.0])
    assert endpoint_atoms == 1
    assert zero_atoms == 0


def test_projective_continuous_keeps_resolved_large_tail_value():
    # A value clearly separated from the endpoint remains a legitimate large
    # continuous projective observation.
    y = 1.0 - 1e-6
    x, endpoint_atoms, _ = projective_continuous(
        np.asarray([4.0 * y]),
        4.0,
        endpoint_tol=1e-10,
    )
    assert endpoint_atoms == 0
    assert x.size == 1
    assert np.isclose(x[0], y / (1.0 - y))


def test_monte_carlo_ks_detects_obviously_shifted_distribution():
    rng = np.random.default_rng(123)
    nulls = [rng.lognormal(mean=0.0, sigma=0.25, size=80) for _ in range(25)]
    actual = rng.lognormal(mean=1.5, sigma=0.25, size=80)
    D, p = _monte_carlo_ks_variable(actual, nulls)
    assert np.isfinite(D)
    assert np.isfinite(p)
    assert D > 0.5
    assert p < 0.1


def test_tail_ks_uses_trained_xmin_only_as_diagnostic_threshold():
    rng = np.random.default_rng(456)
    nulls = [rng.pareto(2.0, size=100) + 1.0 for _ in range(20)]
    actual = rng.pareto(2.0, size=100) + 1.0
    D, p, n_null = _monte_carlo_tail_ks(actual, 1.5, nulls)
    assert np.isfinite(D)
    assert np.isfinite(p)
    assert n_null > 3

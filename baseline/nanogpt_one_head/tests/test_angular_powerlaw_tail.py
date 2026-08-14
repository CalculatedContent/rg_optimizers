from __future__ import annotations

import numpy as np

import rg_nanogpt_one_head.angular_powerlaw_tail as tail


class _FakePowerLaw:
    alpha = 2.25
    xmin = 3.0
    D = 0.04


class _FakeFit:
    def __init__(self, data, **kwargs):
        self.data_seen = np.asarray(data, dtype=float)
        self.kwargs_seen = dict(kwargs)
        self.power_law = _FakePowerLaw()


def test_fit_passes_all_positive_values_without_xmin_or_xmax(monkeypatch):
    captured = {}

    def fake_fit(data, **kwargs):
        fit = _FakeFit(data, **kwargs)
        captured["data"] = fit.data_seen
        captured["kwargs"] = fit.kwargs_seen
        return fit

    monkeypatch.setattr(tail.powerlaw, "Fit", fake_fit)

    values = np.asarray([9.0, -2.0, 1.0, np.nan, 5.0, 3.0, 2.0, 7.0])
    result, package_fit = tail.fit_powerlaw_tail(values, min_tail=3)

    np.testing.assert_allclose(captured["data"], [1.0, 2.0, 3.0, 5.0, 7.0, 9.0])
    assert captured["kwargs"] == {"discrete": False, "verbose": False}
    assert "xmin" not in captured["kwargs"]
    assert "xmax" not in captured["kwargs"]
    assert package_fit is not None
    assert result.success
    assert result.alpha == 2.25
    assert result.xmin == 3.0
    assert result.n_tail == 4
    assert result.n_total == 6
    assert result.xmax_observed == 9.0
    assert np.isclose(result.tail_decades, np.log10(9.0 / 3.0))


def test_min_tail_is_applied_after_package_selects_xmin(monkeypatch):
    calls = {"count": 0}

    def fake_fit(data, **kwargs):
        calls["count"] += 1
        return _FakeFit(data, **kwargs)

    monkeypatch.setattr(tail.powerlaw, "Fit", fake_fit)

    # There are enough total observations to call powerlaw.Fit, but the
    # package-selected xmin leaves only two tail values.  The wrapper rejects
    # the fit only after the package has made that selection.
    values = np.asarray([1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    result, package_fit = tail.fit_powerlaw_tail(values, min_tail=5)

    assert calls["count"] == 1
    assert package_fit is not None
    assert result.n_tail == 2
    assert not result.success

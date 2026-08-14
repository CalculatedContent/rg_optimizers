from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from rg_nanogpt_one_head.angular_weightwatcher_pipeline import _powerlaw_fit


PIPELINE_PATH = (
    EXPERIMENT_ROOT
    / "src"
    / "rg_nanogpt_one_head"
    / "angular_weightwatcher_pipeline.py"
)


def test_angular_tail_fit_uses_powerlaw_package_mle_xmin_search():
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    assert "powerlaw.Fit(" in source
    assert "discrete=False" in source
    assert "fit.power_law.alpha" in source
    assert "fit.power_law.xmin" in source
    assert "fit.power_law.D" in source
    assert '"fit_backend": "powerlaw.Fit"' in source


def test_angular_diagnostics_use_powerlaw_native_pdf_cdf_ccdf_plots():
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    assert "package_fit.plot_pdf" in source
    assert "package_fit.power_law.plot_pdf" in source
    assert "powerlaw.plot_pdf(" in source
    assert "linear_bins=True" in source
    assert "package_fit.plot_cdf" in source
    assert "package_fit.power_law.plot_cdf" in source
    assert "package_fit.plot_ccdf" in source
    assert "package_fit.power_law.plot_ccdf" in source
    assert "powerlaw_pdf_loglog.png" in source
    assert "powerlaw_pdf_linear.png" in source
    assert "powerlaw_cdf.png" in source
    assert "powerlaw_ccdf_loglog.png" in source


def test_powerlaw_fit_selects_the_largest_values_above_xmin():
    rng = np.random.default_rng(1234)
    # Continuous Pareto sample with a small non-tail body.  The exact xmin is
    # data-dependent; the contract we need is that n_tail is the count of the
    # largest observations selected by powerlaw.Fit's xmin.
    tail = 1.0 / np.power(rng.random(2000), 1.0 / 1.5)
    body = rng.uniform(0.05, 0.8, size=400)
    data = np.concatenate([body, tail])

    summary, fit = _powerlaw_fit(data, min_tail=20)

    assert fit is not None
    assert summary.success
    expected_n = int(np.count_nonzero(data >= summary.xmin))
    assert summary.n_tail == expected_n
    assert np.isclose(summary.alpha, float(fit.power_law.alpha))
    assert np.isclose(summary.xmin, float(fit.power_law.xmin))
    assert np.isclose(summary.ks, float(fit.power_law.D))

from pathlib import Path

diagnostics_path = Path("baseline/rg_baselines/diagnostics.py")
diagnostics = diagnostics_path.read_text(encoding="utf-8")

new_clean = """\
def clean_positive_eigenvalues(
    values: Any,
    *,
    expected_dimension: Optional[int] = None,
) -> np.ndarray:
    \"\"\"Return positive eigenvalues in ascending order.

    When ``expected_dimension`` is supplied, fail closed if the ESD is
    incomplete, non-finite, or rank deficient. This preserves
    WeightWatcher's full-M normalization instead of silently renormalizing a
    filtered positive-rank spectrum.
    \"\"\"

    evals = np.asarray(values, dtype=float).reshape(-1)
    if expected_dimension is not None:
        expected = int(expected_dimension)
        if expected < 2:
            raise ValueError("expected spectral dimension must be at least two")
        if evals.size != expected:
            raise ValueError(
                "ESD dimension mismatch: "
                f"expected {expected} eigenvalues, received {evals.size}"
            )
        if not np.all(np.isfinite(evals)):
            raise ValueError("full ESD contains non-finite eigenvalues")
        if np.any(evals <= 0.0):
            positive = int(np.count_nonzero(evals > 0.0))
            raise ValueError(
                "rank-deficient ESD: "
                f"expected {expected} positive eigenvalues, found {positive}"
            )
    else:
        evals = evals[np.isfinite(evals) & (evals > 0.0)]

    evals = np.sort(evals)
    if evals.size < 2:
        raise ValueError("fewer than two finite positive eigenvalues")
    return evals
"""
start = diagnostics.index("def clean_positive_eigenvalues(")
end = diagnostics.index("\ndef _entropy_effective_rank", start)
diagnostics = diagnostics[:start] + new_clean + diagnostics[end + 1 :]

new_metrics_prefix = """\
def spectral_metrics_from_esd(
    raw_evals_ascending: Any,
    normalized_evals_ascending: Any,
    *,
    detx_num: int,
    num_pl_spikes: int,
    erg_gap: int,
    expected_dimension: Optional[int] = None,
) -> dict[str, float | int]:
    \"\"\"Compute transparent metrics from one WeightWatcher ESD.

    ``normalized_evals_ascending`` must be produced by WeightWatcher's own
    ``RMT_Util.rescale_eigenvalues``. The trace-log boundary and gap are not
    recomputed here: the supplied ``detx_num``, ``num_pl_spikes``, and
    ``erg_gap`` must come from ``watcher.analyze(ERG=True)``.

    ``expected_dimension`` is the full spectral dimension
    ``min(weight.shape)``. Strict baseline measurements require all of those
    eigenvalues to be finite and positive so WeightWatcher's normalization is
    not silently changed by positive-eigenvalue filtering.
    \"\"\"

    raw = clean_positive_eigenvalues(
        raw_evals_ascending,
        expected_dimension=expected_dimension,
    )
    normalized = clean_positive_eigenvalues(
        normalized_evals_ascending,
        expected_dimension=expected_dimension,
    )
    if raw.size != normalized.size:
        raise ValueError("raw and normalized ESDs have different sizes")

    count = int(raw.size)
    normalized_sum = float(np.sum(normalized))
    if not np.isclose(
        normalized_sum,
        float(count),
        rtol=1e-10,
        atol=1e-10 * max(count, 1),
    ):
        raise ValueError(
            "WeightWatcher normalization audit failed: "
            f"sum={normalized_sum:.17g}, expected={count}"
        )

    m_detx = int(detx_num)
    m_pl = int(num_pl_spikes)
    if not 1 <= m_detx <= count:
        raise ValueError(
            f"detX_num must lie in [1, {count}], received {m_detx}"
        )
    if not 1 <= m_pl <= count:
        raise ValueError(
            f"num_pl_spikes must lie in [1, {count}], received {m_pl}"
        )

    expected_gap = m_detx - m_pl
    if int(erg_gap) != expected_gap:
        raise ValueError(
            f"WeightWatcher ERG_gap audit failed: {erg_gap} != {m_detx} - {m_pl}"
        )
    m_midpoint = int(math.floor((m_detx + m_pl) / 2.0))
"""
start = diagnostics.index("def spectral_metrics_from_esd(")
body = diagnostics.index("    raw_desc = raw[::-1]", start)
diagnostics = diagnostics[:start] + new_metrics_prefix + "\n" + diagnostics[body:]

old_sum = (
    '        "rescaled_eigenvalue_sum": float(np.sum(normalized)),\n'
    '        "rescale_sum_minus_num_eigenvalues": '
    'float(np.sum(normalized) - count),\n'
)
new_sum = (
    '        "rescaled_eigenvalue_sum": normalized_sum,\n'
    '        "rescale_sum_minus_num_eigenvalues": '
    'float(normalized_sum - count),\n'
)
if diagnostics.count(old_sum) != 1:
    raise RuntimeError("unexpected normalized-sum output source")
diagnostics = diagnostics.replace(old_sum, new_sum, 1)

new_measure = """\
            parameter = parameter_map.get(parameter_name) if parameter_name else None
            if parameter is None:
                raise ValueError(
                    "WeightWatcher layer could not be matched to a model matrix"
                )
            expected_dimension = int(min(parameter.shape))
            raw_esd = clean_positive_eigenvalues(
                _get_esd_compat(
                    watcher,
                    model=model_cpu,
                    layer_id=int(layer_id),
                    params=get_esd_params,
                ),
                expected_dimension=expected_dimension,
            )
            normalized_esd, weight_scale = _rescale_with_weightwatcher(raw_esd)
            computed = spectral_metrics_from_esd(
                raw_esd,
                normalized_esd,
                detx_num=int(detx_num),
                num_pl_spikes=int(num_pl_spikes),
                erg_gap=erg_gap,
                expected_dimension=expected_dimension,
            )

"""
start = diagnostics.index("            raw_esd = clean_positive_eigenvalues(")
end = diagnostics.index("            record = {", start)
diagnostics = diagnostics[:start] + new_measure + diagnostics[end:]

diagnostics = diagnostics.replace(
    '                "layer_rows": int(parameter.shape[0]) '
    'if parameter is not None else np.nan,\n'
    '                "layer_cols": int(parameter.shape[1]) '
    'if parameter is not None else np.nan,\n'
    '                "layer_parameter_count": int(parameter.numel()) '
    'if parameter is not None else np.nan,\n',
    '                "layer_rows": int(parameter.shape[0]),\n'
    '                "layer_cols": int(parameter.shape[1]),\n'
    '                "layer_parameter_count": int(parameter.numel()),\n',
    1,
)
diagnostics_path.write_text(diagnostics, encoding="utf-8")

tests_path = Path("baseline/tests/test_diagnostics.py")
tests_path.write_text(
    """\
import unittest

import numpy as np

from rg_baselines.diagnostics import (
    clean_positive_eigenvalues,
    spectral_metrics_from_esd,
)


class SpectralMetricsTests(unittest.TestCase):
    def test_original_boundaries_and_midpoint(self) -> None:
        raw = np.arange(1.0, 11.0)
        normalized = raw * (len(raw) / raw.sum())
        metrics = spectral_metrics_from_esd(
            raw,
            normalized,
            detx_num=8,
            num_pl_spikes=4,
            erg_gap=4,
            expected_dimension=len(raw),
        )
        self.assertEqual(metrics["m_midpoint"], 6)
        self.assertEqual(metrics["ERG_gap"], 4)
        self.assertAlmostEqual(metrics["rescaled_eigenvalue_sum"], 10.0)
        self.assertAlmostEqual(
            metrics["rescale_sum_minus_num_eigenvalues"],
            0.0,
        )
        self.assertGreater(metrics["midpoint_energy_fraction"], 0.5)

    def test_trace_log_matches_analytic_top_spectrum_value(self) -> None:
        raw = np.asarray([1.0, 2.0, 4.0, 8.0])
        normalized = raw * (len(raw) / raw.sum())
        metrics = spectral_metrics_from_esd(
            raw,
            normalized,
            detx_num=4,
            num_pl_spikes=2,
            erg_gap=2,
            expected_dimension=4,
        )

        retained = normalized[::-1][:3]
        expected_total = float(np.sum(np.log(retained)))
        expected_per_eval = float(np.mean(np.log(retained)))
        self.assertEqual(metrics["m_midpoint"], 3)
        self.assertAlmostEqual(
            metrics["trace_log_midpoint_total"],
            expected_total,
        )
        self.assertAlmostEqual(
            metrics["trace_log_midpoint_per_eval"],
            expected_per_eval,
        )
        self.assertAlmostEqual(
            metrics["geometric_mean_midpoint"],
            float(np.exp(expected_per_eval)),
        )
        self.assertAlmostEqual(
            metrics["trace_log_midpoint_total"],
            3.0 * metrics["trace_log_midpoint_per_eval"],
        )

    def test_gap_mismatch_is_rejected(self) -> None:
        raw = np.arange(1.0, 11.0)
        normalized = raw * (len(raw) / raw.sum())
        with self.assertRaisesRegex(ValueError, "ERG_gap audit failed"):
            spectral_metrics_from_esd(
                raw,
                normalized,
                detx_num=8,
                num_pl_spikes=4,
                erg_gap=3,
                expected_dimension=len(raw),
            )

    def test_out_of_range_boundaries_are_rejected(self) -> None:
        raw = np.arange(1.0, 6.0)
        normalized = raw * (len(raw) / raw.sum())
        for field, detx_num, num_pl_spikes in (
            ("detX_num", 6, 2),
            ("num_pl_spikes", 4, 0),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    spectral_metrics_from_esd(
                        raw,
                        normalized,
                        detx_num=detx_num,
                        num_pl_spikes=num_pl_spikes,
                        erg_gap=detx_num - num_pl_spikes,
                        expected_dimension=len(raw),
                    )

    def test_rank_deficient_full_esd_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-deficient ESD"):
            clean_positive_eigenvalues(
                [0.0, 1.0, 2.0],
                expected_dimension=3,
            )

    def test_incomplete_full_esd_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ESD dimension mismatch"):
            clean_positive_eigenvalues(
                [1.0, 2.0],
                expected_dimension=3,
            )

    def test_incorrect_weightwatcher_normalization_is_rejected(self) -> None:
        raw = np.asarray([1.0, 2.0, 3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "normalization audit failed"):
            spectral_metrics_from_esd(
                raw,
                raw,
                detx_num=4,
                num_pl_spikes=2,
                erg_gap=2,
                expected_dimension=4,
            )


if __name__ == "__main__":
    unittest.main()
""",
    encoding="utf-8",
)

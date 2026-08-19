from __future__ import annotations

import unittest

import numpy as np

from rg_baselines.tangent_rg.ecs_jacobians import (
    feshbach_trace_free_log_jvp,
    feshbach_trace_free_log_map,
    feshbach_trace_free_log_spectrum,
    gap_aware_projector_jvp,
    gap_aware_projector_map,
    gap_aware_projector_spectrum,
    outer_resolvent_jvp,
    outer_resolvent_map,
    outer_resolvent_spectrum,
    outer_trace_free_log_gram_jvp,
    outer_trace_free_log_gram_map,
    outer_trace_free_log_gram_spectrum,
    soft_ecs_projector_jvp,
    soft_ecs_projector_map,
    soft_ecs_projector_spectrum,
)


def _explicit_jacobian(function, base, *, epsilon=1.0e-6):
    columns = []
    for index in range(base.size):
        direction = np.zeros_like(base)
        direction.reshape(-1)[index] = 1.0
        plus = np.asarray(function(base + epsilon * direction), dtype=float)
        minus = np.asarray(function(base - epsilon * direction), dtype=float)
        columns.append(((plus - minus) / (2.0 * epsilon)).reshape(-1))
    return np.column_stack(columns)


def _assert_spectrum(test, numerical, analytic, *, rtol=4.0e-5, atol=4.0e-7):
    singular = np.linalg.svd(numerical, compute_uv=False)
    expected = np.asarray(analytic.singular_amplitudes)
    test.assertEqual(expected.size, analytic.derivative_rank)
    np.testing.assert_allclose(
        singular[: expected.size], expected, rtol=rtol, atol=atol
    )
    if singular.size > expected.size:
        test.assertLess(
            float(singular[expected.size]),
            max(atol * 10.0, float(expected[-1]) * 2.0e-4),
        )
    np.testing.assert_allclose(
        analytic.jt_j_nonzero_eigenvalues,
        analytic.singular_amplitudes**2,
        rtol=0.0,
        atol=0.0,
    )


class ECSWeightOnlyJacobianTests(unittest.TestCase):
    def setUp(self):
        self.weight = np.zeros((4, 6), dtype=float)
        self.weight[:4, :4] = np.diag([5.0, 3.0, 2.0, 1.0])
        self.rng = np.random.default_rng(20260820)
        self.direction = self.rng.normal(size=self.weight.shape)

    def test_gap_aware_projector_jvp_and_exact_spectrum(self):
        mapped = lambda candidate: gap_aware_projector_map(
            self.weight,
            candidate,
            retained_rank=2,
            outer_rank=4,
            rcond=1.0e-12,
        ).value
        epsilon = 1.0e-6
        finite = (
            mapped(self.weight + epsilon * self.direction)
            - mapped(self.weight - epsilon * self.direction)
        ) / (2.0 * epsilon)
        analytic_jvp = gap_aware_projector_jvp(
            self.weight,
            self.direction,
            retained_rank=2,
            outer_rank=4,
            rcond=1.0e-12,
        )
        np.testing.assert_allclose(
            analytic_jvp.jvp, finite, rtol=3.0e-5, atol=3.0e-7
        )
        numerical = _explicit_jacobian(mapped, self.weight)
        analytic = gap_aware_projector_spectrum(
            self.weight,
            retained_rank=2,
            outer_rank=4,
            rcond=1.0e-12,
        )
        _assert_spectrum(self, numerical, analytic)
        self.assertEqual(analytic.derivative_rank, 4)
        radial = gap_aware_projector_jvp(
            self.weight,
            self.weight,
            retained_rank=2,
            outer_rank=4,
            rcond=1.0e-12,
        )
        np.testing.assert_allclose(radial.jvp, 0.0, atol=1.0e-12)

    def test_soft_ecs_projector_jvp_and_exact_spectrum(self):
        center, temperature = 6.5, 10.0
        mapped = lambda candidate: soft_ecs_projector_map(
            candidate,
            lambda_center=center,
            temperature=temperature,
        ).value
        epsilon = 1.0e-6
        finite = (
            mapped(self.weight + epsilon * self.direction)
            - mapped(self.weight - epsilon * self.direction)
        ) / (2.0 * epsilon)
        analytic_jvp = soft_ecs_projector_jvp(
            self.weight,
            self.direction,
            lambda_center=center,
            temperature=temperature,
        )
        np.testing.assert_allclose(
            analytic_jvp.jvp, finite, rtol=4.0e-5, atol=4.0e-7
        )
        numerical = _explicit_jacobian(mapped, self.weight)
        analytic = soft_ecs_projector_spectrum(
            self.weight,
            lambda_center=center,
            temperature=temperature,
            rcond=1.0e-12,
        )
        _assert_spectrum(self, numerical, analytic, rtol=8.0e-5, atol=8.0e-7)
        self.assertEqual(analytic.derivative_rank, 18)
        self.assertEqual(analytic.parameters["right_null_dimension"], 2)

    def test_outer_trace_free_log_gram_jvp_and_exact_spectrum(self):
        mapped = lambda candidate: outer_trace_free_log_gram_map(
            self.weight,
            candidate,
            outer_rank=4,
            rcond=1.0e-12,
        ).value
        epsilon = 1.0e-6
        finite = (
            mapped(self.weight + epsilon * self.direction)
            - mapped(self.weight - epsilon * self.direction)
        ) / (2.0 * epsilon)
        analytic_jvp = outer_trace_free_log_gram_jvp(
            self.weight,
            self.direction,
            outer_rank=4,
            rcond=1.0e-12,
        )
        np.testing.assert_allclose(
            analytic_jvp.jvp, finite, rtol=4.0e-5, atol=4.0e-7
        )
        numerical = _explicit_jacobian(mapped, self.weight)
        analytic = outer_trace_free_log_gram_spectrum(
            self.weight, outer_rank=4, rcond=1.0e-12
        )
        _assert_spectrum(self, numerical, analytic)
        self.assertEqual(analytic.derivative_rank, 9)
        scale = outer_trace_free_log_gram_jvp(
            self.weight, self.weight, outer_rank=4, rcond=1.0e-12
        )
        np.testing.assert_allclose(scale.jvp, 0.0, atol=1.0e-12)

    def test_outer_resolvent_jvp_and_exact_spectrum(self):
        mapped = lambda candidate: outer_resolvent_map(
            self.weight,
            candidate,
            outer_rank=4,
            z=2.0,
            trace_free=True,
            rcond=1.0e-12,
        ).value
        epsilon = 1.0e-6
        finite = (
            mapped(self.weight + epsilon * self.direction)
            - mapped(self.weight - epsilon * self.direction)
        ) / (2.0 * epsilon)
        analytic_jvp = outer_resolvent_jvp(
            self.weight,
            self.direction,
            outer_rank=4,
            z=2.0,
            trace_free=True,
            rcond=1.0e-12,
        )
        np.testing.assert_allclose(
            analytic_jvp.jvp, finite, rtol=4.0e-5, atol=4.0e-7
        )
        numerical = _explicit_jacobian(mapped, self.weight)
        analytic = outer_resolvent_spectrum(
            self.weight,
            outer_rank=4,
            z=2.0,
            trace_free=True,
            rcond=1.0e-12,
        )
        _assert_spectrum(self, numerical, analytic)
        self.assertEqual(analytic.derivative_rank, 9)

    def test_feshbach_jvp_spectrum_and_svd_gauge_collapse(self):
        mapped = lambda candidate: feshbach_trace_free_log_map(
            self.weight,
            candidate,
            retained_rank=2,
            outer_rank=4,
            z=0.5,
            rcond=1.0e-12,
        ).value
        epsilon = 1.0e-6
        finite = (
            mapped(self.weight + epsilon * self.direction)
            - mapped(self.weight - epsilon * self.direction)
        ) / (2.0 * epsilon)
        analytic_jvp = feshbach_trace_free_log_jvp(
            self.weight,
            self.direction,
            retained_rank=2,
            outer_rank=4,
            z=0.5,
            rcond=1.0e-12,
        )
        np.testing.assert_allclose(
            analytic_jvp.jvp, finite, rtol=5.0e-5, atol=5.0e-7
        )
        numerical = _explicit_jacobian(mapped, self.weight)
        analytic = feshbach_trace_free_log_spectrum(
            self.weight,
            retained_rank=2,
            outer_rank=4,
            z=0.5,
            rcond=1.0e-12,
        )
        _assert_spectrum(self, numerical, analytic)
        self.assertEqual(analytic.derivative_rank, 2)
        self.assertFalse(analytic.parameters["first_order_shell_downfolding_active"])
        self.assertLess(analytic.parameters["base_coupling_norm"], 1.0e-12)

    def test_gap_and_spd_failures_are_explicit(self):
        repeated = self.weight.copy()
        repeated[1, 1] = repeated[2, 2]
        with self.assertRaises(np.linalg.LinAlgError):
            gap_aware_projector_spectrum(
                repeated, retained_rank=2, outer_rank=4, rcond=1.0e-12
            )
        with self.assertRaises(ValueError):
            outer_resolvent_spectrum(
                self.weight, outer_rank=4, z=0.0, rcond=1.0e-12
            )
        with self.assertRaises(ValueError):
            feshbach_trace_free_log_spectrum(
                self.weight,
                retained_rank=2,
                outer_rank=4,
                z=np.nan,
                rcond=1.0e-12,
            )

    def test_all_exact_spectra_are_orthogonally_invariant(self):
        left, _ = np.linalg.qr(self.rng.normal(size=(4, 4)))
        right, _ = np.linalg.qr(self.rng.normal(size=(6, 6)))
        rotated = left @ self.weight @ right.T
        cases = (
            lambda value: gap_aware_projector_spectrum(
                value, retained_rank=2, outer_rank=4, rcond=1.0e-12
            ),
            lambda value: soft_ecs_projector_spectrum(
                value,
                lambda_center=6.5,
                temperature=10.0,
                rcond=1.0e-12,
            ),
            lambda value: outer_trace_free_log_gram_spectrum(
                value, outer_rank=4, rcond=1.0e-12
            ),
            lambda value: outer_resolvent_spectrum(
                value,
                outer_rank=4,
                z=2.0,
                trace_free=True,
                rcond=1.0e-12,
            ),
            lambda value: feshbach_trace_free_log_spectrum(
                value,
                retained_rank=2,
                outer_rank=4,
                z=0.5,
                rcond=1.0e-12,
            ),
        )
        for spectrum in cases:
            with self.subTest(function=spectrum):
                np.testing.assert_allclose(
                    spectrum(rotated).singular_amplitudes,
                    spectrum(self.weight).singular_amplitudes,
                    rtol=2.0e-12,
                    atol=2.0e-12,
                )


if __name__ == "__main__":
    unittest.main()

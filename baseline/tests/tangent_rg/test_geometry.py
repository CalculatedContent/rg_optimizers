from __future__ import annotations

import unittest

import numpy as np

from rg_baselines.tangent_rg.polar import (
    explicit_polar_jacobian,
    polar_factor,
    polar_frechet_derivative,
    polar_pullback_spectrum,
)
from rg_baselines.tangent_rg.stiefel import project_stiefel_tangent
from rg_baselines.tangent_rg.two_checkpoint import (
    aligned_rectangular_transfer,
    conditioned_square_transfer,
    finite_difference_beta,
    generalized_gram_log_rates,
    grassmann_flow_rates,
)


class PolarAndStiefelTests(unittest.TestCase):
    def test_analytic_polar_derivative_matches_central_difference_tall_and_wide(self):
        rng = np.random.default_rng(20260819)
        for shape in ((5, 3), (3, 5), (3, 3)):
            matrix = rng.normal(size=shape)
            direction = rng.normal(size=shape)
            analytic = polar_frechet_derivative(matrix, direction)
            epsilon = 1.0e-6
            numerical = (
                polar_factor(matrix + epsilon * direction)
                - polar_factor(matrix - epsilon * direction)
            ) / (2.0 * epsilon)
            relative = np.linalg.norm(numerical - analytic.derivative) / np.linalg.norm(
                analytic.derivative
            )
            self.assertLess(relative, 2.0e-6)
            self.assertLess(analytic.tangent_constraint_residual, 1.0e-10)

    def test_stiefel_projection_is_tangent_and_reconstructs(self):
        rng = np.random.default_rng(13)
        for shape in ((6, 3), (3, 6)):
            factor = polar_factor(rng.normal(size=shape))
            ambient = rng.normal(size=shape)
            record = project_stiefel_tangent(factor, ambient)
            self.assertLess(record.tangent_constraint_residual, 1.0e-10)
            self.assertLess(record.reconstruction_residual, 1.0e-12)
            np.testing.assert_allclose(record.tangent + record.normal, ambient)

    def test_explicit_polar_jacobian_matches_analytic_spectrum(self):
        matrix = np.asarray([[2.0, 0.3], [-0.4, 1.2]])
        analytic = polar_pullback_spectrum(matrix)
        numerical = explicit_polar_jacobian(matrix)
        self.assertEqual(analytic.derivative_rank, 1)
        self.assertEqual(analytic.zero_count, 3)
        self.assertEqual(numerical.numerical_rank, 1)
        self.assertAlmostEqual(
            numerical.singular_values[0], analytic.singular_amplitudes[0], places=5
        )


class TwoCheckpointTests(unittest.TestCase):
    def test_rectangular_exact_transfer_and_aligned_core_recover_known_scale(self):
        rng = np.random.default_rng(903)
        delta_s = 0.4
        rate = 0.27
        scale = np.exp(rate * delta_s)
        for shape in ((7, 3), (3, 7), (4, 4)):
            with self.subTest(shape=shape):
                initial = rng.normal(size=shape)
                final = scale * initial
                record = aligned_rectangular_transfer(
                    initial,
                    final,
                    delta_s,
                    materialize_operator=True,
                )
                self.assertTrue(record.available)
                self.assertTrue(record.operator_materialized)
                self.assertFalse(record.is_training_jacobian)
                self.assertLess(record.relative_reconstruction_residual, 1e-12)
                self.assertLess(record.core_reconstruction_residual, 1e-12)
                self.assertLess(record.unsupported_source_action_residual, 1e-12)
                np.testing.assert_allclose(record.principal_angles, 0.0, atol=2e-8)
                np.testing.assert_allclose(record.core_log_rates, rate, atol=1e-11)
                np.testing.assert_allclose(
                    record.supported_transfer_log_rates,
                    rate,
                    atol=1e-11,
                )
                if shape[0] >= shape[1]:
                    np.testing.assert_allclose(
                        record.operator @ initial,
                        final,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                else:
                    np.testing.assert_allclose(
                        initial @ record.operator,
                        final,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                self.assertEqual(
                    record.structural_zero_count,
                    max(shape) - min(shape),
                )
                self.assertIn("not_jacobian", record.operator_kind)

    def test_rectangular_transfer_spectra_are_common_gauge_invariant(self):
        rng = np.random.default_rng(904)

        def orthogonal(size):
            q, r = np.linalg.qr(rng.normal(size=(size, size)))
            return q * np.sign(np.where(np.diag(r) == 0.0, 1.0, np.diag(r)))[None, :]

        for shape in ((8, 3), (3, 8), (4, 4)):
            with self.subTest(shape=shape):
                initial = rng.normal(size=shape)
                final = initial + 0.07 * rng.normal(size=shape)
                left = orthogonal(shape[0])
                right = orthogonal(shape[1])
                original = aligned_rectangular_transfer(initial, final, 0.3)
                transformed = aligned_rectangular_transfer(
                    left @ initial @ right.T,
                    left @ final @ right.T,
                    0.3,
                )
                self.assertTrue(original.available)
                self.assertTrue(transformed.available)
                self.assertIsNone(original.operator)
                self.assertFalse(original.operator_materialized)
                np.testing.assert_allclose(
                    transformed.supported_transfer_singular_values,
                    original.supported_transfer_singular_values,
                    rtol=2e-11,
                    atol=2e-12,
                )
                np.testing.assert_allclose(
                    transformed.core_singular_values,
                    original.core_singular_values,
                    rtol=2e-11,
                    atol=2e-12,
                )
                np.testing.assert_allclose(
                    transformed.principal_angles,
                    original.principal_angles,
                    rtol=2e-9,
                    atol=2e-9,
                )

    def test_materialized_rectangular_operator_matches_direct_pseudoinverse(self):
        rng = np.random.default_rng(905)
        for shape in ((9, 4), (4, 9), (4, 4)):
            with self.subTest(shape=shape):
                initial = rng.normal(size=shape)
                final = rng.normal(size=shape)
                record = aligned_rectangular_transfer(
                    initial,
                    final,
                    0.5,
                    materialize_operator=True,
                )
                expected = (
                    final @ np.linalg.pinv(initial)
                    if shape[0] >= shape[1]
                    else np.linalg.pinv(initial) @ final
                )
                np.testing.assert_allclose(
                    record.operator,
                    expected,
                    rtol=2e-12,
                    atol=2e-12,
                )
                nonzero = np.linalg.svd(expected, compute_uv=False)
                nonzero = nonzero[: min(shape)]
                np.testing.assert_allclose(
                    record.supported_transfer_singular_values,
                    nonzero,
                    rtol=2e-12,
                    atol=2e-12,
                )
                if shape[0] == shape[1]:
                    square = conditioned_square_transfer(initial, final)
                    np.testing.assert_allclose(
                        record.operator,
                        square.operator,
                        rtol=2e-12,
                        atol=2e-12,
                    )

    def test_rectangular_transfer_rejects_rank_deficient_source(self):
        initial = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        final = initial + np.asarray([[0.1, 0.0], [0.0, 0.1], [0.1, -0.1]])
        record = aligned_rectangular_transfer(initial, final, 1.0)
        self.assertFalse(record.available)
        self.assertIsNone(record.operator)
        self.assertIn("full rectangular rank", record.unavailable_reason)

    def test_known_scale_has_radial_rate_and_zero_subspace_motion(self):
        rng = np.random.default_rng(9)
        initial = rng.normal(size=(6, 4))
        delta_s = 0.25
        rate = 0.3
        final = np.exp(rate * delta_s) * initial
        radial = generalized_gram_log_rates(initial, final, delta_s)
        np.testing.assert_allclose(radial.radial_log_rates, rate, atol=1.0e-10)
        grassmann = grassmann_flow_rates(initial, final, delta_s)
        np.testing.assert_allclose(grassmann.column.principal_angles, 0.0, atol=1e-7)
        np.testing.assert_allclose(grassmann.row.principal_angles, 0.0, atol=1e-7)

    def test_square_transfer_is_conditioned_but_never_called_training_jacobian(self):
        initial = np.diag([1.0, 2.0, 4.0])
        final = 2.0 * initial
        transfer = conditioned_square_transfer(initial, final)
        self.assertTrue(transfer.available)
        self.assertFalse(transfer.is_training_jacobian)
        np.testing.assert_allclose(transfer.operator, 2.0 * np.eye(3))
        self.assertLess(transfer.relative_reconstruction_residual, 1.0e-12)

    def test_rectangular_transfer_is_unavailable_and_secant_is_not_jacobian(self):
        initial = np.ones((3, 2))
        final = initial + 0.1
        transfer = conditioned_square_transfer(initial, final)
        beta = finite_difference_beta(initial, final, 2.0)
        self.assertFalse(transfer.available)
        self.assertFalse(beta.is_jacobian)
        np.testing.assert_allclose(beta.beta_surrogate, 0.05)


if __name__ == "__main__":
    unittest.main()

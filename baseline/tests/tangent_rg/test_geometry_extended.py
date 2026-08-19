from __future__ import annotations

import unittest

import numpy as np

from rg_baselines.tangent_rg.nulls import (
    check_invariances,
    gaussian_null,
    haar_orthogonal,
    haar_polar_null,
    haar_stiefel,
    rotation_null,
    scale_null,
)
from rg_baselines.tangent_rg.polar import (
    central_difference_jacobian,
    explicit_muon_newton_schulz_jacobian,
    muon_newton_schulz_map,
    muon_quintic_orthogonalizer,
)
from rg_baselines.tangent_rg.single_checkpoint import (
    calibrated_training_map_finite_difference,
    centered_log_singular_flow,
    gram_translation_quotient,
    normalized_gram_analytic_spectrum,
    normalized_gram_map,
)
from rg_baselines.tangent_rg.two_checkpoint import (
    check_relative_polar_orthogonal_invariance,
    relative_polar_angular_flow,
)


def _reference_muon_quintic(
    matrix: np.ndarray,
    *,
    steps: int,
    eps: float,
) -> np.ndarray:
    transposed = matrix.shape[0] > matrix.shape[1]
    work = matrix.T.copy() if transposed else matrix.copy()
    work /= max(float(np.linalg.norm(work)), eps)
    for _ in range(steps):
        gram = work @ work.T
        work = (
            3.4445 * work
            + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ work
        )
    return work.T if transposed else work


class MuonFiniteOrthogonalizerTests(unittest.TestCase):
    def test_quintic_map_matches_independent_reference_tall_and_wide(self):
        rng = np.random.default_rng(20260819)
        for shape in ((5, 3), (3, 5), (3, 3)):
            with self.subTest(shape=shape):
                matrix = rng.normal(size=shape)
                expected = _reference_muon_quintic(
                    matrix,
                    steps=5,
                    eps=1.0e-7,
                )
                actual = muon_quintic_orthogonalizer(matrix)
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

                record = muon_newton_schulz_map(matrix)
                np.testing.assert_allclose(record.output, expected)
                self.assertIn("finite_iteration", record.operator_kind)
                self.assertIn("distinct from ideal", record.map_definition)
                self.assertGreaterEqual(record.relative_error_to_ideal_polar, 0.0)

    def test_explicit_quintic_jacobian_matches_directional_difference(self):
        matrix = np.asarray([[1.2, 0.3], [-0.4, 0.9]])
        direction = np.asarray([[0.2, -0.7], [0.5, 0.1]])
        explicit = explicit_muon_newton_schulz_jacobian(matrix)
        epsilon = 1.0e-6
        directional = (
            muon_quintic_orthogonalizer(matrix + epsilon * direction)
            - muon_quintic_orthogonalizer(matrix - epsilon * direction)
        ) / (2.0 * epsilon)
        materialized = explicit.jacobian @ direction.reshape(-1)
        np.testing.assert_allclose(
            materialized.reshape(matrix.shape),
            directional,
            rtol=2.0e-5,
            atol=2.0e-7,
        )
        # Frobenius normalization makes the finite map scale-invariant locally.
        self.assertEqual(explicit.numerical_rank, 3)
        self.assertIn("finite_muon", explicit.operator_kind)
        self.assertIn("do not identify", explicit.map_definition)


class RelativePolarAngularTests(unittest.TestCase):
    def test_square_flow_has_zero_tilt_but_nonzero_twist(self):
        angle = 0.43
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        initial = np.diag([3.0, 2.0, 1.0])
        final = rotation @ np.diag([4.0, 2.5, 1.2])
        record = relative_polar_angular_flow(initial, final, delta_s=0.5)

        np.testing.assert_allclose(record.tilt_angles, 0.0, atol=1e-14)
        np.testing.assert_allclose(record.tilt_geodesic_rates, 0.0, atol=1e-14)
        self.assertEqual(record.tilt_forced_intersection_zeros, 3)
        self.assertEqual(record.tilt_zero_atoms, 3)
        self.assertGreater(float(np.max(record.twist_geodesic_rates)), 0.1)
        self.assertTrue(np.all(record.twist_projective_rates >= 0.0))
        self.assertTrue(record.twist_unique)
        self.assertIn("not_jacobian", record.operator_kind)

    def test_common_left_right_orthogonal_gauge_is_invariant(self):
        rng = np.random.default_rng(711)
        for shape in ((7, 4), (4, 7), (4, 4)):
            with self.subTest(shape=shape):
                initial = rng.normal(size=shape)
                final = initial + 0.05 * rng.normal(size=shape)
                audit = check_relative_polar_orthogonal_invariance(
                    initial,
                    final,
                    delta_s=0.25,
                    rng=91,
                )
                self.assertTrue(audit.passed)
                self.assertTrue(audit.atom_counts_match)
                self.assertLess(audit.maximum_absolute_spectral_error, 1.0e-7)


class SingleCheckpointMapTests(unittest.TestCase):
    def test_normalized_gram_analytic_spectrum_matches_explicit_jacobian(self):
        rng = np.random.default_rng(88)
        for shape in ((4, 2), (2, 4), (2, 2)):
            with self.subTest(shape=shape):
                matrix = rng.normal(size=shape)
                analytic = normalized_gram_analytic_spectrum(matrix)
                numerical = central_difference_jacobian(
                    lambda value: normalized_gram_map(
                        value,
                        side=analytic.side,
                    ).value,
                    matrix,
                    rank_rtol=1.0e-7,
                )
                self.assertEqual(numerical.numerical_rank, analytic.derivative_rank)
                np.testing.assert_allclose(
                    numerical.singular_values[: analytic.derivative_rank],
                    analytic.singular_amplitudes,
                    rtol=2.0e-5,
                    atol=2.0e-7,
                )
                self.assertEqual(
                    analytic.derivative_rank + analytic.zero_count,
                    matrix.size,
                )

    def test_gram_translation_quotient_ignores_scalar_gram_shift(self):
        rng = np.random.default_rng(102)
        scalar_shift = 0.75
        for shape in ((6, 3), (3, 6)):
            with self.subTest(shape=shape):
                matrix = rng.normal(size=shape)
                left, singular_values, right_h = np.linalg.svd(
                    matrix,
                    full_matrices=False,
                )
                shifted = (
                    left * np.sqrt(singular_values**2 + scalar_shift)[None, :]
                ) @ right_h
                original = gram_translation_quotient(matrix)
                translated = gram_translation_quotient(shifted)
                np.testing.assert_allclose(
                    translated.gram,
                    original.gram
                    + scalar_shift * np.eye(original.gram.shape[0]),
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    translated.translated_gram,
                    original.translated_gram,
                    rtol=1e-11,
                    atol=1e-11,
                )
                self.assertAlmostEqual(
                    translated.lambda_min - original.lambda_min,
                    scalar_shift,
                    places=11,
                )

    def test_centered_log_flow_separates_scale_and_anisotropy(self):
        singular_values = np.asarray([4.0, 2.0, 1.0])
        weight = np.diag(singular_values)

        scale_rate = 0.37
        scale = centered_log_singular_flow(
            weight,
            scale_rate * weight,
        )
        np.testing.assert_allclose(
            scale.instantaneous_log_singular_rates,
            scale_rate,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            scale.centered_log_singular_rates,
            0.0,
            atol=1e-14,
        )
        self.assertAlmostEqual(scale.mean_log_singular_rate, scale_rate)

        raw_rates = np.asarray([0.5, -0.2, 0.1])
        anisotropic_update = np.diag(singular_values * raw_rates)
        anisotropic = centered_log_singular_flow(weight, anisotropic_update)
        expected_centered = raw_rates - np.mean(raw_rates)
        np.testing.assert_allclose(
            anisotropic.instantaneous_log_singular_rates,
            raw_rates,
        )
        np.testing.assert_allclose(
            anisotropic.centered_log_singular_rates,
            expected_centered,
        )
        self.assertAlmostEqual(
            float(np.sum(anisotropic.centered_log_singular_rates)),
            0.0,
            places=14,
        )
        self.assertIn("requires captured Z", anisotropic.map_definition)


class CalibratedTrainingMapContractTests(unittest.TestCase):
    @staticmethod
    def _linear_map(weight, batch, optimizer_state):
        return (
            batch["left"] @ weight @ batch["right"]
            + optimizer_state["offset"]
        )

    def test_missing_batch_or_optimizer_state_is_rejected(self):
        weight = np.eye(2)
        direction = np.ones((2, 2))
        batch = {"left": np.eye(2), "right": np.eye(2)}
        state = {"offset": np.zeros((2, 2))}
        with self.assertRaisesRegex(ValueError, "batch"):
            calibrated_training_map_finite_difference(
                self._linear_map,
                weight,
                direction,
                batch=None,
                optimizer_state=state,
                map_definition="known fixed linear map",
            )
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            calibrated_training_map_finite_difference(
                self._linear_map,
                weight,
                direction,
                batch=batch,
                optimizer_state=None,
                map_definition="known fixed linear map",
            )

    def test_calibrated_finite_difference_matches_known_linear_map(self):
        weight = np.asarray([[1.0, 2.0], [-1.0, 0.5]])
        direction = np.asarray([[0.2, -0.4], [0.7, 0.1]])
        batch = {
            "left": np.asarray([[2.0, -0.1], [0.3, 1.5]]),
            "right": np.asarray([[1.0, 0.2], [-0.4, 0.8]]),
        }
        state = {"offset": np.asarray([[0.5, 0.0], [0.0, -0.2]])}
        record = calibrated_training_map_finite_difference(
            self._linear_map,
            weight,
            direction,
            batch=batch,
            optimizer_state=state,
            map_definition="F(W;b,s)=L_b W R_b + offset_s",
        )
        expected_base = self._linear_map(weight, batch, state)
        expected_derivative = batch["left"] @ direction @ batch["right"]
        np.testing.assert_allclose(record.base_output, expected_base)
        np.testing.assert_allclose(
            record.directional_derivative,
            expected_derivative,
            rtol=2e-9,
            atol=2e-10,
        )
        self.assertFalse(record.weight_only)
        self.assertTrue(record.batch_supplied)
        self.assertTrue(record.optimizer_state_supplied)
        self.assertIn("not_w_only", record.operator_kind)


class NullGeneratorTests(unittest.TestCase):
    def test_haar_orthogonal_and_stiefel_constraints(self):
        orthogonal = haar_orthogonal(6, rng=18, proper=True)
        np.testing.assert_allclose(orthogonal.T @ orthogonal, np.eye(6), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(orthogonal)), 1.0, places=12)

        stiefel = haar_stiefel(8, 3, rng=19)
        np.testing.assert_allclose(stiefel.T @ stiefel, np.eye(3), atol=1e-12)

        tall = haar_polar_null((7, 3), rng=20).sample
        wide = haar_polar_null((3, 7), rng=21).sample
        np.testing.assert_allclose(tall.T @ tall, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(wide @ wide.T, np.eye(3), atol=1e-12)

    def test_scale_rotation_and_gaussian_null_properties(self):
        rng = np.random.default_rng(22)
        matrix = rng.normal(loc=0.4, scale=1.3, size=(20, 12))

        scaled = scale_null(matrix, scale=2.5)
        np.testing.assert_allclose(scaled.sample, 2.5 * matrix)

        rotated = rotation_null(matrix, rng=23)
        np.testing.assert_allclose(
            np.linalg.svd(rotated.sample, compute_uv=False),
            np.linalg.svd(matrix, compute_uv=False),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertTrue(rotated.preserved_frobenius_norm)

        gaussian = gaussian_null(matrix, rng=24, match="mean_std")
        gaussian_repeat = gaussian_null(matrix, rng=24, match="mean_std")
        np.testing.assert_array_equal(gaussian.sample, gaussian_repeat.sample)
        self.assertEqual(gaussian.sample.shape, matrix.shape)
        self.assertAlmostEqual(gaussian.metadata["mean"], float(np.mean(matrix)))
        self.assertAlmostEqual(gaussian.metadata["std"], float(np.std(matrix)))

        gaussian_frobenius = gaussian_null(matrix, rng=25, match="frobenius")
        self.assertAlmostEqual(
            float(np.linalg.norm(gaussian_frobenius.sample)),
            float(np.linalg.norm(matrix)),
            places=12,
        )

    def test_normalized_gram_eigenvalues_pass_scale_rotation_audit(self):
        matrix = np.random.default_rng(26).normal(size=(7, 4))
        audit = check_invariances(
            lambda value: normalized_gram_map(value).eigenvalues,
            matrix,
            rng=27,
        )
        self.assertTrue(audit.all_passed)
        self.assertTrue(all(case.passed for case in audit.cases))


if __name__ == "__main__":
    unittest.main()

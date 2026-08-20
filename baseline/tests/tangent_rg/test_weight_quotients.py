"""Algebra and contract tests for weight-only Muon quotient hypotheses."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "rg_baselines"
    / "tangent_rg"
    / "weight_quotients.py"
)
SPEC = importlib.util.spec_from_file_location(
    "tangent_rg_weight_quotients_tests", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
weight_quotients = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = weight_quotients
SPEC.loader.exec_module(weight_quotients)


def matrix_with_singular_values(
    singular_values: list[float],
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    values = np.asarray(singular_values, dtype=float)
    p = int(values.size)
    m = int(rows or p)
    n = int(columns or p)
    if min(m, n) != p:
        raise ValueError("one requested dimension must equal the singular-value count")
    result = np.zeros((m, n), dtype=float)
    result[:p, :p] = np.diag(values)
    return result


def orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    factor, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return factor


def generic_frame_matrix(
    singular_values: list[float], *, rows: int, columns: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diagonal = matrix_with_singular_values(
        singular_values, rows=rows, columns=columns
    )
    return orthogonal(rng, rows) @ diagonal @ orthogonal(rng, columns).T


class WeightQuotientTests(unittest.TestCase):
    def test_midpoint_rank_matches_floor_convention(self) -> None:
        self.assertEqual(
            weight_quotients.midpoint_ecs_rank(11, 20, maximum_rank=32),
            15,
        )
        self.assertEqual(
            weight_quotients.midpoint_ecs_rank(0, 20, maximum_rank=8),
            8,
        )

    def test_midpoint_control_materializes_exact_rank_k_spectrum(self) -> None:
        weight = matrix_with_singular_values([5.0, 4.0, 3.0, 2.0], columns=7)
        result = weight_quotients.ecs_truncation(weight, ecs_rank=3)
        np.testing.assert_allclose(result.singular_values, [5.0, 4.0, 3.0])
        np.testing.assert_allclose(result.gram_eigenvalues, [25.0, 16.0, 9.0])
        self.assertEqual(result.weight.shape, weight.shape)
        self.assertEqual(result.retained_rank, 3)

    def test_canonical_section_remains_exact_rank_after_float32_cast(self) -> None:
        singular = np.linspace(12.0, 1.0, 32).tolist()
        for rows, columns in ((32, 48), (48, 32)):
            with self.subTest(shape=(rows, columns)):
                weight = generic_frame_matrix(
                    singular, rows=rows, columns=columns, seed=rows + columns
                )
                result = weight_quotients.ecs_truncation(weight, ecs_rank=11)
                cast = result.weight.astype(np.float32)
                observed = np.linalg.svd(cast, compute_uv=False)
                self.assertEqual(np.count_nonzero(observed > 0.0), 11)
                off_diagonal = cast.copy()
                diagonal = np.arange(min(rows, columns))
                off_diagonal[diagonal, diagonal] = 0.0
                self.assertEqual(np.count_nonzero(off_diagonal), 0)
                self.assertEqual(
                    result.parameters["orthogonal_orbit_gauge"],
                    "rectangular_diagonal_canonical_section",
                )

    def test_uniform_singular_translation_is_exact_for_planted_shift(self) -> None:
        latent = np.array([7.0, 5.0, 3.0])
        observed = matrix_with_singular_values((latent + 2.0).tolist())
        fraction = 2.0 / float(latent[-1] + 2.0)
        result = weight_quotients.uniform_singular_translation(
            observed, ecs_rank=3, shift_fraction=fraction
        )
        np.testing.assert_allclose(result.singular_values, latent)

    def test_gram_ridge_recovers_commuting_planted_counterterm(self) -> None:
        latent = np.array([8.0, 5.0, 3.0])
        tau = 4.0
        observed_singular = np.sqrt(latent**2 + tau)
        observed = matrix_with_singular_values(observed_singular.tolist())
        fraction = tau / float(observed_singular[-1] ** 2)
        result = weight_quotients.gram_ridge_quotient(
            observed, ecs_rank=3, tau_fraction=fraction
        )
        np.testing.assert_allclose(result.singular_values, latent)

    def test_blockwise_shift_recovers_fixed_planted_bands(self) -> None:
        latent = np.array([9.0, 7.0, 5.0, 4.0])
        shifts = np.array([1.75, 1.75, 1.0, 1.0])
        observed = matrix_with_singular_values((latent + shifts).tolist())
        # Each shift is one fifth of the observed floor in its two-mode band.
        result = weight_quotients.blockwise_singular_quotient(
            observed,
            ecs_rank=4,
            block_count=2,
            shift_fraction=0.2,
        )
        np.testing.assert_allclose(result.singular_values, latent)
        self.assertAlmostEqual(result.parameters["isotonic_correction_norm"], 0.0)

    def test_blockwise_output_is_nonincreasing_after_declared_pava(self) -> None:
        weight = matrix_with_singular_values([10.0, 9.0, 8.0, 7.9])
        result = weight_quotients.blockwise_singular_quotient(
            weight,
            ecs_rank=4,
            block_count=2,
            shift_fraction=0.9,
        )
        self.assertTrue(np.all(np.diff(result.singular_values) <= 1e-12))

    def test_feshbach_is_truncation_at_its_spectral_anchor(self) -> None:
        weight = matrix_with_singular_values([5.0, 4.0, 3.0, 2.0])
        result = weight_quotients.feshbach_downfolding_quotient(
            weight,
            weight,
            ecs_rank=2,
            regularization_ratio=1e-3,
        )
        np.testing.assert_allclose(result.singular_values, [5.0, 4.0])
        self.assertLess(result.parameters["coupling_frobenius_norm"], 1e-12)

    def test_feshbach_uses_nontrivial_anchor_frozen_coupling(self) -> None:
        anchor = matrix_with_singular_values([5.0, 4.0, 3.0, 2.0])
        angle = 0.31
        rotation = np.eye(4)
        rotation[:2, :2] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        weight = rotation @ anchor
        result = weight_quotients.feshbach_downfolding_quotient(
            weight,
            anchor,
            ecs_rank=1,
            regularization_ratio=1e-2,
        )
        self.assertTrue(np.all(np.isfinite(result.gram_eigenvalues)))
        self.assertGreater(result.parameters["coupling_frobenius_norm"], 0.0)
        self.assertLess(result.parameters["linear_solve_relative_residual"], 1e-10)

    def test_feshbach_rejects_degenerate_anchor_boundary(self) -> None:
        anchor = matrix_with_singular_values([5.0, 4.0, 4.0, 2.0])
        with self.assertRaisesRegex(
            weight_quotients.WeightQuotientUnavailable,
            "spectrally degenerate",
        ):
            weight_quotients.feshbach_downfolding_quotient(
                anchor,
                anchor,
                ecs_rank=2,
                regularization_ratio=1e-2,
                minimum_anchor_gap_ratio=1e-5,
            )

    def test_feshbach_spectrum_is_two_sided_orthogonally_invariant(self) -> None:
        anchor = generic_frame_matrix(
            [8.0, 6.0, 4.0, 2.0], rows=4, columns=7, seed=41
        )
        rng = np.random.default_rng(42)
        relative_left = orthogonal(rng, 4)
        weight = relative_left @ anchor
        base = weight_quotients.feshbach_downfolding_quotient(
            weight,
            anchor,
            ecs_rank=2,
            regularization_ratio=1e-2,
        )
        common_left = orthogonal(rng, 4)
        common_right = orthogonal(rng, 7)
        transformed = weight_quotients.feshbach_downfolding_quotient(
            common_left @ weight @ common_right.T,
            common_left @ anchor @ common_right.T,
            ecs_rank=2,
            regularization_ratio=1e-2,
        )
        np.testing.assert_allclose(
            transformed.gram_eigenvalues,
            base.gram_eigenvalues,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_feshbach_matches_direct_schur_spectrum(self) -> None:
        factor = np.array(
            [
                [5.0, 0.0, 0.0, 0.0],
                [0.4, 4.0, 0.0, 0.0],
                [0.3, 0.2, 3.0, 0.0],
                [0.1, 0.2, 0.4, 2.0],
            ]
        )
        gram = factor @ factor.T
        weight = np.linalg.cholesky(gram)
        anchor = matrix_with_singular_values([5.0, 4.0, 3.0, 2.0])
        ratio = 1e-2
        ridge = ratio * 4.0**2
        first = gram[:2, :2]
        coupling = gram[:2, 2:]
        shell = gram[2:, 2:]
        expected = first - coupling @ np.linalg.solve(
            shell + ridge * np.eye(2), coupling.T
        )
        expected_eigenvalues = np.sort(np.linalg.eigvalsh(expected))[::-1]
        result = weight_quotients.feshbach_downfolding_quotient(
            weight,
            anchor,
            ecs_rank=2,
            regularization_ratio=ratio,
        )
        np.testing.assert_allclose(
            result.gram_eigenvalues,
            expected_eigenvalues,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_rectangular_d_transform_is_transpose_equivariant(self) -> None:
        weight = matrix_with_singular_values(
            [12.0, 10.0, 8.0, 3.0, 2.0, 1.0], columns=9
        )
        first = weight_quotients.rectangular_d_transform_quotient(
            weight, ecs_rank=3, minimum_noise_modes=3
        )
        second = weight_quotients.rectangular_d_transform_quotient(
            weight.T, ecs_rank=3, minimum_noise_modes=3
        )
        np.testing.assert_allclose(first.singular_values, second.singular_values)
        self.assertTrue(np.all(first.singular_values <= np.array([12.0, 10.0, 8.0])))

    def test_rectangular_d_transform_rejects_missing_noise_bulk(self) -> None:
        weight = matrix_with_singular_values([5.0, 4.0, 3.0, 2.0])
        with self.assertRaisesRegex(
            weight_quotients.WeightQuotientUnavailable,
            "selected noise modes",
        ):
            weight_quotients.rectangular_d_transform_quotient(
                weight, ecs_rank=3, minimum_noise_modes=2
            )

    def test_rectangular_d_transform_rejects_unseparated_edge(self) -> None:
        weight = matrix_with_singular_values([5.0, 4.0, 4.0, 3.0, 2.0, 1.0])
        with self.assertRaisesRegex(
            weight_quotients.WeightQuotientUnavailable,
            "not separated",
        ):
            weight_quotients.rectangular_d_transform_quotient(
                weight,
                ecs_rank=2,
                minimum_noise_modes=4,
                minimum_relative_separation=1e-3,
            )

    def test_rectangular_d_transform_matches_square_empirical_formula(self) -> None:
        observed = np.array([6.0, 4.0, 2.0, 1.0])
        weight = matrix_with_singular_values(observed.tolist())
        noise = observed[2:]
        expected = []
        for value in observed[:2]:
            phi = np.mean(value / (value**2 - noise**2))
            expected.append(min(value, 1.0 / phi))
        result = weight_quotients.rectangular_d_transform_quotient(
            weight,
            ecs_rank=2,
            minimum_noise_modes=2,
            noise_bulk_fraction=1.0,
        )
        np.testing.assert_allclose(result.singular_values, expected)

    def test_rectangular_d_bulk_fraction_is_an_operative_scan(self) -> None:
        weight = matrix_with_singular_values(
            [12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0]
        )
        lower_half = weight_quotients.rectangular_d_transform_quotient(
            weight,
            ecs_rank=2,
            minimum_noise_modes=3,
            noise_bulk_fraction=0.5,
        )
        full_bulk = weight_quotients.rectangular_d_transform_quotient(
            weight,
            ecs_rank=2,
            minimum_noise_modes=3,
            noise_bulk_fraction=1.0,
        )
        self.assertFalse(
            np.allclose(lower_half.singular_values, full_bulk.singular_values)
        )

    def test_calibrated_mp_shrinker_is_monotone_and_nonexpansive(self) -> None:
        observed = np.array([14.0, 11.0, 8.0, 3.0, 2.0, 1.0])
        weight = matrix_with_singular_values(observed.tolist(), rows=8)
        result = weight_quotients.calibrated_mp_shrinker_quotient(
            weight, ecs_rank=3, minimum_noise_modes=3
        )
        self.assertTrue(np.all(np.diff(result.singular_values) <= 1e-12))
        self.assertTrue(np.all(result.singular_values <= observed[: result.retained_rank]))
        self.assertEqual(
            result.parameters["calibration_target"],
            "discarded_MP_like_bulk_not_WeightWatcher_fit",
        )

    def test_calibrated_mp_shrinker_matches_square_closed_form(self) -> None:
        observed = np.array([10.0, 8.0, 2.0, 1.0])
        weight = matrix_with_singular_values(observed.tolist())
        noise_unit = 1.0
        normalized = observed[:2] / noise_unit
        expected = np.sqrt((normalized**2 - 2.0) ** 2 - 4.0) / normalized
        result = weight_quotients.calibrated_mp_shrinker_quotient(
            weight,
            ecs_rank=2,
            minimum_noise_modes=2,
        )
        np.testing.assert_allclose(result.singular_values, expected)

    def test_scale_covariance(self) -> None:
        weight = matrix_with_singular_values([8.0, 6.0, 4.0, 2.0])
        base = weight_quotients.gram_ridge_quotient(
            weight, ecs_rank=3, tau_fraction=0.5
        )
        scaled = weight_quotients.gram_ridge_quotient(
            7.0 * weight, ecs_rank=3, tau_fraction=0.5
        )
        np.testing.assert_allclose(scaled.singular_values, 7.0 * base.singular_values)

    def test_dispatch_requires_feshbach_anchor(self) -> None:
        weight = matrix_with_singular_values([4.0, 3.0, 2.0])
        with self.assertRaisesRegex(ValueError, "anchor_weight"):
            weight_quotients.apply_weight_quotient(
                "feshbach_downfolding",
                weight,
                ecs_rank=2,
                parameters={"regularization_ratio": 1e-3},
            )


if __name__ == "__main__":
    unittest.main()

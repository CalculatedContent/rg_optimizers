from __future__ import annotations

import unittest

import numpy as np
import torch

from rg_baselines.rectangular_rg import (
    aligned_core_flow_operator,
    grassmann_angular_spectrum,
    rectangular_flow_spectra,
)


class RectangularRGTests(unittest.TestCase):
    @staticmethod
    def _orthonormal(
        ambient: int,
        rank: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        value = torch.randn(
            ambient,
            rank,
            generator=generator,
            dtype=torch.float64,
        )
        basis, _ = torch.linalg.qr(value, mode="reduced")
        return basis

    def test_square_case_reduces_to_exact_relative_jacobian(self) -> None:
        generator = torch.Generator().manual_seed(7)
        previous = torch.randn(8, 8, generator=generator, dtype=torch.float64)
        current = torch.randn(8, 8, generator=generator, dtype=torch.float64)
        previous = previous + 3.0 * torch.eye(8, dtype=torch.float64)
        current = current + 3.0 * torch.eye(8, dtype=torch.float64)

        operator, angles, metadata = aligned_core_flow_operator(
            previous, current
        )
        expected = torch.linalg.solve(previous.T, current.T).T

        self.assertTrue(torch.allclose(operator, expected, atol=1e-10, rtol=1e-10))
        self.assertEqual(metadata["forced_intersection_dimension"], 8)
        angular = grassmann_angular_spectrum(
            angles,
            forced_intersection_dimension=metadata[
                "forced_intersection_dimension"
            ],
        )
        self.assertEqual(angular.size, 0)

    def test_wide_same_subspace_recovers_known_core_flow(self) -> None:
        generator = torch.Generator().manual_seed(11)
        basis = self._orthonormal(11, 7, generator)
        previous_core = torch.randn(
            7, 7, generator=generator, dtype=torch.float64
        ) + 3.0 * torch.eye(7, dtype=torch.float64)
        current_core = torch.randn(
            7, 7, generator=generator, dtype=torch.float64
        ) + 3.0 * torch.eye(7, dtype=torch.float64)
        previous = previous_core @ basis.T
        current = current_core @ basis.T

        result = rectangular_flow_spectra(previous, current)
        expected = torch.linalg.solve(previous_core.T, current_core.T).T

        self.assertTrue(
            torch.allclose(
                result["core_operator"], expected, atol=1e-10, rtol=1e-10
            )
        )
        self.assertEqual(result["forced_intersection_dimension"], 3)
        self.assertEqual(result["maximum_angular_modes"], 4)
        self.assertEqual(result["angular_eigenvalues"].size, 0)

    def test_pure_subspace_motion_has_identity_aligned_core(self) -> None:
        generator = torch.Generator().manual_seed(13)
        previous_basis = self._orthonormal(11, 7, generator)
        current_basis = self._orthonormal(11, 7, generator)
        overlap = previous_basis.T @ current_basis
        left, _, right_h = torch.linalg.svd(overlap, full_matrices=False)
        alignment = right_h.T @ left.T

        core = torch.randn(
            7, 7, generator=generator, dtype=torch.float64
        ) + 3.0 * torch.eye(7, dtype=torch.float64)
        previous = core @ previous_basis.T
        current = (core @ alignment.T) @ current_basis.T

        result = rectangular_flow_spectra(previous, current)
        identity = torch.eye(7, dtype=torch.float64)

        self.assertTrue(
            torch.allclose(
                result["core_operator"], identity, atol=1e-9, rtol=1e-9
            )
        )
        self.assertEqual(result["angular_eigenvalues"].size, 4)
        self.assertTrue(np.all(result["angular_eigenvalues"] > 0.0))

    def test_rank_deficient_pair_is_rejected(self) -> None:
        previous = torch.zeros(5, 8, dtype=torch.float64)
        current = torch.randn(5, 8, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "not numerically full rank"):
            aligned_core_flow_operator(previous, current)


if __name__ == "__main__":
    unittest.main()

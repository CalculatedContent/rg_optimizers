import copy
import unittest

import torch

from full_matrix_log_rg import (
    FullMatrixLogConfig,
    FullMatrixLogRG,
    MatrixLogSupport,
    full_matrix_log_geometry,
    remove_inward_matrix_log_flow,
)


def support_from_weight(weight: torch.Tensor, rank: int | None = None) -> MatrixLogSupport:
    work = weight if weight.shape[0] >= weight.shape[1] else weight.T
    transposed = weight.shape[0] < weight.shape[1]
    _, _, vh = torch.linalg.svd(work.float(), full_matrices=False)
    m = rank or min(work.shape)
    return MatrixLogSupport(
        retained_rank=m,
        normalization_dimension=float(work.shape[1]),
        right_basis=vh[:m].T.contiguous(),
        transposed=transposed,
    )


class FullMatrixLogTests(unittest.TestCase):
    def test_isotropic_zero(self):
        weight = torch.eye(4)
        geometry = full_matrix_log_geometry(
            weight, support=support_from_weight(weight)
        )
        self.assertLess(float(geometry.potential), 1e-10)
        self.assertLess(geometry.gradient_norm_sq, 1e-10)

    def test_trace_zero_can_still_be_anisotropic(self):
        weight = torch.diag(torch.tensor([2.0, 0.5]))
        geometry = full_matrix_log_geometry(
            weight,
            retained_rank=2,
            normalization_dimension=float(torch.sum(weight.square())),
        )
        self.assertAlmostEqual(
            float(torch.sum(geometry.log_eigenvalues)), 0.0, places=5
        )
        self.assertGreater(float(geometry.potential), 0.1)

    def test_cached_basis_matches_direct_svd(self):
        torch.manual_seed(7)
        weight = torch.randn(7, 5)
        support = support_from_weight(weight, rank=4)
        cached = full_matrix_log_geometry(weight, support=support)
        direct = full_matrix_log_geometry(weight, retained_rank=4)
        self.assertTrue(
            torch.allclose(cached.log_eigenvalues, direct.log_eigenvalues, atol=1e-5)
        )
        self.assertTrue(torch.allclose(cached.gradient, direct.gradient, atol=2e-5))

    def test_gradient_matches_finite_difference(self):
        torch.manual_seed(11)
        weight = torch.randn(6, 4, dtype=torch.float64)
        direction = torch.randn_like(weight)
        support = support_from_weight(weight, rank=3)
        support.right_basis = support.right_basis.double()
        geometry = full_matrix_log_geometry(
            weight,
            support=support,
            ridge_relative=1e-10,
        )
        epsilon = 1e-6
        plus = full_matrix_log_geometry(
            weight + epsilon * direction,
            support=support,
            ridge_relative=1e-10,
        ).potential
        minus = full_matrix_log_geometry(
            weight - epsilon * direction,
            support=support,
            ridge_relative=1e-10,
        ).potential
        finite_difference = float((plus - minus) / (2.0 * epsilon))
        analytic = float(torch.sum(geometry.gradient * direction))
        self.assertAlmostEqual(finite_difference, analytic, places=5)

    def test_radial_inward_flow_cancelled(self):
        weight = torch.diag(torch.tensor([2.0, 1.0, 0.5]))
        geometry = full_matrix_log_geometry(weight, retained_rank=3)
        result = remove_inward_matrix_log_flow(
            -0.01 * geometry.gradient,
            geometry,
            mode="radial",
            projection_strength=1.0,
            max_correction_ratio=None,
        )
        self.assertLess(result.base_drift, 0.0)
        self.assertAlmostEqual(result.corrected_drift, 0.0, places=6)

    def test_modewise_removes_all_inward_drifts(self):
        weight = torch.diag(torch.tensor([3.0, 1.0, 0.25]))
        geometry = full_matrix_log_geometry(weight, retained_rank=3)
        delta = torch.diag(torch.tensor([-0.03, 0.02, 0.01]))
        result = remove_inward_matrix_log_flow(
            delta,
            geometry,
            mode="modewise",
            projection_strength=1.0,
            max_correction_ratio=None,
            gram_ridge_relative=1e-10,
        )
        self.assertGreater(result.inward_mode_count, 0)
        self.assertLess(
            result.corrected_inward_mode_norm,
            max(1e-6, 1e-4 * result.base_inward_mode_norm),
        )

    def test_outward_radial_flow_untouched(self):
        weight = torch.diag(torch.tensor([2.0, 1.0, 0.5]))
        geometry = full_matrix_log_geometry(weight, retained_rank=3)
        delta = 0.01 * geometry.gradient
        result = remove_inward_matrix_log_flow(
            delta,
            geometry,
            mode="radial",
            max_correction_ratio=None,
        )
        self.assertFalse(result.applied)
        self.assertTrue(torch.equal(result.corrected_delta, delta))

    def test_rectangular_transposed_parameter(self):
        torch.manual_seed(17)
        weight = torch.randn(3, 7)
        support = support_from_weight(weight, rank=3)
        geometry = full_matrix_log_geometry(weight, support=support)
        delta = -0.01 * geometry.gradient
        result = remove_inward_matrix_log_flow(
            delta, geometry, mode="radial", max_correction_ratio=None
        )
        self.assertEqual(result.corrected_delta.shape, weight.shape)
        self.assertAlmostEqual(result.corrected_drift, 0.0, places=5)

    def test_wrapper_cadence_and_state_roundtrip(self):
        torch.manual_seed(5)
        model = torch.nn.Linear(4, 4, bias=False)
        support = support_from_weight(model.weight, rank=4)
        base = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = FullMatrixLogRG(
            base,
            model.named_parameters(),
            FullMatrixLogConfig(
                mode="modewise",
                apply_every_steps=2,
                max_correction_ratio=None,
                parameter_names=("weight",),
            ),
        )
        optimizer.set_supports({"weight": support})
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)

        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
        self.assertEqual(optimizer.pop_step_stats(), [])

        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
        self.assertTrue(optimizer.pop_step_stats())

        saved = copy.deepcopy(optimizer.state_dict())
        model2 = torch.nn.Linear(4, 4, bias=False)
        base2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
        optimizer2 = FullMatrixLogRG(
            base2,
            model2.named_parameters(),
            optimizer.config,
        )
        optimizer2.load_state_dict(saved)
        self.assertEqual(optimizer2.global_step, 2)
        self.assertIn("weight", optimizer2.get_supports())
        self.assertTrue(
            torch.equal(
                optimizer2.get_supports()["weight"].right_basis,
                support.right_basis.float(),
            )
        )


if __name__ == "__main__":
    unittest.main()

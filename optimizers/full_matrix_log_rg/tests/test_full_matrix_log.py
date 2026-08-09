import copy
import unittest

import torch

from full_matrix_log_rg import (
    FullMatrixLogConfig,
    FullMatrixLogProjectedSGD,
    FullMatrixLogRG,
    MatrixLogSupport,
    full_matrix_log_geometry,
    matrix_log_mode_drifts,
    project_matrix_log_cone,
    remove_inward_matrix_log_flow,
    solve_active_set_nonnegative_qp,
)


def support_from_weight(weight: torch.Tensor, rank: int | None = None) -> MatrixLogSupport:
    work = weight if weight.shape[0] >= weight.shape[1] else weight.T
    transposed = weight.shape[0] < weight.shape[1]
    _, singular_values, vh = torch.linalg.svd(work.float(), full_matrices=False)
    retained = rank or min(work.shape)
    return MatrixLogSupport(
        retained_rank=retained,
        normalization_dimension=float(work.shape[1]),
        right_basis=vh[:retained].T.contiguous(),
        transposed=transposed,
        eigenvalues_ascending=singular_values.square().flip(0),
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

    def test_legacy_modewise_remains_available(self):
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

    def test_active_set_qp_kkt_conditions(self):
        gram = torch.tensor(
            [[2.0, -0.25, 0.0], [-0.25, 1.5, 0.1], [0.0, 0.1, 1.0]],
            dtype=torch.float64,
        )
        linear = torch.tensor([-1.0, 0.3, -0.2], dtype=torch.float64)
        solution = solve_active_set_nonnegative_qp(
            gram,
            linear,
            ridge_relative=0.0,
            tolerance=1e-10,
            max_iterations=64,
        )
        gradient = gram @ solution.multipliers + linear
        self.assertTrue(solution.converged)
        self.assertTrue(torch.all(solution.multipliers >= -1e-10))
        self.assertTrue(torch.all(gradient >= -1e-8))
        self.assertLess(
            float(torch.max(torch.abs(solution.multipliers * gradient))), 1e-8
        )

    def test_cone_projection_handles_mixed_inward_and_outward_modes(self):
        weight = torch.diag(
            torch.tensor([3.0, 1.2, 0.3], dtype=torch.float64)
        )
        geometry = full_matrix_log_geometry(
            weight, retained_rank=3, ridge_relative=1e-12
        )
        delta = torch.diag(
            torch.tensor([-0.05, -0.03, 0.015], dtype=torch.float64)
        )
        base_drifts = matrix_log_mode_drifts(delta, geometry)
        signs = torch.sign(geometry.log_eigenvalues)
        self.assertTrue(torch.any(signs * base_drifts < 0.0))
        self.assertTrue(torch.any(signs * base_drifts > 0.0))

        result = project_matrix_log_cone(
            delta,
            geometry,
            projection_strength=1.0,
            max_correction_ratio=None,
            gram_ridge_relative=1e-12,
            tolerance=1e-9,
            max_iterations=64,
            log_deadband=0.0,
        )
        corrected_drifts = matrix_log_mode_drifts(result.corrected_delta, geometry)
        signed_corrected = signs * corrected_drifts
        self.assertTrue(result.cone_converged)
        self.assertGreater(result.cone_active_set_size, 0)
        self.assertLessEqual(
            float(torch.max(torch.clamp(-signed_corrected, min=0.0))), 5e-7
        )
        self.assertLessEqual(result.max_signed_violation_after, 5e-7)

    def test_normalization_ablation_full_m_vs_self_consistent(self):
        weight = torch.diag(torch.tensor([5.0, 2.0, 0.2, 0.05]))
        support = support_from_weight(weight, rank=2)
        full_m = support.normalization_dimension_for("full_m")
        self_consistent = support.normalization_dimension_for(
            "self_consistent",
            method="participation_ratio",
            gamma=0.0,
        )
        self.assertEqual(full_m, 4.0)
        self.assertGreaterEqual(self_consistent, 2.0)
        self.assertLess(self_consistent, full_m)
        geometry_full = full_matrix_log_geometry(
            weight, support=support, normalization_dimension=full_m
        )
        geometry_sc = full_matrix_log_geometry(
            weight, support=support, normalization_dimension=self_consistent
        )
        self.assertFalse(
            torch.allclose(
                geometry_full.log_eigenvalues,
                geometry_sc.log_eigenvalues,
            )
        )

    def test_rectangular_transposed_parameter(self):
        torch.manual_seed(17)
        weight = torch.randn(3, 7)
        support = support_from_weight(weight, rank=3)
        geometry = full_matrix_log_geometry(weight, support=support)
        delta = -0.01 * geometry.gradient
        result = project_matrix_log_cone(
            delta,
            geometry,
            projection_strength=1.0,
            max_correction_ratio=None,
        )
        self.assertEqual(result.corrected_delta.shape, weight.shape)

    def test_projected_state_matches_unmodified_sgd_when_not_due(self):
        for nesterov in (False, True):
            torch.manual_seed(23)
            control = torch.nn.Linear(4, 3)
            wrapped = torch.nn.Linear(4, 3)
            wrapped.load_state_dict(control.state_dict())
            control_optimizer = torch.optim.SGD(
                control.parameters(),
                lr=0.03,
                momentum=0.9,
                nesterov=nesterov,
                weight_decay=0.01,
            )
            base = torch.optim.SGD(
                wrapped.parameters(),
                lr=0.03,
                momentum=0.9,
                nesterov=nesterov,
                weight_decay=0.01,
            )
            optimizer = FullMatrixLogRG(
                base,
                wrapped.named_parameters(),
                FullMatrixLogConfig(
                    momentum_projection="projected_state",
                    apply_every_steps=1000,
                    require_support=False,
                    parameter_names=("weight",),
                ),
            )
            for _ in range(5):
                x = torch.randn(8, 4)
                y = torch.randn(8, 3)
                for model, stepper in (
                    (control, control_optimizer),
                    (wrapped, optimizer),
                ):
                    stepper.zero_grad()
                    torch.nn.functional.mse_loss(model(x), y).backward()
                    stepper.step()
            for expected, actual in zip(control.parameters(), wrapped.parameters()):
                self.assertTrue(
                    torch.allclose(expected, actual, atol=5e-8, rtol=1e-6)
                )

    def test_projected_nesterov_buffer_matches_applied_step(self):
        torch.manual_seed(29)
        model = torch.nn.Linear(3, 3, bias=False, dtype=torch.float64)
        support = support_from_weight(model.weight, rank=3)
        base = torch.optim.SGD(
            model.parameters(), lr=0.05, momentum=0.9, nesterov=True
        )
        optimizer = FullMatrixLogProjectedSGD(
            base,
            model.named_parameters(),
            FullMatrixLogConfig(
                mode="cone",
                momentum_projection="projected_state",
                normalization="full_m",
                apply_every_steps=1,
                max_correction_ratio=None,
                log_deadband=0.0,
                gram_ridge_relative=1e-12,
                cone_tolerance=1e-9,
                parameter_names=("weight",),
            ),
        )
        optimizer.set_supports({"weight": support})
        x = torch.randn(12, 3, dtype=torch.float64)
        y = torch.randn(12, 3, dtype=torch.float64)
        before = model.weight.detach().clone()
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        gradient = model.weight.grad.detach().clone()
        optimizer.step()

        applied_delta = model.weight.detach() - before
        momentum_buffer = base.state[model.weight]["momentum_buffer"]
        expected_delta = -0.05 * (gradient + 0.9 * momentum_buffer)
        self.assertTrue(
            torch.allclose(applied_delta, expected_delta, atol=1e-9, rtol=1e-7)
        )
        stats = optimizer.pop_step_stats()
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["momentum_projection"], "projected_state")
        self.assertEqual(stats[0]["mode"], "cone")

    def test_wrapper_state_roundtrip_preserves_spectrum(self):
        model = torch.nn.Linear(4, 4, bias=False)
        support = support_from_weight(model.weight, rank=4)
        config = FullMatrixLogConfig(
            mode="cone",
            apply_every_steps=2,
            parameter_names=("weight",),
        )
        base = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, nesterov=True
        )
        optimizer = FullMatrixLogRG(base, model.named_parameters(), config)
        optimizer.set_supports({"weight": support})
        saved = copy.deepcopy(optimizer.state_dict())

        model2 = torch.nn.Linear(4, 4, bias=False)
        base2 = torch.optim.SGD(
            model2.parameters(), lr=0.01, momentum=0.9, nesterov=True
        )
        optimizer2 = FullMatrixLogRG(base2, model2.named_parameters(), config)
        optimizer2.load_state_dict(saved)
        restored = optimizer2.get_supports()["weight"]
        self.assertTrue(torch.equal(restored.right_basis, support.right_basis.float()))
        self.assertTrue(
            torch.equal(
                restored.eigenvalues_ascending,
                support.eigenvalues_ascending.float(),
            )
        )


if __name__ == "__main__":
    unittest.main()

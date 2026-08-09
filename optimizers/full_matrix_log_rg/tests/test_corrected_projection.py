import unittest

import torch

from full_matrix_log_rg import (
    FullMatrixLogConfig,
    FullMatrixLogRG,
    build_support,
    full_matrix_log_geometry,
    mode_drifts,
    project_matrix_log_cone,
)


class CorrectedFullMatrixLogTests(unittest.TestCase):
    def test_active_set_cone_handles_mixed_inward_and_outward_modes(self):
        weight = torch.diag(torch.tensor([3.0, 1.0, 0.25]))
        geometry = full_matrix_log_geometry(weight, retained_rank=3)
        delta = torch.tensor(
            [[0.0078, -0.0045, -0.0064],
             [-0.0241, 0.0209, -0.0127],
             [0.0115, 0.0108, -0.0078]]
        )
        signed_before = torch.sign(geometry.log_eigenvalues) * mode_drifts(delta, geometry)
        self.assertTrue((signed_before < 0).any())
        self.assertTrue((signed_before > 0).any())
        result = project_matrix_log_cone(
            delta,
            geometry,
            max_correction_ratio=None,
            tolerance=1e-7,
        )
        signed_after = torch.sign(geometry.log_eigenvalues) * mode_drifts(
            result.corrected_delta, geometry
        )
        self.assertTrue(result.converged)
        self.assertGreater(result.active_set_size, 0)
        self.assertGreaterEqual(float(signed_after.min()), -2e-5)

    def test_full_m_and_self_consistent_normalizations_are_distinct(self):
        weight = torch.diag(torch.tensor([4.0, 2.0, 0.5, 0.1]))
        support = build_support(weight, 2)
        self.assertLess(
            support.dimension("self_consistent"),
            support.dimension("full_m"),
        )
        full_m = full_matrix_log_geometry(
            weight, support=support, normalization="full_m"
        )
        self_consistent = full_matrix_log_geometry(
            weight, support=support, normalization="self_consistent"
        )
        self.assertFalse(
            torch.allclose(full_m.log_eigenvalues, self_consistent.log_eigenvalues)
        )

    def test_wrapper_projects_weight_and_nesterov_momentum_state(self):
        model = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.diag(torch.tensor([3.0, 1.0, 0.25])))
        initial = model.weight.detach().clone()
        base = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, nesterov=True
        )
        wrapper = FullMatrixLogRG(
            base,
            model.named_parameters(),
            FullMatrixLogConfig(
                mode="radial",
                momentum_projection="projected_state",
                normalization="full_m",
                apply_every_steps=1,
                max_correction_ratio=None,
                parameter_names=("weight",),
            ),
        )
        support = build_support(model.weight, 3)
        wrapper.set_supports({"weight": support})
        geometry = full_matrix_log_geometry(
            model.weight, support=support, normalization="full_m"
        )
        desired_delta = -0.01 * geometry.gradient
        model.weight.grad = -desired_delta / (0.1 * (1.0 + 0.9))
        wrapper.step()
        stats = wrapper.pop_step_stats()[0]
        self.assertTrue(stats["momentum_state_projected"])
        self.assertTrue(torch.allclose(model.weight, initial, atol=2e-5))
        self.assertIn("momentum_buffer", base.state[model.weight])


if __name__ == "__main__":
    unittest.main()

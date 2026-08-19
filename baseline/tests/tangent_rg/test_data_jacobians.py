from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from rg_baselines.tangent_rg.data_jacobians import (
    embed_grassmann_coordinate,
    grassmann_parameter_output_jacobian,
    grassmann_tangent_basis,
    input_output_jacobian_spectrum,
    per_example_quotient_loss_jacobian,
    pullback_grassmann_direction,
    quotient_generalized_gauss_newton,
    quotient_observable,
    step_quotient_jacobian_sketch,
)


class _TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 4)
        self.fc2 = torch.nn.Linear(4, 3)

    def forward(self, inputs):
        return self.fc2(torch.relu(self.fc1(inputs)))


class DataDependentJacobianTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260820)
        self.model = _TinyMLP().double().eval()
        with torch.no_grad():
            self.model.fc1.weight.zero_()
            self.model.fc1.weight[:, :4] = torch.diag(
                torch.tensor([5.0, 3.0, 2.0, 1.0], dtype=torch.float64)
            )
            self.model.fc1.bias.fill_(2.0)
            self.model.fc2.weight.copy_(
                torch.tensor(
                    [
                        [0.4, -0.2, 0.1, 0.3],
                        [-0.1, 0.5, 0.2, -0.4],
                        [0.2, 0.1, -0.3, 0.6],
                    ],
                    dtype=torch.float64,
                )
            )
            self.model.fc2.bias.zero_()
        self.inputs = torch.tensor(
            [
                [0.2, -0.1, 0.3, 0.4, 0.1],
                [-0.2, 0.4, 0.1, -0.1, 0.3],
                [0.1, 0.2, -0.2, 0.3, -0.4],
            ],
            dtype=torch.float64,
        )
        self.targets = torch.tensor([0, 2, 1], dtype=torch.long)
        self.basis = grassmann_tangent_basis(
            self.model.fc1.weight.detach(),
            retained_rank=2,
            outer_rank=4,
            rcond=1.0e-12,
        )

    def test_embedding_and_pullback_are_adjoint(self):
        coordinate = torch.randn(2, 2, dtype=torch.float64)
        direction = torch.randn(4, 5, dtype=torch.float64)
        embedded = embed_grassmann_coordinate(self.basis, coordinate)
        pulled = pullback_grassmann_direction(self.basis, direction)
        self.assertAlmostEqual(
            float(torch.sum(direction * embedded)),
            float(torch.sum(pulled * coordinate)),
            places=11,
        )
        self.assertEqual(self.basis.coordinate_dimension, 4)
        self.assertGreater(self.basis.retained_boundary_relative_gap, 0.0)

    def test_grassmann_boundary_gap_is_required(self):
        repeated = self.model.fc1.weight.detach().clone()
        repeated[1, 1] = repeated[2, 2]
        with self.assertRaises(np.linalg.LinAlgError):
            grassmann_tangent_basis(
                repeated,
                retained_rank=2,
                outer_rank=4,
                rcond=1.0e-12,
            )

    def test_parameter_output_jacobian_matches_finite_difference(self):
        record = grassmann_parameter_output_jacobian(
            self.model,
            self.inputs,
            parameter_name="fc1.weight",
            basis=self.basis,
            chunk_size=2,
        )
        direction = torch.randn(2, 2, dtype=torch.float64)
        direction /= torch.linalg.vector_norm(direction)
        epsilon = 1.0e-6

        def evaluate(sign):
            changed = copy.deepcopy(self.model)
            with torch.no_grad():
                changed.fc1.weight.add_(
                    sign
                    * epsilon
                    * embed_grassmann_coordinate(self.basis, direction)
                )
            return changed(self.inputs)

        finite = (evaluate(1.0) - evaluate(-1.0)) / (2.0 * epsilon)
        analytic = torch.einsum(
            "bcsq,sq->bc",
            record.jacobian.reshape(3, 3, 2, 2),
            direction,
        )
        torch.testing.assert_close(analytic, finite, rtol=2.0e-7, atol=2.0e-9)
        self.assertEqual(record.spectrum.input_dimension, 4)
        np.testing.assert_allclose(
            record.spectrum.jt_j_nonzero_eigenvalues,
            record.spectrum.singular_amplitudes**2,
        )

    def test_per_example_loss_pullback_and_empirical_fisher(self):
        logits = grassmann_parameter_output_jacobian(
            self.model,
            self.inputs,
            parameter_name="fc1.weight",
            basis=self.basis,
            chunk_size=2,
        )
        record = per_example_quotient_loss_jacobian(logits, self.targets)
        expected = []
        for index in range(3):
            self.model.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                self.model(self.inputs[index : index + 1]),
                self.targets[index : index + 1],
            )
            gradient = torch.autograd.grad(loss, self.model.fc1.weight)[0]
            expected.append(
                pullback_grassmann_direction(self.basis, gradient).reshape(-1)
            )
        expected_matrix = torch.stack(expected)
        torch.testing.assert_close(
            record.quotient_gradients, expected_matrix, rtol=2.0e-10, atol=2.0e-10
        )
        torch.testing.assert_close(
            record.empirical_fisher,
            expected_matrix.mT @ expected_matrix,
            rtol=2.0e-10,
            atol=2.0e-10,
        )

    def test_quotient_ggn_eigenvalues_and_energy_convention(self):
        logits = grassmann_parameter_output_jacobian(
            self.model,
            self.inputs,
            parameter_name="fc1.weight",
            basis=self.basis,
            chunk_size=2,
        )
        record = quotient_generalized_gauss_newton(logits)
        factor = record.weighted_logit_jacobian
        explicit = factor.mT @ factor
        eigenvalues = torch.linalg.eigvalsh(explicit).detach().cpu().numpy()
        eigenvalues = np.sort(eigenvalues[eigenvalues > 1.0e-12])[::-1]
        np.testing.assert_allclose(
            record.ggn_nonzero_eigenvalues, eigenvalues, rtol=2.0e-9, atol=2.0e-11
        )
        np.testing.assert_allclose(
            record.spectrum.singular_amplitudes,
            record.ggn_nonzero_eigenvalues,
        )
        np.testing.assert_allclose(
            record.spectrum.jt_j_nonzero_eigenvalues,
            record.ggn_nonzero_eigenvalues**2,
        )
        self.assertEqual(record.spectrum.parameters["loss_reduction"], "mean")

    def test_quotient_ggn_is_the_loss_gradient_jacobian_when_logits_are_linear(self):
        from torch.func import functional_call

        logit_record = grassmann_parameter_output_jacobian(
            self.model,
            self.inputs,
            parameter_name="fc1.weight",
            basis=self.basis,
            chunk_size=2,
        )
        record = quotient_generalized_gauss_newton(
            logit_record, loss_reduction="mean"
        )
        parameters = {
            name: value.detach() for name, value in self.model.named_parameters()
        }
        buffers = {
            name: value.detach() for name, value in self.model.named_buffers()
        }
        base_weight = parameters["fc1.weight"]

        def quotient_loss(coordinate_flat):
            coordinate = coordinate_flat.reshape(2, 2)
            local = dict(parameters)
            local["fc1.weight"] = base_weight + embed_grassmann_coordinate(
                self.basis, coordinate
            )
            logits = functional_call(
                self.model, (local, buffers), (self.inputs,)
            )
            return torch.nn.functional.cross_entropy(
                logits, self.targets, reduction="mean"
            )

        zero = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        exact_hessian = torch.autograd.functional.hessian(quotient_loss, zero)
        ggn = record.weighted_logit_jacobian.mT @ record.weighted_logit_jacobian
        torch.testing.assert_close(ggn, exact_hessian, rtol=2.0e-9, atol=2.0e-10)

    def test_input_output_baseline_is_block_diagonal_union(self):
        record = input_output_jacobian_spectrum(
            self.model, self.inputs, maximum_examples=2
        )
        self.assertEqual(record.input_dimension, 10)
        self.assertEqual(record.output_dimension, 6)
        self.assertGreater(record.derivative_rank, 0)
        self.assertLessEqual(record.derivative_rank, 6)

    def test_step_quotient_sketch_is_an_actual_restricted_derivative(self):
        base = self.model.fc1.weight.detach().cpu().numpy()
        directions = np.eye(4).reshape(4, 2, 2)[:3]

        def step_map(coordinate):
            tensor = torch.as_tensor(coordinate, dtype=torch.float64)
            return base + embed_grassmann_coordinate(
                self.basis, tensor
            ).detach().cpu().numpy()

        first = step_quotient_jacobian_sketch(
            step_map,
            directions,
            retained_rank=2,
            outer_rank=4,
            epsilon=1.0e-4,
            rcond=1.0e-12,
            map_definition="identity step map used only for formula validation",
        )
        second = step_quotient_jacobian_sketch(
            step_map,
            directions,
            retained_rank=2,
            outer_rank=4,
            epsilon=5.0e-5,
            rcond=1.0e-12,
            map_definition="identity step map used only for formula validation",
        )
        np.testing.assert_allclose(
            first.combined_response,
            second.combined_response,
            rtol=2.0e-5,
            atol=2.0e-7,
        )
        self.assertEqual(first.probe_count, 3)
        self.assertEqual(first.combined_spectrum.input_dimension, 3)
        self.assertEqual(
            first.combined_spectrum.parameters["full_coordinate_dimension"], 4
        )
        self.assertIn("restricted", first.map_definition)

    def test_quotient_observable_is_scale_invariant(self):
        weight = self.model.fc1.weight.detach().cpu().numpy()
        original = quotient_observable(
            weight, retained_rank=2, outer_rank=4, rcond=1.0e-12
        )
        scaled = quotient_observable(
            7.0 * weight, retained_rank=2, outer_rank=4, rcond=1.0e-12
        )
        np.testing.assert_allclose(
            scaled.centered_log_singular,
            original.centered_log_singular,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            scaled.row_projector,
            original.row_projector,
            rtol=2.0e-12,
            atol=2.0e-12,
        )


if __name__ == "__main__":
    unittest.main()

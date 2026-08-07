from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecs_trace_wall import ECSProbeLossTraceWall, MLP3, TraceWallConfig


class LinearClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc1(inputs)


class OptimizerTests(unittest.TestCase):
    def _make_case(self) -> tuple[
        LinearClassifier,
        ECSProbeLossTraceWall,
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        torch.manual_seed(401)
        model = LinearClassifier()
        base = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9)
        config = TraceWallConfig(
            parameter_names=("fc1.weight",),
            correction_interval_steps=1,
            correction_start_step=1,
            probe_batch_size=8,
            probe_batches_per_correction=1,
            correction_to_base_step_ratio=0.5,
            minimum_weight_fraction=1e-4,
            maximum_weight_fraction=5e-2,
            min_ecs_rank=2,
            svd_device="cpu",
            strict=True,
        )
        wrapper = ECSProbeLossTraceWall(model, base, config)
        inputs = torch.randn(16, 4)
        teacher = torch.tensor(
            [[1.4, -0.7, 0.3, 0.1], [-0.4, 1.1, 0.2, -0.8], [0.1, -0.2, 0.9, 1.0]]
        )
        targets = (inputs @ teacher.T).argmax(dim=1)
        batches = [(inputs[:8], targets[:8]), (inputs[8:], targets[8:])]
        return model, wrapper, batches

    def test_correction_lowers_ecs_truncated_probe_loss(self) -> None:
        model, wrapper, batches = self._make_case()
        train_inputs, train_targets = batches[0]
        wrapper.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_inputs), train_targets)
        loss.backward()
        record = wrapper.step(probe_batches=batches, loss_function=F.cross_entropy)

        self.assertTrue(record.attempted)
        self.assertTrue(record.applied)
        self.assertEqual(record.reason, "accepted")
        self.assertLess(record.probe_loss_after, record.probe_loss_before)
        self.assertLess(record.directional_derivative, 0.0)
        self.assertEqual(len(record.layers), 1)
        self.assertGreater(record.layers[0].accepted_correction_norm, 0.0)
        self.assertLess(record.layers[0].projection_identity_error, 5e-5)

    def test_standard_mlp3_wide_and_tall_layers_run_together(self) -> None:
        torch.manual_seed(409)
        model = MLP3()
        base = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        wrapper = ECSProbeLossTraceWall(
            model,
            base,
            TraceWallConfig(
                correction_interval_steps=1,
                correction_start_step=1,
                probe_batch_size=4,
                probe_batches_per_correction=2,
                strict=True,
            ),
        )
        inputs = torch.randn(8, 1, 28, 28)
        targets = torch.randint(0, 10, (8,))
        wrapper.zero_grad(set_to_none=True)
        F.cross_entropy(model(inputs[:4]), targets[:4]).backward()
        record = wrapper.step(
            probe_batches=[(inputs[:4], targets[:4]), (inputs[4:], targets[4:])],
            loss_function=F.cross_entropy,
        )
        self.assertTrue(record.applied)
        self.assertEqual(
            {layer.parameter_name for layer in record.layers},
            {"fc1.weight", "fc2.weight", "fc3.weight"},
        )
        ranks = {layer.parameter_name: layer.ecs_rank for layer in record.layers}
        self.assertLessEqual(ranks["fc1.weight"], 512)
        self.assertLessEqual(ranks["fc2.weight"], 512)
        self.assertLessEqual(ranks["fc3.weight"], 10)
        self.assertLess(record.probe_loss_after, record.probe_loss_before)

    def test_wrapper_state_round_trip_restores_step_and_rank_cache(self) -> None:
        model, wrapper, batches = self._make_case()
        wrapper.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batches[0][0]), batches[0][1])
        loss.backward()
        wrapper.step(probe_batches=batches, loss_function=F.cross_entropy)
        state = copy.deepcopy(wrapper.state_dict())

        other_model = LinearClassifier()
        other_base = torch.optim.SGD(other_model.parameters(), lr=0.04, momentum=0.9)
        other = ECSProbeLossTraceWall(other_model, other_base, wrapper.config)
        other.load_state_dict(state)
        self.assertEqual(other.global_step, wrapper.global_step)
        self.assertEqual(other.previous_ranks, wrapper.previous_ranks)


if __name__ == "__main__":
    unittest.main()

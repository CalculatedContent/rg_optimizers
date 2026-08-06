import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from wwpgd_local_delta import LocalDeltaECSConfig, LocalDeltaECSOptimizer
from wwpgd_local_delta.ecs import _select_candidate_index


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def one_step(model: nn.Module, optimizer: LocalDeltaECSOptimizer) -> None:
    x = torch.randn(16, 5)
    y = torch.randint(0, 2, (16,))
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()


class HardeningTests(unittest.TestCase):
    def test_module_name_filter_resolves_weight_parameter(self):
        model = TinyMLP()
        base = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = LocalDeltaECSOptimizer(
            base,
            model.named_parameters(),
            config=LocalDeltaECSConfig(
                parameter_name_filter=("fc1",),
                min_retained=2,
            ),
        )
        self.assertEqual(
            list(optimizer._selected_matrix_parameters()),
            ["fc1.weight"],
        )

    def test_ambiguous_suffix_filter_is_rejected(self):
        model = TinyMLP()
        base = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with self.assertRaisesRegex(ValueError, "Ambiguous parameter filter"):
            LocalDeltaECSOptimizer(
                base,
                model.named_parameters(),
                config=LocalDeltaECSConfig(
                    parameter_name_filter=("weight",),
                    min_retained=2,
                ),
            )

    def test_double_begin_epoch_is_rejected(self):
        model = TinyMLP()
        base = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = LocalDeltaECSOptimizer(
            base,
            model.named_parameters(),
            config=LocalDeltaECSConfig(min_retained=2),
        )
        optimizer.begin_epoch()
        with self.assertRaisesRegex(RuntimeError, "previous epoch snapshot"):
            optimizer.begin_epoch()

    def test_mid_epoch_state_dict_restores_epoch_snapshot(self):
        torch.manual_seed(7)
        model = TinyMLP()
        base = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = LocalDeltaECSConfig(min_retained=2)
        optimizer = LocalDeltaECSOptimizer(
            base, model.named_parameters(), config=config
        )
        optimizer.begin_epoch()
        one_step(model, optimizer)

        wrapper_state = copy.deepcopy(optimizer.state_dict())
        model_state = copy.deepcopy(model.state_dict())

        restored_model = TinyMLP()
        restored_model.load_state_dict(model_state)
        restored_base = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
        restored = LocalDeltaECSOptimizer(
            restored_base,
            restored_model.named_parameters(),
            config=config,
        )
        restored.load_state_dict(wrapper_state)
        stats = restored.apply_epoch_delta_correction(epoch=0)
        self.assertTrue(stats)
        self.assertTrue(all(row["status"] == "ok" for row in stats))

    def test_previous_rank_breaks_equal_residual_tie(self):
        ranks = [2, 3]
        residuals = [0.1, -0.1]
        selected, _, _ = _select_candidate_index(
            ranks,
            residuals,
            numeric_eps=1e-12,
            reference_rank=2,
        )
        self.assertEqual(selected, 0)
        selected, _, _ = _select_candidate_index(
            ranks,
            residuals,
            numeric_eps=1e-12,
            reference_rank=3,
        )
        self.assertEqual(selected, 1)

    def test_optional_state_synchronization_for_adamw_and_sgd(self):
        cases = (
            (
                "exp_avg",
                lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3),
            ),
            (
                "momentum_buffer",
                lambda model: torch.optim.SGD(
                    model.parameters(), lr=1e-2, momentum=0.9
                ),
            ),
        )
        for expected_key, factory in cases:
            with self.subTest(expected_key=expected_key):
                torch.manual_seed(11)
                model = TinyMLP()
                base = factory(model)
                optimizer = LocalDeltaECSOptimizer(
                    base,
                    model.named_parameters(),
                    config=LocalDeltaECSConfig(
                        min_retained=2,
                        synchronize_optimizer_state=True,
                    ),
                )
                optimizer.begin_epoch()
                one_step(model, optimizer)
                stats = optimizer.apply_epoch_delta_correction(epoch=0)
                self.assertTrue(
                    any(
                        expected_key in row["optimizer_state_adjusted_keys"]
                        for row in stats
                        if row["status"] == "ok"
                    )
                )


if __name__ == "__main__":
    unittest.main()

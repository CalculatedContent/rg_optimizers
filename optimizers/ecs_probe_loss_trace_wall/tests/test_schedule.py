from __future__ import annotations

import unittest

import torch

from ecs_trace_wall import WarmupCosineSchedule


class ScheduleTests(unittest.TestCase):
    def test_linear_warmup_and_cosine_endpoint(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        schedule = WarmupCosineSchedule(
            optimizer,
            total_steps=20,
            warmup_steps=4,
            minimum_ratio=0.05,
        )
        factors = [schedule.factor(step) for step in range(20)]
        self.assertEqual(factors[:4], [0.25, 0.5, 0.75, 1.0])
        self.assertAlmostEqual(factors[4], 1.0, places=12)
        self.assertAlmostEqual(factors[-1], 0.05, places=12)
        self.assertTrue(all(a >= b for a, b in zip(factors[4:], factors[5:])))

    def test_apply_changes_optimizer_learning_rate(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        schedule = WarmupCosineSchedule(
            optimizer,
            total_steps=10,
            warmup_steps=2,
            minimum_ratio=0.1,
        )
        self.assertAlmostEqual(schedule.apply(0), 5e-4)
        self.assertAlmostEqual(schedule.apply(1), 1e-3)
        self.assertAlmostEqual(schedule.apply(9), 1e-4)


if __name__ == "__main__":
    unittest.main()

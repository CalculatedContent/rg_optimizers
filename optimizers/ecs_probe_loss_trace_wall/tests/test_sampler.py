from __future__ import annotations

import unittest

from ecs_trace_wall import RotatingSubsetSampler


class RotatingSubsetSamplerTests(unittest.TestCase):
    def test_draws_rotate_without_replacement(self) -> None:
        sampler = RotatingSubsetSampler(10, seed=101)
        first = sampler.take(6)
        second = sampler.take(4)
        self.assertEqual(len(set(first.indices)), 6)
        self.assertEqual(len(set(second.indices)), 4)
        self.assertEqual(set(first.indices) | set(second.indices), set(range(10)))
        self.assertFalse(set(first.indices) & set(second.indices))
        self.assertEqual(sampler.cycle, 1)
        self.assertEqual(sampler.position, 0)

    def test_cross_cycle_draw_is_still_a_true_subset(self) -> None:
        sampler = RotatingSubsetSampler(10, seed=202)
        sampler.take(8)
        crossing = sampler.take(6)
        self.assertEqual(len(crossing.indices), 6)
        self.assertEqual(len(set(crossing.indices)), 6)
        self.assertEqual(crossing.cycle_start, 0)
        self.assertEqual(crossing.cycle_end, 1)

    def test_state_round_trip_reproduces_future_draws(self) -> None:
        first = RotatingSubsetSampler(17, seed=303)
        first.take(9)
        state = first.state_dict()
        expected = first.take(7)

        restored = RotatingSubsetSampler(17, seed=999)
        restored.load_state_dict(state)
        actual = restored.take(7)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()

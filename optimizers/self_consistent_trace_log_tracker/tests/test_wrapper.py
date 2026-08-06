import unittest

import torch

from rg_sc_trace_log.ecs import AdaptiveSupportState
from rg_sc_trace_log.geometry import adaptive_trace_log_geometry
from rg_sc_trace_log.wrapper import (
    SelfConsistentTraceLogConfig,
    SelfConsistentTraceLogRGWrapper,
)


class OneMatrix(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(11)
        self.weight = torch.nn.Parameter(
            torch.randn(9, 6, dtype=torch.float64) + 0.2
        )


class WrapperTests(unittest.TestCase):
    def _make_wrapper(self):
        model = OneMatrix()
        base = torch.optim.SGD(model.parameters(), lr=1.0)
        wrapper = SelfConsistentTraceLogRGWrapper(
            base,
            model.named_parameters(),
            config=SelfConsistentTraceLogConfig(
                mode="one_sided",
                correction_scale=1.0,
                max_correction_ratio=None,
                apply_every_steps=1,
                refresh_ecs_every_steps=0,
                bootstrap_without_weightwatcher=False,
                ridge_relative=0.0,
            ),
        )
        state = AdaptiveSupportState(
            ecs_rank=4,
            normalization_dimension=5.0,
            bulk_effective_count=1.0,
            trace_log_per_eval=0.0,
            status="test",
            pl_rank=4,
            working_rank=4,
        )
        wrapper.set_support_states({"weight": state}, replace=True)
        return model, wrapper

    def test_wrapper_cancels_completed_contracting_step(self) -> None:
        model, wrapper = self._make_wrapper()
        before = model.weight.detach().clone()
        geometry = adaptive_trace_log_geometry(
            before,
            fixed_ecs_rank=4,
            fixed_normalization_dimension=5.0,
            ridge_relative=0.0,
        )
        # SGD(lr=1) proposes delta=-gradient: a pure contracting direction.
        model.weight.grad = geometry.gradient.clone()
        wrapper.step()
        self.assertTrue(torch.allclose(model.weight, before, atol=1e-9, rtol=0.0))
        stats = wrapper.pop_step_stats()
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["status"], "ok")
        self.assertAlmostEqual(stats[0]["corrected_trace_log_drift"], 0.0, places=8)

    def test_wrapper_leaves_completed_expanding_step(self) -> None:
        model, wrapper = self._make_wrapper()
        before = model.weight.detach().clone()
        geometry = adaptive_trace_log_geometry(
            before,
            fixed_ecs_rank=4,
            fixed_normalization_dimension=5.0,
            ridge_relative=0.0,
        )
        model.weight.grad = -geometry.gradient.clone()
        expected = before + geometry.gradient
        wrapper.step()
        self.assertTrue(torch.allclose(model.weight, expected, atol=1e-9, rtol=0.0))
        stats = wrapper.pop_step_stats()
        self.assertEqual(stats[0]["status"], "skipped")
        self.assertEqual(stats[0]["reason"], "no contracting component")

    def test_state_dict_round_trip_preserves_support(self) -> None:
        _, first = self._make_wrapper()
        state = first.state_dict()
        _, second = self._make_wrapper()
        second.load_state_dict(state)
        restored = second.get_support_states()["weight"]
        self.assertEqual(restored.ecs_rank, 4)
        self.assertAlmostEqual(restored.normalization_dimension, 5.0)


if __name__ == "__main__":
    unittest.main()

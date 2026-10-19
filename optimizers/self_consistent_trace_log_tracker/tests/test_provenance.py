"""Provenance logging fields on SC trace-log step stats (logging only)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from rg_sc_trace_log.ecs import AdaptiveSupportState
from rg_sc_trace_log.wrapper import (
    SelfConsistentTraceLogConfig,
    SelfConsistentTraceLogRGWrapper,
)


class SCProvenanceTests(unittest.TestCase):
    def _make_wrapper(self, **config_kwargs):
        torch.manual_seed(0)
        model = nn.Linear(6, 9, bias=False).double()
        base = torch.optim.SGD(model.parameters(), lr=0.05)
        cfg = SelfConsistentTraceLogConfig(
            mode="one_sided",
            min_retained=2,
            min_ecs_size=2,
            correction_scale=1.0,
            max_correction_ratio=None,
            ridge_relative=0.0,
            bootstrap_without_weightwatcher=False,
            refresh_ecs_every_steps=0,
            **config_kwargs,
        )
        wrapper = SelfConsistentTraceLogRGWrapper(
            base,
            model.named_parameters(),
            config=cfg,
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

    def test_ok_rows_carry_provenance_and_dose(self):
        model, wrapper = self._make_wrapper(warmup_steps=0, apply_every_steps=1)
        model.zero_grad(set_to_none=True)
        (model.weight ** 2).sum().backward()
        wrapper.step()
        stats = wrapper.pop_step_stats()
        self.assertTrue(stats)
        for row in stats:
            self.assertEqual(row["actuator_id"], "self_consistent_trace_log_tracker")
            self.assertEqual(row["ecs_backend"], "self_consistent_F_m")
            self.assertEqual(
                row["dose_definition"],
                "correction_frobenius_over_base_step_delta_frobenius",
            )
            self.assertIn(
                row["status"],
                {"ok", "skipped", "geometry_failed", "geometry_skipped"},
            )
            self.assertIn("is_first_due", row)
            if row["status"] == "ok":
                self.assertIsNotNone(row["dose_value"])
                self.assertGreaterEqual(float(row["dose_value"]), 0.0)
                self.assertIs(row["is_first_apply"], True)
            elif row["status"] in {
                "skipped",
                "geometry_failed",
                "geometry_skipped",
            }:
                self.assertIsNone(row["dose_value"])
                self.assertIs(row["is_first_apply"], False)

    def test_first_apply_respects_warmup_and_cadence(self):
        model, wrapper = self._make_wrapper(warmup_steps=2, apply_every_steps=2)
        self.assertEqual(wrapper._first_due_step(), 4)
        for _ in range(4):
            model.zero_grad(set_to_none=True)
            (model.weight ** 2).sum().backward()
            wrapper.step()
        stats = wrapper.pop_step_stats()
        self.assertTrue(stats)
        for row in stats:
            self.assertTrue(row["is_first_due"])
            if row["dose_value"] is not None:
                self.assertTrue(row["is_first_apply"])
            else:
                self.assertFalse(row["is_first_apply"])
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            (model.weight ** 2).sum().backward()
            wrapper.step()
        stats2 = wrapper.pop_step_stats()
        self.assertTrue(stats2)
        self.assertTrue(all(not row["is_first_due"] for row in stats2))
        self.assertTrue(all(not row["is_first_apply"] for row in stats2))

    def test_state_dict_preserves_applied_parameters(self):
        model, wrapper = self._make_wrapper(warmup_steps=0, apply_every_steps=1)
        model.zero_grad(set_to_none=True)
        (model.weight ** 2).sum().backward()
        wrapper.step()
        stats = wrapper.pop_step_stats()
        if any(r.get("dose_value") is not None for r in stats):
            state = wrapper.state_dict()
            self.assertIn("applied_parameters", state)
            _, second = self._make_wrapper(warmup_steps=0, apply_every_steps=1)
            second.load_state_dict(state)
            self.assertTrue(second._applied_parameters)


if __name__ == "__main__":
    unittest.main()

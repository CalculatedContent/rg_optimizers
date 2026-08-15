"""Provenance logging fields on trace-log step stats (logging only)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from rg_trace_log.wrapper import TraceLogConfig, TraceLogRGWrapper


class TraceLogProvenanceTests(unittest.TestCase):
    def _make_wrapper(self, **config_kwargs):
        torch.manual_seed(0)
        model = nn.Linear(6, 9, bias=False).double()
        base = torch.optim.SGD(model.parameters(), lr=0.05)
        cfg = TraceLogConfig(
            mode="one_sided",
            normalization="raw",
            min_retained=2,
            correction_scale=1.0,
            max_correction_ratio=None,
            **config_kwargs,
        )
        wrapper = TraceLogRGWrapper(
            base,
            model.named_parameters(),
            config=cfg,
        )
        # retain most modes so geometry succeeds
        wrapper.set_supports({"weight": 4})
        return model, wrapper

    def test_ok_rows_carry_provenance_and_dose(self):
        model, wrapper = self._make_wrapper(warmup_steps=0, apply_every_steps=1)
        # force a contracting step: set grad opposite a random weight move
        model.zero_grad(set_to_none=True)
        loss = (model.weight ** 2).sum()
        loss.backward()
        wrapper.step()
        stats = wrapper.pop_step_stats()
        self.assertTrue(stats)
        for row in stats:
            self.assertEqual(row["actuator_id"], "trace_log_tracker")
            self.assertEqual(row["ecs_backend"], "midpoint_pl_detx")
            self.assertEqual(
                row["dose_definition"],
                "correction_frobenius_over_base_step_delta_frobenius",
            )
            self.assertIn(row["status"], {"ok", "skipped", "geometry_failed"})
            self.assertIn("is_first_due", row)
            if row["status"] == "ok":
                self.assertIsNotNone(row["dose_value"])
                self.assertGreaterEqual(float(row["dose_value"]), 0.0)
                self.assertIs(row["is_first_apply"], True)
            elif row["status"] in {"skipped", "geometry_failed"}:
                self.assertIsNone(row["dose_value"])
                self.assertIs(row["is_first_apply"], False)

    def test_first_apply_respects_warmup_and_cadence(self):
        model, wrapper = self._make_wrapper(warmup_steps=2, apply_every_steps=2)
        # first due step is 4 ( >2 and multiple of 2)
        self.assertEqual(wrapper._first_due_step(), 4)
        for _ in range(4):
            model.zero_grad(set_to_none=True)
            (model.weight ** 2).sum().backward()
            wrapper.step()
        # steps 1-3: no correction stats (not due); step 4: stats with first_apply
        # pop after each? step clears; only last step has stats
        stats = wrapper.pop_step_stats()
        self.assertTrue(stats)
        # first due may skip with null dose; first_apply only if applied
        for row in stats:
            self.assertTrue(row["is_first_due"])
            if row["dose_value"] is not None:
                self.assertTrue(row["is_first_apply"])
            else:
                self.assertFalse(row["is_first_apply"])
        # one more due step (6): not first due; not first apply if already applied
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            (model.weight ** 2).sum().backward()
            wrapper.step()
        stats2 = wrapper.pop_step_stats()
        self.assertTrue(stats2)
        self.assertTrue(all(not row["is_first_due"] for row in stats2))
        self.assertTrue(all(not row["is_first_apply"] for row in stats2))


if __name__ == "__main__":
    unittest.main()

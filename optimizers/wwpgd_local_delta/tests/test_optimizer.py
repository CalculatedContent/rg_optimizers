import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from wwpgd_local_delta import LocalDeltaECSConfig, LocalDeltaECSOptimizer


class WrapperTests(unittest.TestCase):
    @staticmethod
    def _model():
        return nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2))

    def _run_once(self, base_factory):
        torch.manual_seed(0)
        model = self._model()
        base = base_factory(model)
        config = LocalDeltaECSConfig(correction_fraction=0.25, min_retained=2)
        self.assertEqual(config.reference, "epoch_end")
        opt = LocalDeltaECSOptimizer(base, model.named_parameters(), config=config)
        x = torch.randn(16, 5)
        y = torch.randint(0, 2, (16,))
        before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.ndim == 2
        }
        opt.begin_epoch()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        stats = opt.apply_epoch_delta_correction(epoch=0)
        self.assertTrue(stats)
        self.assertTrue(
            all(row.get("status") == "ok" for row in stats if "parameter" in row)
        )
        after = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.ndim == 2
        }
        self.assertTrue(
            any(not torch.allclose(before[name], after[name]) for name in before)
        )
        for row in stats:
            if row.get("status") != "ok":
                continue
            self.assertLess(row["damping_error"], 1e-5)
            self.assertLess(row["pythagorean_error"], 1e-5)
            self.assertFalse(row["optimizer_state_adjusted"])
            self.assertEqual(row["actuator_id"], "wwpgd_local_delta")
            self.assertEqual(row["ecs_backend"], "self_consistent_local_geometry")
            self.assertEqual(
                row["dose_definition"],
                "removed_frobenius_over_base_epoch_delta_frobenius",
            )
            self.assertEqual(row["dose_value"], row["removed_fraction_of_base"])
            self.assertIs(row["is_first_apply"], True)

    def test_adamw_wrapper(self):
        self._run_once(
            lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3)
        )

    def test_sgd_momentum_wrapper(self):
        self._run_once(
            lambda model: torch.optim.SGD(
                model.parameters(), lr=1e-2, momentum=0.9
            )
        )

    def test_first_apply_respects_warmup_and_cadence(self):
        model = self._model()
        opt = LocalDeltaECSOptimizer(
            torch.optim.SGD(model.parameters(), lr=1e-2),
            model.named_parameters(),
            config=LocalDeltaECSConfig(
                correction_fraction=0.25,
                min_retained=2,
                warmup_epochs=2,
                apply_every_epochs=2,
            ),
        )

        for epoch in range(3):
            opt.begin_epoch()
            self.assertEqual(opt.apply_epoch_delta_correction(epoch=epoch), [])

        opt.begin_epoch()
        first_stats = opt.apply_epoch_delta_correction(epoch=3)
        self.assertTrue(first_stats)
        self.assertTrue(all(row["is_first_apply"] for row in first_stats))

        opt.begin_epoch()
        self.assertEqual(opt.apply_epoch_delta_correction(epoch=4), [])
        opt.begin_epoch()
        later_stats = opt.apply_epoch_delta_correction(epoch=5)
        self.assertTrue(later_stats)
        self.assertTrue(all(not row["is_first_apply"] for row in later_stats))

    def test_geometry_failure_rows_keep_provenance(self):
        model = self._model()
        with torch.no_grad():
            model[0].weight.fill_(float("nan"))
        opt = LocalDeltaECSOptimizer(
            torch.optim.SGD(model.parameters(), lr=1e-2),
            model.named_parameters(),
            config=LocalDeltaECSConfig(
                correction_fraction=0.25,
                min_retained=2,
                parameter_name_filter=("0.weight",),
                strict_numerics=False,
            ),
        )
        opt.begin_epoch()
        stats = opt.apply_epoch_delta_correction(epoch=0)
        self.assertEqual(len(stats), 1)
        row = stats[0]
        self.assertEqual(row["status"], "geometry_failed")
        self.assertEqual(row["actuator_id"], "wwpgd_local_delta")
        self.assertEqual(row["ecs_backend"], "self_consistent_local_geometry")
        self.assertEqual(
            row["dose_definition"],
            "removed_frobenius_over_base_epoch_delta_frobenius",
        )
        self.assertIsNone(row["dose_value"])
        self.assertIs(row["is_first_apply"], True)

    def test_skipped_rows_keep_provenance(self):
        model = self._model()
        opt = LocalDeltaECSOptimizer(
            torch.optim.SGD(model.parameters(), lr=1e-2),
            model.named_parameters(),
            config=LocalDeltaECSConfig(
                correction_fraction=0.25,
                min_retained=2,
                strict_epoch_lifecycle=False,
            ),
        )
        row = opt.apply_epoch_delta_correction(epoch=0)[0]
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(row["actuator_id"], "wwpgd_local_delta")
        self.assertEqual(row["ecs_backend"], "self_consistent_local_geometry")
        self.assertIsNone(row["dose_value"])
        self.assertIs(row["is_first_apply"], True)


if __name__ == "__main__":
    unittest.main()

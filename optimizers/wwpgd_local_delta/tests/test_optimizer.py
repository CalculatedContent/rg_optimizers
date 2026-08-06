import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from wwpgd_local_delta import LocalDeltaECSConfig, LocalDeltaECSOptimizer


class WrapperTests(unittest.TestCase):
    def _run_once(self, base_factory):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2))
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


if __name__ == "__main__":
    unittest.main()

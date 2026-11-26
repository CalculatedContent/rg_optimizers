"""Numerical invariants and exact checkpoint continuation, without downloads."""

import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from california_housing.optimizer import MagnitudeGraft, MATRIX_NAMES, make_optimizer
from california_housing.train import MLP, learning_rate, train


def _same_nested(test, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _same_nested(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            _same_nested(test, a, b)
    else:
        test.assertEqual(left, right)


class TrainingTests(unittest.TestCase):
    def test_graft_norm_direction_and_moments(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        model = MLP()
        native = copy.deepcopy(model)
        reference = copy.deepcopy(model)
        ratio = 17.0
        graft = make_optimizer(model, {"optimizer": "modified_adamw", "peak_lr": .001,
                                       "reference_lr_ratio": ratio})
        native_opt = make_optimizer(native, {"optimizer": "adamw", "peak_lr": .001})
        sgdm = torch.optim.SGD(reference.parameters(), lr=.017, momentum=.9,
                               dampening=0, nesterov=False, weight_decay=0)
        self.assertEqual(graft.matrix_names, MATRIX_NAMES)
        for index, lr in enumerate((.001, .0002, .0007, .0001)):
            rng = torch.Generator().manual_seed(501 + index)
            native.load_state_dict(model.state_dict())
            reference.load_state_dict(model.state_dict())
            for p in model.parameters():
                p.grad = torch.randn(p.shape, generator=rng)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for p, q, r in zip(model.parameters(), native.parameters(), reference.parameters()):
                q.grad = p.grad.clone()
                r.grad = p.grad.clone()
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            graft.param_groups[0]["lr"] = native_opt.param_groups[0]["lr"] = lr
            sgdm.param_groups[0]["lr"] = ratio * lr
            graft.step()
            native_opt.step()
            sgdm.step()
            for (name, p), q, r in zip(model.named_parameters(), native.parameters(),
                                       reference.parameters()):
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(graft.optimizer.state[p][key],
                                               native_opt.state[q][key], rtol=0, atol=0)
                if name not in MATRIX_NAMES:
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                    continue
                delta = (p - before[name]).double()
                native_delta = (q - before[name]).double()
                target = ratio * lr * sgdm.state[r]["momentum_buffer"].double().norm()
                expected = native_delta / native_delta.norm() * target
                torch.testing.assert_close(delta, expected, rtol=8e-4, atol=3e-8)
                torch.testing.assert_close(delta.norm(), target, rtol=2e-4, atol=2e-8)

    def test_zero_missing_and_optimizer_state_restore(self):
        torch.manual_seed(5)
        model = MLP()
        config = {"optimizer": "modified_adamw", "peak_lr": .001, "reference_lr_ratio": 10.0}
        optimizer = make_optimizer(model, config)
        first = dict(model.named_parameters())[MATRIX_NAMES[0]]
        second = dict(model.named_parameters())[MATRIX_NAMES[1]]
        old_first, old_second = first.detach().clone(), second.detach().clone()
        first.grad = torch.zeros_like(first)
        optimizer.step()
        torch.testing.assert_close(first, old_first, rtol=0, atol=0)
        torch.testing.assert_close(second, old_second, rtol=0, atol=0)
        self.assertNotIn(second, optimizer.optimizer.state)
        for p in model.parameters():
            p.grad = torch.ones_like(p) * .001
        optimizer.step()
        restored = copy.deepcopy(model)
        restored_optimizer = make_optimizer(restored, config)
        restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for p, q in zip(model.parameters(), restored.parameters()):
            p.grad = torch.ones_like(p) * .002
            q.grad = p.grad.clone()
        optimizer.step()
        restored_optimizer.step()
        _same_nested(self, model.state_dict(), restored.state_dict())
        with self.assertRaises(ValueError):
            make_optimizer(restored, {**config, "reference_lr_ratio": 11}).load_state_dict(
                optimizer.state_dict())

    def test_exact_resume_for_all_three_optimizers(self):
        rng = np.random.default_rng(12)
        arrays = {"y_scale": np.array(1.7), "y_mean": np.array(2.1)}
        for split, count in (("train", 32), ("validation", 16), ("test", 16)):
            arrays[f"x_{split}"] = rng.normal(size=(count, 8)).astype(np.float32)
            arrays[f"y_{split}"] = rng.normal(size=count).astype(np.float32)
            arrays[f"y_{split}_original"] = arrays[f"y_{split}"].astype(np.float64) * 1.7 + 2.1
        plan = {"total_updates": 16, "warmup_updates": 2, "minimum_lr_ratio": .1,
                "batch_size": 8, "eval_every": 4, "checkpoint_updates": [8, 16], "grad_clip": 1.0}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text("{}\n")
            with patch("california_housing.train.load_data", return_value=(arrays, {})):
                for name in ("sgdm", "adamw", "modified_adamw"):
                    config = {"optimizer": name, "peak_lr": .001,
                              "reference_lr_ratio": 10.0 if name == "modified_adamw" else None}
                    full = train(plan, config, root, root / name / "full", 1340, phase="calibration")
                    train(plan, config, root, root / name / "resume", 1340,
                          phase="calibration", stop_step=8)
                    continued = train(plan, config, root, root / name / "resume", 1340,
                                      phase="calibration")
                    full_checkpoint = torch.load(root / name / "full" / "resume.pt", weights_only=False)
                    resume_checkpoint = torch.load(root / name / "resume" / "resume.pt", weights_only=False)
                    _same_nested(self, full_checkpoint["model"], resume_checkpoint["model"])
                    _same_nested(self, full_checkpoint["optimizer"], resume_checkpoint["optimizer"])
                    _same_nested(self, full_checkpoint["minibatch_rng"], resume_checkpoint["minibatch_rng"])
                    self.assertEqual(full["minibatch_sha256"], continued["minibatch_sha256"])
                    self.assertEqual(full["initialization_sha256"], continued["initialization_sha256"])
                    diagnostic = continued["history"][-1]["optimizer_diagnostics"]
                    self.assertTrue(diagnostic["available"])
                    self.assertEqual(set(diagnostic["matrices"]), set(MATRIX_NAMES))
                    for item in diagnostic["matrices"].values():
                        self.assertTrue(item["finite"])
                        self.assertGreater(item["realized_update_norm"], 0)
                        if name == "sgdm":
                            # The diagnostic measures the realized float32 weight
                            # difference, including subtraction/rounding at tiny LR.
                            self.assertGreater(item["cosine_with_own_sgdm_direction"], .9999)
                    for a, b in zip(full["history"], continued["history"]):
                        self.assertEqual(a["step"], b["step"])
                        self.assertEqual(a["train"], b["train"])
                        self.assertEqual(a["validation"], b["validation"])
                        self.assertEqual(a["optimizer_diagnostics"], b["optimizer_diagnostics"])
                        self.assertIsNone(a["test"])
                        self.assertIsNone(a["spectral"])
                    with self.assertRaises(ValueError):
                        train(plan, {**config, "peak_lr": .002}, root,
                              root / name / "resume", 1340, phase="calibration")

    def test_schedule_and_parameter_count(self):
        plan = {"total_updates": 20000, "warmup_updates": 100, "minimum_lr_ratio": .1}
        self.assertAlmostEqual(learning_rate(1, .001, plan), .00001)
        self.assertAlmostEqual(learning_rate(100, .001, plan), .001)
        self.assertAlmostEqual(learning_rate(20000, .001, plan), .0001)
        self.assertGreater(learning_rate(2000, .001, plan), .0009)
        self.assertEqual(sum(p.numel() for p in MLP().parameters()), 50817)

    def test_concurrent_worker_is_rejected_before_data_or_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (root / ".run.lock").open("a") as held:
                fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch("california_housing.train.load_data") as loader:
                    with self.assertRaisesRegex(RuntimeError, "refusing duplicate training"):
                        train({}, {"optimizer": "sgdm", "peak_lr": .001}, root / "missing", root, 1340)
                    loader.assert_not_called()
                fcntl.flock(held.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    unittest.main()

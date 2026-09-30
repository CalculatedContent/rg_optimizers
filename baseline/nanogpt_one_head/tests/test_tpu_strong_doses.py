"""CPU-only checks for explicit TPU canary repetition; no XLA initialization."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
MODULE = PACKAGE / "src/rg_nanogpt_one_head/tpu_memorization_sweep.py"
spec = importlib.util.spec_from_file_location("tested_tpu_sweep", MODULE)
sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sweep
spec.loader.exec_module(sweep)


class StrongDosesTests(unittest.TestCase):
    def setUp(self):
        path = PACKAGE.parent / "experiments/nanogpt_one_head_2026_08_21_baseline/configs/harmful_memorization_0pct.yaml"
        base = yaml.safe_load(path.read_text())
        self.configs = {}
        for load, (_, fraction) in sweep.LOADS.items():
            cfg = deepcopy(base)
            cfg["memorization"]["harmful_load_fraction"] = fraction
            self.configs[load] = cfg

    def test_default_is_unchanged(self):
        before = deepcopy(self.configs)
        self.assertEqual(sweep.with_canary_doses(self.configs, None), before)
        self.assertEqual(self.configs, before)

    def test_only_tracked_doses_and_protocol_identity_change(self):
        original = deepcopy(self.configs)
        doses = [0, 64, 256, 1024, 4096]
        changed = sweep.with_canary_doses(self.configs, doses)
        self.assertEqual(self.configs, original)
        for load, cfg in changed.items():
            self.assertEqual(cfg["memorization"]["doses"], doses)
            self.assertEqual(cfg["protocol"]["tracked_canary_doses"], doses)
            self.assertNotEqual(cfg["protocol"], original[load]["protocol"])
            restored = deepcopy(cfg)
            restored["memorization"]["doses"] = original[load]["memorization"]["doses"]
            restored["protocol"] = original[load]["protocol"]
            self.assertEqual(restored, original[load])
        self.assertEqual(8 * sum(doses), 43520)
        self.assertEqual(len(sweep.build_tasks()), 25)

    def test_invalid_or_overcapacity_doses_are_rejected(self):
        bad = ([0, 1, 4, 16], [1, 4, 16, 64, 256], [0, 1, 1, 4, 16],
               [0, 4, 1, 16, 64], [0, -1, 4, 16, 64], [0, True, 4, 16, 64],
               [0, 1.0, 4, 16, 64], [0, 1, 4, 16, 100000])
        for doses in bad:
            with self.subTest(doses=doses), self.assertRaises(ValueError):
                sweep.with_canary_doses(self.configs, list(doses))

    def test_combined_background_and_tracked_capacity_is_checked(self):
        # Tracked presentations fit alone but exceed capacity at 10% background.
        with self.assertRaisesRegex(ValueError, "10pct"):
            sweep.with_canary_doses(self.configs, [0, 1, 4, 16, 75000])

    def test_cli_and_frozen_plan_preserve_the_registered_doses(self):
        args = sweep.parser().parse_args([
            "plan", "--code", "/repo", "--data-root", "/data", "--root", "/new",
            "--hardware-block", "v5e", "--canary-doses", "0,64,256,1024,4096"])
        configs = sweep.with_canary_doses(self.configs, args.canary_doses)
        plan = sweep.make_plan(args, configs, {"commit": "new-commit"}, {})
        task = sweep.Task(**plan["tasks"][0])
        self.assertEqual(sweep.task_config(plan, task)["memorization"]["doses"], args.canary_doses)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            sweep.freeze_plan(path, plan)
            baseline = sweep.make_plan(args, self.configs, {"commit": "new-commit"}, {})
            with self.assertRaises(ValueError):
                sweep.freeze_plan(path, baseline)


if __name__ == "__main__":
    unittest.main()

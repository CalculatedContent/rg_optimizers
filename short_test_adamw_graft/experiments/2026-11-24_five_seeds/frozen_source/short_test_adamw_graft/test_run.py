"""Exercise checkpointing, endpoint selection, and reporting without downloading the corpus."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import report
import run


class TestTrainingPath(unittest.TestCase):
    def test_stage_resume_and_report(self):
        # Use a small token fixture with the production model, optimizer, and WW.
        rng = np.random.default_rng(5)
        data = {key: rng.integers(0, 65, size=1000, dtype=np.uint16)
                for key in ("train", "validation", "test")}
        plan = dict(run.SCHEDULE, max_steps=3, checkpoint_every=1, eval_every=1)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            job = root / "graft" / "seed_1340"
            run.trial(root, data, "graft", 1340, plan, 2, "fixture")
            self.assertFalse((job / "result.json").exists())
            self.assertEqual(json.loads((job / "progress.json").read_text())["step"], 2)
            run.trial(root, data, "graft", 1340, plan, 3, "fixture")
            result = json.loads((job / "result.json").read_text())
            self.assertEqual([s["step"] for s in result["history"]], list(range(4)))
            best = min(result["history"][1:], key=lambda s: s["validation_loss"])
            self.assertEqual(result["validation_selected"]["step"], best["step"])
            self.assertEqual(result["endpoint"]["metrics"]["test"]["evaluated_targets"], 999)
            rows, _, _ = report.build_report(root)
            endpoint = next(r for r in rows if r["arm"] == "graft" and r["seed"] == 1340
                            and r["checkpoint"] == "endpoint")
            self.assertEqual(endpoint["valid_matrix_fits"], 18)
            self.assertGreater(endpoint["confidence_tail_weighted_alpha"], 0)


if __name__ == "__main__":
    unittest.main()

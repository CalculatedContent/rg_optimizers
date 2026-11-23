"""Exercise missing seeds, fit coverage, different checkpoint steps and paired CIs."""

import csv
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import report


def spectrum():
    matrices = []
    for i in range(18):
        fit = dict(status="success", alpha=2 + i, D=0.1, num_pl_spikes=4 if i % 2 else 1,
                   num_evals=16, raw_extra=i)
        matrices.append({"matrix": f"m{i}", "aliases": [f"m{i}"], "shape": [16, 64], "fits": [fit]})
    return {"matrices": matrices}


def state(step, loss, alpha):
    metrics = {split: {report.LOSS: loss, "next_character_accuracy": 0.4} for split in report.SPLITS}
    return dict(step=step, alpha=alpha, metrics=metrics)


def sample(arm, seed, loss):
    alpha = spectrum()
    return dict(arm=arm, seed=seed, endpoint=state(20000, loss, alpha),
                validation_selected=state(16000, loss, alpha),
                history=[dict(step=s, alpha=alpha, validation_loss=loss) for s in (12000, 16000, 20000)])


class ReportTest(unittest.TestCase):
    def test_weighted_alpha_requires_every_matrix(self):
        spec = spectrum()
        stats, raw = report.extract_spectrum(spec)
        expected = sum((2 + i) * (2 if i % 2 else 1) for i in range(18)) / 27
        self.assertAlmostEqual(stats[report.ALPHA], expected)
        self.assertEqual(raw[7]["fit_raw_extra"], 7)
        spec["matrices"][0]["fits"][0].update(D=-0.3)
        stats, raw = report.extract_spectrum(spec)
        self.assertEqual(stats["valid_matrix_fits"], 17)
        self.assertIsNone(stats[report.ALPHA])
        self.assertEqual(raw[0]["fit_D"], -0.3)
        spec["matrices"][1]["fits"] = []
        self.assertIsNone(report.extract_spectrum(spec)[1][1]["confidence_weight"])

    def test_report_on_partial_then_complete_cohort(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            def run(arm, seed, loss):
                folder = root / arm / f"seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "result.json").write_text(json.dumps(sample(arm, seed, loss)))

            run("graft", 1340, 1)
            run("sgdm", 1340, 2)
            run("sgdm", 1341, 2)
            failed = root / "adamw" / "seed_1341"
            failed.mkdir(parents=True)
            (failed / "numerical_failure.json").write_text(json.dumps(dict(step=400, error="NaN")))
            active = root / "graft" / "seed_1342"
            active.mkdir()
            (active / "progress.json").write_text(json.dumps(dict(step=12000, best_step=8000)))
            states, means, pairs = report.build_report(root)
            self.assertEqual(len(states), 3 * 5 * 2)
            self.assertTrue(any(r["status"] == "failed" and "400" in r["detail"] for r in states))
            self.assertTrue(any(r["status"] == "in_progress" and "12000" in r["detail"] for r in states))
            self.assertTrue(any(r["status"] == "missing" for r in states))
            with (root / "per_matrix.csv").open() as stream:
                raw = list(csv.DictReader(stream))
            self.assertEqual(len(raw), 3 * 3 * 18)
            self.assertEqual(raw[18]["checkpoints"], "history|validation_selected")
            self.assertEqual(raw[20]["fit_raw_extra"], "2")
            def effects(rows):
                return next(r for r in rows if r["checkpoint"] == "endpoint" and r["level"] == "mean"
                            and r["baseline"] == "sgdm" and r["metric"] == "test_" + report.LOSS)
            partial = effects(pairs)
            self.assertEqual((partial["status"], partial["n"], partial["graft_minus_baseline"]), ("partial", 1, -1))
            self.assertIsNone(partial["ci95_low"])
            for i, seed in enumerate(report.SEEDS):
                run("sgdm", seed, 2)
                run("graft", seed, 1 + i)
            states, means, pairs = report.build_report(root)
            full = effects(pairs)
            self.assertEqual((full["status"], full["n"], full["graft_minus_baseline"]), ("complete", 5, 1))
            self.assertAlmostEqual(full["ci95_high"], 1 + 2.77644511 * math.sqrt(0.5))
            self.assertEqual(sum(r["n"] for r in means if r["checkpoint"] == "endpoint" and
                                 r["metric"] == report.ALPHA), 10)


if __name__ == "__main__":
    unittest.main()

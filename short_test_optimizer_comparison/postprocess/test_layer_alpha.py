"""Synthetic regression fixtures, not measured optimizer outcomes."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import layer_alpha as analysis


def spectrum(values):
    rows = [dict(matrix=name,
                 aliases=[name, analysis.HEAD] if name == analysis.TOKEN else [name],
                 shape=[128, 128], fits=[dict(status="success", alpha=value)])
            for name, value in zip(analysis.MATRICES, values)]
    return dict(unique_matrices=18, successful_fits=len(values),
                mean_alpha=sum(values) / 18 if len(values) == 18 else None,
                below_two=sum(value < 2 for value in values), matrices=rows)


def checkpoint(values, step, loss):
    return dict(step=step, alpha=spectrum(values), metrics={
        split: dict(loss_nats_per_character=loss, perplexity=10.0,
                    next_character_accuracy=0.4, evaluated_targets=100)
        for split in ("train", "validation", "test")
    })


def result(seed, optimizer, values, selected=None):
    return dict(completed=True,
                job=dict(phase="confirmation", seed=seed,
                         recipe=dict(optimizer=optimizer, id=optimizer.lower(), peak_lr=0.01)),
                design_sha256="synthetic-design", test_disclosure="Synthetic fixture; no measured result.",
                endpoint=checkpoint(values, 20000, 2.0 if optimizer == "AdamW" else 1.5),
                validation_selected=checkpoint(values if selected is None else selected, 19000, 1.4))


class LayerComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.design = dict(confirmation_seeds=list(analysis.DEFAULT_SEEDS), architecture=dict(n_layer=4))
        (self.root / "design.json").write_text(json.dumps(self.design))

    def save(self, payload, directory=None):
        payload = copy.deepcopy(payload)
        payload["design_sha256"] = analysis.digest(self.root / "design.json")
        job = payload["job"]
        directory = directory or f"{job['recipe']['optimizer']}_seed{job['seed']}"
        path = self.root / "confirmation" / directory / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        (path.parent / "job.json").write_text(json.dumps(job))
        return path

    def paired(self, a=None, s=None, seeds=(1340,), a_selected=None, s_selected=None):
        for seed in seeds:
            self.save(result(seed, "AdamW", [2.2] * 18 if a is None else a, a_selected))
            self.save(result(seed, "SGDM", [2.2] * 18 if s is None else s, s_selected))

    @staticmethod
    def stat(report, metric, checkpoint="endpoint", matrix=None):
        return next(r for r in report["paired_statistics"] if r["checkpoint"] == checkpoint
                    and r["matrix"] == matrix and r["metric"] == metric)

    def test_equal_means_do_not_hide_below_two_layers(self):
        self.paired(a=[1.8] * 9 + [2.6] * 9, seeds=analysis.DEFAULT_SEEDS)
        report = analysis.build_report(self.root)
        self.assertTrue(report["complete"])
        count = self.stat(report, "valid_below_two_count")
        self.assertEqual(count["adamw_minus_sgdm_mean"], 9)
        self.assertEqual(self.stat(report, "valid_below_two_fraction")["adamw_minus_sgdm_mean"], 0.5)
        self.assertAlmostEqual(self.stat(report, "mean_alpha")["adamw_minus_sgdm_mean"], 0)
        self.assertEqual(self.stat(report, "mean_alpha_below_two")["adamw_minus_sgdm_mean"], 0)
        self.assertEqual(self.stat(report, "any_layer_below_two")["adamw_minus_sgdm_mean"], 1)
        self.assertEqual(count["n_pairs"], 5)
        named = self.stat(report, "alpha", matrix=analysis.TOKEN)
        self.assertEqual(named["n_pairs"], 5)  # Never 18*5 or 2*5.
        self.assertAlmostEqual(named["adamw_minus_sgdm_mean"], -0.4)
        self.assertEqual(named["test_loss_nats_per_character_adamw_minus_sgdm_mean"], 0.5)
        # Performance is retained even when one recipe performs worse; no test filter.
        self.assertEqual(len(report["paired_layers"]), 2 * 5 * 18)

    def test_checkpoint_types_are_not_pooled(self):
        self.paired(a=[1.9] * 18, s=[2.0] * 18, a_selected=[2.1] * 18, s_selected=[1.9] * 18)
        report = analysis.build_report(self.root)
        self.assertEqual(self.stat(report, "mean_alpha_below_two")["adamw_minus_sgdm_mean"], 1)
        selected = self.stat(report, "mean_alpha_below_two", "validation_selected")
        self.assertEqual(selected["adamw_minus_sgdm_mean"], -1)
        self.assertEqual(selected["n_pairs"], 1)
        self.assertIsNone(selected["ci95_low"])
        self.assertEqual(self.stat(report, "test_loss_nats_per_character", "validation_selected")["adamw_minus_sgdm_mean"], 0)

    def test_strict_threshold_and_tied_alias_once(self):
        raw = spectrum([2.0] * 18)
        alias = copy.deepcopy(raw["matrices"][0])
        alias["matrix"] = analysis.HEAD
        raw["matrices"].append(alias)
        summary, matrices, issues = analysis.normalize_spectrum(raw)
        self.assertEqual(len(matrices), 18)
        self.assertEqual(summary["valid_fit_count"], 18)
        self.assertEqual(summary["duplicate_rows_collapsed"], 1)
        self.assertEqual(summary["valid_below_two_count"], 0)
        self.assertEqual(summary["valid_below_two_fraction"], 0)
        self.assertFalse(summary["mean_alpha_below_two"])
        self.assertFalse(summary["any_layer_below_two"])
        self.assertEqual(issues, [])

    def test_conflicting_tied_duplicates_are_invalid_not_selected(self):
        raw = spectrum([2.2] * 18)
        alias = copy.deepcopy(raw["matrices"][0])
        alias.update(matrix=analysis.HEAD, fits=[dict(status="success", alpha=1.2)])
        raw["matrices"].append(alias)
        summary, matrices, _ = analysis.normalize_spectrum(raw)
        self.assertEqual(summary["valid_fit_count"], 17)
        self.assertIsNone(summary["mean_alpha"])
        self.assertEqual(matrices[0]["status"], "conflicting_duplicate_fits")

    def test_invalid_fits_and_valid_fit_denominators(self):
        invalid = [[], [{"status": "failed", "alpha": 1.0}],
                   [{"status": "success", "alpha": value} for value in [1.5, 1.6]]]
        invalid += [[dict(status="success", alpha=value)] for value in (None, "1.8", True, float("nan"), float("inf"))]
        for fits in invalid:
            with self.subTest(fits=fits):
                raw = spectrum([1.8] * 9 + [2.6] * 9)
                raw["matrices"][0]["fits"] = fits
                summary, _, _ = analysis.normalize_spectrum(raw)
                self.assertEqual(summary["valid_fit_count"], 17)
                self.assertEqual(summary["valid_below_two_count"], 8)
                self.assertAlmostEqual(summary["valid_below_two_fraction"], 8 / 17)
                self.assertIsNone(summary["mean_alpha"])
                self.assertIsNone(summary["mean_alpha_below_two"])
                self.assertTrue(summary["any_layer_below_two"])

    def test_absent_fits_are_not_evidence_of_no_below_two_layers(self):
        raw = spectrum([2.2] * 17)
        summary, _, _ = analysis.normalize_spectrum(raw)
        self.assertEqual(summary["missing_fit_count"], 1)
        self.assertIsNone(summary["any_layer_below_two"])
        self.assertFalse(summary["any_valid_layer_below_two"])
        self.assertEqual(summary["missing_matrices"], [analysis.MATRICES[-1]])
        empty, _, _ = analysis.normalize_spectrum(None)
        self.assertEqual(empty["valid_fit_count"], 0)
        self.assertIsNone(empty["valid_below_two_fraction"])
        self.assertIsNone(empty["any_valid_layer_below_two"])
        self.assertIsNone(empty["mean_valid_alpha"])

    def test_partial_fit_pairing_is_metric_specific(self):
        self.paired(seeds=(1340, 1341))
        path = self.root / "confirmation" / "AdamW_seed1341" / "result.json"
        payload = analysis.read_json(path)
        payload["endpoint"]["alpha"]["matrices"].pop(0)
        payload["endpoint"]["metrics"]["test"]["loss_nats_per_character"] = None
        path.write_text(json.dumps(payload))
        report = analysis.build_report(self.root)
        self.assertEqual(self.stat(report, "mean_alpha")["paired_seeds"], [1340])
        self.assertEqual(self.stat(report, "alpha", matrix=analysis.TOKEN)["paired_seeds"], [1340])
        other = self.stat(report, "alpha", matrix=analysis.MATRICES[1])
        self.assertEqual(other["paired_seeds"], [1340, 1341])
        self.assertEqual(other["test_loss_nats_per_character_adamw_minus_sgdm_seeds"], [1340])
        self.assertEqual(self.stat(report, "valid_below_two_fraction")["n_pairs"], 2)
        self.assertFalse(report["complete"])

    def test_missing_pairs_and_numerical_failures_remain_visible(self):
        self.paired()
        path = self.save(result(1341, "AdamW", [2.2] * 18))
        path.unlink()
        (path.parent / "numerical_failure.json").write_text('{"error":"Nonfinite loss"}')
        self.save(result(1342, "AdamW", [2.2] * 18))
        report = analysis.build_report(self.root)
        stat = self.stat(report, "mean_alpha")
        self.assertEqual(stat["paired_seeds"], [1340])
        self.assertEqual(stat["missing_pair_seeds"], [1341, 1342, 1343, 1344])
        failed = next(r for r in report["audit"]["runs"] if r["seed"] == 1341 and r["optimizer"] == "AdamW")
        self.assertEqual(failed["status"], "numerical_failure")

    def test_duplicate_seed_files_are_excluded_not_extra_replicates(self):
        self.paired(seeds=(1340, 1341))
        self.save(result(1340, "AdamW", [2.2] * 18), "copy_of_seed1340")
        report = analysis.build_report(self.root)
        self.assertEqual(self.stat(report, "mean_alpha")["paired_seeds"], [1341])
        duplicates = [e for e in report["audit"]["input_files"] if "duplicate_optimizer_seed" in e["issues"]]
        self.assertEqual(len(duplicates), 2)
        self.assertTrue(all(e["status"] == "excluded" for e in duplicates))

    def test_unexpected_seeds_and_calibration_are_excluded(self):
        self.paired()
        calibration = result(1341, "AdamW", [2.2] * 18)
        calibration["job"]["phase"] = "calibration"
        self.save(calibration)
        self.save(result(2027, "SGDM", [2.2] * 18))
        report = analysis.build_report(self.root)
        self.assertEqual(len(report["models"]), 4)
        self.assertFalse(report["complete"])

    def test_design_mismatch_and_mixed_recipes_are_exposed(self):
        self.paired(seeds=(1340, 1341))
        path = self.root / "confirmation" / "AdamW_seed1341" / "result.json"
        payload = analysis.read_json(path)
        payload["design_sha256"] = "different"
        path.write_text(json.dumps(payload))
        report = analysis.build_report(self.root)
        self.assertEqual(self.stat(report, "mean_alpha")["paired_seeds"], [1340])
        payload["design_sha256"] = analysis.digest(self.root / "design.json")
        payload["job"]["recipe"]["peak_lr"] = 9.0
        path.write_text(json.dumps(payload))
        report = analysis.build_report(self.root)
        self.assertEqual(self.stat(report, "mean_alpha")["n_pairs"], 0)

    def test_no_manifest_checks_consistent_hashes_and_reports_seed_default(self):
        self.paired()
        (self.root / "design.json").unlink()
        report = analysis.build_report(self.root)
        self.assertEqual(report["audit"]["seed_source"], "frozen_five_seed_default")
        self.assertEqual(self.stat(report, "mean_alpha")["n_pairs"], 1)
        path = self.root / "confirmation" / "SGDM_seed1340" / "result.json"
        payload = analysis.read_json(path)
        payload["design_sha256"] = "different"
        path.write_text(json.dumps(payload))
        report = analysis.build_report(self.root)
        self.assertEqual(self.stat(report, "mean_alpha")["n_pairs"], 0)
        self.assertTrue(all("mixed_design_hashes" in e["issues"] for e in report["audit"]["input_files"]))

    def test_missing_checkpoint_is_not_filled_from_other_checkpoint(self):
        self.paired()
        path = self.root / "confirmation" / "AdamW_seed1340" / "result.json"
        payload = analysis.read_json(path)
        del payload["validation_selected"]
        path.write_text(json.dumps(payload))
        report = analysis.build_report(self.root)
        self.assertEqual(self.stat(report, "mean_alpha")["n_pairs"], 1)
        self.assertEqual(self.stat(report, "mean_alpha", "validation_selected")["n_pairs"], 0)

    def test_student_t_uncertainty_uses_pair_differences(self):
        stat = analysis.paired_statistic({s: (s + 10, 10) for s in range(1, 6)}, list(range(1, 6)))
        self.assertEqual(stat["n_pairs"], 5)
        self.assertEqual(stat["adamw_minus_sgdm_mean"], 3)
        self.assertAlmostEqual(stat["ci95_low"], 1.03675684, places=7)
        self.assertAlmostEqual(stat["ci95_high"], 4.96324316, places=7)
        self.assertIsNone(analysis.paired_statistic({}, [1])["ci95_low"])
        self.assertIsNone(analysis.paired_statistic({1: (2, 1)}, [1])["ci95_low"])

    def test_cli_writes_auditable_reports_and_preserves_inputs(self):
        self.paired(seeds=analysis.DEFAULT_SEEDS)
        before = {str(p): analysis.digest(p) for p in self.root.rglob("*.json")}
        output = self.root / "analysis"
        command = [sys.executable, str(Path(analysis.__file__)), "--study", str(self.root),
                   "--output", str(output), "--require-complete"]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(json.loads(process.stdout)["complete"])
        self.assertEqual(before, {path: analysis.digest(path) for path in before})
        with (output / "paired_layers.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 180)
        text = (output / "report.md").read_text()
        self.assertIn("## endpoint", text)
        self.assertIn("## validation_selected", text)
        self.assertIn("Synthetic fixture", text)
        (self.root / "confirmation" / "SGDM_seed1344" / "result.json").unlink()
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertFalse(analysis.read_json(output / "layer_comparison.json")["complete"])

    def test_empty_and_malformed_inputs_produce_missing_pair_audit(self):
        report = analysis.build_report(self.root)
        self.assertFalse(report["complete"])
        self.assertEqual(self.stat(report, "mean_alpha")["n_pairs"], 0)
        path = self.root / "confirmation" / "broken" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text("{bad JSON}")
        report = analysis.build_report(self.root)
        self.assertIn("unreadable_result", report["audit"]["input_files"][0]["issues"][0])
        analysis.write_report(report, self.root / "analysis")
        with self.assertRaises(ValueError):
            analysis.write_report(report, self.root / "confirmation" / "overwrite")

    def test_frozen_source_manifest_is_unchanged(self):
        repository = Path(analysis.__file__).resolve().parents[2]
        frozen = analysis.read_json(repository / "short_test_optimizer_comparison" / "frozen_design_20261021.json")
        comparison = repository / "short_test_optimizer_comparison"
        baseline = repository / "short_test"
        actual = {"comparison": {p.name: analysis.digest(p) for p in comparison.glob("*.py")},
                  "baseline": {str(p.relative_to(baseline)): analysis.digest(p) for p in baseline.rglob("*.py")}}
        self.assertEqual(actual, frozen["source_sha256"])


if __name__ == "__main__":
    unittest.main()

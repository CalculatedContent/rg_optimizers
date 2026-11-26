"""Statistical/provenance gates, not optimizer performance predictions."""

import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from california_housing import analyze, spectral


def matrix_rows(offset=0.0):
    return [{"matrix": name, "shape": [128, 128],
             "fit": {"status": "success", "alpha": 1.5 + index * 0.5 + offset,
                     "D": 0.02 + index * 0.01, "num_pl_spikes": 16 * (index + 1),
                     "num_evals": 128, "alpha_weighted": 999}}
            for index, name in enumerate(spectral.ELIGIBLE_MATRICES)]


def record(seed, optimizer):
    matrices = matrix_rows((seed - 1340) * 0.01)
    spec = {"matrices": matrices, **spectral.aggregate_matrices(matrices)}
    metric = {"rmse": 0.5, "mse": 0.25, "mae": 0.4}
    return {"seed": seed, "optimizer": optimizer, "phase": "confirmation", "status": "complete",
            "completed_updates": 20000, "plan_sha256": "frozen-plan", "data_manifest_sha256": "fixed-data",
            "initialization_sha256": f"init-{seed}", "minibatch_sha256": f"batches-{seed}",
            "config": {"optimizer": optimizer, "batch_size": 64, "hidden_dim": 128},
            "history": [{"step": step, "train": metric.copy(), "validation": metric.copy(),
                         "test": metric.copy(),
                         "spectral": copy.deepcopy(spec), "elapsed_seconds": step / 100}
                        for step in analyze.STEPS]}


def write_cohort(root, local_reproduction=False):
    root = Path(root)
    plan = {"confirmation_seeds": list(analyze.SEEDS), "development_seed": 1339}
    (root / "plan.json").write_text(json.dumps(plan))
    data = root / "calibration" / "data_manifest.json"
    data.parent.mkdir(parents=True)
    data.write_text(json.dumps({"fixture": "synthetic unit test, not an experiment"}))
    source = {name: f"test-sha-{name}" for name in ("data.py", "train.py", "optimizer.py", "spectral.py")}
    manifest = root / "SOURCE_SHA256.json"
    import hashlib
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"schema_version": 1, "files": {
        **{f"california_housing/{key}": value for key, value in source.items()},
        "california_housing/plan.json": digest(root / "plan.json")}}))
    frozen = {"calibration_complete": True, "github_run_id": "test-run", "launch_commit": "test-launch",
              "confirmation_seeds": list(analyze.SEEDS), "development_seed": 1339,
              "selected": {arm: record(1340, arm)["config"] for arm in analyze.ARMS},
              "data_manifest_sha256": digest(data), "source_manifest_sha256": digest(manifest),
              "plan_file_sha256": digest(root / "plan.json")}
    if local_reproduction:
        frozen.update(github_run_id=None, launch_commit=None)
    frozen_path = root / "frozen_selection.json"
    frozen_path.write_text(json.dumps(frozen))
    execution = {"github": {"GITHUB_RUN_ID": "test-run", "GITHUB_SHA": "test-launch"},
                 "precision": "float32", "torch_num_threads": 1, "torch_num_interop_threads": 1,
                 "deterministic_algorithms": True,
                 "thread_environment": {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                 "source_sha256": source, "source_bundle_sha256": analyze._canonical_sha256(source),
                 "python": "3.12.0", "packages": {key: "fixture" for key in ("torch", "numpy", "scipy", "scikit-learn")}}
    execution["packages"].update(weightwatcher="0.7.7", powerlaw="2.0.0")
    if local_reproduction:
        execution["github"].update(GITHUB_RUN_ID=None, GITHUB_SHA=None)
    for seed in analyze.SEEDS:
        for optimizer in analyze.ARMS:
            path = Path(root) / "runs" / f"seed_{seed}" / optimizer / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            value = record(seed, optimizer)
            value.update(plan_sha256=analyze._canonical_sha256(plan), data_manifest_sha256=digest(data),
                         frozen_selection_sha256=digest(frozen_path), source_manifest_sha256=digest(manifest),
                         source_bundle_sha256=execution["source_bundle_sha256"], execution=copy.deepcopy(execution))
            path.write_text(json.dumps(value), encoding="utf-8")
            # Opaque nonempty fixtures: checkpoint loading is a trainer test.
            for name in ("checkpoint_step_2000.pt", "checkpoint_step_20000.pt", "resume.pt"):
                (path.parent / name).write_bytes(b"unit-test fixture")


class SpectralTests(unittest.TestCase):
    def test_ordinary_alpha_and_confidence_tail_weights(self):
        matrices = matrix_rows()
        result = spectral.aggregate_matrices(matrices)
        weights = [math.sqrt(16 * (i + 1)) / (0.04 + i * 0.01) for i in range(3)]
        expected = sum((1.5 + i * 0.5) * weights[i] for i in range(3)) / sum(weights)
        self.assertTrue(result["complete"])
        self.assertAlmostEqual(result["confidence_tail_alpha"], expected)
        self.assertAlmostEqual(result["layer_alpha_sd"], 0.5)
        self.assertNotEqual(result["confidence_tail_alpha"], 999)

    def test_invalid_or_duplicate_matrix_prevents_survivor_average(self):
        for corruption in ("missing", "duplicate", "failure", "invalid_tail"):
            matrices = matrix_rows()
            if corruption == "missing":
                matrices.pop()
            elif corruption == "duplicate":
                matrices[2]["matrix"] = matrices[1]["matrix"]
            elif corruption == "failure":
                matrices[1]["fit"]["status"] = "failed"
            else:
                matrices[1]["fit"]["num_pl_spikes"] = 129
            summary = spectral.aggregate_matrices(matrices)
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["confidence_tail_alpha"])

    def test_rng_preserved_and_input_readout_excluded(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                dimensions = (8, 128, 128, 128, 128, 1)
                for i in range(5):
                    setattr(self, f"fc{i+1}", torch.nn.Linear(dimensions[i], dimensions[i+1]))

        class Frame:
            def to_json(self, orient):
                return json.dumps([matrix_rows()[0]["fit"]])

        class Watcher:
            def __init__(self, _):
                pass

            def analyze(self):
                torch.rand(3)
                np.random.random(3)
                random.random()
                return Frame()

        model = Model()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        torch_state, numpy_state, python_state = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        with patch("weightwatcher.WeightWatcher", Watcher):
            result = spectral.analyze_model(model)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in state.items()))
        self.assertEqual({item["matrix"] for item in result["excluded_matrices"]}, {"fc1.weight", "fc5.weight"})
        self.assertEqual(result["n_valid"], 3)
        for item in result["matrices"]:
            self.assertEqual(len(item["eigenvalues"]), 128)
            expected = np.linalg.svd(state[item["matrix"]].double().numpy(), compute_uv=False) ** 2
            np.testing.assert_allclose(item["eigenvalues"], np.sort(expected), rtol=1e-12)


class CohortAnalysisTests(unittest.TestCase):
    def test_student_t_uses_five_seeds_and_sample_sd(self):
        stats = analyze.seed_statistics([1, 2, 3, 4, 5])
        self.assertEqual(stats["n"], 5)
        self.assertEqual(stats["degrees_of_freedom"], 4)
        self.assertEqual(stats["mean"], 3)
        self.assertAlmostEqual(stats["sample_sd"], math.sqrt(2.5))
        self.assertAlmostEqual(stats["ci95_half_width"], 2.7764451051977987 / math.sqrt(2))
        self.assertIsNone(analyze.seed_statistics([1])["ci95_half_width"])
        self.assertFalse(analyze.seed_statistics([1, 2, 3, 4])["complete_five_seeds"])

    def test_complete_cohort_and_paired_effects(self):
        with tempfile.TemporaryDirectory() as root:
            write_cohort(root)
            records, rows, errors, warnings, _ = analyze.audit_records(root)
            self.assertEqual(len(records), 15)
            self.assertEqual(len(rows), 15 * 41)
            self.assertEqual(errors, [])
            self.assertEqual(warnings, [])
            _, paired = analyze.summarize_rows(rows)
            alpha = next(row for row in paired if row["metric"] == "confidence_tail_alpha")
            self.assertEqual(alpha["n"], 5)
            self.assertEqual(alpha["mean"], 0)

    def test_missing_seed_does_not_claim_complete(self):
        with tempfile.TemporaryDirectory() as root:
            write_cohort(root)
            (Path(root) / "runs/seed_1344/adamw/result.json").unlink()
            with patch.object(analyze, "_plot"):
                summary = analyze.write_outputs(root, allow_incomplete=True)
            self.assertFalse(summary["complete_five_seeds"])
            self.assertEqual(summary["n_completed_runs"], 14)
            with patch.object(analyze, "_plot"), self.assertRaises(ValueError):
                analyze.write_outputs(root)

    def test_mismatched_initialization_and_tampered_alpha_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_cohort(root)
            path = Path(root) / "runs/seed_1340/modified_adamw/result.json"
            altered = json.loads(path.read_text())
            altered["initialization_sha256"] = "different"
            altered["history"][4]["spectral"]["confidence_tail_alpha"] = 2.0
            path.write_text(json.dumps(altered))
            _, _, errors, _, _ = analyze.audit_records(root)
            self.assertTrue(any("initialization fingerprints differ" in error for error in errors))
            self.assertTrue(any("stored alpha disagrees" in error for error in errors))

    def test_primary_provenance_and_retained_checkpoints_are_required(self):
        for failure in ("test_metrics", "checkpoint", "fingerprint", "github_run", "frozen_recipe", "source"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                write_cohort(root)
                path = Path(root) / "runs/seed_1340/modified_adamw/result.json"
                value = json.loads(path.read_text())
                if failure == "test_metrics":
                    value["history"][4]["test"] = None
                elif failure == "checkpoint":
                    (path.parent / "checkpoint_step_2000.pt").unlink()
                elif failure == "fingerprint":
                    value.pop("minibatch_sha256")
                elif failure == "github_run":
                    value["execution"]["github"]["GITHUB_RUN_ID"] = "local-or-different-run"
                elif failure == "frozen_recipe":
                    value["config"]["batch_size"] = 999
                else:
                    value["source_manifest_sha256"] = "different-source"
                path.write_text(json.dumps(value))
                _, _, errors, _, _ = analyze.audit_records(root)
                self.assertTrue(errors, f"audit accepted missing/invalid {failure}")

    def test_local_reproduction_is_explicit_and_cannot_replace_primary(self):
        with tempfile.TemporaryDirectory() as root:
            write_cohort(root, local_reproduction=True)
            self.assertTrue(analyze.audit_records(root)[2])
            self.assertEqual(analyze.audit_records(root, local_reproduction=True)[2], [])
            with patch.object(analyze, "_plot"):
                result = analyze.write_outputs(root, local_reproduction=True)
            self.assertEqual(result["cohort_type"], "local_reproduction")
            self.assertTrue(result["complete_five_seeds"])
        forbidden = Path(analyze.__file__).resolve().parent / "results" / "any-primary-campaign"
        with self.assertRaises(ValueError):
            analyze.audit_records(forbidden, local_reproduction=True)
        with tempfile.TemporaryDirectory() as root:
            write_cohort(root)
            self.assertTrue(analyze.audit_records(root, local_reproduction=True)[2])


if __name__ == "__main__":
    unittest.main()

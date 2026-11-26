"""Failure gates for exact 20k reuse and a strictly capped new cohort."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from california_housing_benchmark import train as base
from california_housing_benchmark.spectral import aggregate_matrices
from california_housing_sgdm_search.long_train import BASE_HASHES
from california_housing_sgdm_20k import prepare, publish, verify


def plan():
    return json.loads((Path(__file__).parents[1] / "plan.json").read_text())


def records(valid_spectral=True):
    source = plan()["parent_source"]
    candidate = plan()["candidates"][0]
    old_plan = json.loads((verify.REPOSITORY / "california_housing_sgdm_search/plan.json").read_text())
    fit = {"alpha": 2.2, "D": 0.1, "num_pl_spikes": 32, "num_evals": 784, "status": "success"}
    if not valid_spectral:
        fit.update(alpha=-1, D=-1, num_pl_spikes=-1, status="failed")
    matrices = [{"matrix": "hidden.weight", "shape": [784, 784], "fit": fit}]
    spectral = {**aggregate_matrices(matrices, ["hidden.weight"]),
                "matrices": matrices, "eligible_matrices": ["hidden.weight"]}
    def row(step):
        return {"step": step, "validation": {"rmse": .5}, "test": None,
                "weights_sha256": "a" * 64, "spectral_weights_sha256": "a" * 64,
                "spectral": deepcopy(spectral)}
    record = {
        "status": "complete", "completed_updates": 50000, "total_updates": 50000,
        "optimizer": "sgdm", "seed": 3338, "phase": "development", "config": candidate["config"],
        "plan_sha256": base.canonical_sha256(old_plan), "data_manifest_sha256": "data-hash",
        "source_manifest_sha256": prepare.PARENT_SOURCE_MANIFEST_SHA256,
        "source_bundle_sha256": base.canonical_sha256({**BASE_HASHES, "long_train.py": prepare.PARENT_TRAIN_SHA256}),
        "execution": {"github": {"GITHUB_RUN_ID": source["run_id"], "GITHUB_SHA": source["launch_commit"],
                                    "GITHUB_REPOSITORY": source["repository"]}},
        "history": [row(step) for step in [0, 2000, 5000, 10000, 15000, 20000, 50000]],
        "epoch_size": 52,
        "epoch_history": [{"step": step, "validation": {"rmse": .5}} for step in range(52, 50001, 52)],
    }
    execution = {"github_run_id": source["run_id"], "launch_commit": source["launch_commit"],
                 "candidate_id": candidate["id"], "seed": 3338, "phase": "development",
                 "source_manifest_sha256": prepare.PARENT_SOURCE_MANIFEST_SHA256,
                 "data_manifest_sha256": "data-hash"}
    return record, execution, candidate, source


def validate(value):
    record, execution, candidate, source = value
    return prepare.validate_development(record, execution, candidate=candidate, seed=3338,
                                        source=source, data_hash="data-hash")


def test_plan_requires_hard_cap_and_unchanged_full_grid():
    source = plan()
    verify.validate_plan(source)
    for key, wrong in (("total_updates", 200000), ("development_updates", 50000),
                       ("checkpoint_updates", [2000, 20000, 100000])):
        changed = deepcopy(source)
        changed[key] = wrong
        with pytest.raises(ValueError):
            verify.validate_plan(changed)
    changed = deepcopy(source)
    changed["candidates"].pop()
    with pytest.raises(ValueError, match="full grid"):
        verify.validate_plan(changed)


def test_reuse_keeps_original_budget_and_runs_zero_new_updates():
    value = records()
    original = deepcopy(value[0])
    summary = validate(value)
    assert value[0] == original
    assert summary["source_completed_updates"] == 50000
    assert summary["analysis_max_step"] == 20000
    assert summary["new_updates"] == 0
    assert summary["n_epoch_observations_through_cap"] == 384
    assert summary["n_validation_observations_through_cap"] == 386


def test_measured_failed_spectral_fit_is_retained_not_retrained():
    value = records(valid_spectral=False)
    value[0]["status"] = "running"
    value[0]["completed_updates"] = 23192
    summary = validate(value)
    assert summary["alpha_at_cap"] is None
    assert summary["spectral_valid_at_cap"] is False
    assert summary["new_updates"] == 0
    assert summary["source_status"] == "running"


def test_reuse_refuses_missing_endpoint_and_validation_epochs():
    value = records()
    value[0]["history"] = [row for row in value[0]["history"] if row["step"] != 20000]
    with pytest.raises(ValueError, match="measurement prefix"):
        validate(value)
    value = records()
    value[0]["epoch_history"].pop(5)
    with pytest.raises(ValueError, match="epoch coverage"):
        validate(value)
    value = records()
    value[0]["completed_updates"] = 19999
    with pytest.raises(ValueError, match="fallback is forbidden"):
        validate(value)


def test_reuse_refuses_wrong_runtime_and_test_leakage():
    value = records()
    value[0]["execution"]["github"]["GITHUB_RUN_ID"] = "wrong"
    with pytest.raises(ValueError, match="execution differs"):
        validate(value)
    value = records()
    value[0]["history"][1]["test"] = {"rmse": .4}
    with pytest.raises(ValueError, match="test measurements"):
        validate(value)


def test_reuse_refuses_mismatched_performance_spectral_weights():
    value = records()
    value[0]["history"][-2]["spectral_weights_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different model weights"):
        validate(value)


def test_archive_scope_excludes_previous_results_and_source():
    assert publish.valid_paths([publish.PREFIX])
    for name in ("california_housing_sgdm_search/results/anything", "california_housing_sgdm_20k/plan.json",
                 publish.PREFIX + "/../other", "/" + publish.PREFIX):
        with pytest.raises(ValueError, match="outside"):
            publish.valid_paths([name])


def test_archive_cap_applies_to_new_training_not_original_inputs(tmp_path):
    prior = tmp_path / publish.PREFIX / "inputs/development/sgdm_00/seed_3338/result.json"
    prior.parent.mkdir(parents=True)
    prior.write_text('{"completed_updates":50000,"total_updates":50000}')
    publish.check_new_training_cap(tmp_path, prior)
    fresh = tmp_path / publish.PREFIX / "runs/seed_3340/result.json"
    fresh.parent.mkdir(parents=True)
    fresh.write_text('{"completed_updates":20001,"total_updates":20000}')
    with pytest.raises(RuntimeError, match="authorized 20k cap"):
        publish.check_new_training_cap(tmp_path, fresh)
    fresh.write_text('{"completed_updates":20000,"total_updates":20000}')
    publish.check_new_training_cap(tmp_path, fresh)


def test_original_inputs_are_immutable_and_checkpoints_are_retained(tmp_path):
    source = tmp_path / "source"
    original = source / publish.PREFIX / "inputs/development/sgdm_00/seed_3338/result.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"completed_updates":20000}')
    archived = tmp_path / "archived.json"
    archived.write_text('{"completed_updates":50000}')
    with pytest.raises(RuntimeError, match="Frozen input"):
        publish.check_immutable(source, original, archived)
    assert publish.ignore_archived_files(tmp_path, ["resume.pt", "checkpoint_step_20000.pt", ".run.lock"]) == {".run.lock"}


def test_another_workflow_cannot_replace_cohort_identity(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "plan.json").write_text('{"experiment_id":"test"}')
    (package / "SOURCE_SHA256.json").write_text('{}')
    monkeypatch.setattr(verify, "PACKAGE", package)
    monkeypatch.setenv("GITHUB_RUN_ID", "first")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_REPOSITORY", verify.REPOSITORY_NAME)
    root = tmp_path / "outputs"
    verify.claim_cohort(root, initialize=True)
    verify.claim_cohort(root)
    monkeypatch.setenv("GITHUB_RUN_ID", "second")
    with pytest.raises(RuntimeError, match="another workflow"):
        verify.claim_cohort(root, initialize=True)


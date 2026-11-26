"""Focused failure gates for continuing retained results, not restarting them."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from california_housing_benchmark import train as base
from california_housing_sgdm_search.long_train import BASE_HASHES
from california_housing_sgdm_200k import prepare, publish, verify


def source_plan():
    return json.loads((Path(__file__).parents[1] / "plan.json").read_text())


def effective_plan():
    source = source_plan()
    plan = deepcopy(source)
    plan.update(source_plan_sha256="source-plan-hash", continuation={
        "repository": prepare.PARENT_REPOSITORY,
        "run_id": prepare.PARENT_RUN_ID,
        "launch_commit": prepare.PARENT_LAUNCH_COMMIT,
        "archive_commit": "a" * 40, "checkpoint_step": 100000,
        "checkpoints": {str(seed): {"sha256": "c" * 64, "git_blob_sha": "b" * 40,
                                     "repository_path": f"{source['parent_source']['results_root']}/continuations/seed_{seed}/checkpoint_step_100000.pt"}
                         for seed in prepare.PARENT_SEEDS}})
    return source, plan


def parent_records():
    config = source_plan()["continuation_config"]
    old_plan = json.loads((verify.REPOSITORY / "california_housing_sgdm_search/plan.json").read_text())
    record = {
        "status": "complete", "optimizer": "sgdm", "seed": 2340,
        "phase": "continuation", "completed_updates": 100000, "total_updates": 100000,
        "start_step": 20000, "new_updates": 80000, "config": config,
        "data_manifest_sha256": "data-hash", "source_manifest_sha256": prepare.PARENT_SOURCE_MANIFEST_SHA256,
        "source_bundle_sha256": base.canonical_sha256({**BASE_HASHES, "long_train.py": prepare.PARENT_TRAIN_SHA256}),
        "plan_sha256": base.canonical_sha256(old_plan), "final_weights_sha256": "weights-hash",
        "history": [{"step": 100000, "weights_sha256": "weights-hash", "spectral_weights_sha256": "weights-hash"}],
        "continuity": {"verified": True},
        "execution": {"github": {"GITHUB_RUN_ID": prepare.PARENT_RUN_ID,
                                    "GITHUB_SHA": prepare.PARENT_LAUNCH_COMMIT,
                                    "GITHUB_REPOSITORY": prepare.PARENT_REPOSITORY}},
    }
    shared = {"github_run_id": prepare.PARENT_RUN_ID, "launch_commit": prepare.PARENT_LAUNCH_COMMIT,
              "seed": 2340, "phase": "continuation", "data_manifest_sha256": "data-hash",
              "source_manifest_sha256": prepare.PARENT_SOURCE_MANIFEST_SHA256}
    completion = {**shared, "status": "complete", "completed_updates": 100000}
    return record, shared, completion


def validate_records(record, execution, completion):
    prepare.validate_parent_record(record, execution, completion, seed=2340,
                                    config=source_plan()["continuation_config"], data_hash="data-hash")


def test_effective_plan_only_adds_pinned_parent_identities():
    source, plan = effective_plan()
    verify.validate_effective_plan(plan, source, "source-plan-hash")
    changed = deepcopy(plan)
    changed["continuation_config"]["lr"] = 0.1
    with pytest.raises(ValueError, match="scientific setting"):
        verify.validate_effective_plan(changed, source, "source-plan-hash")
    changed = deepcopy(plan)
    changed["continuation"]["run_id"] = "different-run"
    with pytest.raises(ValueError, match="provenance"):
        verify.validate_effective_plan(changed, source, "source-plan-hash")
    changed = deepcopy(plan)
    del changed["continuation"]["checkpoints"]["2344"]
    with pytest.raises(ValueError, match="seeds"):
        verify.validate_effective_plan(changed, source, "source-plan-hash")


def test_parent_requires_complete_expected_cohort_and_same_model():
    record, execution, completion = parent_records()
    validate_records(record, execution, completion)
    for key, wrong in (("status", "running"), ("completed_updates", 99999),
                       ("source_bundle_sha256", "changed-source")):
        changed = deepcopy(record)
        changed[key] = wrong
        with pytest.raises(ValueError, match="identity/completeness"):
            validate_records(changed, execution, completion)
    changed = deepcopy(record)
    changed["execution"]["github"]["GITHUB_RUN_ID"] = "different-run"
    with pytest.raises(ValueError, match="runtime provenance"):
        validate_records(changed, execution, completion)
    changed = deepcopy(record)
    changed["history"][0]["spectral_weights_sha256"] = "different-weights"
    with pytest.raises(ValueError, match="same weights"):
        validate_records(changed, execution, completion)


def test_parent_checkpoint_must_be_full_at_exact_step(tmp_path):
    path = tmp_path / "checkpoint.pt"
    record, _, _ = parent_records()
    for checkpoint in ({"checkpoint_kind": "weights_only", "completed_updates": 100000},
                       {"checkpoint_kind": "full_resume", "completed_updates": 20000}):
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="full optimizer/RNG"):
            prepare.validate_parent_checkpoint(path, record, seed=2340,
                      config=source_plan()["continuation_config"], data_hash="data-hash")


def test_publication_scope_excludes_parent_and_training_source():
    assert publish.valid_paths([publish.PREFIX, publish.PREFIX + "/continuations/seed_2340"])
    for path in ("california_housing_sgdm_search/results/anything", "california_housing_sgdm_200k/plan.json",
                 publish.PREFIX + "/../other", "/" + publish.PREFIX):
        with pytest.raises(ValueError, match="outside"):
            publish.valid_paths([path])


def test_archive_refuses_completed_rollback_and_changed_measurements():
    archived = {"status": "complete", "completed_updates": 200000, "seed": 2340,
                "history": [{"step": 200000, "alpha": 2.1}]}
    for changed in ({**archived, "status": "running"}, {**archived, "completed_updates": 100000},
                    {**archived, "history": [{"step": 200000, "alpha": 2.0}]}):
        with pytest.raises(RuntimeError):
            publish.check_record_monotonic(changed, archived, "seed2340")
    publish.check_record_monotonic(archived, archived, "seed2340")


def test_archive_keeps_every_resume_checkpoint_and_freezes_inputs(tmp_path):
    names = ["resume.pt", "checkpoint_step_100000.pt", "checkpoint_step_200000.pt", "best_checkpoint.pt", ".run.lock"]
    assert publish.ignore_archived_files(tmp_path, names) == {".run.lock"}
    source = tmp_path / "source"
    incoming = source / publish.PREFIX / "parent_snapshot.json"
    incoming.parent.mkdir(parents=True)
    incoming.write_text('{"archive_commit":"new"}')
    archived = tmp_path / "archived_snapshot.json"
    archived.write_text('{"archive_commit":"old"}')
    with pytest.raises(RuntimeError, match="Frozen input"):
        publish.check_immutable(source, incoming, archived)


def test_cohort_claim_cannot_start_another_workflow(tmp_path, monkeypatch):
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


def test_parent_path_set_has_no_unrelated_runs():
    paths = prepare.parent_paths(source_plan())
    assert len(paths) == 22
    for seed in prepare.PARENT_SEEDS:
        assert f"parent/seed_{seed}/checkpoint100k.pt" in paths.values()
    assert all("development" not in path and "runs/seed_334" not in path for path in paths)


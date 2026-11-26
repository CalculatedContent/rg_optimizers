"""Protocol tests: full-grid selection, archive recovery, and frozen execution."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest
import torch

from california_housing_benchmark import train as base
from california_housing_sgdm_search import run_one, search


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    package = tmp_path / "source"
    original = search.PACKAGE_ROOT
    for name in search.REQUIRED_SOURCE_FILES:
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / name, destination)
    monkeypatch.setattr(search, "PACKAGE_ROOT", package)
    monkeypatch.setenv("GITHUB_RUN_ID", "test-new-cohort")
    monkeypatch.setenv("GITHUB_SHA", "test-frozen-source")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    root = tmp_path / "results"
    (root / "data").mkdir(parents=True)
    data = '{"unit_test_fixture":true}\n'
    (root / "data" / "manifest.json").write_text(data)
    (root / "data_manifest.json").write_text(data)
    plan = json.loads((package / "california_housing_sgdm_search/plan.json").read_text())

    def update_plan(value):
        content = json.dumps(value, indent=2) + "\n"
        (root / "plan.json").write_text(content)
        (package / "california_housing_sgdm_search/plan.json").write_text(content)
        files = {name: search.digest(package / name) for name in search.REQUIRED_SOURCE_FILES}
        (root / "SOURCE_SHA256.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    update_plan(plan)
    return {"root": root, "plan": plan, "update_plan": update_plan}


def fake_backend(campaign, *, failures=(), scores=None):
    """Writes explicit tiny fixture states; performs no optimizer updates."""
    calls = []
    root = campaign["root"]
    failures = set(failures)
    scores = scores or {}

    def train(plan, config, data_dir, output_dir, seed, phase, parent_checkpoint=None, resume=True):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        cid = next((candidate["id"] for candidate in plan["candidates"] if candidate["config"] == config), None)
        calls.append((phase, cid, seed, parent_checkpoint))
        identity = search.load_context(root)[2]
        total = plan["development_updates"] if phase == "development" else plan["total_updates"]
        score = scores.get((cid, seed), 0.8)
        best_weights = {"fixture.weight": torch.tensor([float(seed), score], dtype=torch.float32)}
        final_weights = {"fixture.weight": torch.tensor([float(seed), score + 1], dtype=torch.float32)}
        best_hash, final_hash = base.model_hash(best_weights), base.model_hash(final_weights)
        start = 20000 if phase == "continuation" else 0
        metrics = {"rmse": score, "mse": score ** 2, "mae": score, "r2": .5, "mape_percent": 1.0}
        best = {"step": start + 1, "epoch": 1, "validation": metrics,
                "train": metrics, "test": None if phase == "development" else metrics,
                "spectral": None, "weights_sha256": best_hash}
        window = {"start": start, "end": total, "observed_through": total, "complete": True}
        best["selection_window"] = window
        record = {"schema_version": 1, "fixture_only": True, "seed": seed, "phase": phase,
                  "config": config, "status": "complete", "completed_updates": total,
                  "plan_sha256": base.canonical_sha256(plan),
                  "source_manifest_sha256": identity["source_manifest_sha256"],
                  "data_manifest_sha256": identity["data_manifest_sha256"],
                  "execution": {"github": {"GITHUB_RUN_ID": os.getenv("GITHUB_RUN_ID"),
                                            "GITHUB_SHA": os.getenv("GITHUB_SHA")}},
                  "resume_executions": [], "best_validation": best, "selection_window": window,
                  "best_weights_sha256": best_hash, "final_weights_sha256": final_hash, "history": []}
        if (cid, seed) in failures:
            record.update(status="failed", completed_updates=7, best_validation=None,
                          error="FloatingPointError: synthetic non-finite-gradient fixture")
            search.write_json(output / "result.json", record)
            raise FloatingPointError("synthetic non-finite-gradient fixture")
        search.write_json(output / "result.json", record)
        state = {"checkpoint_kind": "full_resume",
                 "identity": {"seed": seed, "phase": phase, "config": config},
                 "model": final_weights, "best_weights": best_weights,
                 "completed_updates": total, "result": record}
        base.atomic_checkpoint(output / "resume.pt", state)
        for step in plan["checkpoint_updates"]:
            if start <= step <= total:
                base.atomic_checkpoint(output / f"checkpoint_step_{step}.pt",
                                       {**state, "completed_updates": step})
        base.atomic_checkpoint(output / "best_checkpoint.pt",
                               {"model": best_weights, "weights_sha256": best_hash,
                                "selected_step": best["step"], "selected_epoch": 1,
                                "identity": state["identity"]})
        return record
    return train, calls


def fill_grid(campaign, monkeypatch, *, omit_last=False):
    scores = {(candidate["id"], seed): .8 + i / 1000
              for i, candidate in enumerate(campaign["plan"]["candidates"])
              for seed in campaign["plan"]["development_seeds"]}
    # Equal means; fixed ID must break the tie. A survivor of a failed pair must not win.
    scores.update({("sgdm_00", 3339): .01, ("sgdm_01", 3338): .4,
                   ("sgdm_01", 3339): .6, ("sgdm_02", 3338): .5, ("sgdm_02", 3339): .5})
    train, calls = fake_backend(campaign, failures={("sgdm_00", 3338)}, scores=scores)
    monkeypatch.setattr(search, "train_long", train)
    pairs = [(candidate["id"], seed) for candidate in campaign["plan"]["candidates"]
             for seed in campaign["plan"]["development_seeds"]]
    for cid, seed in pairs[:-1] if omit_last else pairs:
        search.run(campaign["root"], cid, seed)
    return calls, pairs[-1]


def test_completed_development_archive_is_reused_without_full_state(campaign, monkeypatch):
    train, calls = fake_backend(campaign)
    monkeypatch.setattr(search, "train_long", train)
    root = campaign["root"]
    first = search.run(root, "sgdm_00", 3338)
    directory = root / "development/sgdm_00/seed_3338"
    (directory / "resume.pt").unlink()
    for path in directory.glob("checkpoint_step_*.pt"):
        path.unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed archive must never replay training")
    monkeypatch.setattr(search, "train_long", forbidden)
    second = search.run(root, "sgdm_00", 3338)
    assert first == second and len(calls) == 1
    assert set(first["retained_files"]) == {"best_checkpoint.pt", "final_weights.pt"}


def test_completed_export_failure_is_repaired_without_training(campaign, monkeypatch):
    train, _ = fake_backend(campaign)
    monkeypatch.setattr(search, "train_long", train)
    root = campaign["root"]
    search.run(root, "sgdm_01", 3338)
    directory = root / "development/sgdm_01/seed_3338"
    (directory / "trial.json").unlink()  # Interrupted after core completion, before a durable trial.
    (directory / "final_weights.pt").unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("Export recovery must not call the trainer")
    monkeypatch.setattr(search, "train_long", forbidden)
    result = search.run(root, "sgdm_01", 3338)
    assert result["status"] == "complete"
    assert search.digest(directory / "final_weights.pt") == result["retained_files"]["final_weights.pt"]


def test_full_grid_required_and_winner_uses_both_seeds_with_id_tie_break(campaign, monkeypatch):
    calls, last = fill_grid(campaign, monkeypatch, omit_last=True)
    root = campaign["root"]
    with pytest.raises(FileNotFoundError):
        search.freeze(root)
    assert not (root / "frozen_selection.json").exists()
    search.run(root, *last)
    frozen = search.freeze(root)
    assert len(calls) == 64 and len(frozen["trials"]) == 32
    assert frozen["selected_candidate_id"] == "sgdm_01"
    assert frozen["selected_mean_best_validation_rmse"] == .5
    failed = frozen["trials"][0]
    assert failed["eligible"] is False and failed["mean_best_validation_rmse"] is None
    assert [row["status"] for row in failed["seed_results"]] == ["numerical_failure", "complete"]
    assert (root / "development/sgdm_00/trial.json").is_file()
    assert search.freeze(root) == frozen
    count = len(calls)
    repeated = search.run(root, "sgdm_00", 3338)
    assert repeated["status"] == "numerical_failure" and len(calls) == count


def test_confirmation_requires_full_frozen_grid_then_reuses_completed_run(campaign, monkeypatch):
    train, calls = fake_backend(campaign)
    monkeypatch.setattr(run_one, "train_long", train)
    root = campaign["root"]
    with pytest.raises(FileNotFoundError):
        run_one.run_one(root, "confirmation", 3340)
    assert not calls
    fill_grid(campaign, monkeypatch)
    frozen = search.freeze(root)
    result = run_one.run_one(root, "confirmation", 3340)
    assert calls == [("confirmation", "sgdm_01", 3340, None)]
    assert result["config"] == frozen["selected"]
    assert result["frozen_selection_sha256"] == search.digest(root / "frozen_selection.json")
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed confirmation must not restart")
    monkeypatch.setattr(run_one, "train_long", forbidden)
    assert run_one.run_one(root, "confirmation", 3340) == result
    changed = deepcopy(frozen)
    changed["trials"] = changed["trials"][:-1]
    search.write_json(root / "frozen_selection.json", changed)
    with pytest.raises(ValueError, match="full grid"):
        run_one.run_one(root, "confirmation", 3341)


def test_continuation_can_start_before_grid_freeze_and_passes_parent(campaign, monkeypatch):
    root = campaign["root"]
    parent = root / "parent/seed_2340/checkpoint20k.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"explicit unit-test checkpoint; backend is mocked")
    plan = deepcopy(campaign["plan"])
    plan["continuation"]["checkpoints"]["2340"]["sha256"] = search.digest(parent)
    campaign["plan"] = plan
    campaign["update_plan"](plan)
    train, calls = fake_backend(campaign)
    monkeypatch.setattr(run_one, "train_long", train)
    result = run_one.run_one(root, "continuation", 2340)
    assert not (root / "frozen_selection.json").exists()
    assert calls == [("continuation", "sgdm_17", 2340, parent)]
    assert result["config"] == plan["continuation_config"]
    assert result["frozen_selection_sha256"] is None
    assert (root / "continuations/seed_2340/completion.json").exists()


def test_changed_data_or_runtime_cohort_cannot_reuse_a_completed_trial(campaign, monkeypatch):
    train, calls = fake_backend(campaign)
    monkeypatch.setattr(search, "train_long", train)
    root = campaign["root"]
    search.run(root, "sgdm_00", 3338)
    monkeypatch.setenv("GITHUB_RUN_ID", "different-cohort")
    with pytest.raises(ValueError, match="github_run_id"):
        search.run(root, "sgdm_00", 3338)
    assert len(calls) == 1
    monkeypatch.setenv("GITHUB_RUN_ID", "test-new-cohort")
    (root / "data/manifest.json").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="data manifests"):
        search.run(root, "sgdm_00", 3338)
    assert len(calls) == 1

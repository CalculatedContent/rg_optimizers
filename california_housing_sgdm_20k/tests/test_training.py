"""Hard-cap, provenance, and duplicate-prevention tests; no model training."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import types

import pytest
import torch

from california_housing_benchmark import train as base
from california_housing_sgdm_20k import run_one as module


def plan_fixture():
    config = {"optimizer": "sgdm", "lr": .03, "momentum": .9, "weight_decay": 0,
              "dampening": 0, "nesterov": False, "dropout": .3690073279296028}
    return {"total_updates": 20000, "development_updates": 20000,
            "confirmation_seeds": list(module.SEEDS), "eval_every": 5000,
            "checkpoint_updates": [2000, 20000], "candidates": [{"id": "sgdm_test", "config": config}]}


@pytest.fixture
def context(tmp_path, monkeypatch):
    plan = plan_fixture()
    identity = {"plan_file_sha256": "plan-file", "source_inventory_sha256": "inventory",
                "data_manifest_sha256": "data", "source_manifest_sha256": "orchestration-manifest",
                "github_run_id": "unit-run", "launch_commit": "unit-commit"}
    frozen = {"selected": plan["candidates"][0]["config"]}
    (tmp_path / "frozen_selection.json").write_text(json.dumps(frozen))
    verify = types.ModuleType("california_housing_sgdm_20k.verify")
    verify.load_context = lambda root: (Path(root), plan, identity)
    analysis = types.ModuleType("california_housing_sgdm_20k.analysis")
    analysis.validate_frozen_selection = lambda root, supplied_plan, supplied_identity: frozen
    monkeypatch.setitem(sys.modules, verify.__name__, verify)
    monkeypatch.setitem(sys.modules, analysis.__name__, analysis)
    monkeypatch.setenv("GITHUB_REPOSITORY", module.REPOSITORY)
    calls = []

    def trainer(supplied_plan, config, data, directory, seed, phase, parent_checkpoint, resume):
        calls.append({"plan": supplied_plan, "config": config, "seed": seed,
                      "phase": phase, "parent": parent_checkpoint, "resume": resume})
        model = {"matrix.weight": torch.tensor([[1., 2.], [3., 4.]])}
        model_hash = base.model_hash(model)
        sources = {name: module.sha256(module.PACKAGE.parent / "california_housing_benchmark" / name)
                   for name in ("data.py", "train.py", "optimizer.py", "spectral.py")}
        sources["long_train.py"] = module.sha256(module.PACKAGE.parent / "california_housing_sgdm_search/long_train.py")
        execution = {"source_sha256": sources, "source_manifest_sha256": module.training_manifest_sha256(),
                     "precision": "float32", "torch_num_threads": 1, "torch_num_interop_threads": 1,
                     "github": {"GITHUB_RUN_ID": identity["github_run_id"], "GITHUB_SHA": identity["launch_commit"],
                                "GITHUB_REPOSITORY": module.REPOSITORY}}
        engine_identity = {"optimizer": "sgdm", "seed": seed, "phase": phase, "config": config,
                           "plan_sha256": base.canonical_sha256(supplied_plan),
                           "data_manifest_sha256": identity["data_manifest_sha256"],
                           "source_bundle_sha256": base.canonical_sha256(sources),
                           "source_manifest_sha256": module.training_manifest_sha256(),
                           "parent_checkpoint_sha256": None}
        history = [{"step": step, "epoch": step // 52, "weights_sha256": model_hash,
                    "spectral_weights_sha256": model_hash,
                    "train": {"rmse": .3}, "validation": {"rmse": .4}, "test": {"rmse": .5}}
                   for step in (0, 2000, 5000, 10000, 15000, 20000)]
        window = {"start": 0, "end": 20000, "observed_through": 20000, "complete": True}
        best = {**history[-1], "selection_window": window}
        result = {**engine_identity, "status": "complete", "start_step": 0, "total_updates": 20000,
                  "completed_updates": 20000, "new_updates": 20000, "execution": execution,
                  "resume_executions": [], "history": history, "epoch_history": [],
                  "primary_endpoint": history[-1], "best_validation": best,
                  "primary_estimand": "best_validation", "selection_window": window,
                  "final_weights_sha256": model_hash, "best_weights_sha256": model_hash}
        for step in (2000, 20000):
            state = {"checkpoint_kind": "full_resume", "identity": engine_identity,
                     "model": model, "optimizer": {}, "rng": base.rng_state(),
                     "batch_generator": torch.Generator().get_state(), "epoch_permutation": None,
                     "next_offset": 0, "completed_updates": step, "completed_epochs": step // 52,
                     "minibatch_sha256": "0" * 64, "epoch_loss_sum": 0., "epoch_examples": 0,
                     "best_weights": model, "best_state": {"step": 20000}, "result": result}
            torch.save(state, directory / f"checkpoint_step_{step}.pt")
            if step == 20000:
                torch.save(state, directory / "resume.pt")
        base.atomic_json(directory / "result.json", result)
        return result

    monkeypatch.setattr(module, "train_long", trainer)
    return tmp_path, plan, identity, calls


def test_hard_cap_and_confirmation_seed_guard():
    plan = plan_fixture()
    module.check_cap(plan, 3340)
    for key, value in (("total_updates", 20001), ("development_updates", 50000),
                       ("checkpoint_updates", [2000, 20000, 50000])):
        bad = deepcopy(plan)
        bad[key] = value
        with pytest.raises(ValueError):
            module.check_cap(bad, 3340)
    for seed in (2338, 2340, 3338, 3339):
        with pytest.raises(ValueError, match="fresh confirmation"):
            module.check_cap(plan, seed)


def test_unchanged_engine_receives_only_fresh20k_and_sources_remain_distinct(context):
    root, plan, identity, calls = context
    result = module.run_one(root, 3340)
    assert len(calls) == 1
    assert calls[0]["plan"]["total_updates"] == 20000
    assert calls[0]["phase"] == "confirmation"
    assert calls[0]["parent"] is None
    assert result["source_manifest_sha256"] == module.training_manifest_sha256()
    assert result["orchestration_source_manifest_sha256"] == identity["source_manifest_sha256"]
    assert result["primary_estimand"] == "best_validation"
    assert result["report_primary_estimand"] == "fixed_20000_updates"
    completed = json.loads((root / "runs/seed_3340/completion.json").read_text())
    assert set(completed["retained_files"]) == {"checkpoint_step_2000.pt", "checkpoint_step_20000.pt",
                                               "resume.pt", "best_checkpoint.pt", "final_weights.pt"}


def test_completed_confirmation_never_retrains(context):
    root, plan, identity, calls = context
    first = module.run_one(root, 3340)
    second = module.run_one(root, 3340)
    assert first == second
    assert len(calls) == 1


def test_missing_completed_archive_requires_restore_not_retraining(context):
    root, plan, identity, calls = context
    module.run_one(root, 3340)
    (root / "runs/seed_3340/checkpoint_step_2000.pt").unlink()
    with pytest.raises(ValueError, match="Restore missing retained"):
        module.run_one(root, 3340)
    assert len(calls) == 1


def test_over_cap_result_and_resume_rejected_before_training(context):
    root, plan, identity, calls = context
    directory = root / "runs/seed_3340"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({"completed_updates": 50000, "status": "complete"}))
    with pytest.raises(ValueError, match="above20k"):
        module.run_one(root, 3340)
    assert not calls
    (directory / "result.json").unlink()
    torch.save({"completed_updates": 20001}, directory / "resume.pt")
    with pytest.raises(ValueError, match="above20k"):
        module.run_one(root, 3340)
    assert not calls


def test_result_source_and_metric_binding_are_checked(context):
    root, plan, identity, calls = context
    result = module.run_one(root, 3340)
    for mutate in (lambda r: r.update(source_manifest_sha256=identity["source_manifest_sha256"]),
                   lambda r: r["history"][1].update(spectral_weights_sha256="wrong"),
                   lambda r: r["history"].append(deepcopy(r["history"][0])),
                   lambda r: r["epoch_history"].append({"step": 50000})):
        bad = deepcopy(result)
        mutate(bad)
        with pytest.raises(ValueError):
            module.validate_result(bad, plan, identity, result["config"], 3340)

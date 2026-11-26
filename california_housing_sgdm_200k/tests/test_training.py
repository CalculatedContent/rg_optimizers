"""Small numerical checks for exact second-stage continuation and recovery."""
from copy import deepcopy
import hashlib
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from california_housing_benchmark import train as base
from california_housing_sgdm_search import long_train as previous
from california_housing_sgdm_200k import long_train as trainer
from california_housing_sgdm_200k import run_one as wrapper


def saved(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def config():
    return {"optimizer": "sgdm", "lr": .003, "momentum": .9,
            "weight_decay": 0, "dampening": 0, "nesterov": False, "dropout": .37}


def plan(**changes):
    value = {"model": {"n_blocks": 2, "d_block": 12, "dropout": .37},
             "batch_size": 5, "eval_every": 5, "eval_batch_size": 32768,
             "total_updates": 31, "checkpoint_updates": [20, 25, 31],
             "measure_spectra": False, "grad_clip": 1.0,
             "continuation_seeds": [2340], "continuation_config": config()}
    value.update(changes)
    return value


def record_for(path, repository_path):
    payload = path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(),
            "git_blob_sha": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
            "repository_path": repository_path}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(804)
    arrays = {}
    for split, n in (("train", 32), ("validation", 9), ("test", 11)):
        x = rng.normal(size=(n, 8)).astype(np.float32)
        arrays[f"x_{split}"] = x
        arrays[f"y_{split}_original"] = (2 + .2 * x[:, 0] - .1 * x[:, 1]).astype(np.float32)
    arrays["y_mean"] = np.float32(arrays["y_train_original"].mean())
    arrays["y_scale"] = np.float32(arrays["y_train_original"].std())
    for split in ("train", "validation", "test"):
        arrays[f"y_{split}"] = ((arrays[f"y_{split}_original"] - arrays["y_mean"])
                                / arrays["y_scale"]).astype(np.float32)
    (tmp_path / "manifest.json").write_text(json.dumps({"fixture": True}))
    monkeypatch.setattr(base, "load_data", lambda directory: (arrays, {"fixture": True}))
    current = {"run_id": "45753214421", "commit": trainer.BASE_SOURCE_COMMIT}
    original_metadata = base.execution_metadata
    def metadata():
        value = original_metadata()
        value["github"].update(GITHUB_RUN_ID=current["run_id"], GITHUB_SHA=current["commit"],
                               GITHUB_REPOSITORY=trainer.PARENT_REPOSITORY)
        return value
    monkeypatch.setattr(base, "execution_metadata", metadata)
    monkeypatch.setattr(trainer, "PARENT_CHECKPOINT_STEP", 20)
    monkeypatch.setattr(trainer, "TOTAL_UPDATES", 31)
    return SimpleNamespace(root=tmp_path, current=current, monkeypatch=monkeypatch)


def make_parent(fixture):
    root = fixture.root
    grandparent_plan = plan(total_updates=31, safety_max_updates=50, patience=100,
                            checkpoint_updates=[7, 31])
    without_dropout = {key: value for key, value in config().items() if key != "dropout"}
    base.train(grandparent_plan, without_dropout, root, root / "grandparent", 2340,
               phase="confirmation", stop_step=7)
    grandparent = root / "grandparent/checkpoint_step_7.pt"
    first_contract = {"repository": trainer.PARENT_REPOSITORY, "run_id": "45753214421",
                      "launch_commit": trainer.BASE_SOURCE_COMMIT, "archive_commit": "fixture-first-archive",
                      "checkpoint_step": 7, "checkpoints": {"2340": record_for(grandparent, "first/checkpoint.pt")}}
    previous_plan = plan(total_updates=20, checkpoint_updates=[7, 15, 20], continuation=first_contract)
    fixture.current.update(run_id=trainer.PARENT_RUN_ID, commit=trainer.PARENT_LAUNCH_COMMIT)
    fixture.monkeypatch.setattr(trainer, "PARENT_PLAN_SHA256", base.canonical_sha256(previous_plan))
    previous.train_long(previous_plan, config(), root, root / "parent100k", 2340,
                        "continuation", parent_checkpoint=grandparent)
    path = root / "parent100k/checkpoint_step_20.pt"
    contract = {"repository": trainer.PARENT_REPOSITORY, "run_id": trainer.PARENT_RUN_ID,
                "launch_commit": trainer.PARENT_LAUNCH_COMMIT, "archive_commit": "fixture-second-archive",
                "checkpoint_step": 20, "checkpoints": {"2340": record_for(path, "second/checkpoint.pt")}}
    fixture.current.update(run_id="fixture-next-run", commit="fixture-next-launch")
    return path, plan(continuation=contract)


def test_second_continuation_and_pause_equal_uninterrupted_updates(fixture, monkeypatch):
    parent, p = make_parent(fixture)
    original_bytes = parent.read_bytes()
    previous.train_long(plan(), config(), fixture.root, fixture.root / "continuous", 2340, "confirmation")
    calls = []
    original_factory = base.make_optimizer
    def factory(model, cfg):
        optimizer = original_factory(model, cfg)
        original_step = optimizer.step
        def step(*args, **kwargs):
            calls.append(1)
            return original_step(*args, **kwargs)
        optimizer.step = step
        return optimizer
    monkeypatch.setattr(base, "make_optimizer", factory)
    trainer.train_long(p, config(), fixture.root, fixture.root / "extension", 2340,
                       "continuation", parent_checkpoint=parent, stop_step=25)
    result = trainer.train_long(p, config(), fixture.root, fixture.root / "extension", 2340,
                                "continuation", parent_checkpoint=parent)
    assert len(calls) == 11
    assert parent.read_bytes() == original_bytes
    expected = saved(fixture.root / "continuous/resume.pt")
    actual = saved(fixture.root / "extension/resume.pt")
    assert trainer.checkpoint_fingerprints(actual) == trainer.checkpoint_fingerprints(expected)
    fingerprint = trainer.checkpoint_fingerprints(saved(parent))
    assert result["continuity"] == {"parent": fingerprint, "start": fingerprint, "verified": True}
    assert result["parent"]["identity"]["config"] == config()  # includes the original dropout field
    assert result["parent_checkpoint_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert result["start_step"] == 20 and result["new_updates"] == 11
    assert result["primary_estimand"] == "fixed_updates"
    assert result["primary_endpoint"]["step"] == 31
    assert result["selection_window"] == {"start": 20, "end": 31, "observed_through": 31, "complete": True}
    assert all((fixture.root / f"extension/checkpoint_step_{step}.pt").is_file() for step in [20, 25, 31])


def test_old_best_does_not_enter_new_selection_window(fixture, monkeypatch):
    def constant_metrics(model, arrays, split, batch_size=32768):
        return {"rmse": 1., "mse": 1., "mae": 1., "r2": 0.,
                "mape_percent": 1., "standardized_mse": 1.}
    monkeypatch.setattr(base, "evaluate", constant_metrics)
    parent, p = make_parent(fixture)
    assert saved(parent)["result"]["best_validation"]["step"] == 7
    result = trainer.train_long(p, config(), fixture.root, fixture.root / "extension", 2340,
                                "continuation", parent_checkpoint=parent)
    assert result["best_validation"]["step"] == 20
    assert result["parent"]["original_best_validation"]["step"] == 7
    assert result["best_validation"]["selection_window"]["start"] == 20


@pytest.mark.parametrize("change,match", [
    ("config", "optimizer settings"), ("source", "frozen 100k trainer"),
    ("run", "execution provenance"), ("plan", "frozen 100k protocol"),
    ("phase", "continuation phase"),
])
def test_pinned_parent_recipe_source_and_cohort_cannot_change(fixture, change, match):
    parent, p = make_parent(fixture)
    state = saved(parent)
    c = config()
    if change == "config":
        c["lr"] *= 2
    elif change == "source":
        state["identity"]["source_bundle_sha256"] = base.canonical_sha256(trainer.BASE_HASHES)
    elif change == "run":
        state["result"]["execution"]["github"]["GITHUB_RUN_ID"] = "different-run"
    elif change == "plan":
        state["plan"]["grad_clip"] = .5
    elif change == "phase":
        state["identity"]["phase"] = "confirmation"
    changed = fixture.root / "changed.pt"
    torch.save(state, changed)
    p["continuation"]["checkpoints"]["2340"] = record_for(changed, "second/checkpoint.pt")
    with pytest.raises(ValueError, match=match):
        trainer.train_long(p, c, fixture.root, fixture.root / "invalid", 2340,
                           "continuation", parent_checkpoint=changed)


def test_parent_bytes_and_completed_run_are_preserved(fixture, monkeypatch):
    parent, p = make_parent(fixture)
    corrupt = fixture.root / "corrupt.pt"
    corrupt.write_bytes(parent.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="file hash"):
        trainer.train_long(p, config(), fixture.root, fixture.root / "wrong", 2340,
                           "continuation", parent_checkpoint=corrupt)
    output = fixture.root / "extension"
    result = trainer.train_long(p, config(), fixture.root, output, 2340, "continuation", parent_checkpoint=parent)
    original = (output / "resume.pt").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed continuation must not construct another optimizer")
    monkeypatch.setattr(base, "make_optimizer", forbidden)
    assert trainer.train_long(p, config(), fixture.root, output, 2340, "continuation", parent_checkpoint=parent) == result
    assert (output / "resume.pt").read_bytes() == original
    (output / "checkpoint_step_25.pt").unlink()
    with pytest.raises(ValueError, match="retained checkpoints"):
        trainer.train_long(p, config(), fixture.root, output, 2340, "continuation", parent_checkpoint=parent)


def test_spectral_measurement_preserves_rng_and_same_model_pairing(fixture, monkeypatch):
    parent, p = make_parent(fixture)
    def spectra(model):
        torch.rand(3)
        np.random.normal(size=3)
        random.random()
        return {"fixture": True, "model_hash": base.model_hash(model), "confidence_tail_alpha": 2.25}
    monkeypatch.setattr(trainer, "measure_spectra", spectra)
    with_spectra = {**p, "measure_spectra": True}
    result = trainer.train_long(with_spectra, config(), fixture.root, fixture.root / "spectral", 2340,
                                "continuation", parent_checkpoint=parent)
    trainer.train_long(p, config(), fixture.root, fixture.root / "control", 2340,
                       "continuation", parent_checkpoint=parent)
    assert trainer.checkpoint_fingerprints(saved(fixture.root / "spectral/resume.pt")) == trainer.checkpoint_fingerprints(saved(fixture.root / "control/resume.pt"))
    for row in [*result["history"], result["best_validation"]]:
        assert row["weights_sha256"] == row["spectral_weights_sha256"] == row["spectral"]["model_hash"]


def test_wrapper_reuses_complete_run_and_retains_every_full_checkpoint(fixture, monkeypatch):
    parent, p = make_parent(fixture)
    root = fixture.root / "wrapper"
    staged = root / "parent/seed_2340/checkpoint100k.pt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(parent.read_bytes())
    (root / "data").mkdir()
    (root / "data/manifest.json").write_bytes((fixture.root / "manifest.json").read_bytes())
    execution = trainer.execution_metadata()
    identity = {"source_manifest_sha256": execution["source_manifest_sha256"],
                "data_manifest_sha256": trainer.file_sha256(root / "data/manifest.json"),
                "plan_file_sha256": "fixture-effective-file", "effective_plan_sha256": base.canonical_sha256(p),
                "source_plan_sha256": "fixture-source", "github_run_id": fixture.current["run_id"],
                "launch_commit": fixture.current["commit"]}
    monkeypatch.setitem(sys.modules, "california_housing_sgdm_200k.verify",
                        SimpleNamespace(load_context=lambda directory: (root, p, identity)))
    result = wrapper.run_one(root, 2340)
    output = root / "continuations/seed_2340"
    completion = json.loads((output / "completion.json").read_text())
    assert set(completion["retained_files"]) == {
        "checkpoint_step_20.pt", "checkpoint_step_25.pt", "checkpoint_step_31.pt",
        "resume.pt", "best_checkpoint.pt", "final_weights.pt"}
    original = (output / "resume.pt").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed wrapper must not call training")
    monkeypatch.setattr(wrapper, "train_long", forbidden)
    assert wrapper.run_one(root, 2340) == result
    assert (output / "resume.pt").read_bytes() == original
    (output / "checkpoint_step_25.pt").unlink()
    with pytest.raises(ValueError, match="restore its archive"):
        wrapper.run_one(root, 2340)

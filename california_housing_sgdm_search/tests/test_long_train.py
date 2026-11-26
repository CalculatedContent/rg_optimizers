"""Meaningful invariants for fixed-budget SGDM and exact parent continuation."""
from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from california_housing_benchmark import train as base
from california_housing_sgdm_search import long_train as trainer


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
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
    original_metadata = base.execution_metadata
    def metadata():
        value = original_metadata()
        value["github"].update(
            GITHUB_RUN_ID="45753214421", GITHUB_SHA=trainer.BASE_SOURCE_COMMIT,
            GITHUB_REPOSITORY="CalculatedContent/rg_optimizers")
        return value
    monkeypatch.setattr(base, "execution_metadata", metadata)
    return tmp_path


def plan(**changes):
    value = {
        "model": {"n_blocks": 2, "d_block": 12, "dropout": .37},
        "batch_size": 5, "total_updates": 31, "development_updates": 31,
        "eval_every": 5, "checkpoint_updates": [5, 14, 31],
        "measure_spectra": False, "grad_clip": 1.0,
    }
    value.update(changes)
    return value


def config():
    return {"optimizer": "sgdm", "lr": .003, "weight_decay": .001,
            "momentum": .99, "dampening": 0.0, "nesterov": False}


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def make_parent(data_dir, seed=3337, parent_step=20, patience=100):
    parent_plan = plan(total_updates=31, safety_max_updates=50, patience=patience,
                       checkpoint_updates=[parent_step, 31])
    base.train(parent_plan, config(), data_dir, data_dir / "parent", seed,
               phase="confirmation", stop_step=parent_step)
    path = data_dir / "parent" / f"checkpoint_step_{parent_step}.pt"
    payload = path.read_bytes()
    contract = {
        "repository": "CalculatedContent/rg_optimizers", "run_id": "45753214421",
        "launch_commit": trainer.BASE_SOURCE_COMMIT, "archive_commit": "fixture-archive",
        "checkpoint_step": parent_step,
        "checkpoints": {str(seed): {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "git_blob_sha": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
            "repository_path": f"parent/seed_{seed}/checkpoint_step_{parent_step}.pt"}},
    }
    return path, contract


def test_mid_epoch_resume_preserves_all_training_state(tiny_data):
    p, c = plan(), config()
    continuous = trainer.train_long(p, c, tiny_data, tiny_data / "continuous", 3337, "confirmation")
    trainer.train_long(p, c, tiny_data, tiny_data / "resumed", 3337, "confirmation", stop_step=5)
    resumed = trainer.train_long(p, c, tiny_data, tiny_data / "resumed", 3337, "confirmation")
    a, b = (read_checkpoint(tiny_data / path / "resume.pt") for path in ("continuous", "resumed"))
    assert trainer.checkpoint_fingerprints(a) == trainer.checkpoint_fingerprints(b)
    assert a["best_state"] == b["best_state"]
    assert base.model_hash(a["best_weights"]) == base.model_hash(b["best_weights"])
    assert continuous["epoch_history"] == resumed["epoch_history"]
    assert resumed["status"] == "complete"
    assert resumed["completed_updates"] == 31
    assert resumed["best_validation"]["selection_window"]["complete"]


def test_parent_continuation_and_its_resume_equal_uninterrupted_training(tiny_data, monkeypatch):
    trainer.train_long(plan(), config(), tiny_data, tiny_data / "continuous", 3337, "confirmation")
    parent_path, contract = make_parent(tiny_data)
    original_parent = parent_path.read_bytes()
    parent = read_checkpoint(parent_path)
    p = plan(continuation=contract)
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
    trainer.train_long(p, config(), tiny_data, tiny_data / "extension", 3337,
                       "continuation", parent_checkpoint=parent_path, stop_step=25)
    result = trainer.train_long(p, config(), tiny_data, tiny_data / "extension", 3337,
                                "continuation", parent_checkpoint=parent_path)
    assert len(calls) == 11  # No replay of the first 20 updates.
    assert parent_path.read_bytes() == original_parent
    expected = read_checkpoint(tiny_data / "continuous/resume.pt")
    actual = read_checkpoint(tiny_data / "extension/resume.pt")
    assert trainer.checkpoint_fingerprints(actual) == trainer.checkpoint_fingerprints(expected)
    fingerprints = trainer.checkpoint_fingerprints(parent)
    assert result["continuity"] == {"parent": fingerprints, "start": fingerprints, "verified": True}
    assert result["parent"]["identity"] == parent["identity"]
    assert result["parent_checkpoint_sha256"] == hashlib.sha256(original_parent).hexdigest()
    assert result["primary_estimand"] == "fixed_updates"
    assert result["start_step"] == 20 and result["new_updates"] == 11
    assert result["selection_window"] == {"start": 20, "end": 31, "observed_through": 31, "complete": True}
    assert result["best_validation"]["step"] >= 20
    assert result["primary_endpoint"]["step"] == 31


def test_continuation_does_not_inherit_a_frozen_old_best(tiny_data, monkeypatch):
    def constant_metrics(model, arrays, split, batch_size=32768):
        return {"rmse": 1.0, "mse": 1.0, "mae": 1.0, "r2": 0.0,
                "mape_percent": 1.0, "standardized_mse": 1.0}
    monkeypatch.setattr(base, "evaluate", constant_metrics)
    parent_path, contract = make_parent(tiny_data, patience=1)
    parent = read_checkpoint(parent_path)
    assert parent["result"]["primary_best_validation"]["step"] == 7
    result = trainer.train_long(plan(continuation=contract), config(), tiny_data,
                                tiny_data / "extension", 3337, "continuation",
                                parent_checkpoint=parent_path)
    assert result["best_validation"]["step"] == 20
    assert result["selection_window"]["start"] == 20
    assert result["parent"]["original_primary_best_validation"]["step"] == 7
    assert "continuation window" in result["best_validation"]["selection_scope"]


def test_development_never_reads_test_and_spectra_preserve_rng(tiny_data, monkeypatch):
    original_evaluate = base.evaluate
    def checked(model, arrays, split, batch_size=32768):
        assert split != "test", "Development must never measure test data"
        return original_evaluate(model, arrays, split, batch_size)
    monkeypatch.setattr(base, "evaluate", checked)
    spectral_calls = []
    def fake_spectra(model):
        digest = base.model_hash(model)
        spectral_calls.append(digest)
        # Fitting is allowed to use randomness, but it must not change training RNG.
        torch.rand(3)
        np.random.normal(size=3)
        random.random()
        return {"fixture": True, "model_hash": digest, "confidence_tail_alpha": 2.25}
    monkeypatch.setattr(trainer, "measure_spectra", fake_spectra)
    p = plan(measure_spectra=True)
    result = trainer.train_long(p, config(), tiny_data, tiny_data / "spectral", 3338, "development")
    control = trainer.train_long(plan(), config(), tiny_data, tiny_data / "control", 3338, "development")
    a, b = (read_checkpoint(tiny_data / path / "resume.pt") for path in ("spectral", "control"))
    assert trainer.checkpoint_fingerprints(a) == trainer.checkpoint_fingerprints(b)
    assert all(row["test"] is None for row in result["history"])
    assert result["best_validation"]["test"] is None
    assert len(spectral_calls) == 4  # Three declared fixed checkpoints + selected best.
    for row in result["history"]:
        assert (row["spectral"] is not None) == (row["step"] in {5, 14, 31})
        if row["spectral"] is not None:
            assert row["weights_sha256"] == row["spectral_weights_sha256"] == row["spectral"]["model_hash"]
    best = result["best_validation"]
    assert best["weights_sha256"] == best["spectral_weights_sha256"] == best["spectral"]["model_hash"]
    retained = read_checkpoint(tiny_data / "spectral/best_checkpoint.pt")
    assert base.model_hash(retained["model"]) == best["weights_sha256"] == result["best_weights_sha256"]
    assert control["best_validation"]["spectral"] is None


def test_selection_continues_after_sixteen_plateau_epochs(tiny_data, monkeypatch):
    counter = {"steps": 0}
    original_factory = base.make_optimizer
    def factory(model, cfg):
        optimizer = original_factory(model, cfg)
        original_step = optimizer.step
        def step(*args, **kwargs):
            value = original_step(*args, **kwargs)
            counter["steps"] += 1
            return value
        optimizer.step = step
        return optimizer
    def controlled_metrics(model, arrays, split, batch_size=32768):
        value = 2.0 if counter["steps"] < 40 else 0.25
        return {"rmse": value, "mse": value ** 2, "mae": value,
                "r2": 0.0, "mape_percent": value, "standardized_mse": value ** 2}
    monkeypatch.setattr(base, "make_optimizer", factory)
    monkeypatch.setattr(base, "evaluate", controlled_metrics)
    p = plan(total_updates=44, batch_size=16, checkpoint_updates=[20, 40, 44], patience=16)
    result = trainer.train_long(p, config(), tiny_data, tiny_data / "plateau", 3340, "confirmation")
    assert result["status"] == "complete" and result["completed_updates"] == 44
    assert len(result["epoch_history"]) == 22
    assert all(row["validation"]["rmse"] == 2 for row in result["epoch_history"][:19])
    assert result["best_validation"]["step"] == 40
    assert result["best_validation"]["validation"]["rmse"] == .25
    assert result["patience_stopping"] is False
    retained = read_checkpoint(tiny_data / "plateau/best_checkpoint.pt")
    at40 = read_checkpoint(tiny_data / "plateau/checkpoint_step_40.pt")
    assert base.model_hash(retained["model"]) == base.model_hash(at40["model"])


def test_complete_run_is_reused_without_constructing_an_optimizer(tiny_data, monkeypatch):
    output = tiny_data / "complete"
    result = trainer.train_long(plan(), config(), tiny_data, output, 3341, "confirmation")
    original_payload = (output / "resume.pt").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed run must not start another optimizer")
    monkeypatch.setattr(base, "make_optimizer", forbidden)
    repeated = trainer.train_long(plan(), config(), tiny_data, output, 3341, "confirmation")
    assert repeated == result
    assert (output / "resume.pt").read_bytes() == original_payload


def test_parent_recipe_and_bytes_cannot_be_changed(tiny_data):
    parent_path, contract = make_parent(tiny_data)
    p = plan(continuation=contract)
    changed = {**config(), "lr": .01}
    with pytest.raises(ValueError, match="optimizer settings"):
        trainer.train_long(p, changed, tiny_data, tiny_data / "wrong_recipe", 3337,
                           "continuation", parent_checkpoint=parent_path)
    corrupt = tiny_data / "corrupt.pt"
    corrupt.write_bytes(parent_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="file hash"):
        trainer.train_long(p, config(), tiny_data, tiny_data / "wrong_parent", 3337,
                           "continuation", parent_checkpoint=corrupt)

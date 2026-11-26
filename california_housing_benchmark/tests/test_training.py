"""Numerical invariants: nonzero decay, dropout replay, and frozen selection."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch import nn

from california_housing_benchmark import train as trainer
from california_housing_benchmark.optimizer import make_optimizer


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    generator = np.random.default_rng(80)
    arrays = {}
    for part, n in (("train", 32), ("validation", 9), ("test", 11)):
        x = generator.normal(size=(n, 8)).astype(np.float32)
        y = (2 + .2 * x[:, 0] - .1 * x[:, 1]).astype(np.float32)
        arrays[f"x_{part}"] = x
        arrays[f"y_{part}_original"] = y
    arrays["y_mean"] = np.float32(arrays["y_train_original"].mean())
    arrays["y_scale"] = np.float32(arrays["y_train_original"].std())
    for part in ("train", "validation", "test"):
        arrays[f"y_{part}"] = ((arrays[f"y_{part}_original"] - arrays["y_mean"])
                               / arrays["y_scale"]).astype(np.float32)
    (tmp_path / "manifest.json").write_text(json.dumps({"fixture": True}))
    monkeypatch.setattr(trainer, "load_data", lambda directory: (arrays, {"fixture": True}))
    return tmp_path


def plan(**overrides):
    value = {"model": {"n_blocks": 2, "d_block": 12, "dropout": .37},
             "batch_size": 5, "total_updates": 20, "safety_max_updates": 40,
             "patience": 100, "eval_every": 5, "checkpoint_updates": [5, 20],
             "measure_spectra": False, "grad_clip": 1.0,
             "calibration": {"max_updates": 20}}
    value.update(overrides)
    return value


def config(arm):
    return {"optimizer": arm, "lr": .003, "weight_decay": .2,
            "betas": [.9, .999], "reference_lr_ratio": 3.0}


@pytest.mark.parametrize("arm", ["sgdm", "adamw", "modified_adamw"])
def test_mid_epoch_dropout_resume_is_bitwise_exact(tiny_data, arm):
    p, c = plan(), config(arm)
    trainer.train(p, c, tiny_data, tiny_data / "continuous", 133, stop_step=20)
    trainer.train(p, c, tiny_data, tiny_data / "resumed", 133, stop_step=5)
    trainer.train(p, c, tiny_data, tiny_data / "resumed", 133, stop_step=20)
    a = torch.load(tiny_data / "continuous/resume.pt", weights_only=False)
    b = torch.load(tiny_data / "resumed/resume.pt", weights_only=False)
    assert a["completed_updates"] == b["completed_updates"] == 20
    assert a["minibatch_sha256"] == b["minibatch_sha256"]
    assert a["next_offset"] == b["next_offset"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert torch.equal(a["batch_generator"], b["batch_generator"])
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert a["result"]["epoch_history"] == b["result"]["epoch_history"]


def test_graft_preserves_direction_moments_and_nonzero_decay():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 1))
    native_model = deepcopy(model)
    native = make_optimizer(native_model, config("adamw"))
    graft = make_optimizer(model, config("modified_adamw"))
    for _ in range(3):
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        native_before = {name: p.detach().clone() for name, p in native_model.named_parameters()}
        for (name, p), (_, q) in zip(model.named_parameters(), native_model.named_parameters()):
            grad = torch.randn_like(p)
            p.grad, q.grad = grad.clone(), grad.clone()
        native.step()
        graft.step()
        for (name, p), (_, q) in zip(model.named_parameters(), native_model.named_parameters()):
            s, n = graft.optimizer.state[p], native.state[q]
            assert torch.equal(s["exp_avg"], n["exp_avg"])
            assert torch.equal(s["exp_avg_sq"], n["exp_avg_sq"])
            if p.ndim == 2:
                delta = p - before[name] * (1 - .003 * .2)
                native_delta = q - native_before[name] * (1 - .003 * .2)
                expected_norm = .009 * torch.linalg.vector_norm(s["exp_avg"]) / .1
                assert torch.allclose(delta.norm(), expected_norm, rtol=1e-5, atol=2e-7)
                assert torch.nn.functional.cosine_similarity(delta.flatten(), native_delta.flatten(), dim=0) > .99999
            else:
                assert torch.equal(p, q)


def test_first_patience_selection_is_frozen_during_extension(tiny_data, monkeypatch):
    def constant_metrics(model, arrays, split, batch_size=32768):
        return {"rmse": 1.0, "mse": 1.0, "mae": 1.0, "r2": 0.0,
                "mape_percent": 1.0, "standardized_mse": 1.0}
    monkeypatch.setattr(trainer, "evaluate", constant_metrics)
    result = trainer.train(plan(patience=1), config("adamw"), tiny_data,
                           tiny_data / "selection", 19)
    assert result["paper_stopping"]["step"] == 14  # 7 batches/epoch, first bad epoch is epoch2.
    assert result["primary_best_validation"]["step"] == 7
    assert result["completed_updates"] == 20
    assert result["status"] == "complete"
    checkpoint = torch.load(tiny_data / "selection/best_checkpoint.pt", weights_only=False)
    trainer.train(plan(patience=1), config("adamw"), tiny_data,
                  tiny_data / "prefix", 19, stop_step=7)
    prefix = torch.load(tiny_data / "prefix/resume.pt", weights_only=False)
    assert all(torch.equal(checkpoint["model"][key], prefix["model"][key]) for key in prefix["model"])


def test_calibration_never_reads_test_and_marks_cap(tiny_data, monkeypatch):
    original = trainer.evaluate
    def checked(model, arrays, split, batch_size=32768):
        assert split != "test"
        return original(model, arrays, split, batch_size)
    monkeypatch.setattr(trainer, "evaluate", checked)
    result = trainer.train(plan(), config("adamw"), tiny_data,
                           tiny_data / "calibration", 22, phase="calibration")
    assert result["calibration_capped"]
    assert result["primary_best_validation"]["test"] is None
    assert all(row["test"] is None and row["spectral"] is None for row in result["history"])


def test_resume_rejects_recipe_change(tiny_data):
    trainer.train(plan(), config("sgdm"), tiny_data, tiny_data / "identity", 37, stop_step=5)
    changed = config("sgdm") | {"lr": .004}
    with pytest.raises(ValueError, match="identity differs"):
        trainer.train(plan(), changed, tiny_data, tiny_data / "identity", 37, stop_step=20)


def test_native_initialization_and_update_match_vendored_upstream_model():
    """The published Model AST independently checks the recovered CPU port."""
    import ast
    from pathlib import Path

    import delu
    import tabm

    source_path = Path(__file__).resolve().parents[1] / "reference/upstream/bin/ffn.py"
    assert source_path.is_file(), f"Required frozen author source is absent: {source_path}"
    parsed = ast.parse(source_path.read_text())
    model_node = next(node for node in parsed.body
                      if isinstance(node, ast.ClassDef) and node.name == "Model")
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + [model_node],
        type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "tabm": tabm}
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    author_model = namespace["Model"]
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    backbone = {"n_blocks": 2, "d_block": 784, "dropout": 0.3690073279296028}
    delu.random.seed(701)
    reference = author_model(n_num_features=8, cat_cardinalities=[], n_classes=None,
                             type="mlp", num_embeddings_bins=None, backbone=backbone)
    delu.random.seed(701)
    ported = trainer.MLP(backbone)
    assert sum(p.numel() for p in ported.parameters()) == 623281
    assert reference.state_dict().keys() == ported.state_dict().keys()
    assert all(torch.equal(value, ported.state_dict()[key])
               for key, value in reference.state_dict().items())

    lr, wd = 0.0002722214565241805, 0.4555843429087667
    reference_optimizer = torch.optim.AdamW([
        {"params": [p for name, p in reference.named_parameters() if name.endswith("bias")],
         "weight_decay": 0.0},
        {"params": [p for name, p in reference.named_parameters() if not name.endswith("bias")],
         "weight_decay": wd},
    ], lr=lr)
    ported_optimizer = make_optimizer(ported, {
        "optimizer": "adamw", "lr": lr, "weight_decay": wd,
        "betas": [.9, .999], "eps": 1e-8})
    fixture_generator = torch.Generator().manual_seed(701)
    x = torch.randn(32, 8, generator=fixture_generator)
    y = torch.randn(32, generator=fixture_generator)
    dropout_state = torch.get_rng_state()

    reference.train()
    torch.set_rng_state(dropout_state)
    reference_prediction = reference(x, None).squeeze(-1)
    reference_loss = nn.functional.mse_loss(reference_prediction, y)
    reference_optimizer.zero_grad()
    reference_loss.backward()
    nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    reference_optimizer.step()

    ported.train()
    torch.set_rng_state(dropout_state)
    ported_prediction = ported(x)
    ported_loss = nn.functional.mse_loss(ported_prediction, y)
    ported_optimizer.zero_grad()
    ported_loss.backward()
    nn.utils.clip_grad_norm_(ported.parameters(), 1.0)
    ported_optimizer.step()
    assert torch.equal(reference_prediction, ported_prediction)
    assert torch.equal(reference_loss, ported_loss)
    assert all(torch.equal(value, ported.state_dict()[key])
               for key, value in reference.state_dict().items())

"""Numerical/data/recovery invariants, all with explicit tiny smoke dimensions."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
from ..checkpoint import load_checkpoint, recover_checkpoint_aliases, save_checkpoint, sha256_file
from ..config import smoke_config, TrainingConfig, BENCHMARK_VALIDATION_TOKENS
from ..dataset import TokenStream, required_names
from ..train import Runtime, make_model, make_optimizer, parse_args, run, target_met, train_step, evaluate


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def test_stream_prefix_does_not_depend_on_batch_and_crosses_shards():
    arrays = [np.arange(7), np.arange(7, 17)]
    one = TokenStream(arrays)
    two = TokenStream(arrays)
    x1, y1 = one.next_batch(2, 8)
    pieces = [two.next_batch(1, 4) for _ in range(4)]
    assert torch.equal(x1.flatten(), torch.cat([pair[0].flatten() for pair in pieces]))
    assert torch.equal(y1.flatten(), torch.cat([pair[1].flatten() for pair in pieces]))
    assert one.state_dict() == two.state_dict()
    x2, y2 = one.next_batch(1, 4)
    assert x2.tolist() == [[16, 0, 1, 2]]
    assert y2.tolist() == [[0, 1, 2, 3]]
    saved = one.state_dict()
    restored = TokenStream(arrays)
    restored.load_state_dict(saved)
    assert torch.equal(one.next_batch(1, 4)[0], restored.next_batch(1, 4)[0])
    with pytest.raises(ValueError, match="consumed"):
        restored.load_state_dict({**saved, "consumed": saved["consumed"] + 1})


def test_shard_plan_includes_next_target_token():
    manifest = {"files": {"fineweb_val_000000.bin": {"size": 1044},
                           "fineweb_train_000001.bin": {"size": 1044},
                           "fineweb_train_000002.bin": {"size": 1044}}}
    assert len(required_names(manifest, 9)) == 2
    assert len(required_names(manifest, 10)) == 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accumulation_is_one_global_mean_grafted_adamw_update(dtype):
    config = replace(smoke_config(), beta1=0.9, weight_decay=0.00003)
    small = replace(config, microbatch_sequences=1)
    rt = Runtime("cpu")
    first, second = make_model(config, rt).to(dtype=dtype), make_model(small, rt).to(dtype=dtype)
    opt1, opt2 = make_optimizer(first, config), make_optimizer(second, small)
    assert isinstance(opt1, torch.optim.AdamW)
    assert opt1.defaults["betas"] == (config.beta1, config.beta2)
    assert len(opt1.grafted_parameter_names) == 4 * config.n_layer
    assert len({id(p) for group in opt1.param_groups for p in group["params"]}) == len(list(first.parameters()))
    arrays = [np.arange(1024) % config.vocab_size]
    train_step(first, opt1, TokenStream(arrays), config, rt, 0)
    train_step(second, opt2, TokenStream(arrays), small, rt, 0)
    # Different FP32 reduction orders perturb near-zero embedding gradients;
    # Adam's epsilon denominator can amplify that roundoff in an update. Check
    # the mean gradients/moments directly and also the higher precision result.
    gradient_tolerance = {"atol": 3e-8, "rtol": 2e-5} if dtype == torch.float32 else {"atol": 1e-14, "rtol": 1e-11}
    weight_tolerance = {"atol": 1e-6, "rtol": 2e-5} if dtype == torch.float32 else {"atol": 1e-12, "rtol": 1e-11}
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a.grad, b.grad, **gradient_tolerance)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(opt1.state[a][key], opt2.state[b][key], **gradient_tolerance)
        torch.testing.assert_close(a, b, **weight_tolerance)


def test_validation_measures_exact_offset_and_count():
    config = replace(smoke_config(), validation_offset=32)
    rt = Runtime("cpu")
    model = make_model(config, rt)
    tokens = np.arange(500) % config.vocab_size
    result = evaluate(model, tokens, config, rt, 0)
    total = 0.0
    model.eval()
    with torch.no_grad():
        for offset in range(32, 160, 64):
            buf = torch.tensor(tokens[offset:offset + 65])
            _, loss = model(buf[:-1].reshape(2, 32), buf[1:].reshape(2, 32))
            total += float(loss) / 2
    assert result["evaluation_tokens"] == 128
    assert result["validation_offset"] == 32
    assert result["val_nll"] == pytest.approx(total, abs=1e-6)
    assert result["target_met"] is False


def test_target_requires_production_and_exact_official_evaluation():
    row = {"validation_kind": "benchmark", "validation_offset": 0,
           "evaluation_tokens": BENCHMARK_VALIDATION_TOKENS,
           "full_benchmark_evaluation": True, "val_nll": 3.28}
    assert target_met(TrainingConfig(validation_kind="benchmark", validation_tokens=BENCHMARK_VALIDATION_TOKENS,
                                     validation_offset=0), row)
    assert not target_met(smoke_config(), row)
    for key, value in (("validation_kind", "tuning"), ("validation_offset", 1),
                       ("evaluation_tokens", 1024), ("val_nll", float("nan")), ("val_nll", 3.280001)):
        assert not target_met(TrainingConfig(), {**row, key: value})


def _args(tmp_path, config, name, *extra):
    path = tmp_path / "config.json"
    config.write_json(path)
    return parse_args(["--config", str(path), "--data-dir", str(tmp_path / "data"),
                       "--run-dir", str(tmp_path / name), "--device", "cpu", "--no-spectral",
                       "--cpu-threads", "1", *extra])


def test_full_resume_matches_uninterrupted_weights_moments_and_step_counters(tmp_path):
    config = replace(smoke_config(), budget_tokens=256)
    complete = run(_args(tmp_path, config, "complete"))
    paused = run(_args(tmp_path, config, "resumed", "--max-steps-this-session", "2"))
    assert paused["status"] == "paused" and paused["step"] == 2
    resumed = run(_args(tmp_path, config, "resumed", "--resume"))
    assert complete["step"] == resumed["step"] == 4
    first = load_checkpoint(tmp_path / "complete/checkpoints/latest.pt", config)
    second = load_checkpoint(tmp_path / "resumed/checkpoints/latest.pt", config)
    for name in first["model"]:
        assert torch.equal(first["model"][name], second["model"][name]), name
    for index, state in first["optimizer"]["state"].items():
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(state[key], second["optimizer"]["state"][index][key])
    assert first["data_cursor"] == second["data_cursor"]
    assert torch.equal(first["rng"]["torch"], second["rng"]["torch"])
    assert first["next_lr_factor"] == second["next_lr_factor"] == 0
    assert first["validation"]["val_nll"] == second["validation"]["val_nll"]
    assert (tmp_path / "resumed/checkpoints/best.pt").exists()
    assert (tmp_path / "resumed/checkpoints/previous.pt").exists()
    with pytest.raises(ValueError, match="configuration changed"):
        load_checkpoint(tmp_path / "complete/checkpoints/latest.pt", replace(config, peak_lr=0.1))


def test_checkpoint_corruption_fails_closed(tmp_path):
    config = smoke_config()
    run(_args(tmp_path, config, "corrupt"))
    path = tmp_path / "corrupt/checkpoints/latest.pt"
    with path.open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        load_checkpoint(path, config)


def test_existing_run_requires_explicit_resume(tmp_path):
    config = smoke_config()
    args = _args(tmp_path, config, "existing")
    run(args)
    with pytest.raises(FileExistsError, match="already contains"):
        run(args)


def test_first_target_alias_is_retained_and_partial_aliases_are_repaired(tmp_path):
    payload = {"config_sha256": "test-config", "run_id": "test-run", "step": 2, "tokens_seen": 128,
               "model": {"weight": torch.ones(4)}, "best_validation": {"step": 2},
               "first_target_validation": {"step": 2}}
    save_checkpoint(tmp_path, payload, best=True, first_target=True)
    original = sha256_file(tmp_path / "checkpoints/first_target.pt")
    (tmp_path / "checkpoints/best.pt").unlink()
    (tmp_path / "checkpoints/first_target.json").unlink()
    assert set(recover_checkpoint_aliases(tmp_path, payload)) == {"best", "first_target"}
    assert recover_checkpoint_aliases(tmp_path, payload) == []
    assert sha256_file(tmp_path / "checkpoints/first_target.pt") == original
    later = {**payload, "step": 4, "tokens_seen": 256, "model": {"weight": torch.zeros(4)},
             "best_validation": {"step": 4}}
    save_checkpoint(tmp_path, later, best=True)
    assert sha256_file(tmp_path / "checkpoints/first_target.pt") == original
    assert json.loads((tmp_path / "checkpoints/first_target.json").read_text())["step"] == 2

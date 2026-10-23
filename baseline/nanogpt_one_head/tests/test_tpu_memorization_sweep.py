from copy import deepcopy
import csv
from pathlib import Path

import pytest
import torch

from rg_nanogpt_one_head.canary_eval_batched import (
    EVALUATOR_VERSION, FIELDS, evaluate_batched, score_batch, write_rows,
)
from rg_nanogpt_one_head.tpu_memorization_sweep import (
    LOADS, SEEDS, REVISION, build_tasks, check_root, child_environment,
    exclusive_lock, freeze_plan, task_config, validate_study_config,
)
from rg_nanogpt_one_head.model import GPT, GPTConfig
from rg_nanogpt_one_head.random_canaries import RandomCanaryExperiment


def cfg():
    return {
        "model": dict(vocab_size=41, block_size=32, n_layer=1, n_head=1,
                      n_embd=16, dropout=0.0, bias=False, tie_weights=True),
        "training": dict(batch_size=4, grad_accum_steps=8),
        "memorization": dict(enabled=True, data_seed=20260925, doses=[0, 1, 4],
                             canaries_per_dose=2, prefix_tokens=8, suffix_tokens=4,
                             acquisition_fraction=0.5, harmful_load_fraction=0.01,
                             harmful_load_dose=64),
    }


def full_cfg():
    return {
        "dataset": dict(name="HuggingFaceFW/fineweb-edu", config="sample-10BT",
                        revision=REVISION, tokenizer="gpt2", train_tokens=80000000,
                        val_tokens=1000000, test_tokens=1000000),
        "model": dict(vocab_size=50257, block_size=256, n_layer=1, n_head=1,
                      n_embd=128, dropout=0.0, bias=False, tie_weights=True),
        "training": dict(batch_size=4, grad_accum_steps=8, target_epochs=4.0,
                         epoch_interval=0.25, eval_interval_steps=500,
                         checkpoint_interval_steps=500, eval_batches=64),
        "memorization": dict(enabled=True, data_seed=20260925, doses=[0, 1, 4, 16, 64],
                             canaries_per_dose=8, prefix_tokens=64, suffix_tokens=32,
                             acquisition_fraction=0.5, harmful_load_fraction=0.001,
                             harmful_load_dose=64),
        "weightwatcher": dict(enabled=True, ERG=True, randomize=True, strict=True,
                              fix_fingers="clip_xmax", max_fingers=10, require_raw_alpha=True),
        "optimizer_profiles": {"muon_clip": {"family": "muon_clip"}},
        "runtime": {"matmul_precision": "highest"},
    }


def test_default_plan_is_full_25_runs_using_all_four_chips():
    tasks = build_tasks()
    assert len(tasks) == 25
    assert len({t.key for t in tasks}) == 25
    assert {t.chip for t in tasks} == {0, 1, 2, 3}
    assert all(t.optimizer == "muon_clip" for t in tasks)
    for seed in SEEDS:
        assert {t.load for t in tasks if t.seed == seed} == set(LOADS)
    assert max(sum(t.chip == c for t in tasks) for c in range(4)) == 7


def test_explicit_adamw_extension_keeps_muonclip():
    tasks = build_tasks(optimizers=("muon_clip", "adamw"))
    assert len(tasks) == 50
    assert len({t.key for t in tasks}) == 50


@pytest.mark.parametrize("kwargs", [
    {"chips": [0, 0]}, {"chips": [4]}, {"chips": []}, {"seeds": [1, 1]},
    {"loads": ["64pct"]}, {"optimizers": ["muon"]}, {"seeds": [-1]},
])
def test_bad_plan_refused(kwargs):
    with pytest.raises(ValueError):
        build_tasks(**kwargs)


def test_full_protocol_validation_and_smoke_rejection():
    spec = full_cfg()
    validate_study_config(spec, 0.001)
    for section, key, value in [("dataset", "train_tokens", 4000000),
                                ("training", "target_epochs", 0.2),
                                ("memorization", "enabled", False),
                                ("weightwatcher", "enabled", False),
                                ("model", "n_head", 4)]:
        changed = deepcopy(spec)
        changed[section][key] = value
        with pytest.raises(ValueError):
            validate_study_config(changed, 0.001)


def test_task_config_preserves_training_and_injection():
    original = full_cfg()
    plan = {"configs": {"0pct": original}, "source": {"commit": "abc"},
            "canary_batch_size": 40, "seeds": list(SEEDS), "hardware_block": "v5e-a"}
    task = build_tasks()[0]
    before = deepcopy(plan)
    changed = task_config(plan, task)
    assert plan == before
    for key in ("dataset", "model", "training", "memorization", "optimizer_profiles", "weightwatcher"):
        assert changed[key] == original[key]
    assert changed["runtime"]["tpu_memorization_execution"]["canary_evaluator"] == EVALUATOR_VERSION


def test_chip_environment_isolated_before_import(tmp_path):
    base = {"RANK": "3", "WORLD_SIZE": "4", "PJRT_LOCAL_PROCESS_COUNT": "4",
            "PYTHONPATH": "/old/checkout", "OTHER": "preserved"}
    env = child_environment(base, 2, tmp_path, Path("/repo"), "v5e-test")
    assert env["TPU_VISIBLE_CHIPS"] == "2"
    assert env["TPU_PROCESS_BOUNDS"] == "1,1,1"
    assert env["TPU_CHIPS_PER_PROCESS_BOUNDS"] == "1,1,1"
    assert "WORLD_SIZE" not in env and "RANK" not in env
    assert env["OTHER"] == "preserved" and base["RANK"] == "3"
    assert env["PYTHONPATH"] == "/repo/baseline/nanogpt_one_head/src"
    with pytest.raises(ValueError):
        child_environment({"XLA_USE_BF16": "1"}, 0, tmp_path, Path("/repo"), "v5e-test")


def test_plan_freeze_refuses_changed_conditions_without_overwrite(tmp_path):
    path = tmp_path / "plan.json"
    freeze_plan(path, {"version": 1})
    data = path.read_bytes()
    freeze_plan(path, {"version": 1})
    with pytest.raises(ValueError):
        freeze_plan(path, {"version": 2})
    assert path.read_bytes() == data


def test_legacy_root_and_source_output_refused(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    with pytest.raises(ValueError):
        check_root(code / "results", code, True)
    root = tmp_path / "old_mac_run"
    (root / "load_0pct").mkdir(parents=True)
    with pytest.raises(ValueError):
        check_root(root, code, True)
    assert (root / "load_0pct").exists()


def test_second_writer_is_blocked(tmp_path):
    with exclusive_lock(tmp_path / "lock"):
        with pytest.raises(RuntimeError):
            with exclusive_lock(tmp_path / "lock"):
                pytest.fail("acquired an already-held lock")


@pytest.mark.parametrize("seed", [1337, 2027, 4099])
def test_batched_metrics_match_historical_per_canary_scorer(tmp_path, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    spec = cfg()
    model = GPT(GPTConfig(**spec["model"])).eval()
    experiment = RandomCanaryExperiment(spec, seed=seed, total_steps=100, run_dir=tmp_path)
    expected = torch.tensor([experiment._score_one(model, c["tokens"], torch.device("cpu"))
                             for c in experiment.canaries])
    tokens = torch.stack([c["tokens"] for c in experiment.canaries])
    actual = score_batch(model, tokens, prefix=8, suffix=4, device=torch.device("cpu"))
    torch.testing.assert_close(actual[:, 0], expected[:, 0], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual[:, 1:], expected[:, 1:], rtol=0, atol=0)


def test_batch_padding_restores_model_mode_and_rng_and_preserves_inventory(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    spec = cfg()
    model = GPT(GPTConfig(**spec["model"])).train()
    experiment = RandomCanaryExperiment(spec, seed=1337, total_steps=100, run_dir=tmp_path)
    schedule = dict(experiment.schedule)
    inventory = [(c["id"], c["dose"], c["tokens"].clone()) for c in experiment.canaries]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    shapes = []
    original_hidden = model.hidden_states
    def hidden(tokens):
        shapes.append(tuple(tokens.shape))
        return original_hidden(tokens)
    model.hidden_states = hidden
    summary = evaluate_batched(experiment, model, device=torch.device("cpu"),
                               step=0, epoch=0.0, batch_size=4)
    assert model.training
    assert len(shapes) == 2 * (1 + spec["memorization"]["suffix_tokens"])
    assert set(shapes) == {(4, 11), (4, 12)}
    assert set(summary) == {"exact_match", "token_accuracy", "zero_exact_match"}
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(model.state_dict()[key], state[key], rtol=0, atol=0)
    assert experiment.schedule.keys() == schedule.keys()
    for c, (cid, dose, tokens) in zip(experiment.canaries, inventory):
        assert (c["id"], c["dose"]) == (cid, dose)
        torch.testing.assert_close(c["tokens"], tokens)
    rows = list(csv.DictReader(experiment.csv_path.open()))
    assert len(rows) == len(experiment.canaries)
    assert {r["id"] for r in rows} == {c["id"] for c in experiment.canaries}
    assert tuple(rows[0]) == FIELDS


def test_replayed_canary_suffix_is_not_duplicated(tmp_path):
    path = tmp_path / "random_canary_metrics.csv"
    def rows(step):
        return [dict(zip(FIELDS, (step, step / 100, "dose0_canary0", 0, 3.0, 0.0, 0.0, 0.0)))]
    write_rows(path, rows(10))
    write_rows(path, rows(20))
    write_rows(path, rows(15))
    write_rows(path, rows(15))
    result = list(csv.DictReader(path.open()))
    assert [int(r["step"]) for r in result] == [10, 15]


def test_mode_restored_after_scoring_failure(tmp_path):
    spec = cfg()
    model = GPT(GPTConfig(**spec["model"])).train()
    experiment = RandomCanaryExperiment(spec, seed=1337, total_steps=100, run_dir=tmp_path)
    experiment.prefix = 100
    with pytest.raises(ValueError):
        evaluate_batched(experiment, model, device=torch.device("cpu"), step=0, epoch=0.0)
    assert model.training
    assert not experiment.csv_path.exists()


def test_full_gpt2_vocabulary_canary_parity(tmp_path):
    """The actual 50,257-vocabulary/128-width model, not just a toy shape."""
    torch.set_num_threads(2)
    torch.manual_seed(1337)
    model = GPT(GPTConfig()).eval()
    spec = {"model": vars(model.cfg), "training": {"batch_size": 4, "grad_accum_steps": 8},
            "memorization": {"data_seed": 20260925, "doses": [0, 64],
                             "canaries_per_dose": 1, "prefix_tokens": 64,
                             "suffix_tokens": 32, "acquisition_fraction": 0.5,
                             "harmful_load_fraction": 0.001}}
    experiment = RandomCanaryExperiment(spec, seed=1337, total_steps=39063, run_dir=tmp_path)
    reference = torch.tensor([experiment._score_one(model, c["tokens"], torch.device("cpu"))
                              for c in experiment.canaries])
    actual = score_batch(model, torch.stack([c["tokens"] for c in experiment.canaries]),
                         prefix=64, suffix=32, device=torch.device("cpu"))
    torch.testing.assert_close(actual[:, 0], reference[:, 0], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual[:, 1:], reference[:, 1:], rtol=0, atol=0)

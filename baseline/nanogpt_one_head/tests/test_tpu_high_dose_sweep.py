from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rg_nanogpt_one_head import tpu_high_dose_sweep as high

CODE = Path(__file__).resolve().parents[3]


def config():
    return high.load_study(CODE)


def reference():
    high.load_study(CODE)  # registers the existing MuonClip extension
    from rg_nanogpt_one_head.config import load_config
    return load_config(CODE / high.CONFIG_DIR / "harmful_memorization_0pct.yaml")


def test_full_study_keeps_model_steps_and_raw_primary():
    cfg = config()
    from rg_nanogpt_one_head.config import max_steps, epoch_step_map, tokens_per_step
    assert cfg["model"]["vocab_size"] == 50257
    assert tokens_per_step(cfg) == 8192
    assert max_steps(cfg, 80000000) == 39063
    assert len(epoch_step_map(cfg, 80000000)) == 33
    assert cfg["weightwatcher"]["require_raw_alpha"] is True
    assert cfg["weightwatcher"]["fix_fingers"] == "clip_xmax"  # raw also retained


def test_five_jobs_and_four_independent_chips():
    jobs = high.tasks()
    assert len(jobs) == 5
    assert Counter(t.chip for t in jobs) == {0: 2, 1: 1, 2: 1, 3: 1}
    assert [t.seed for t in jobs] == high.SEEDS
    assert {t.load for t in jobs} == {"highdose"}
    assert {t.optimizer for t in jobs} == {"muon_clip"}


@pytest.mark.parametrize("section,key,value", [
    ("model", "n_embd", 256), ("training", "batch_size", 8),
    ("training", "target_epochs", 2), ("training", "epoch_interval", 0.25),
    ("memorization", "doses", [0, 64]),
    ("memorization", "acquisition_fraction", 1.0),
    ("memorization", "harmful_load_fraction", 0.1),
    ("weightwatcher", "require_raw_alpha", False),
    ("runtime", "deterministic_algorithms", False),
])
def test_unregistered_changes_are_rejected(section, key, value):
    cfg = config()
    cfg[section][key] = value
    with pytest.raises(ValueError, match="unregistered"):
        high.validate_study(cfg, reference())


@pytest.mark.parametrize("seed", high.SEEDS)
def test_exact_doses_and_no_unexposed_injection(tmp_path, seed):
    from rg_nanogpt_one_head.random_canaries import RandomCanaryExperiment
    experiment = RandomCanaryExperiment(config(), seed=seed, total_steps=39063, run_dir=tmp_path)
    counts = Counter(item["id"] for item in experiment.schedule.values())
    assert len(experiment.canaries) == 48
    assert len(experiment.schedule) == 15872
    for canary in experiment.canaries:
        assert counts[canary["id"]] == canary["dose"]
    assert all(0 <= step < 19532 and 0 <= micro < 8 and 0 <= row < 4
               for step, micro, row in experiment.schedule)
    manifest = json.loads((tmp_path / "random_canary_manifest.json").read_text())
    assert manifest["harmful_bank_size"] == 0
    assert manifest["harmful_presentations"] == 0


def previous(root, n=25):
    tasks = [high.base.asdict(t) for t in high.base.build_tasks()]
    (root / "receipts").mkdir(parents=True)
    (root / "tpu_sweep_plan.json").write_text(json.dumps({"tasks": tasks}))
    for t in tasks[:n]:
        done = {"completed": True, "optimizer_steps": 39063, "seed": t["seed"],
                "optimizer": t["optimizer"], "fingerprint": f"fp-{t['index']}"}
        p = root / f"load_{t['load']}" / "results" / t["optimizer"] / f"seed_{t['seed']}"
        p.mkdir(parents=True)
        (p / "run_complete.json").write_text(json.dumps(done))
        (root / "receipts" / f"task_{t['index']:03d}.json").write_text(
            json.dumps({"task": t, "completion": done}))
    return tasks


def test_previous_campaign_guard_is_read_only(tmp_path):
    previous(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    high.require_previous_complete(tmp_path)
    after = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_pending_old_job_blocks_launch(tmp_path):
    previous(tmp_path, n=24)
    with pytest.raises(RuntimeError, match="24/25"):
        high.require_previous_complete(tmp_path)


def test_receipt_mismatch_blocks_launch(tmp_path):
    previous(tmp_path)
    p = tmp_path / "receipts/task_000.json"
    receipt = json.loads(p.read_text())
    receipt["completion"]["fingerprint"] = "not-the-checkpoint"
    p.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="24/25"):
        high.require_previous_complete(tmp_path)


def test_plan_can_be_consumed_by_qualified_worker(tmp_path, monkeypatch):
    prior = tmp_path / "old"
    previous(prior)
    from rg_nanogpt_one_head import data
    monkeypatch.setattr(data, "validate_prepared_data", lambda *args: {"splits": {"train": 80000000}})
    monkeypatch.setattr(high.base, "source_info", lambda path: {"commit": "unit-test"})
    monkeypatch.setattr(high.base, "check_root", lambda *args: None)
    args = SimpleNamespace(code=CODE, root=tmp_path / "new", data_root=tmp_path / "data",
                           previous_root=prior, hardware_block="test-block")
    plan = high.make(args)
    assert plan["canary_batch_size"] == 48
    assert plan["scientific_hypothesis"]["primary_spectral_variable"] == "alpha_raw"
    for task in high.tasks():
        cfg = high.base.task_config(plan, task)
        assert cfg["memorization"]["doses"] == high.DOSES
        assert cfg["runtime"]["tpu_memorization_execution"]["canary_batch_size"] == 48
        assert cfg["training"]["target_epochs"] == 4.0


def test_original_executor_dispatches_highdose_plan_on_registered_chips(tmp_path, monkeypatch):
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            self.pid = 1000 + len(calls)
        def wait(self):
            return 0
    monkeypatch.setattr(high.base.subprocess, "Popen", Process)
    monkeypatch.setattr(high.base, "exclusive_lock", lambda path: nullcontext())
    plan = {"root": str(tmp_path), "code": str(CODE), "chips": high.CHIPS,
            "hardware_block": "cpu-contract-test", "tasks": [high.base.asdict(t) for t in high.tasks()]}
    p = tmp_path / "tpu_sweep_plan.json"
    p.write_text(json.dumps(plan))
    assert high.base.run_sweep(p, 0) == 0
    assert len(calls) == 5
    assert Counter(v[1]["env"]["TPU_VISIBLE_CHIPS"] for v in calls) == {"0": 2, "1": 1, "2": 1, "3": 1}
    assert all("rg_nanogpt_one_head.tpu_memorization_sweep" in command for command, _ in calls)


def test_batched_48_scorer_against_scalar_reference(tmp_path):
    from rg_nanogpt_one_head.model import GPT, GPTConfig
    from rg_nanogpt_one_head.random_canaries import RandomCanaryExperiment
    from rg_nanogpt_one_head.canary_eval_batched import score_batch
    torch.set_num_threads(1)
    torch.manual_seed(1337)
    cfg = config()
    # Small CPU model tests padding/evaluator semantics, not TPU throughput.
    cfg["model"].update(vocab_size=97, n_embd=16)
    model = GPT(GPTConfig(**cfg["model"])).eval()
    experiment = RandomCanaryExperiment(cfg, seed=1337, total_steps=39063, run_dir=tmp_path)
    tokens = torch.stack([item["tokens"] for item in experiment.canaries])
    values = score_batch(model, tokens, prefix=64, suffix=32, device=torch.device("cpu"))
    assert values.shape == (48, 4)
    for i in (0, 7, 40, 47):
        scalar = experiment._score_one(model, tokens[i], torch.device("cpu"))
        assert values[i].tolist() == pytest.approx(scalar, abs=1e-5)

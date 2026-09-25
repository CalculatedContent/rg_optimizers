from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from test_one_head import tiny_config, write_tiny_data
from rg_nanogpt_one_head.engine import run_one


SCRIPTS = Path(__file__).resolve().parents[2] / "experiments" / "nanogpt_one_head_2026_08_21_baseline" / "scripts"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("visible_canary_launch", SCRIPTS / "start_canaries.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_canary_and_baseline_configs_allowed_but_changes_rejected(launcher, tmp_path):
    runner = launcher.campaign
    runner._validate_protocol_config(runner.DEFAULT_CONFIG)
    cfg = runner._validate_protocol_config(launcher.CONFIG)
    assert cfg["memorization"]["enabled"] is True
    cfg["memorization"]["doses"] = [0, 2]
    modified = tmp_path / "modified.yaml"
    modified.write_text(yaml.safe_dump(cfg))
    with pytest.raises(runner.CampaignError, match="frozen dated campaign"):
        runner._validate_protocol_config(modified)


def test_quiet_child_has_heartbeat_and_nonzero_exit_is_failure(launcher, capsys):
    log = io.StringIO()
    with pytest.raises(launcher.campaign.CampaignError, match="exit_code=7"):
        launcher._run_stage(
            "test", [sys.executable, "-u", "-c", "import time; time.sleep(.12); print('child error'); raise SystemExit(7)"],
            environment=os.environ.copy(), log=log, heartbeat=.03,
        )
    output = capsys.readouterr().out
    assert "[heartbeat] stage=test" in output
    assert "NOT proof of a completed training update" in output
    assert "child error" in output and "exit_code=7" in output
    assert output == log.getvalue()


def test_doctor_failure_never_prepares_or_trains(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.campaign, "_require_clean_git", lambda: {})
    calls = []
    def fail(name, *args, **kwargs):
        calls.append(name)
        raise launcher.campaign.CampaignError("test doctor failure")
    monkeypatch.setattr(launcher, "_run_stage", fail)
    with pytest.raises(launcher.campaign.CampaignError, match="doctor failure"):
        launcher._workflow(tmp_path, {}, SimpleNamespace(reuse_root=None), io.StringIO())
    assert calls == ["doctor"]


def test_workflow_selects_canary_config_and_both_arms(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.campaign, "_require_clean_git", lambda: {})
    calls = []
    monkeypatch.setattr(launcher, "_run_stage", lambda name, cmd, **kw: calls.append((name, cmd, kw)))
    launcher._workflow(tmp_path, {}, SimpleNamespace(reuse_root=None), io.StringIO())
    assert [name for name, _, _ in calls] == ["doctor", "prepare", "training", "verify"]
    for _, cmd, kw in calls:
        assert cmd[cmd.index("--config") + 1] == str(launcher.CONFIG)
        assert kw["environment"]["RG_NANOGPT_LOG_EVERY_STEP"] == "1"
    train = calls[2][1]
    assert train[train.index("--optimizers") + 1] == "adamw,muon_clip"
    assert train[train.index("--seeds") + 1] == "1337"


def test_existing_root_is_preserved(launcher, tmp_path):
    marker = tmp_path / "checkpoint_latest.pt"
    marker.write_bytes(b"preserve this")
    assert launcher.main(["--root", str(tmp_path)]) == 2
    assert marker.read_bytes() == b"preserve this"
    assert not (tmp_path / "console.log").exists()


def test_reuse_refuses_active_preparation(launcher, tmp_path):
    source = tmp_path / "old"
    lock = launcher.campaign._acquire_exclusive_lock(source / "logs" / "prepare.log.lock")
    try:
        with pytest.raises(launcher.campaign.CampaignError, match="another campaign process"):
            launcher._reuse_data(source, {}, {}, io.StringIO())
    finally:
        launcher.campaign._release_exclusive_lock(lock)


def test_every_completed_step_logged_without_changing_weights(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = tiny_config("adamw")
    cfg["training"]["grad_accum_steps"] = 2
    cfg["training"]["target_epochs"] = .04  # three optimizer updates
    cfg["training"]["eval_interval_steps"] = 500
    cfg["training"]["checkpoint_interval_steps"] = 500
    cfg["memorization"] = {
        "enabled": True, "data_seed": 7, "doses": [0, 1],
        "canaries_per_dose": 1, "prefix_tokens": 2, "suffix_tokens": 2,
        "acquisition_fraction": .5,
    }
    data = tmp_path / "data"
    write_tiny_data(data, cfg)
    monkeypatch.setattr("rg_nanogpt_one_head.train_loop.run_weightwatcher", lambda *a, **kw: {})
    monkeypatch.setattr("rg_nanogpt_one_head.run_utils.evaluate_bleu", lambda *a, **kw: {"bleu": 0.0})
    weights = []
    for value in ("0", "1"):
        monkeypatch.setenv("RG_NANOGPT_LOG_EVERY_STEP", value)
        directory = run_one(
            cfg=cfg, data_root=data, results_root=tmp_path / value,
            optimizer_name="adamw", seed=13, device="cpu", progress=True,
        )
        output = capsys.readouterr().out
        steps = re.findall(r"\[one-head-step\].*?step=(\d+)/3", output)
        assert steps == (["1", "2", "3"] if value == "1" else [])
        if value == "1":
            assert "batch_loss=" in output and "eta_training_min=" in output
            assert "evaluating canaries" in output and "running WeightWatcher" in output
        weights.append(torch.load(directory / "checkpoint_final.pt", weights_only=False)["model"])
    assert weights[0].keys() == weights[1].keys()
    assert all(torch.equal(weights[0][key], weights[1][key]) for key in weights[0])

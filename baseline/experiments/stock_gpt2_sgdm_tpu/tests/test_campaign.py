"""Campaign contracts and end-to-end orchestration with explicit simulated runs."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import campaign
from config import TrainingConfig, FULL_BUDGET_TOKENS, BENCHMARK_VALIDATION_TOKENS, smoke_config
import report


class FakePublisher:
    def __init__(self, *, final_complete=True):
        self.calls = []
        self.final_complete = final_complete

    def receipt(self, action, final=False):
        self.calls.append((action, final))
        return {"status": "published", "commit_sha": f"{len(self.calls):040x}",
                "complete": self.final_complete if final else False}

    def preflight(self, campaign_manifest=None):
        return self.receipt("preflight", True)

    def publish_run(self, run_dir, run_key=None, final=False):
        return self.receipt("run:" + run_key, final)

    def publish_campaign(self, campaign_dir, final=False):
        return self.receipt("campaign", final)


def plan(tmp_path, monkeypatch, *, smoke=False, no_publish=False):
    monkeypatch.setattr(campaign, "source_commit", lambda *a, **k: "a" * 40)
    argv = ["plan", "--campaign-dir", str(tmp_path / "campaign"), "--data-dir", str(tmp_path / "data"),
            "--publication-retry-seconds", "0", "--poll-seconds", "0.01", "--min-free-gib", "0"]
    if smoke:
        argv += ["--smoke", "--device", "cpu"]
    if no_publish:
        argv += ["--no-publish"]
    args = campaign.parser().parse_args(argv)
    return campaign.create_plan(args)


def result_for(config, *, target=False, diverged=False):
    step = min(config.max_steps, 4) if target else 1 if diverged else config.max_steps
    tokens = step * config.tokens_per_step
    nll = 3.27 if target else 3.6 + config.peak_lr / 100
    validation = {"step": step, "tokens_seen": tokens, "val_nll": nll,
                  "val_perplexity": __import__("math").exp(nll), "evaluation_tokens": config.validation_tokens,
                  "validation_offset": config.validation_offset, "validation_kind": config.validation_kind,
                  "full_benchmark_evaluation": config.mode == "production" and config.validation_kind == "benchmark"}
    interval = config.spectral_interval_tokens // config.tokens_per_step
    steps = sorted({0, step, *range(interval, step, interval)})
    status = {"status": "diverged" if diverged else "target_reached" if target else "budget_complete_target_not_met",
              "terminal": True, "mode": config.mode, "config_sha256": config.config_sha256,
              "candidate_id": config.candidate_id, "stage_id": config.stage_id, "seed": config.seed,
              "run_id": config.run_name, "step": step, "tokens_seen": tokens,
              "full_training_recipe_completed": step == config.max_steps,
              "target_met": target, "target_ever_met": target, "best_validation": validation,
              "first_target_validation": validation if target else None,
              "validation": validation, "expected_snapshot_steps": steps}
    tracking = {"complete": True, "training_done": True, "failed": 0, "pending": 0,
                "completed": len(steps), "completed_steps": steps, "missing_expected_steps": [],
                "config_sha256": config.config_sha256, "run_id": config.run_name}
    return status, tracking


def write_fake_run(spec, *, target=False, diverged=False):
    config = TrainingConfig.from_json(spec["config"])
    root = Path(spec["run_dir"])
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    config.write_json(root / "config.json")
    status, tracking = result_for(config, target=target, diverged=diverged)
    now = time.time()
    for name, value in {"TRAINING_STATUS.json": status, "TRACKING_STATUS.json": tracking,
                        "LAUNCH.json": {"supervisor_pid": 99999999, "started_unix": now - 1},
                        "LAUNCHER_STATUS.json": {"supervision_finished": True, "finished_unix": now},
                        "TRAINING_DONE": {"status": status["status"]}}.items():
        campaign.atomic_json(root / name, value)


def test_plan_is_finite_automatic_and_does_not_start_or_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "_publisher", lambda *a: pytest.fail("Plan must not contact GitHub"))
    monkeypatch.setattr(campaign.launch, "start_local", lambda *a: pytest.fail("Plan must not launch"))
    manifest = plan(tmp_path, monkeypatch)
    assert manifest["maximum_planned_training_tokens"] == 36_536_582_144
    assert manifest["max_full_budget_multiplier"] == 2
    assert manifest["production_stage_plan"][-1]["stop_at_target"] is True
    assert manifest["github_branch"] == "sgdm-tpu-results"
    assert manifest["github_prefix"].endswith("/results/campaign")
    assert len(list((tmp_path / "campaign/search/screen/configs").glob("*.json"))) == 11
    assert not (tmp_path / "campaign/runs").exists()


def test_production_publication_cannot_be_disabled(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="require incremental"):
        plan(tmp_path, monkeypatch, no_publish=True)


def test_early_full_target_is_valid_but_short_tuning_run_is_not():
    config = TrainingConfig(validation_kind="benchmark", validation_offset=0,
                            validation_tokens=BENCHMARK_VALIDATION_TOKENS, stop_at_target=True)
    status, tracking = result_for(config, target=True)
    assert status["step"] < config.max_steps
    assert campaign.verify_result(config, status, tracking, full=True)["target_met"]
    status["validation"]["evaluation_tokens"] -= 1024
    with pytest.raises(campaign.CampaignFailure, match="Validation"):
        campaign.verify_result(config, status, tracking, full=True)
    tuning = TrainingConfig()
    bad, tracked = result_for(tuning, target=True)
    with pytest.raises(campaign.CampaignFailure, match="Incomplete training horizon"):
        campaign.verify_result(tuning, bad, tracked)


def test_missing_or_wrong_identity_diagnostics_prevent_success():
    config = TrainingConfig(validation_kind="benchmark", validation_offset=0,
                            validation_tokens=BENCHMARK_VALIDATION_TOKENS, stop_at_target=True)
    status, tracking = result_for(config, target=True)
    with pytest.raises(campaign.CampaignFailure, match="incomplete"):
        campaign.verify_result(config, status, {**tracking, "complete": False}, full=True)
    with pytest.raises(campaign.CampaignFailure, match="another run"):
        campaign.verify_result(config, status, {**tracking, "run_id": "wrong"}, full=True)
    with pytest.raises(campaign.CampaignFailure, match="coverage"):
        campaign.verify_result(config, status, {**tracking, "completed_steps": [0], "completed": 1}, full=True)


def test_full_search_promotes_automatically_and_qualifies_only_after_publication(tmp_path, monkeypatch):
    manifest = plan(tmp_path, monkeypatch)
    publisher = FakePublisher()
    calls, preparations = [], []
    monkeypatch.setattr(report, "make_plots", lambda *a, **k: None)
    def starter(spec):
        config = TrainingConfig.from_json(spec["config"])
        calls.append(config)
        write_fake_run(spec, target=config.stage_id == "full")
    runner = campaign.Campaign(manifest, publisher=publisher, run_starter=starter,
                               data_preparer=lambda path, budget: preparations.append(budget))
    status = runner.execute()
    assert status["status"] == "succeeded" and status["successful"] is True
    assert [sum(c.stage_id == stage for c in calls) for stage in ("screen", "refine", "full")] == [11, 4, 1]
    assert calls[-1].stop_at_target is True
    assert preparations == [134_217_728, 1_073_741_824, FULL_BUDGET_TOKENS]
    assert all(calls[index].config_sha256 != calls[-1].config_sha256 for index in range(len(calls) - 1))
    assert publisher.calls[0][0] == "preflight"
    assert publisher.calls[-1] == ("campaign", True)
    assert status["publication_complete"] is True
    assert (tmp_path / "campaign/FINAL_PUBLICATION.json").exists()


def test_one_bounded_fresh_extended_attempt_follows_a_missed_target(tmp_path, monkeypatch):
    manifest = plan(tmp_path, monkeypatch)
    publisher = FakePublisher()
    calls = []
    monkeypatch.setattr(report, "make_plots", lambda *a, **k: None)
    def starter(spec):
        config = TrainingConfig.from_json(spec["config"])
        calls.append((config, spec["resume"]))
        write_fake_run(spec, target=config.stage_id == "extended")
    result = campaign.Campaign(manifest, publisher=publisher, run_starter=starter,
                               data_preparer=lambda *a: None).execute()
    assert result["successful"]
    assert len(calls) == 17
    last, resumed = calls[-1]
    assert last.stage_id == "extended" and last.budget_tokens == 2 * FULL_BUDGET_TOKENS
    assert last.stop_at_target and not resumed
    assert calls[-2][0].config_sha256 != last.config_sha256


def test_budget_exhaustion_is_not_success_even_with_complete_publication(tmp_path, monkeypatch):
    manifest = plan(tmp_path, monkeypatch)
    monkeypatch.setattr(report, "make_plots", lambda *a, **k: None)
    result = campaign.Campaign(manifest, publisher=FakePublisher(), run_starter=write_fake_run,
                               data_preparer=lambda *a: None).execute()
    assert result["status"] == "budget_exhausted_target_not_met"
    assert result["workflow_complete"] and result["publication_complete"]
    assert not result["successful"] and not result["benchmark_target_met"]
    assert len(result["run_records"]) == 17


def test_unconfirmed_final_publication_cannot_finish_campaign(tmp_path, monkeypatch):
    manifest = plan(tmp_path, monkeypatch)
    monkeypatch.setattr(report, "make_plots", lambda *a, **k: None)
    publisher = FakePublisher(final_complete=False)
    runner = campaign.Campaign(manifest, publisher=publisher, run_starter=write_fake_run,
                               data_preparer=lambda *a: None)
    with pytest.raises(campaign.CampaignFailure, match="publication is incomplete"):
        runner.execute()
    assert runner.state["status"] == "failed" and not runner.state["successful"]


def test_publication_retry_pauses_training_and_is_bounded(tmp_path, monkeypatch):
    manifest = plan(tmp_path, monkeypatch)
    paused = []
    monkeypatch.setattr(campaign, "_stop_supervisor", lambda root: paused.append(root))
    runner = campaign.Campaign(manifest, publisher=FakePublisher(), sleep=lambda seconds: None)
    count = [0]
    def unavailable():
        count[0] += 1
        raise RuntimeError("HTTP403: write access lost")
    with pytest.raises(campaign.CampaignFailure, match="bounded retries"):
        runner.publish(unavailable, active_run=tmp_path / "run")
    assert count[0] == manifest["publication_retries"] == len(paused)


def test_smoke_workflow_never_claims_production_target(tmp_path, monkeypatch):
    manifest = plan(tmp_path, monkeypatch, smoke=True, no_publish=True)
    monkeypatch.setattr(report, "make_plots", lambda *a, **k: None)
    calls = []
    def starter(spec):
        calls.append(spec)
        write_fake_run(spec)
    result = campaign.Campaign(manifest, run_starter=starter).execute()
    assert result["status"] == "smoke_completed_not_a_benchmark"
    assert not result["successful"] and not result["benchmark_target_met"]
    assert len(calls) == 3
    assert result["workflow_complete"] and not result["publication_complete"]


def test_old_launcher_completion_is_not_reused_after_resume(tmp_path):
    campaign.atomic_json(tmp_path / "LAUNCH.json", {"started_unix": 100, "supervisor_pid": 123})
    campaign.atomic_json(tmp_path / "LAUNCHER_STATUS.json", {"supervision_finished": True, "finished_unix": 99})
    assert not campaign.current_launcher_state(tmp_path)["supervision_finished"]
    campaign.atomic_json(tmp_path / "LAUNCHER_STATUS.json", {"supervision_finished": True, "finished_unix": 101})
    assert campaign.current_launcher_state(tmp_path)["supervision_finished"]


def test_changed_source_refuses_campaign_resume(tmp_path, monkeypatch):
    manifest = plan(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"different": "source"})
    with pytest.raises(campaign.CampaignFailure, match="source changed"):
        campaign.Campaign(manifest, publisher=FakePublisher()).execute()


def test_explicit_changed_settings_are_rejected_for_existing_campaign(tmp_path, monkeypatch, capsys):
    manifest = plan(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="--seed differs"):
        campaign.main(["plan", "--campaign-dir", manifest["campaign_dir"], "--seed", "43"])
    with pytest.raises(ValueError, match="--max-full-budget-multiplier differs"):
        campaign.main(["resume", "--campaign-dir", manifest["campaign_dir"],
                       "--max-full-budget-multiplier", "1"])
    assert campaign.main(["plan", "--campaign-dir", manifest["campaign_dir"], "--seed=42"]) == 0
    assert json.loads(capsys.readouterr().out)["campaign"]["seed"] == 42


def test_recycled_pid_is_never_signaled(tmp_path, monkeypatch):
    run = tmp_path / "run"
    campaign.atomic_json(run / "LAUNCH.json", {"supervisor_pid": 123456, "launch_id": "launch-abc"})
    command = [b"/usr/bin/python", b"-u", b"/unrelated/service.py"]
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"\0".join(command) + b"\0"
                        if str(path) == "/proc/123456/cmdline" else original_read(path))
    signals = []
    monkeypatch.setattr(campaign.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    campaign._stop_supervisor(run)
    assert not signals and not campaign.run_supervisor_alive(run)
    command[:] = [b"/usr/bin/python", b"-u", str(campaign.HERE / "launch.py").encode(),
                  b"_supervise", b"--spec", str(run / "launches/launch-abc.json").encode()]
    campaign._stop_supervisor(run)
    assert signals == [(123456, campaign.signal.SIGTERM)]


def test_reports_are_rebuilt_before_final_publication(tmp_path, monkeypatch):
    manifest = plan(tmp_path, monkeypatch, smoke=True)
    class ReportCheckingPublisher(FakePublisher):
        def publish_run(self, run_dir, run_key=None, final=False):
            if final:
                summary = campaign.read_json(Path(run_dir) / "reports/summary.json")
                status = campaign.read_json(Path(run_dir) / "TRAINING_STATUS.json")
                assert summary["step"] == status["step"]
                assert summary["run_id"] == status["run_id"]
            return super().publish_run(run_dir, run_key=run_key, final=final)
    monkeypatch.setattr(report, "make_plots", lambda *a, **k: None)
    result = campaign.Campaign(manifest, publisher=ReportCheckingPublisher(), run_starter=write_fake_run).execute()
    assert result["workflow_complete"] and not result["benchmark_target_met"]


def test_source_commit_requires_every_hashed_file_to_belong_to_head(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    experiment = repo / "baseline/experiments/stock_gpt2_sgdm_tpu"
    dependency = repo / "baseline/gpt2_small/muon_speedrun"
    experiment.mkdir(parents=True)
    dependency.mkdir(parents=True)
    (experiment / "train.py").write_text("# committed test fixture\n")
    (experiment / "requirements-cpu.txt").write_text("# committed requirements fixture\n")
    for name in ("runtime.py", "stock_model.py", "stock_config.py"):
        (dependency / name).write_text("# committed dependency fixture\n")
    git = ["git", "-C", str(repo)]
    campaign.subprocess.run(git + ["init", "--quiet"], check=True, capture_output=True)
    campaign.subprocess.run(git + ["add", "--all"], check=True, capture_output=True)
    campaign.subprocess.run(git + ["-c", "user.name=Campaign Test", "-c", "user.email=campaign-test@example.invalid",
                                  "commit", "--quiet", "--no-gpg-sign", "-m", "Source fixture"],
                            check=True, capture_output=True)
    monkeypatch.setattr(campaign, "REPO", repo)
    monkeypatch.setattr(campaign, "HERE", experiment)
    commit = campaign.source_commit()
    assert len(commit) == 40
    # Untracked prose is not executable source and is outside this check.
    (experiment / "notes.md").write_text("Temporary test notes\n")
    assert campaign.source_commit() == commit
    (experiment / "uncommitted_helper.py").write_text("# not present in source commit\n")
    with pytest.raises(ValueError, match="not in the pinned HEAD commit.*uncommitted_helper.py"):
        campaign.source_commit()


@pytest.mark.parametrize("resume", [False, True])
def test_source_change_during_campaign_blocks_new_or_resumed_launch(tmp_path, monkeypatch, resume):
    manifest = plan(tmp_path, monkeypatch)
    stage, _ = campaign.tune._load_stage(tmp_path / "campaign/search/screen")
    first_id = stage["candidates"][0]["candidate_id"]
    if resume:
        checkpoint = tmp_path / "campaign/runs/screen" / first_id / "checkpoints/latest.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"Unopened checkpoint fixture: launch must be refused first")
    def change_source_after_entry_check(*args):
        monkeypatch.setattr(campaign, "source_hashes", lambda: {"changed_during_campaign.py": "different"})
    calls = []
    runner = campaign.Campaign(manifest, publisher=FakePublisher(),
                               data_preparer=change_source_after_entry_check,
                               run_starter=lambda spec: calls.append(spec))
    with pytest.raises(campaign.CampaignFailure, match="source changed before launching"):
        runner.execute()
    assert calls == [] and runner.state["attempts"] == {}
    assert runner.state["status"] == "failed" and not runner.state["successful"]
    if resume:
        assert checkpoint.read_bytes().startswith(b"Unopened checkpoint fixture")

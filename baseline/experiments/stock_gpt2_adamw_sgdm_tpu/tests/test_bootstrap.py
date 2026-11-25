"""The one-command entry point does not allocate or mutate shared environments."""
import argparse
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
from .. import run_speedrun as bootstrap


def args(tmp_path, **changes):
    values = dict(action="plan", work_dir=tmp_path, campaign_id="test-campaign", smoke=False,
                  github_repo="CalculatedContent/rg_optimizers", github_branch="adamw-sgdm-tpu-results",
                  max_full_budget_multiplier=2, tracking_threads=2, min_free_gib=150,
                  gcs_uri=None, source_sha=None, no_publish=False)
    return argparse.Namespace(**{**values, **changes})


def test_plan_is_read_only_and_describes_complete_automatic_pipeline(tmp_path, monkeypatch, capsys):
    target = tmp_path / "unused"
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("plan invoked a process"))
    assert bootstrap.main(["plan", "--work-dir", str(target), "--campaign-id", "run-a"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["executes"] is False and result["allocates_tpu"] is False
    assert result["endpoint"]["loss_nats_per_token"] == 3.28
    assert "incremental full layer publication" in result["automatic_steps"]
    assert result["github_results_branch"] == "adamw-sgdm-tpu-results"
    assert result["optimizer"] == "adamw_sgdm_graft_v1"
    assert result["maximum_planned_training_tokens"] == 35_194_404_864
    assert result["result_prefix"].startswith("baseline/experiments/stock_gpt2_adamw_sgdm_tpu/results/")
    assert not target.exists()


def test_start_command_uses_one_controller_and_explicit_cap(tmp_path):
    value = args(tmp_path, action="start")
    command = bootstrap.campaign_command(value, tmp_path / "campaigns/run-a", "run-a", "/venv/bin/python")
    assert command[:4] == ["/venv/bin/python", "-u", str(HERE / "campaign.py"), "start"]
    assert command[command.index("--max-full-budget-multiplier") + 1] == "2"
    assert command[command.index("--data-dir") + 1] == str(tmp_path / "fineweb")
    assert command[command.index("--min-free-gib") + 1] == "30"
    assert "--no-publish" not in command


def test_existing_environment_keeps_venv_symlink_and_does_not_reinstall(tmp_path):
    environment = tmp_path / "venv-cpu"
    (environment / "bin").mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home=/base/python\n")
    (environment / "bin/python").symlink_to(sys.executable)
    bootstrap.atomic_json(environment / "ADAMW_SGDM_ENVIRONMENT.json", {"schema_version": 1,
        "mode": "smoke", "requirements_sha256": bootstrap.requirements_hash(True)})
    def forbidden(*a, **k):
        pytest.fail("Unchanged environment should not reinstall")
    result = bootstrap.ensure_environment(tmp_path, smoke=True, runner=forbidden)
    assert result == str(environment / "bin/python")
    assert result != str(Path(sys.executable).resolve())


def test_unknown_environment_directory_is_not_overwritten(tmp_path):
    (tmp_path / "venv-tpu").mkdir()
    with pytest.raises(RuntimeError, match="non-venv"):
        bootstrap.ensure_environment(tmp_path, runner=lambda *a, **k: pytest.fail("must not install"))


def test_resume_resolves_the_saved_campaign_and_uses_immutable_options(tmp_path):
    bootstrap.atomic_json(tmp_path / "LATEST_CAMPAIGN.json", {"campaign_id": "saved-run"})
    value = args(tmp_path, action="resume", campaign_id=None)
    work, root, campaign_id = bootstrap.choose_campaign(value)
    assert campaign_id == "saved-run" and root == work / "campaigns/saved-run"
    command = bootstrap.campaign_command(value, root, campaign_id, "/venv/bin/python")
    assert command[-3:] == ["resume", "--campaign-dir", str(root)]
    assert "--max-full-budget-multiplier" not in command


def test_production_cannot_disable_publication(tmp_path):
    with pytest.raises(SystemExit):
        bootstrap.main(["start", "--work-dir", str(tmp_path), "--no-publish"])


def test_failed_mount_check_precedes_environment_install(tmp_path, monkeypatch):
    from .. import preflight
    def bad_mount(*args, **kwargs):
        assert kwargs == {"device": "tpu", "min_free_gib": 150}
        raise RuntimeError("not persistent")
    monkeypatch.setattr(preflight, "check_output", bad_mount)
    monkeypatch.setattr(bootstrap, "ensure_environment", lambda *a, **k: pytest.fail("should not install"))
    with pytest.raises(RuntimeError, match="not persistent"):
        bootstrap.main(["start", "--work-dir", str(tmp_path / "unused"), "--campaign-id", "run-a"])
    assert not (tmp_path / "unused").exists()


def test_python_environment_keeps_auth_and_device_but_removes_import_injection(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated/packages")
    monkeypatch.setenv("PYTHONHOME", "/unrelated/python")
    monkeypatch.setenv("PYTHONUSERBASE", "/unrelated/user")
    monkeypatch.setenv("GH_TOKEN", "unit-test-placeholder")
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    environment = bootstrap.isolated_python_environment()
    assert not {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"}.intersection(environment)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["GH_TOKEN"] == "unit-test-placeholder"
    assert environment["PJRT_DEVICE"] == "TPU"


def test_installation_uses_the_isolated_environment(tmp_path, monkeypatch):
    environment = tmp_path / "venv-cpu"
    (environment / "bin").mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home=/base/python\n")
    (environment / "bin/python").symlink_to(sys.executable)
    monkeypatch.setenv("PYTHONPATH", "/unrelated/packages")
    calls = []
    bootstrap.ensure_environment(tmp_path, smoke=True, runner=lambda *a, **k: calls.append((a, k)))
    assert len(calls) == 1
    assert "PYTHONPATH" not in calls[0][1]["env"]
    assert calls[0][1]["env"]["PYTHONNOUSERSITE"] == "1"


def test_explicit_resume_budget_change_fails_before_installation(tmp_path, monkeypatch):
    root = tmp_path / "campaigns/saved"
    bootstrap.atomic_json(tmp_path / "LATEST_CAMPAIGN.json", {"campaign_id": "saved"})
    bootstrap.atomic_json(root / "CAMPAIGN.json", {"optimizer": bootstrap.OPTIMIZER, "mode": "production", "max_full_budget_multiplier": 2,
                                                  "min_free_gib": 30, "publication_required": True})
    monkeypatch.setattr(bootstrap, "ensure_environment", lambda *a, **k: pytest.fail("Must not install"))
    with pytest.raises(ValueError, match="max-full-budget-multiplier differs"):
        bootstrap.main(["resume", "--work-dir", str(tmp_path), "--max-full-budget-multiplier", "1"])
    with pytest.raises(ValueError, match="smoke differs"):
        bootstrap.main(["resume", "--work-dir", str(tmp_path), "--smoke"])


def test_resume_defaults_use_saved_mode_and_sanitize_controller_environment(tmp_path, monkeypatch, capsys):
    from .. import preflight
    root = tmp_path / "campaigns/saved"
    bootstrap.atomic_json(tmp_path / "LATEST_CAMPAIGN.json", {"campaign_id": "saved"})
    bootstrap.atomic_json(root / "CAMPAIGN.json", {"optimizer": bootstrap.OPTIMIZER, "mode": "smoke", "max_full_budget_multiplier": 1,
                                                  "min_free_gib": 0, "publication_required": False})
    mounts, environments, commands = [], [], []
    monkeypatch.setattr(preflight, "check_output", lambda *a, **k: mounts.append((a, k)))
    monkeypatch.setattr(bootstrap, "ensure_environment", lambda *a, **k:
                        environments.append((a, k)) or "/isolated/bin/python")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/packages")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k:
                        commands.append((a, k)) or argparse.Namespace(returncode=0))
    assert bootstrap.main(["resume", "--work-dir", str(tmp_path)]) == 0
    assert mounts[0][1] == {"device": "cpu", "min_free_gib": 0}
    assert environments[0][1]["smoke"] is True
    assert "PYTHONPATH" not in commands[0][1]["env"]
    assert "--max-full-budget-multiplier" not in commands[0][0][0]


def test_explicit_resume_free_space_cannot_weaken_saved_minimum(tmp_path):
    saved = {"optimizer": bootstrap.OPTIMIZER, "mode": "production", "publication_required": True, "min_free_gib": 30}
    value = args(tmp_path, action="resume", min_free_gib=0, _provided={"--min-free-gib"})
    with pytest.raises(ValueError, match="cannot lower"):
        bootstrap.validate_resume_options(value, saved)
    value.min_free_gib = 200
    bootstrap.validate_resume_options(value, saved)


def test_resume_refuses_another_optimizer_before_environment_changes(tmp_path, monkeypatch):
    root = tmp_path / "campaigns/plain-sgdm"
    bootstrap.atomic_json(tmp_path / "LATEST_CAMPAIGN.json", {"campaign_id": "plain-sgdm"})
    bootstrap.atomic_json(root / "CAMPAIGN.json", {"mode": "production", "optimizer": "sgdm"})
    monkeypatch.setattr(bootstrap, "ensure_environment", lambda *a, **k: pytest.fail("Must not install"))
    with pytest.raises(ValueError, match="different optimizer experiment"):
        bootstrap.main(["resume", "--work-dir", str(tmp_path)])


def test_default_identity_and_generated_campaign_name_are_separate(tmp_path):
    value = args(tmp_path, campaign_id=None)
    _, _, campaign_id = bootstrap.choose_campaign(value)
    assert campaign_id.startswith("adamw-sgdm-")
    assert bootstrap.DEFAULT_WORK_DIR.name == "stock_gpt2_adamw_sgdm_tpu"
    assert bootstrap.DEFAULT_BRANCH == "adamw-sgdm-tpu-results"

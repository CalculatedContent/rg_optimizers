"""Archive safety and immutable-parent restoration; no network or training."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from california_housing_sgdm_search import prepare, publish


@pytest.mark.parametrize("path", [
    "california_housing_benchmark/results/2026-11-26_published_mlp",
    "california_housing_sgdm_search/plan.json",
    publish.PREFIX + "/../elsewhere",
    "/tmp/" + publish.PREFIX,
])
def test_publication_cannot_escape_new_result_prefix(path):
    with pytest.raises(ValueError):
        publish.valid_paths([path])


def test_pinned_download_verifies_both_hashes_and_reuses_bytes(tmp_path, monkeypatch):
    payload = b"immutable checkpoint fixture"
    destination = tmp_path / "checkpoint.pt"
    calls = []
    def open_response(request, timeout):
        assert "Authorization" not in request.headers
        calls.append(request.full_url)
        return BytesIO(payload)
    monkeypatch.setattr(prepare.urllib.request, "urlopen", open_response)
    expected = hashlib.sha256(payload).hexdigest()
    prepare.download_verified(prepare.REPOSITORY, "a" * 40, "parent/checkpoint.pt",
                              destination, expected, prepare.git_blob_sha(payload), len(payload))
    prepare.download_verified(prepare.REPOSITORY, "a" * 40, "parent/checkpoint.pt",
                              destination, expected, prepare.git_blob_sha(payload), len(payload))
    assert len(calls) == 1
    assert destination.read_bytes() == payload
    with pytest.raises(ValueError, match="Git blob"):
        prepare.download_verified(prepare.REPOSITORY, "a" * 40, "parent/checkpoint.pt",
                                  destination, expected, "f" * 40, len(payload))
    assert destination.read_bytes() == payload


def test_corrupt_download_never_becomes_a_prepared_input(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **k: BytesIO(b"bad"))
    destination = tmp_path / "prepared.npz"
    with pytest.raises(ValueError, match="SHA-256"):
        prepare.download_verified(prepare.REPOSITORY, "a" * 40, "data/prepared.npz",
                                  destination, hashlib.sha256(b"good").hexdigest())
    assert not destination.exists()


def make_development(path, status="complete", phase="development"):
    path.mkdir(parents=True)
    (path / "result.json").write_text(json.dumps({"phase": phase, "status": status}))
    for name in ("best_checkpoint.pt", "final_weights.pt", "resume.pt",
                 "checkpoint_step_20000.pt", "history.csv", "workflow.log", "fits.json"):
        (path / name).write_bytes(name.encode())


def test_completed_development_omits_new_full_states_but_preserves_prior_recovery(tmp_path):
    source, staged = tmp_path / "source", tmp_path / "staged"
    make_development(source)
    shutil.copytree(source, staged)
    ignored = publish.ignore_archived_files(str(source), [p.name for p in source.iterdir()])
    assert ignored == {"resume.pt", "checkpoint_step_20000.pt"}
    shutil.copytree(source, staged, dirs_exist_ok=True, ignore=publish.ignore_archived_files)
    assert (staged / "resume.pt").exists()
    fresh = tmp_path / "fresh"
    shutil.copytree(source, fresh, ignore=publish.ignore_archived_files)
    assert {p.name for p in fresh.iterdir()} == {
        "result.json", "best_checkpoint.pt", "final_weights.pt",
        "history.csv", "workflow.log", "fits.json"}
    assert (source / "resume.pt").exists()
    assert (source / "checkpoint_step_20000.pt").exists()


@pytest.mark.parametrize("status,phase", [
    ("failed", "development"), ("paused", "development"),
    ("complete", "continuation"), ("complete", "confirmation")])
def test_other_runs_retain_every_checkpoint(tmp_path, status, phase):
    directory = tmp_path / "run"
    make_development(directory, status, phase)
    assert publish.ignore_archived_files(str(directory), [p.name for p in directory.iterdir()]) == set()


def test_complete_development_requires_both_retained_weights(tmp_path):
    directory = tmp_path / "run"
    make_development(directory)
    (directory / "final_weights.pt").unlink()
    with pytest.raises(RuntimeError, match="lacks retained weights"):
        publish.ignore_archived_files(str(directory), [p.name for p in directory.iterdir()])


def test_archive_refuses_progress_rollback_or_changed_completed_measurements():
    record = {"phase": "development", "seed": 3338, "config": {"lr": .03},
              "status": "complete", "completed_updates": 50000,
              "history": [{"step": 50000, "validation": {"rmse": .5}}]}
    publish.check_record_monotonic(dict(record), record, "fixture")
    with pytest.raises(RuntimeError, match="older archived progress"):
        publish.check_record_monotonic(record | {"completed_updates": 10000}, record, "fixture")
    with pytest.raises(RuntimeError, match="identity"):
        publish.check_record_monotonic(record | {"config": {"lr": .1}}, record, "fixture")
    with pytest.raises(RuntimeError, match="measurement"):
        publish.check_record_monotonic(record | {"history": []}, record, "fixture")


def test_two_concurrent_git_writers_merge_without_touching_local_outputs(tmp_path, monkeypatch):
    """A real local bare Git server rejects a stale push; the retry keeps both writers."""
    real_git = publish.git
    server, initial = tmp_path / "server.git", tmp_path / "initial"
    server.mkdir(); initial.mkdir()
    real_git(["init", "--bare", "-q"], cwd=server)
    real_git(["init", "-q"], cwd=initial)
    for key, value in (("user.name", "Archive test"), ("user.email", "archive@example.invalid")):
        real_git(["config", key, value], cwd=initial)
    (initial / "initial.txt").write_text("initial")
    real_git(["add", "initial.txt"], cwd=initial)
    real_git(["commit", "-q", "-m", "initial"], cwd=initial)
    real_git(["branch", "-M", "study"], cwd=initial)
    real_git(["remote", "add", "origin", str(server)], cwd=initial)
    real_git(["push", "-q", "origin", "study"], cwd=initial)

    source = tmp_path / "local_outputs"
    chosen = publish.PREFIX + "/worker_a"
    output = source / chosen
    output.mkdir(parents=True)
    (output / "metrics.json").write_text('{"metric": 0.5}\n')
    (source / "do_not_reset.txt").write_text("retain local work")
    interfered = False

    def with_race(args, *, cwd, env=None, check=True):
        nonlocal interfered
        if args[:3] == ["remote", "add", "origin"]:
            args = [*args[:3], str(server)]
        if args and args[0] == "push" and not interfered:
            interfered = True
            other = tmp_path / "concurrent"
            other.mkdir()
            real_git(["clone", "-q", "--branch", "study", str(server), "."], cwd=other)
            for key, value in (("user.name", "Other worker"), ("user.email", "other@example.invalid")):
                real_git(["config", key, value], cwd=other)
            path = other / publish.PREFIX / "worker_b" / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"metric": 0.6}\n')
            real_git(["add", "."], cwd=other)
            real_git(["commit", "-q", "-m", "other worker"], cwd=other)
            real_git(["push", "-q", "origin", "study"], cwd=other)
        return real_git(args, cwd=cwd, env=env, check=check)

    monkeypatch.setattr(publish, "git", with_race)
    monkeypatch.setenv("GH_TOKEN", "local-test-placeholder")
    monkeypatch.setattr(publish.time, "sleep", lambda seconds: None)
    commit = publish.restore_or_publish("publish", "study", [chosen], source=source, max_attempts=3)
    assert interfered and len(commit) == 40
    head = real_git(["rev-parse", "refs/heads/study"], cwd=server).stdout.decode().strip()
    assert head == commit
    for worker in ("worker_a", "worker_b"):
        payload = real_git(["show", f"study:{publish.PREFIX}/{worker}/metrics.json"], cwd=server).stdout
        assert json.loads(payload)["metric"] in (.5, .6)
    assert (source / "do_not_reset.txt").read_text() == "retain local work"
    assert (output / "metrics.json").read_text() == '{"metric": 0.5}\n'

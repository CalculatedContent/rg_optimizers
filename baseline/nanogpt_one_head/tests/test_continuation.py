from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from rg_nanogpt_one_head.continuation import pause_reason
from rg_nanogpt_one_head.completion import (
    _validate_test_monitoring, _validate_clip_snapshot_schedule, CompletedRunValidationError,
)
from rg_nanogpt_one_head.muonclip_continue import series_lock, prune_completed_segments
import rg_nanogpt_one_head.muonclip_continue as series
import rg_nanogpt_one_head.muonclip_resilient as resilient


def test_real_continuation_resume_and_segment_retention(tmp_path):
    result = subprocess.run([sys.executable, "tests/continuation_scenario.py", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "CONTINUATION_SCENARIO_PASSED" in result.stdout


def test_supervisor_honors_pause_without_retry(tmp_path, monkeypatch):
    args = SimpleNamespace(config=tmp_path / "cfg.yaml", seed=13, data_root=tmp_path / "data",
        results_root=tmp_path / "results", device="cpu", max_no_progress_failures=3, retry_delay_seconds=0)
    calls = []
    def pause(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(returncode=75)
    monkeypatch.setattr(resilient.subprocess, "run", pause)
    assert resilient.run_resilient(args) == 75
    assert len(calls) == 1


def test_duplicate_writer_lock(tmp_path):
    with series_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already holds"):
            with series_lock(tmp_path):
                pass


def test_low_disk_pause(tmp_path, monkeypatch):
    monkeypatch.setattr("rg_nanogpt_one_head.continuation.shutil.disk_usage", lambda _: SimpleNamespace(free=100))
    assert "free disk" in pause_reason({"training": {"min_free_disk_gb": 1}}, tmp_path)


def test_retention_rejects_outside_series(tmp_path):
    state = {"seed": 13, "keep_segments": 2, "completed_segments": [{"directory": "../outside"}] * 3}
    with pytest.raises(RuntimeError, match="outside owned"):
        prune_completed_segments(tmp_path, state)


def test_periodic_probe_validator_rejects_missing_measurement():
    frame = pd.DataFrame({"step": [0, 4]})
    with pytest.raises(CompletedRunValidationError, match="missing periodic"):
        _validate_test_monitoring(frame, "metrics.csv", 2, 4)


def test_open_ended_series_stops_at_saved_boundary(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(series, "series_environment", lambda: {"source": "fixed"})
    monkeypatch.setattr(series, "make_continuation_config", lambda *args, **kwargs: {
        "continuation": {"global_step_offset": 123}, "training": {"max_steps": 5},
    })
    monkeypatch.setattr(series, "pause_reason", lambda *args: None)
    monkeypatch.setattr(series, "validate_completed_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(series, "prune_completed_segments", lambda *args: None)
    calls = []
    def worker(args):
        calls.append(1)
        run_dir = Path(args.results_root) / "muon_clip/seed_13"
        run_dir.mkdir(parents=True)
        (run_dir / "run_complete.json").write_text(json.dumps({
            "fingerprint": "verified", "optimizer_steps": 5, "global_step": 128,
        }))
        (tmp_path / "STOP").touch()
        return 0
    monkeypatch.setattr(series, "run_resilient", worker)
    state = {"environment": {"source": "fixed"}, "additional_steps": None,
        "completed_steps": 0, "completed_segments": [], "segment_steps": 5,
        "latest_checkpoint": "parent.pt", "learning_rate": 2e-5, "test_interval_steps": 2,
        "min_free_disk_gb": 0, "seed": 13, "data_root": str(tmp_path), "device": "cpu",
        "max_no_progress_failures": 3}
    assert series.drive_series(tmp_path, state, 1) == 75
    assert len(calls) == 1 and state["completed_steps"] == 5
    assert state["status"] == "paused" and state["active_segment"] is None


def test_series_environment_change_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(series, "series_environment", lambda: {"source": "changed"})
    with pytest.raises(RuntimeError, match="source/dependencies changed"):
        series.drive_series(tmp_path, {"environment": {"source": "original"}}, 1)


def _snapshot_manifest(steps):
    # The production TPU shape and cadence, including a short final segment.
    return {
        "continuation": {"global_step_offset": 2150000},
        "data_metadata": {"splits": {"train": 80000000}},
        "model": {"block_size": 256},
        "training": {"batch_size": 32, "grad_accum_steps": 1, "max_steps": steps,
                     "target_epochs": steps * 8192 / 80000000, "epoch_interval": 1.024},
    }


@pytest.mark.parametrize("steps,expected", [
    (20, (0, 20)),
    (25000, (0, 10000, 20000, 25000)),
    (1000000, tuple(range(0, 1000001, 10000))),
])
def test_clip_continuation_requires_its_planned_grid(steps, expected):
    _validate_clip_snapshot_schedule(_snapshot_manifest(steps), expected)


@pytest.mark.parametrize("observed", [(0, 25000), (0, 10000, 25000), (0, 10000, 20000)])
def test_clip_continuation_missing_snapshots_are_rejected(observed):
    with pytest.raises(CompletedRunValidationError, match="exact configured snapshot grid"):
        _validate_clip_snapshot_schedule(_snapshot_manifest(25000), observed)


def test_historical_clip_campaign_still_requires_ten_snapshots():
    with pytest.raises(CompletedRunValidationError, match="fewer than ten"):
        _validate_clip_snapshot_schedule({}, (0, 20))
    _validate_clip_snapshot_schedule({}, tuple(range(10)))

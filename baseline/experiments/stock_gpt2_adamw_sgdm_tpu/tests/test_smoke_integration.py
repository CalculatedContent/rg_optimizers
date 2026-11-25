"""Exercise actual detached CPU supervision and native spectral export together."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).resolve().parents[1]
from ..report import build_report, make_plots


def test_launcher_trainer_metrics_and_report(tmp_path):
    root = tmp_path / "run"
    command = [sys.executable, str(HERE / "launch.py"), "start", "--config", str(HERE / "configs/smoke.json"),
               "--data-dir", str(tmp_path / "data"), "--run-dir", str(root), "--device", "cpu",
               "--min-free-gib", "0", "--tracking-threads", "2", "--drain-timeout", "90"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    launched = json.loads(result.stdout)
    state = {}
    try:
        until = time.monotonic() + 90
        while time.monotonic() < until:
            path = root / "LAUNCHER_STATUS.json"
            if path.exists():
                state = json.loads(path.read_text())
                if state.get("supervision_finished"):
                    break
            time.sleep(0.2)
        assert state.get("supervision_finished"), (root / "logs/supervisor.log").read_text()
        assert state["training_exit_code"] == 0, (root / "logs/training.log").read_text()
        assert state["tracking_exit_code"] == 0, (root / "logs/tracking.log").read_text()
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["optimizer"] == "adamw_sgdm_graft_v1"
        assert manifest["optimizer_implementation"] == "TensorLrAdamWSGDMGraft"
        assert manifest["graft"]["physical_matrix_count"] == 8
        tracking = state["tracking_status"]
        assert tracking["complete"] and tracking["completed_steps"] == [0, 2]
        assert tracking["failed"] == tracking["pending"] == 0
        assert not list((root / "tracking/snapshots").glob("*.pt"))
        report, measurements = build_report(root)
        assert report["mode"] == "smoke" and report["target_met"] is False
        assert report["layer_summary_matches_final_training_step"]
        assert len(measurements) == 2
        assert all(len(row["layers"]) == 14 for row in measurements)
        assert report["final_layer_summary"]["matrix_count"] == 12
        assert all(row["measured_seconds"] > 0 for row in measurements)
        make_plots(root, measurements)
        assert (root / "reports/training_and_layer_metrics.png").stat().st_size > 1000
        assert (root / "reports/training_and_layer_metrics.pdf").stat().st_size > 1000
        assert state["backup_status"]["status"] == "not_requested"
        # An incomplete newer endpoint must remove previously generated final tables.
        status_path = root / "TRAINING_STATUS.json"
        current = json.loads(status_path.read_text())
        current.update(step=3, tokens_seen=192)
        status_path.write_text(json.dumps(current))
        missing, _ = build_report(root)
        assert missing["final_layer_summary"] is None
        assert not (root / "reports/final_layers.json").exists()
        # Mixed-run results are rejected before they can become a report.
        current["run_id"] = "unrelated-run"
        status_path.write_text(json.dumps(current))
        with pytest.raises(ValueError, match="provenance"):
            build_report(root)
    finally:
        if not state.get("supervision_finished"):
            try:
                os.kill(launched["supervisor_pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass

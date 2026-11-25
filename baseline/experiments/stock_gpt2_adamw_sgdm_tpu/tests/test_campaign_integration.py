"""Real CPU training/WeightWatcher/selection; only the GitHub transport is fake.

The tiny synthetic model can never qualify for the FineWeb benchmark. This test
checks the new multi-run operational chain rather than claiming convergence.
"""
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]

from .. import campaign
from ..publish_metrics import Publisher
from .test_publish_metrics import MemoryGitHub


class ObservedPublisher(Publisher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_publications = []

    def publish_run(self, run_dir, run_key=None, final=False):
        receipt = super().publish_run(run_dir, run_key=run_key, final=final)
        self.run_publications.append({"run_key": run_key, "final": final,
                                      "receipt": receipt,
                                      "tracking": campaign.read_json(Path(run_dir) / "TRACKING_STATUS.json", {})})
        return receipt


def test_real_campaign_trains_selects_measures_and_publishes_incrementally(tmp_path):
    root = tmp_path / "campaign"
    head = subprocess.run(["git", "-C", str(campaign.REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    source_sha = head.stdout.strip() if head.returncode == 0 else "a" * 40
    args = campaign.parser().parse_args([
        "plan", "--campaign-dir", str(root), "--data-dir", str(tmp_path / "data"),
        "--campaign-id", "real-cpu-smoke", "--smoke", "--device", "cpu",
        "--source-sha", source_sha, "--min-free-gib", "0", "--poll-seconds", "0.1",
        "--publication-retry-seconds", "0", "--drain-timeout", "180", "--probe-timeout", "120",
    ])
    manifest = campaign.create_plan(args)
    assert manifest["optimizer"] == "adamw_sgdm_graft_v1"
    assert manifest["github_branch"] == "adamw-sgdm-tpu-results"
    assert "/stock_gpt2_adamw_sgdm_tpu/results/" in manifest["github_prefix"]
    api = MemoryGitHub(repo=manifest["github_repo"], branch=manifest["github_branch"],
                       source_sha=manifest["source_sha"])
    publisher = ObservedPublisher(manifest["github_repo"], manifest["github_branch"],
                                  manifest["github_prefix"], manifest["source_sha"],
                                  manifest["campaign_id"], api=api)
    state = campaign.Campaign(manifest, publisher=publisher).execute()

    assert state["workflow_complete"] and state["publication_complete"]
    assert state["status"] == "smoke_completed_not_a_benchmark"
    assert state["successful"] is False and state["benchmark_target_met"] is False
    records = state["run_records"]
    assert len(records) == 3
    assert sum(r["stage"] == "screen" for r in records.values()) == 2
    assert sum(r["stage"] == "full" for r in records.values()) == 1
    assert sum(r["tokens_seen"] for r in records.values()) == 256
    selection = json.loads((root / "search/full/selection.json").read_text())
    best_screen = min((r for r in records.values() if r["stage"] == "screen"),
                      key=lambda r: (r["validation"]["val_nll"], r["candidate_id"]))
    assert selection["selected_candidate_ids"] == [best_screen["candidate_id"]]

    remote = api.head_files()
    prefix = manifest["github_prefix"]
    assert remote["unrelated.txt"] == b"keep me\n"
    final = json.loads(remote[prefix + "/campaign/FINAL_REPORT.json"])
    assert final["benchmark_target_met"] is False
    assert final["all_runs_published"] and final["recorded_training_tokens"] == 256
    assert final["optimality_claim"] is False
    assert json.loads((root / "FINAL_PUBLICATION.json").read_text())["complete"] is True

    for key, record in records.items():
        run_dir = Path(record["run_dir"])
        run_manifest = json.loads((run_dir / "manifest.json").read_text())
        assert run_manifest["optimizer"] == "adamw_sgdm_graft_v1"
        assert run_manifest["optimizer_implementation"] == "TensorLrAdamWSGDMGraft"
        assert run_manifest["graft"]["physical_matrix_count"] == 4
        assert len(run_manifest["graft"]["parameter_names"]) == 4
        assert record["publication"]["complete"] is True
        assert record["tracking"]["complete"]
        assert record["tracking"]["failed"] == record["tracking"]["pending"] == 0
        assert math.isclose(record["validation"]["val_perplexity"],
                            math.exp(record["validation"]["val_nll"]), rel_tol=1e-12)
        launch = json.loads((run_dir / "LAUNCH.json").read_text())
        assert launch["resume"] is False
        assert (run_dir / "reports/training_and_layer_metrics.png").stat().st_size > 1000
        assert (run_dir / "reports/training_and_layer_metrics.pdf").stat().st_size > 1000
        assert not list((run_dir / "tracking/snapshots").glob("*.pt"))
        steps = record["tracking"]["completed_steps"]
        assert steps == [0, record["step"]]
        for step in steps:
            relative = f"tracking/measurements/{step:09d}.json"
            data = remote[f"{prefix}/runs/{key}/{relative}"]
            assert data == (run_dir / relative).read_bytes()
            measurement = json.loads(data)
            assert len(measurement["layers"]) == 8  # Six block matrices, two embeddings.
            assert measurement["summary"]["matrix_count"] == 6
            assert all(isinstance(layer["native_raw"], dict) and
                       isinstance(layer["native_clip_xmax"], dict) for layer in measurement["layers"])
            assert f"{prefix}/runs/{key}/tracking/layers/{step:09d}.csv" in remote
            assert f"{prefix}/runs/{key}/tracking/groups/{step:09d}.csv" in remote
        assert f"{prefix}/runs/{key}/tracking/layers.csv" in remote
        assert f"{prefix}/runs/{key}/reports/summary.json" in remote
        assert json.loads(remote[f"{prefix}/runs/{key}/manifest.json"])["optimizer"] == "adamw_sgdm_graft_v1"
        incremental = [p for p in publisher.run_publications if p["run_key"] == key and not p["final"]]
        assert incremental and all(p["receipt"].get("commit_sha") for p in incremental)
        assert any(p["receipt"]["published_steps"] == [0] for p in incremental)
    assert not any(name.endswith(".pt") or "/logs/" in name for name in remote)

    # A completed campaign is a no-op on resume; it must not retrain or republish.
    request_count = len(api.requests)
    resumed = campaign.Campaign(manifest, publisher=publisher).execute()
    assert resumed["workflow_complete"]
    assert len(api.requests) == request_count

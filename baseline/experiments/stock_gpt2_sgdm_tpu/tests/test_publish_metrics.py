"""Publication safety and exact scientific-artifact retention without network."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import smoke_config
from metrics import collect, expected_matrix_names, write_json as metric_json
from publish_metrics import (GitHubAPI, GitHubRedirectHandler, Publisher, PublishError, blob_sha, build_run_bundle,
                             github_token, json_bytes, safe_error, sha256)

SOURCE = "a" * 40
REPO = "Owner/project"
BRANCH = "sgdm-tpu-results"
PREFIX = "baseline/experiments/stock_gpt2_sgdm_tpu/results/campaign-1"


class MemoryGitHub:
    """Git objects with real blob hashes and non-force compare/parent semantics."""
    def __init__(self, repo=REPO, branch=BRANCH, source_sha=SOURCE):
        self.repo, self.branch, self.source_sha = repo, branch, source_sha
        self.blobs = {}
        self.trees = {}
        self.commits = {}
        self.refs = {}
        self.requests = []
        self.secrets = ("ghp_abcdefghijklmnopqrstuvwxyz1234567890",)
        self.fail = None
        self.race_on_patch = False
        self.race_used = False
        tree = self.store_tree({"unrelated.txt": b"keep me\n"})
        self.commits[source_sha] = {"tree": {"sha": tree}, "parents": []}

    def store_tree(self, files):
        entries = {}
        for path, data in files.items():
            sha = blob_sha(data)
            self.blobs[sha] = data
            entries[path] = {"path": path, "mode": "100644", "type": "blob", "sha": sha}
        return self.tree_entries(entries)

    def tree_entries(self, entries):
        sha = hashlib.sha1(json_bytes(entries)).hexdigest()
        self.trees[sha] = copy.deepcopy(entries)
        return sha

    def head_files(self):
        head = self.refs[self.branch]
        entries = self.trees[self.commits[head]["tree"]["sha"]]
        return {path: self.blobs[item["sha"]] for path, item in entries.items()}

    def concurrent_commit(self, path="other-campaign.json", data=b"{}\n"):
        old = self.refs[self.branch]
        entries = copy.deepcopy(self.trees[self.commits[old]["tree"]["sha"]])
        sha = blob_sha(data)
        self.blobs[sha] = data
        entries[path] = {"path": path, "mode": "100644", "type": "blob", "sha": sha}
        tree = self.tree_entries(entries)
        commit = hashlib.sha1((tree + old + "concurrent").encode()).hexdigest()
        self.commits[commit] = {"tree": {"sha": tree}, "parents": [old]}
        self.refs[self.branch] = commit

    def request(self, method, path, payload=None):
        self.requests.append((method, path, copy.deepcopy(payload)))
        if self.fail and method in ("POST", "PATCH"):
            raise self.fail
        base = "/repos/" + self.repo
        assert path.startswith(base)
        suffix = path[len(base):]
        if method == "GET" and suffix == "":
            return {"permissions": {"push": True}}
        if method == "GET" and suffix.startswith("/git/ref/heads/"):
            branch = suffix[len("/git/ref/heads/"):]
            if branch not in self.refs:
                raise PublishError("Not found", status_code=404)
            return {"ref": "refs/heads/" + branch, "object": {"type": "commit", "sha": self.refs[branch]}}
        if method == "GET" and suffix.startswith("/git/commits/"):
            return copy.deepcopy(self.commits[suffix[len("/git/commits/"):]])
        if method == "GET" and suffix.startswith("/git/trees/"):
            sha = suffix[len("/git/trees/"):].split("?")[0]
            return {"sha": sha, "truncated": False, "tree": list(copy.deepcopy(self.trees[sha]).values())}
        if method == "POST" and suffix == "/git/refs":
            branch = payload["ref"][len("refs/heads/"):]
            if branch in self.refs:
                raise PublishError("Already exists", status_code=422)
            self.refs[branch] = payload["sha"]
            return {"ref": payload["ref"], "object": {"sha": payload["sha"]}}
        if method == "POST" and suffix == "/git/blobs":
            assert payload["encoding"] == "base64"
            data = base64.b64decode(payload["content"])
            sha = blob_sha(data)
            self.blobs[sha] = data
            return {"sha": sha}
        if method == "POST" and suffix == "/git/trees":
            entries = copy.deepcopy(self.trees[payload["base_tree"]])
            for item in payload["tree"]:
                assert item["sha"] is not None
                entries[item["path"]] = item
            return {"sha": self.tree_entries(entries)}
        if method == "POST" and suffix == "/git/commits":
            sha = hashlib.sha1(json_bytes(payload)).hexdigest()
            self.commits[sha] = {"tree": {"sha": payload["tree"]}, "parents": payload["parents"]}
            return {"sha": sha}
        if method == "PATCH" and suffix.startswith("/git/refs/heads/"):
            branch = suffix[len("/git/refs/heads/"):]
            assert payload["force"] is False
            if self.race_on_patch and not self.race_used:
                self.race_used = True
                self.concurrent_commit()
            if self.commits[payload["sha"]]["parents"] != [self.refs[branch]]:
                raise PublishError("Reference update is not a fast forward", status_code=422)
            self.refs[branch] = payload["sha"]
            return {"ref": "refs/heads/" + branch, "object": {"sha": payload["sha"]}}
        raise AssertionError((method, path, payload))


def publisher(api=None, **changes):
    values = dict(repo=REPO, branch=BRANCH, prefix=PREFIX, source_sha=SOURCE,
                  campaign_id="campaign-1", api=api or MemoryGitHub())
    values.update(changes)
    return Publisher(**values)


def setup_run(root):
    root.mkdir(parents=True)
    config = smoke_config()
    config.write_json(root / "config.json")
    manifest = {"run_id": "test-run", "config_sha256": config.config_sha256,
                "spectral_tracking": {"n_layers": config.n_layer, "include_embeddings": config.include_embeddings,
                                      "expected_matrices": expected_matrix_names(config.n_layer, config.include_embeddings)}}
    metric_json(root / "manifest.json", manifest)
    metric_json(root / "TRAINING_STATUS.json", {"status": "training", "terminal": False, "step": 0,
                                               "run_id": "test-run", "config_sha256": config.config_sha256})
    return config, manifest


def add_measurement(root, config, manifest, step, *, terminal=False, status="budget_complete_target_not_met"):
    identity = {"run_id": "test-run", "step": step, "tokens_seen": step * config.tokens_per_step}
    fingerprint = hashlib.sha256(str(step).encode()).hexdigest()
    layers = [{**identity, "matrix_name": name, "alpha_raw": 2.2, "alpha_clip_xmax": 2.1,
               "native_raw": {"alpha": 2.2, "D": 0.1, "ERG_gap": 2, "unusual_native_field": [1, 2, 3]},
               "native_clip_xmax": {"alpha": 2.1, "D": 0.2, "ERG_gap": 3}}
              for name in expected_matrix_names(config.n_layer, config.include_embeddings)]
    validation = {**identity, "val_nll": 5.0, "val_perplexity": 148.413159,
                  "evaluation_tokens": config.validation_tokens, "validation_kind": "tuning", "validation_offset": 0}
    result = {**identity, "manifest": manifest, "content_sha256": fingerprint, "snapshot_sha256": "f" * 64,
              "layers": layers, "groups": [{**identity, "group": "Q", "alpha_raw_mean": 2.2}],
              "summary": {**identity, "alpha_raw_mean": 2.2}, "validation": validation,
              "coverage": {"status": "complete"}}
    metric_json(root / f"tracking/measurements/{step:09d}.json", result)
    metric_json(root / f"tracking/receipts/{step:09d}.json", {"content_sha256": fingerprint})
    steps = sorted(int(path.stem) for path in (root / "tracking/measurements").glob("*.json"))
    metric_json(root / "tracking/EXPECTED_STEPS.json", {"steps": steps})
    metric_json(root / "TRAINING_STATUS.json", {**identity, "status": status if terminal else "training",
                                               "terminal": terminal, "config_sha256": config.config_sha256})
    if terminal:
        metric_json(root / "TRAINING_DONE", {"status": status, "step": step})
    collect(root)
    return result


def test_incremental_native_layers_csv_final_and_idempotence(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    first = add_measurement(root, config, manifest, 0)
    api = MemoryGitHub()
    pub = publisher(api)
    proof = pub.preflight({"campaign": "immutable"})
    assert proof["status"] == "ready"
    result = pub.publish_run(root, "screen/candidate")
    assert result["published_steps"] == [0] and not result["complete"]
    files = api.head_files()
    base = PREFIX + "/runs/screen/candidate/"
    assert json.loads(files[base + "tracking/measurements/000000000.json"]) == first
    assert b"unusual_native_field" in files[base + "tracking/layers/000000000.csv"]
    assert base + "tracking/layers.csv" not in files
    original_head = api.refs[BRANCH]
    unchanged = pub.publish_run(root, "screen/candidate")
    assert unchanged["changed_files"] == 0 and api.refs[BRANCH] == original_head
    add_measurement(root, config, manifest, 2, terminal=True)
    final = pub.publish_run(root, "screen/candidate", final=True)
    assert final["complete"] and final["published_steps"] == [0, 2]
    assert base + "tracking/layers.csv" in api.head_files()
    assert base + "tracking/groups.csv" in api.head_files()
    assert api.head_files()["unrelated.txt"] == b"keep me\n"
    assert pub.publish_run(root, "screen/candidate", final=True)["changed_files"] == 0
    receipt = json.loads((root / "PUBLISH_STATUS.json").read_text())
    assert receipt["complete"] and receipt["commit_sha"]
    assert not any(path.endswith(".pt") for path in api.head_files())


def test_concurrent_ref_advance_preserves_other_campaign(tmp_path):
    config, manifest = setup_run(tmp_path / "run")
    add_measurement(tmp_path / "run", config, manifest, 0)
    api = MemoryGitHub()
    pub = publisher(api)
    pub.preflight()
    api.race_on_patch = True
    result = pub.publish_run(tmp_path / "run")
    assert result["status"] == "published" and api.race_used
    assert api.head_files()["other-campaign.json"] == b"{}\n"
    assert all(not request[2].get("force", False) for request in api.requests if request[0] == "PATCH")


def test_permission_probe_runs_on_resume_and_never_changes_iam():
    api = MemoryGitHub()
    pub = publisher(api)
    pub.preflight()
    count = len([path for path in api.head_files() if "/preflight/" in path])
    pub.preflight()
    assert len([path for path in api.head_files() if "/preflight/" in path]) == count + 1
    api.fail = PublishError("Forbidden", status_code=403)
    with pytest.raises(PublishError, match="Forbidden"):
        pub.preflight()
    assert all("/collaborators" not in path for _, path, _ in api.requests)


def test_transient_failure_retains_backlog_then_retries(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    add_measurement(root, config, manifest, 0)
    api = MemoryGitHub()
    pub = publisher(api)
    pub.preflight()
    pub.publish_run(root)
    add_measurement(root, config, manifest, 2, terminal=True)
    api.fail = PublishError("Temporary server error", retryable=True, status_code=503)
    with pytest.raises(PublishError):
        pub.publish_run(root, final=True)
    state = json.loads((root / "PUBLISH_STATUS.json").read_text())
    assert state["retryable"] and state["pending_steps"] == [2]
    assert state["last_successful_commit"] and state["local_artifacts_retained"]
    api.fail = None
    assert pub.publish_run(root, final=True)["complete"]


def test_remote_snapshot_rewrite_rejected(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    add_measurement(root, config, manifest, 0)
    pub = publisher()
    pub.preflight()
    pub.publish_run(root)
    path = root / "tracking/measurements/000000000.json"
    changed = json.loads(path.read_text())
    changed["layers"][0]["native_raw"]["alpha"] = 9.0
    metric_json(path, changed)
    collect(root)
    with pytest.raises(PublishError, match="Immutable"):
        pub.publish_run(root)


@pytest.mark.parametrize("changes", [{"branch": "main"}, {"branch": "x/../main"}, {"prefix": "../results/campaign-1"},
                                    {"prefix": "baseline/code"}, {"campaign_id": "../x"}, {"source_sha": "main"}])
def test_destination_boundaries(changes):
    with pytest.raises(PublishError):
        publisher(**changes)


def test_namespace_collision_refuses_other_campaign_source():
    api = MemoryGitHub()
    pub = publisher(api)
    pub.preflight()
    other = "b" * 40
    api.commits[other] = api.commits[SOURCE]
    with pytest.raises(PublishError, match="different immutable"):
        publisher(api, source_sha=other).preflight()


def test_partial_or_wrong_run_artifacts_not_published(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    add_measurement(root, config, manifest, 0)
    path = root / "tracking/layers.csv"
    path.write_text(path.read_text() + "partial row\n")
    with pytest.raises(PublishError, match="CSV generation"):
        build_run_bundle(root, "run")
    collect(root)
    measurement = root / "tracking/measurements/000000000.json"
    value = json.loads(measurement.read_text())
    value["run_id"] = "another-run"
    metric_json(measurement, value)
    with pytest.raises(PublishError, match="identity mismatch"):
        build_run_bundle(root, "run")


def test_missing_native_matrix_row_rejected(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    add_measurement(root, config, manifest, 0)
    path = root / "tracking/measurements/000000000.json"
    value = json.loads(path.read_text())
    value["layers"].pop()
    metric_json(path, value)
    with pytest.raises(PublishError, match="matrix rows"):
        build_run_bundle(root, "run")


def test_terminal_divergence_can_archive_requested_measurements(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    add_measurement(root, config, manifest, 0, terminal=True, status="diverged")
    status_path = root / "TRAINING_STATUS.json"
    state = json.loads(status_path.read_text())
    state.update(step=1, tokens_seen=config.tokens_per_step)
    metric_json(status_path, state)
    pub = publisher()
    pub.preflight()
    assert pub.publish_run(root, final=True)["complete"]
    state["status"] = "target_reached"
    metric_json(status_path, state)
    with pytest.raises(PublishError, match="every declared"):
        pub.publish_run(root, final=True)


def test_early_target_endpoint_does_not_require_full_horizon(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    add_measurement(root, config, manifest, 0)
    add_measurement(root, config, manifest, 1, terminal=True, status="target_reached")
    pub = publisher()
    pub.preflight()
    assert pub.publish_run(root, final=True)["complete"]


def test_token_errors_are_redacted_and_secret_metadata_is_refused(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    add_measurement(root, config, manifest, 0)
    api = MemoryGitHub()
    pub = publisher(api)
    pub.preflight()
    token = api.secrets[0]
    api.fail = PublishError("Failed request used " + token, retryable=True)
    with pytest.raises(PublishError) as error:
        pub.publish_run(root)
    assert token not in str(error.value)
    assert token not in (root / "PUBLISH_STATUS.json").read_text()
    api.fail = None
    manifest["credentials"] = token
    metric_json(root / "manifest.json", manifest)
    with pytest.raises(PublishError, match="Credential-like"):
        pub.publish_run(root)
    assert token not in (root / "PUBLISH_STATUS.json").read_text()
    assert safe_error("Authorization " + token, api.secrets).endswith("[REDACTED]")


def test_credentials_use_environment_or_capture_gh_without_argv_token(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "token-from-env")
    assert github_token() == "token-from-env"
    monkeypatch.delenv("GH_TOKEN")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr("publish_metrics.shutil.which", lambda _: "/usr/bin/gh")
    calls = []
    class Result:
        returncode = 0
        stdout = "token-from-gh\n"
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return Result()
    monkeypatch.setattr("publish_metrics.subprocess.run", run)
    assert github_token() == "token-from-gh"
    assert calls[0][0] == ["gh", "auth", "token", "--hostname", "github.com"]
    assert calls[0][1]["capture_output"] is True


def test_campaign_metadata_only_and_no_run_directory_creation(tmp_path):
    pub = publisher()
    pub.preflight()
    absent = tmp_path / "not-created"
    assert pub.publish_run(absent)["status"] == "waiting_for_artifacts"
    assert not absent.exists()
    root = tmp_path / "campaign"
    root.mkdir()
    metric_json(root / "CAMPAIGN.json", {"campaign_id": "campaign-1"})
    metric_json(root / "CAMPAIGN_STATUS.json", {"status": "complete"})
    (root / "logs").mkdir()
    (root / "logs/token.log").write_text(pub.api.secrets[0])
    (root / "checkpoints").mkdir()
    (root / "checkpoints/latest.pt").write_bytes(b"large weights")
    receipt = pub.publish_campaign(root, final=True)
    assert receipt["complete"] and receipt["commit_sha"]
    assert not any("token.log" in name or name.endswith(".pt") for name in pub.api.head_files())


def test_symlinked_artifacts_rejected(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    add_measurement(root, config, manifest, 0)
    path = root / "tracking/measurements/000000000.json"
    moved = tmp_path / "moved.json"
    path.replace(moved)
    path.symlink_to(moved)
    with pytest.raises(PublishError, match="Symlinked"):
        build_run_bundle(root, "run")


def test_stale_final_report_and_missing_expected_inventory_are_rejected(tmp_path):
    root = tmp_path / "run"
    config, manifest = setup_run(root)
    add_measurement(root, config, manifest, 2, terminal=True)
    metric_json(root / "reports/summary.json", {"run_id": "test-run", "step": 1, "tokens_seen": 64})
    with pytest.raises(PublishError, match="Final report identity"):
        build_run_bundle(root, "run", final=True)
    (root / "reports/summary.json").unlink()
    (root / "tracking/EXPECTED_STEPS.json").unlink()
    with pytest.raises(PublishError, match="every declared"):
        build_run_bundle(root, "run", final=True)


def test_cross_origin_redirect_cannot_forward_authorization():
    with pytest.raises(PublishError, match="another origin"):
        GitHubRedirectHandler().redirect_request(None, None, 302, "Found", {}, "https://evil.example/path")


def test_watch_waits_for_healthy_diagnostic_drain_before_finalizing(tmp_path, monkeypatch, capsys):
    import publish_metrics
    root = tmp_path / "run"
    root.mkdir()
    metric_json(root / "TRAINING_DONE", {"status": "target_reached"})
    metric_json(root / "TRACKING_STATUS.json", {"complete": False, "pending": 1})
    finals = []
    class WaitingPublisher:
        secrets = ()
        def publish_run(self, run_dir, run_key=None, final=False):
            finals.append(final)
            if final:
                return {"status": "published", "complete": True, "commit_sha": SOURCE, "url": "mock"}
            return {"status": "waiting_for_artifacts", "complete": False, "commit_sha": None}
    monkeypatch.setattr(publish_metrics, "Publisher", lambda *args: WaitingPublisher())
    monkeypatch.setattr(publish_metrics.time, "sleep", lambda _: metric_json(
        root / "TRACKING_STATUS.json", {"complete": True, "pending": 0}))
    result = publish_metrics.main(["watch", "--repo", REPO, "--branch", BRANCH, "--prefix", PREFIX,
        "--campaign-id", "campaign-1", "--source-sha", SOURCE, "--run-dir", str(root), "--max-failures", "1"])
    assert result == 0 and finals == [False, True]
    assert '"status": "failed"' not in capsys.readouterr().out

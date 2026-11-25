"""Incremental, verified GitHub publication of completed scientific artifacts.

No tensor checkpoints, arbitrary logs, environment dumps or credentials are
published. An explicit results branch and unique campaign subtree are required.
The Git database API preserves unrelated files and never force-updates a ref.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "baseline.experiments.stock_gpt2_adamw_sgdm_tpu"

SCHEMA_VERSION = 1
MAX_FILE_BYTES = 80 * 1024 * 1024
SECRET_KEYS = {"token", "access_token", "gh_token", "github_token", "authorization",
               "password", "secret", "credentials", "credential", "api_key", "private_key"}
TOKEN_PATTERN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,})\b")


class PublishError(RuntimeError):
    def __init__(self, message, *, retryable=False, status_code=None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def local_lock(root):
    root = Path(root)
    if not root.is_dir():
        raise PublishError("Publication source directory has not been created", retryable=True)
    with (root / "PUBLISH.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


@contextmanager
def export_lock(root):
    tracking = Path(root) / "tracking"
    if not tracking.is_dir():
        yield
        return
    with (tracking / ".collect.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        yield


def safe_error(error, secrets=()):
    message = str(error)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return TOKEN_PATTERN.sub("[REDACTED]", message)[:1500]


def github_token():
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    if shutil.which("gh"):
        result = subprocess.run(["gh", "auth", "token", "--hostname", "github.com"],
                                text=True, capture_output=True, timeout=15, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    raise PublishError("GitHub credentials unavailable. Authenticate gh or provide GH_TOKEN/GITHUB_TOKEN in the process environment.")


class GitHubAPI:
    def __init__(self, token=None, *, timeout=45):
        self._token = token or github_token()
        self.timeout = timeout

    @property
    def secrets(self):
        return (self._token,)

    def request(self, method, path, payload=None):
        if not path.startswith("/repos/") or "://" in path or ".." in path.split("/"):
            raise PublishError("Invalid GitHub API path")
        data = json_bytes(payload) if payload is not None else None
        request = urllib.request.Request("https://api.github.com" + path, data=data, method=method,
            headers={"Accept": "application/vnd.github+json", "Authorization": "Bearer " + self._token,
                     "X-GitHub-Api-Version": "2026-03-10", "User-Agent": "rg-adamw-sgdm-metric-publisher",
                     "Content-Type": "application/json"})
        try:
            opener = urllib.request.build_opener(GitHubRedirectHandler())
            with opener.open(request, timeout=self.timeout) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read()).get("message", "GitHub request failed")
            except (ValueError, AttributeError):
                message = "GitHub request failed"
            retryable = exc.code in (408, 409, 429, 500, 502, 503, 504)
            if exc.code == 403 and (exc.headers.get("Retry-After") or exc.headers.get("X-RateLimit-Remaining") == "0"):
                retryable = True
            raise PublishError(f"GitHub HTTP {exc.code}: {safe_error(message, self.secrets)}",
                               retryable=retryable, status_code=exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PublishError("GitHub transport failure: " + safe_error(exc, self.secrets), retryable=True) from None


class GitHubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https" or target.netloc != "api.github.com":
            raise PublishError("Refusing to forward GitHub authorization to another origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def valid_component(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value)) and value not in (".", "..")


def checked_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise PublishError("Expected an ordinary relative artifact path")
    parts = value.split("/")
    if any(not valid_component(part) for part in parts):
        raise PublishError("Artifact path contains an unsafe component")
    if any(part in (".github", ".git", "__pycache__", "snapshots", "checkpoints", "xla-cache") for part in parts):
        raise PublishError("Reserved/non-scientific artifact path")
    return str(PurePosixPath(value))


def checked_file(root, relative):
    relative = checked_relative(relative)
    path = Path(root) / relative
    if any(item.is_symlink() for item in (path, *path.parents) if item != Path(root).parent):
        raise PublishError("Symlinked scientific artifacts are not published")
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise PublishError(f"Scientific artifact exceeds {MAX_FILE_BYTES} byte publication limit: {relative}")
    return data


def reject_secrets(value, *, secrets=()):
    def walk(item):
        if isinstance(item, dict):
            for key, val in item.items():
                if str(key).lower().replace("-", "_") in SECRET_KEYS and val not in (None, "", "[REDACTED]"):
                    raise PublishError("Credential-like field found in scientific metadata; publication refused")
                walk(val)
        elif isinstance(item, list):
            for val in item:
                walk(val)
        elif isinstance(item, str):
            if TOKEN_PATTERN.search(item) or any(secret and secret in item for secret in secrets):
                raise PublishError("Credential-like text found in scientific metadata; publication refused")
    walk(value)


def _read_json(root, relative):
    return json.loads(checked_file(root, relative))


def csv_bytes(rows):
    columns = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    if columns:
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return buffer.getvalue().encode()


def build_run_bundle(run_dir, run_key, *, final=False):
    """Read one coherent committed export generation, with no numerical imports."""
    root = Path(run_dir)
    from .config import TrainingConfig
    from .metrics import expected_matrix_names
    config = TrainingConfig.from_dict(_read_json(root, "config.json"), require_hash=True)
    manifest = _read_json(root, "manifest.json")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or manifest.get("config_sha256") != config.config_sha256:
        raise PublishError("Run manifest/configuration identity mismatch")
    identity = {"schema_version": SCHEMA_VERSION, "run_key": run_key, "run_id": run_id,
                "config_sha256": config.config_sha256}
    files = {"run_identity.json": json_bytes(identity), "config.json": checked_file(root, "config.json"),
             "manifest.json": checked_file(root, "manifest.json")}
    immutable = set(files)
    if (root / "data_receipts.json").exists():
        files["data_receipts.json"] = checked_file(root, "data_receipts.json")
        immutable.add("data_receipts.json")
    training = _read_json(root, "TRAINING_STATUS.json") if (root / "TRAINING_STATUS.json").exists() else {}
    if training and (training.get("run_id") != run_id or training.get("config_sha256") != config.config_sha256):
        raise PublishError("Training status belongs to another immutable run")
    if training:
        files["TRAINING_STATUS.json"] = json_bytes(training)
    exports_path = root / "tracking/EXPORTS.json"
    if not exports_path.exists():
        if final:
            raise PublishError("Final publication requires committed spectral exports", retryable=True)
        return files, immutable, [], False, {**identity, "waiting_for_exports": True, "terminal": training.get("terminal", False)}
    before = checked_file(root, "tracking/EXPORTS.json")
    exports = json.loads(before)
    steps = exports.get("completed_steps")
    if not isinstance(steps, list) or any(type(step) is not int or step < 0 for step in steps) or steps != sorted(set(steps)):
        raise PublishError("Invalid committed spectral step inventory")
    names = expected_matrix_names(config.n_layer, config.include_embeddings)
    measurement_files = []
    for step in steps:
        name = f"tracking/measurements/{step:09d}.json"
        raw = checked_file(root, name)
        result = json.loads(raw)
        if (result.get("step") != step or result.get("tokens_seen") != step * config.tokens_per_step
                or result.get("run_id") != run_id or result.get("manifest", {}).get("config_sha256") != config.config_sha256):
            raise PublishError("Measurement step/token/run/configuration identity mismatch")
        if not re.fullmatch(r"[a-f0-9]{64}", result.get("content_sha256", "")):
            raise PublishError("Measurement has no immutable weight/validation fingerprint")
        validation = result.get("validation", {})
        if validation.get("step") != step or validation.get("tokens_seen") != step * config.tokens_per_step:
            raise PublishError("Measurement validation is not paired with its exact weights")
        layers = result.get("layers", [])
        actual = [row.get("matrix_name") for row in layers]
        if len(actual) != len(names) or set(actual) != set(names):
            raise PublishError("Measurement omits or duplicates declared matrix rows")
        for row in layers:
            if (row.get("step") != step or row.get("tokens_seen") != result["tokens_seen"] or row.get("run_id") != run_id
                    or not isinstance(row.get("native_raw"), dict) or not isinstance(row.get("native_clip_xmax"), dict)):
                raise PublishError("Layer identity/native-fit fields are incomplete")
        files[name] = raw
        immutable.add(name)
        measurement_files.append(name)
        for table in ("layers", "groups"):
            table_name = f"tracking/{table}/{step:09d}.csv"
            files[table_name] = csv_bytes(result[table])
            immutable.add(table_name)
        receipt_name = f"tracking/receipts/{step:09d}.json"
        if (root / receipt_name).exists():
            receipt = _read_json(root, receipt_name)
            if receipt.get("content_sha256") != result["content_sha256"]:
                raise PublishError("Measurement receipt fingerprint mismatch")
            if receipt.get("measurement_sha256") not in (None, sha256(raw)):
                raise PublishError("Measurement bytes differ from committed receipt")
            # Receipts can gain pruning metadata, so they are mutable, unlike measurements.
            files[receipt_name] = json_bytes(receipt)
    for name in ("layers.csv", "summary.csv", "groups.csv"):
        raw = checked_file(root, "tracking/" + name)
        if sha256(raw) != exports.get("files", {}).get(name):
            raise PublishError("Spectral CSV generation changed while publishing; retry", retryable=True)
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        if any(int(row["step"]) not in steps or row.get("run_id") != run_id for row in rows):
            raise PublishError("CSV rows do not match the committed snapshot generation")
        if name == "layers.csv" and len(rows) != len(steps) * len(names):
            raise PublishError("Layer CSV does not preserve every committed matrix row")
        if name == "summary.csv" and [int(row["step"]) for row in rows] != steps:
            raise PublishError("Summary CSV does not preserve every committed snapshot")
        # Individual layer/group snapshots are immutable small CSVs. Rewriting
        # a growing aggregate at every checkpoint would create quadratic Git
        # history/upload volume; aggregate layers/groups are published once at
        # the requested final checkpoint. Summary remains small and incremental.
        if name == "summary.csv" or final:
            files["tracking/" + name] = raw
    if checked_file(root, "tracking/EXPORTS.json") != before:
        raise PublishError("Spectral export generation changed while publishing; retry", retryable=True)
    files["tracking/LOCAL_EXPORTS.json"] = before
    files["tracking/EXPORTS.json"] = json_bytes({"schema_version": SCHEMA_VERSION,
        "completed_steps": steps, "source_local_exports_sha256": sha256(before),
        "files": {name.removeprefix("tracking/"): sha256(data) for name, data in files.items()
                  if name.startswith("tracking/") and name.endswith(".csv")}})
    tracking = _read_json(root, "TRACKING_STATUS.json")
    if (tracking.get("run_id") not in (None, run_id) or tracking.get("config_sha256") not in (None, config.config_sha256)
            or tracking.get("completed_steps") != steps):
        raise PublishError("Tracking status and export generation differ; retry", retryable=True)
    files["TRACKING_STATUS.json"] = json_bytes(tracking)
    expected_present = (root / "tracking/EXPECTED_STEPS.json").exists()
    expected = _read_json(root, "tracking/EXPECTED_STEPS.json").get("steps", []) if expected_present else []
    if (not isinstance(expected, list) or any(type(step) is not int or step < 0 for step in expected)
            or expected != sorted(set(expected))):
        raise PublishError("Invalid required spectral step inventory")
    if expected:
        files["tracking/EXPECTED_STEPS.json"] = json_bytes({"steps": expected})
    done = (root / "TRAINING_DONE").exists()
    if done:
        files["TRAINING_DONE"] = checked_file(root, "TRAINING_DONE")
    if (root / "DIAGNOSTICS_RECOVERY.json").exists():
        files["DIAGNOSTICS_RECOVERY.json"] = checked_file(root, "DIAGNOSTICS_RECOVERY.json")
    complete = (done and expected_present and bool(expected) and training.get("terminal") is True and tracking.get("complete") is True
                and tracking.get("failed") == tracking.get("pending") == 0 and set(expected).issubset(steps))
    if training.get("status") not in ("diverged", "failed_nonfinite"):
        complete = complete and training.get("step") in steps
    if final and not complete:
        raise PublishError("Final publication requires terminal training and every declared spectral checkpoint", retryable=True)
    if final:
        for name in ("reports/summary.json", "reports/final_layers.json", "reports/final_groups.json"):
            if not (root / name).exists():
                continue
            raw = checked_file(root, name)
            report = json.loads(raw)
            rows = [report] if isinstance(report, dict) else report
            if not isinstance(rows, list) or any(not isinstance(row, dict) or row.get("run_id") != run_id
                    or row.get("step") != training.get("step") or row.get("tokens_seen") != training.get("tokens_seen") for row in rows):
                raise PublishError("Final report identity differs from the training endpoint")
            files[name] = raw
        for name in ("reports/training_and_layer_metrics.png", "reports/training_and_layer_metrics.pdf"):
            if (root / name).exists():
                if "reports/summary.json" not in files:
                    raise PublishError("Plot export lacks its matching final report identity")
                files[name] = checked_file(root, name)
    meta = {**identity, "terminal": training.get("terminal", False), "requested_final": final,
            "final_artifacts_complete": complete, "published_steps": steps,
            "measurement_files": measurement_files, "export_generation_sha256": sha256(before)}
    return files, immutable, steps, complete, meta


class Publisher:
    def __init__(self, repo, branch, prefix, source_sha, campaign_id, api=None, *, max_conflict_retries=4):
        if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise PublishError("Expected an explicit owner/repository")
        if not isinstance(branch, str) or branch in ("main", "master", "HEAD") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,199}", branch) or any(
                part in ("", ".", "..") or part.endswith(".lock") or ".." in part for part in branch.split("/")):
            raise PublishError("Use an explicit dedicated results branch, not main/master")
        if not valid_component(campaign_id) or not re.fullmatch(r"[a-f0-9]{40}", source_sha or ""):
            raise PublishError("Campaign ID and pinned 40-character source commit are required")
        prefix = checked_relative(prefix)
        if not prefix.endswith("results/" + campaign_id) or prefix.split("/")[-2] != "results":
            raise PublishError("Publication prefix must end with results/<campaign_id>")
        if max_conflict_retries < 1:
            raise PublishError("Conflict retry count must be positive")
        self.repo, self.branch, self.prefix, self.source_sha, self.campaign_id = repo, branch, prefix, source_sha, campaign_id
        self.api = api or GitHubAPI()
        self.max_conflict_retries = max_conflict_retries
        self.base = "/repos/" + repo
        self.namespace = {"schema_version": SCHEMA_VERSION, "repo": repo, "branch": branch,
                          "prefix": prefix, "source_sha": source_sha, "campaign_id": campaign_id,
                          "purpose": "incremental scientific metrics; no tensor checkpoints or credentials"}

    @property
    def secrets(self):
        return getattr(self.api, "secrets", ())

    def _head(self):
        response = self.api.request("GET", self.base + "/git/ref/heads/" + urllib.parse.quote(self.branch, safe="/"))
        if response.get("ref") != "refs/heads/" + self.branch or response.get("object", {}).get("type") != "commit":
            raise PublishError("GitHub returned a different branch identity")
        return response["object"]["sha"]

    def _tree(self, head):
        commit = self.api.request("GET", self.base + "/git/commits/" + head)
        tree_sha = commit["tree"]["sha"]
        tree = self.api.request("GET", self.base + "/git/trees/" + tree_sha + "?recursive=1")
        if tree.get("truncated"):
            raise PublishError("GitHub tree listing is truncated; refusing an incomplete collision check")
        return tree_sha, {entry["path"]: entry for entry in tree["tree"]}

    def _paths(self, files):
        result = {}
        for relative, data in files.items():
            relative = checked_relative(relative)
            if not isinstance(data, bytes) or len(data) > MAX_FILE_BYTES:
                raise PublishError("Invalid/oversized publication payload")
            if relative.endswith(".json") or Path(relative).name == "TRAINING_DONE":
                reject_secrets(json.loads(data), secrets=self.secrets)
            elif relative.endswith((".csv", ".md")):
                reject_secrets(data.decode(), secrets=self.secrets)
            elif not relative.startswith("runs/") or not relative.endswith((".png", ".pdf")):
                raise PublishError("Unsupported scientific artifact type")
            result[self.prefix + "/" + relative] = data
        return result

    def _sync(self, files, immutable, message, *, trigger_paths=None):
        files = self._paths(files)
        immutable = {self.prefix + "/" + checked_relative(path) for path in immutable}
        trigger_paths = None if trigger_paths is None else {self.prefix + "/" + checked_relative(path) for path in trigger_paths}
        hashes = {path: blob_sha(data) for path, data in files.items()}
        namespace_path = self.prefix + "/PUBLISHING.json"
        for attempt in range(self.max_conflict_retries):
            head = self._head()
            tree_sha, entries = self._tree(head)
            expected_namespace = blob_sha(json_bytes(self.namespace))
            existing_namespace = entries.get(namespace_path)
            if existing_namespace and existing_namespace.get("sha") != expected_namespace:
                raise PublishError("Results prefix belongs to a different immutable campaign/source")
            if not existing_namespace and namespace_path not in files:
                raise PublishError("Publish preflight has not established this campaign namespace")
            if not existing_namespace and any(path.startswith(self.prefix + "/") for path in entries):
                raise PublishError("Results prefix is already occupied without its ownership receipt")
            changed = []
            for path, digest in hashes.items():
                existing = entries.get(path)
                if existing and (existing.get("type") != "blob" or existing.get("mode") != "100644"):
                    raise PublishError("Remote artifact has a non-file or nonstandard mode")
                if existing and existing["sha"] != digest and path in immutable:
                    raise PublishError("Immutable remote scientific artifact collision: " + path)
                if existing is None or existing["sha"] != digest:
                    changed.append(path)
            if not changed or (trigger_paths is not None and not set(changed) & trigger_paths):
                verified = {path: entries[path]["sha"] for path in hashes if path in entries and entries[path]["sha"] == hashes[path]}
                return {"commit_sha": head, "changed_files": 0, "verified_files": verified}
            updates = []
            for path in changed:
                blob = self.api.request("POST", self.base + "/git/blobs", {"content": base64.b64encode(files[path]).decode(), "encoding": "base64"})
                if blob.get("sha") != hashes[path]:
                    raise PublishError("GitHub blob checksum did not match local scientific bytes")
                updates.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
            tree = self.api.request("POST", self.base + "/git/trees", {"base_tree": tree_sha, "tree": updates})
            commit = self.api.request("POST", self.base + "/git/commits", {"message": message, "tree": tree["sha"], "parents": [head]})
            if self._head() != head:
                continue
            try:
                self.api.request("PATCH", self.base + "/git/refs/heads/" + urllib.parse.quote(self.branch, safe="/"),
                                 {"sha": commit["sha"], "force": False})
            except PublishError as exc:
                if exc.status_code in (409, 422) and self._head() != head:
                    continue
                raise
            published_head = self._head()
            _, published = self._tree(published_head)
            if any(published.get(path, {}).get("sha") != digest for path, digest in hashes.items()):
                raise PublishError("Published artifact readback differs from staged bytes", retryable=True)
            return {"commit_sha": published_head, "changed_files": len(changed), "verified_files": hashes}
        raise PublishError("Results branch advanced repeatedly; local artifacts retained for retry", retryable=True)

    def preflight(self, campaign_manifest=None):
        """Prove actual write permission before expensive work, without changing IAM."""
        repository = self.api.request("GET", self.base)
        if repository.get("permissions", {}).get("push") is False:
            raise PublishError("GitHub repository reports no write permission")
        self.api.request("GET", self.base + "/git/commits/" + self.source_sha)
        try:
            self._head()
        except PublishError as exc:
            if exc.status_code != 404:
                raise
            try:
                self.api.request("POST", self.base + "/git/refs", {"ref": "refs/heads/" + self.branch, "sha": self.source_sha})
            except PublishError as create_error:
                if create_error.status_code != 422:
                    raise
                self._head()  # Another publisher may have created this results branch.
        files = {"PUBLISHING.json": json_bytes(self.namespace)}
        if campaign_manifest is not None:
            files["campaign_manifest.json"] = json_bytes(campaign_manifest)
        # A same-prefix resume must still prove present write permission. A
        # unique, harmless receipt is retained as an auditable permission probe.
        files["preflight/" + uuid.uuid4().hex + ".json"] = json_bytes({"schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id, "source_sha": self.source_sha, "checked_unix": time.time()})
        result = self._sync(files, set(files), "Initialize AdamW SGDM-graft metric archive: " + self.campaign_id)
        return {"status": "ready", "repo": self.repo, "branch": self.branch, "prefix": self.prefix,
                "source_sha": self.source_sha, **result}

    def _receipt(self, root, action):
        path = Path(root) / "PUBLISH_STATUS.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        try:
            status = action()
            status.update(repo=self.repo, branch=self.branch, prefix=self.prefix,
                          source_sha=self.source_sha, campaign_id=self.campaign_id, updated_unix=time.time())
            write_json(path, status)
            return status
        except (PublishError, OSError, ValueError, KeyError) as exc:
            retryable = getattr(exc, "retryable", isinstance(exc, (OSError, json.JSONDecodeError)))
            state = {"status": "failed", "complete": False, "retryable": retryable,
                     "error": safe_error(exc, self.secrets), "updated_unix": time.time(),
                     "repo": self.repo, "branch": self.branch, "prefix": self.prefix,
                     "source_sha": self.source_sha, "campaign_id": self.campaign_id,
                     "last_successful_commit": previous.get("commit_sha", previous.get("last_successful_commit")),
                     "published_steps": previous.get("published_steps", []), "local_artifacts_retained": True}
            exports = Path(root) / "tracking/EXPORTS.json"
            if exports.exists():
                try:
                    available = json.loads(exports.read_text()).get("completed_steps", [])
                    state["pending_steps"] = sorted(set(available) - set(state["published_steps"]))
                except (ValueError, TypeError):
                    state["pending_steps"] = None
            write_json(path, state)
            raise PublishError(state["error"], retryable=retryable, status_code=getattr(exc, "status_code", None)) from None

    def publish_run(self, run_dir, run_key=None, final=False):
        root = Path(run_dir)
        run_key = checked_relative(run_key or root.name)
        if any(part in ("reports", "tracking") for part in run_key.split("/")):
            raise PublishError("Run key collides with reserved artifact directories")
        if not all((root / name).is_file() for name in ("config.json", "manifest.json")):
            if final:
                raise PublishError("Final publication requires initialized run artifacts", retryable=True)
            return {"status": "waiting_for_artifacts", "complete": False, "published_steps": [],
                    "pending_steps": [], "run_key": run_key, "commit_sha": None}
        with local_lock(root):
            def publish():
                with export_lock(root):
                    files, immutable, steps, complete, meta = build_run_bundle(root, run_key, final=final)
                base = "runs/" + run_key + "/"
                files["publication_manifest.json"] = json_bytes({**meta, "file_sha256": {name: sha256(data) for name, data in files.items()}})
                payload = {base + name: data for name, data in files.items()}
                immutable_paths = {base + name for name in immutable}
                triggers = immutable_paths | ({base + "TRAINING_STATUS.json"} if meta.get("terminal") else set())
                result = self._sync(payload, immutable_paths, f"Save AdamW SGDM-graft layer metrics: {self.campaign_id}/{run_key} through step {max(steps) if steps else 'initialization'}",
                                    trigger_paths=None if final else triggers)
                required = {self.prefix + "/" + base + name for name in immutable if name.startswith("tracking/measurements/")}
                if not required.issubset(result["verified_files"]):
                    raise PublishError("Not every completed snapshot has verified remote bytes", retryable=True)
                return {"status": "published", "complete": bool(final and complete), "run_key": run_key,
                        "run_id": meta["run_id"], "config_sha256": meta["config_sha256"],
                        "published_steps": steps, "pending_steps": [], "waiting_for_exports": meta.get("waiting_for_exports", False),
                        "url": f"https://github.com/{self.repo}/tree/{self.branch}/{self.prefix}/{base.rstrip('/')}", **result}
            return self._receipt(root, publish)

    def publish_campaign(self, campaign_dir, final=False):
        root = Path(campaign_dir)
        with local_lock(root):
            def publish():
                files = {}
                # Campaign owns these metadata trees; run artifacts have their own strict path.
                top_names = {"CAMPAIGN.json", "CAMPAIGN_STATUS.json", "CAMPAIGN_REPORT.json", "FINAL_REPORT.json", "FINAL_PUBLICATION.json",
                             "campaign.json", "campaign_spec.json", "manifest.json", "plan.json", "selection.json",
                             "winner.json", "README.md", "report.json", "summary.json"}
                for path in sorted(root.rglob("*")):
                    relative = path.relative_to(root)
                    if not path.is_file() or any(part in ("runs", "logs", "checkpoints", "snapshots", "xla-cache", "__pycache__") for part in relative.parts):
                        continue
                    accepted = (len(relative.parts) == 1 and path.name in top_names) or (
                        relative.parts[0] in ("search", "plans", "selections", "reports") and path.suffix == ".json")
                    if accepted:
                        raw = checked_file(root, relative.as_posix())
                        if path.suffix == ".json":
                            json.loads(raw)
                        files["campaign/" + relative.as_posix()] = raw
                if not files:
                    raise PublishError("No campaign metadata found for publication")
                files["campaign/PUBLICATION.json"] = json_bytes({"schema_version": SCHEMA_VERSION,
                    "campaign_id": self.campaign_id, "final": final,
                    "file_sha256": {name: sha256(data) for name, data in files.items()}})
                result = self._sync(files, set(), "Save AdamW SGDM-graft campaign status: " + self.campaign_id)
                return {"status": "published", "complete": bool(final), "url": f"https://github.com/{self.repo}/tree/{self.branch}/{self.prefix}", **result}
            return self._receipt(root, publish)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "once", "watch", "campaign"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-key")
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--max-failures", type=int, default=5)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.max_failures < 1:
        parser.error("Poll interval and maximum failures must be positive")
    if args.action in ("once", "watch") and args.run_dir is None:
        parser.error("Run publication requires --run-dir")
    if args.action == "campaign" and args.campaign_dir is None:
        parser.error("Campaign publication requires --campaign-dir")
    publisher = Publisher(args.repo, args.branch, args.prefix, args.source_sha, args.campaign_id)
    if args.action == "preflight":
        print(json.dumps(publisher.preflight(), indent=2))
        return 0
    failures = 0
    while True:
        try:
            if args.action == "campaign":
                status = publisher.publish_campaign(args.campaign_dir, final=args.final)
            else:
                # A terminal trainer can leave a healthy CPU diagnostic queue.
                # Draining that queue is not a failed publication attempt.
                tracking_path = args.run_dir / "TRACKING_STATUS.json"
                tracking_complete = (tracking_path.exists()
                                     and json.loads(tracking_path.read_text()).get("complete") is True)
                final = args.final or (args.action == "watch" and (args.run_dir / "TRAINING_DONE").exists()
                                       and tracking_complete)
                status = publisher.publish_run(args.run_dir, run_key=args.run_key, final=final)
            print(json.dumps({key: status.get(key) for key in ("status", "complete", "commit_sha", "url")}), flush=True)
            failures = 0
            if args.action != "watch" or status["complete"]:
                return 0
        except PublishError as exc:
            failures += 1
            print(json.dumps({"status": "failed", "retryable": exc.retryable,
                              "error": safe_error(exc, publisher.secrets), "consecutive_failures": failures}), flush=True)
            if args.action != "watch" or not exc.retryable or failures >= args.max_failures:
                return 1
        time.sleep(min(args.poll_seconds, 60))


if __name__ == "__main__":
    raise SystemExit(main())

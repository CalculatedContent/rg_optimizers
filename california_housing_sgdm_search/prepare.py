"""Restore exact published-cohort inputs for the SGDM extension.

This module never refits preprocessing or starts training. Every parent result
and checkpoint is pinned by commit, SHA-256 and Git blob SHA-1. Prepared data
retain the original byte-identical archive and preprocessing manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = "CalculatedContent/rg_optimizers"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha(payload):
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def verify_payload(payload, expected_sha256, expected_git_blob=None, expected_size=None):
    if expected_size is not None and len(payload) != expected_size:
        raise ValueError("Pinned input byte count differs")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Pinned input SHA-256 differs")
    if expected_git_blob is not None and git_blob_sha(payload) != expected_git_blob:
        raise ValueError("Pinned input Git blob SHA differs")


def write_immutable(path, payload):
    """Reuse identical bytes; refuse replacing a conflicting local input."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink input: {path}")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing frozen input differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def download_verified(repository, commit, repository_path, destination,
                      expected_sha256, expected_git_blob=None, expected_size=None):
    """Public immutable download; no environment token is sent to a raw host."""
    if repository != REPOSITORY:
        raise ValueError("Unexpected source repository")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Parent must be an immutable commit SHA")
    path = PurePosixPath(repository_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe parent repository path")
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError(f"Refusing a symlink input: {destination}")
    if destination.exists():
        verify_payload(destination.read_bytes(), expected_sha256,
                       expected_git_blob, expected_size)
        return destination
    url = (f"https://raw.githubusercontent.com/{repository}/{commit}/"
           + urllib.parse.quote(str(path), safe="/"))
    limit = expected_size if expected_size is not None else 100 * 1024 * 1024
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "rg-optimizers-reproduction"})
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read(limit + 1)
            if len(payload) > limit:
                raise ValueError("Pinned download exceeds its byte limit")
            verify_payload(payload, expected_sha256, expected_git_blob, expected_size)
            write_immutable(destination, payload)
            return destination
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(min(15, 2 ** attempt + random.random()))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(min(15, 2 ** attempt + random.random()))
    raise RuntimeError("Unreachable download state")


def verify_sources(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe source-manifest path")
        actual = PACKAGE.parent / path
        if sha256(actual) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return manifest


def prepare(root, plan_path=None, parent_manifest_path=None, source_manifest_path=None,
            downloader=download_verified, recover_raw=None):
    root = Path(root)
    plan_path = Path(plan_path or PACKAGE / "plan.json")
    parent_manifest_path = Path(parent_manifest_path or PACKAGE / "PARENT_MANIFEST.json")
    source_manifest_path = Path(source_manifest_path or PACKAGE / "SOURCE_SHA256.json")
    plan = json.loads(plan_path.read_text())
    parent = json.loads(parent_manifest_path.read_text())
    verify_sources(source_manifest_path)
    if parent["repository"] != REPOSITORY:
        raise ValueError("Wrong parent repository")
    continuation = plan["continuation"]
    for key, other in (("archive_commit", "archive_commit"),
                       ("github_run_id", "run_id"), ("launch_commit", "launch_commit")):
        if parent[key] != continuation[other]:
            raise ValueError(f"Plan and parent manifest disagree: {key}")
    seeds = sorted(map(int, parent["seeds"]))
    if seeds != sorted(plan["continuation_seeds"]):
        raise ValueError("Wrong continuation seed set")
    cohort_path = root / "preparation.json"
    if cohort_path.exists():
        previous = json.loads(cohort_path.read_text())
        current_run = os.environ.get("GITHUB_RUN_ID")
        if previous.get("github_run_id") and current_run and previous["github_run_id"] != current_run:
            raise ValueError("This result root already belongs to another GitHub cohort")

    for name, source in (("plan.json", plan_path),
                         ("SOURCE_SHA256.json", source_manifest_path),
                         ("parent_manifest.json", parent_manifest_path)):
        write_immutable(root / name, source.read_bytes())

    parent_root = plan["parent_results_root"]
    data_dir = root / "data"
    downloader(REPOSITORY, parent["archive_commit"], f"{parent_root}/data/manifest.json",
               data_dir / "manifest.json", parent["data_manifest_sha256"])
    manifest = json.loads((data_dir / "manifest.json").read_text())
    write_immutable(root / "data_manifest.json", (data_dir / "manifest.json").read_bytes())
    downloader(REPOSITORY, parent["archive_commit"], f"{parent_root}/data/prepared.npz",
               data_dir / "prepared.npz", manifest["prepared_npz_sha256"])

    # The unchanged original loader verifies raw arrays before reusing its
    # prepared archive. Recover those arrays without refitting anything.
    if recover_raw is None:
        from california_housing_benchmark.data import ensure_raw_data
        recover_raw = ensure_raw_data
    recover_raw(data_dir)
    for name, expected in manifest["raw_file_sha256"].items():
        if Path(name).name != name or sha256(data_dir / "raw" / name) != expected:
            raise ValueError(f"Original raw array changed: {name}")
    original_source = PACKAGE.parent / "california_housing_benchmark" / "data.py"
    if sha256(original_source) != manifest["preparation_source_sha256"]:
        raise ValueError("Original preprocessing source differs from the parent data manifest")

    tasks = []
    for seed in seeds:
        row = parent["seeds"][str(seed)]
        if row["seed"] != seed:
            raise ValueError("Manifest row has a different seed")
        checkpoint = continuation["checkpoints"][str(seed)]
        if (checkpoint["sha256"] != row["checkpoint_sha256"]
                or checkpoint["git_blob_sha"] != row["checkpoint_git_blob_sha"]
                or checkpoint["repository_path"] != row["checkpoint_repository_path"]):
            raise ValueError("Plan and parent checkpoint identities disagree")
        directory = root / "parent" / f"seed_{seed}"
        for filename, path_key, sha_key, blob_key, size in (
            ("result.json", "result_repository_path", "result_sha256", "result_git_blob_sha", None),
            ("checkpoint20k.pt", "checkpoint_repository_path", "checkpoint_sha256",
             "checkpoint_git_blob_sha", row["checkpoint_bytes"]),
            ("native_adamw_result.json", "native_adamw_result_repository_path",
             "native_adamw_result_sha256", "native_adamw_result_git_blob_sha", None),
        ):
            tasks.append((REPOSITORY, parent["archive_commit"], row[path_key],
                          directory / filename, row[sha_key], row[blob_key], size))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda args: downloader(*args), tasks))

    for seed in seeds:
        for name, optimizer in (("result.json", "sgdm"), ("native_adamw_result.json", "adamw")):
            result = json.loads((root / "parent" / f"seed_{seed}" / name).read_text())
            github = result["execution"]["github"]
            if (result["seed"] != seed or result["optimizer"] != optimizer
                    or result["status"] != "complete" or result["completed_updates"] < 20000
                    or result["data_manifest_sha256"] != parent["data_manifest_sha256"]
                    or str(github["GITHUB_RUN_ID"]) != str(parent["github_run_id"])
                    or github["GITHUB_SHA"] != parent["launch_commit"]):
                raise ValueError(f"Parent result provenance differs: {seed}/{optimizer}")

    provenance = {
        "schema_version": 1, "complete": True,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "launch_commit": os.environ.get("GITHUB_SHA"),
        "plan_file_sha256": sha256(plan_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "parent_manifest_sha256": sha256(parent_manifest_path),
        "data_manifest_sha256": parent["data_manifest_sha256"],
        "prepared_npz_sha256": manifest["prepared_npz_sha256"],
        "parent_archive_commit": parent["archive_commit"],
        "parent_seeds": seeds,
        "preprocessing": "Restored exact parent prepared bytes; no split or transform was refitted.",
    }
    write_immutable(cohort_path, (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(provenance, indent=2), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--parent-manifest")
    parser.add_argument("--source-manifest")
    args = parser.parse_args()
    prepare(args.root, args.plan, args.parent_manifest, args.source_manifest)


if __name__ == "__main__":
    main()

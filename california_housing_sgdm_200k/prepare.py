"""Freeze the five existing 100k checkpoints before any 200k training starts.

A clean sparse Git checkout pins one immutable parent archive commit. Existing
snapshot metadata is reused on recovery. No optimizer setting, seed, data split,
transform, or checkpoint is selected from test or spectral results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from california_housing_benchmark.publish import auth_env, git
from california_housing_sgdm_search.prepare import git_blob_sha, write_immutable
from .publish import _safe_git_error
from .verify import (PACKAGE, REPOSITORY, claim_cohort, read_json, sha256,
                     validate_effective_plan, verify_source)


PARENT_REPOSITORY = "CalculatedContent/rg_optimizers"
PARENT_RUN_ID = "45758895218"
PARENT_LAUNCH_COMMIT = "f89e5e00a3b9ad09c906b272fbc1d627ccecd714"
PARENT_STEP = 100000
PARENT_SEEDS = [2340, 2341, 2342, 2343, 2344]
PARENT_TRAIN_SHA256 = "93d8c691bb78b7c8d7d98c9606a93c806299141436e3303846619494f94938e5"
PARENT_SOURCE_MANIFEST_SHA256 = "72c3aaf260b8b06bca40d6dcf61486b09791283f3d65dea4402dc3b085c7908d"


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def validate_parent_source(plan):
    source = plan["parent_source"]
    expected = {"repository": PARENT_REPOSITORY, "run_id": PARENT_RUN_ID,
                "launch_commit": PARENT_LAUNCH_COMMIT, "checkpoint_step": PARENT_STEP}
    if any(str(source.get(key)) != str(value) for key, value in expected.items()):
        raise ValueError("Unexpected parent source for the authorized 200k continuation")
    if sorted(plan["continuation_seeds"]) != PARENT_SEEDS:
        raise ValueError("The original five SGDM seeds must all continue")
    if plan["start_step"] != PARENT_STEP or plan["total_updates"] != 200000:
        raise ValueError("The authorized continuation is exactly 100k to 200k")


def parent_paths(plan):
    """Map only required parent paths into the new isolated input directory."""
    prefix = plan["parent_source"]["results_root"]
    paths = {
        f"{prefix}/data/manifest.json": "data/manifest.json",
        f"{prefix}/data/prepared.npz": "data/prepared.npz",
    }
    for seed in plan["continuation_seeds"]:
        source = f"{prefix}/continuations/seed_{seed}"
        destination = f"parent/seed_{seed}"
        for filename in ("result.json", "github_execution.json", "completion.json"):
            paths[f"{source}/{filename}"] = f"{destination}/{filename}"
        paths[f"{source}/checkpoint_step_{PARENT_STEP}.pt"] = f"{destination}/checkpoint100k.pt"
    return paths


def restore_parent_snapshot(root, plan):
    """Restore byte-identical blobs from one commit; never change a pinned head."""
    root = Path(root)
    source = plan["parent_source"]
    snapshot_path = root / "parent_snapshot.json"
    frozen = read_json(snapshot_path) if snapshot_path.exists() else None
    if frozen is not None:
        for key, value in (("repository", source["repository"]),
                           ("branch", source["branch"]),
                           ("github_run_id", source["run_id"]),
                           ("launch_commit", source["launch_commit"])):
            if frozen.get(key) != value:
                raise ValueError(f"Pinned parent snapshot identity differs: {key}")
        commit = frozen.get("archive_commit", "")
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Pinned parent snapshot must identify an immutable commit")
    revision = frozen["archive_commit"] if frozen else source["branch"]
    paths = parent_paths(plan)
    env = auth_env()
    with tempfile.TemporaryDirectory(prefix="sgdm200k-parent-") as temporary:
        work = Path(temporary)
        try:
            git(["init", "-q"], cwd=work)
            git(["check-ref-format", "--branch", source["branch"]], cwd=work)
            git(["remote", "add", "origin", f"https://github.com/{PARENT_REPOSITORY}.git"], cwd=work)
            git(["fetch", "--depth=1", "--filter=blob:none", "origin", revision], cwd=work, env=env)
            commit = git(["rev-parse", "FETCH_HEAD"], cwd=work).stdout.decode().strip()
            if frozen and commit != frozen["archive_commit"]:
                raise ValueError("Git fetched a different parent snapshot")
            snapshot = {"schema_version": 1, "repository": source["repository"],
                        "branch": source["branch"], "github_run_id": source["run_id"],
                        "launch_commit": source["launch_commit"], "archive_commit": commit}
            write_immutable(snapshot_path, json_bytes(snapshot))
            git(["sparse-checkout", "init", "--no-cone"], cwd=work)
            git(["sparse-checkout", "set", "--no-cone", *["/" + name for name in paths]], cwd=work)
            git(["checkout", "--detach", "FETCH_HEAD"], cwd=work, env=env)
            metadata = {}
            for repository_path, destination in paths.items():
                path = work / repository_path
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"Pinned 100k parent file is missing: {repository_path}; restore it, do not retrain")
                payload = path.read_bytes()
                if len(payload) > 100 * 1024 * 1024:
                    raise ValueError("Parent input exceeds the declared archive limit")
                expected_git_blob = git(["rev-parse", f"{commit}:{repository_path}"], cwd=work).stdout.decode().strip()
                if git_blob_sha(payload) != expected_git_blob:
                    raise ValueError(f"Parent checkout bytes differ from Git object: {repository_path}")
                write_immutable(root / destination, payload)
                metadata[destination] = {"repository_path": repository_path,
                                         "git_blob_sha": expected_git_blob,
                                         "sha256": hashlib.sha256(payload).hexdigest(),
                                         "bytes": len(payload)}
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Cannot restore the pinned 100k archive: {_safe_git_error(exc)}") from exc
    return snapshot, metadata


def validate_parent_record(record, execution, completion, *, seed, config, data_hash):
    """Reject incomplete/different cohorts before reading trusted full states."""
    from california_housing_benchmark import train as base
    from california_housing_sgdm_search.long_train import BASE_HASHES
    source_bundle = base.canonical_sha256({**BASE_HASHES, "long_train.py": PARENT_TRAIN_SHA256})
    old_plan_hash = base.canonical_sha256(read_json(REPOSITORY / "california_housing_sgdm_search/plan.json"))
    required = {
        "status": "complete", "optimizer": "sgdm", "seed": seed,
        "phase": "continuation", "completed_updates": PARENT_STEP,
        "total_updates": PARENT_STEP, "start_step": 20000, "new_updates": 80000,
        "config": config, "data_manifest_sha256": data_hash,
        "source_bundle_sha256": source_bundle,
        "source_manifest_sha256": PARENT_SOURCE_MANIFEST_SHA256,
        "plan_sha256": old_plan_hash,
    }
    for key, value in required.items():
        if record.get(key) != value:
            raise ValueError(f"Parent result identity/completeness differs: seed {seed}, {key}")
    github = (record.get("execution") or {}).get("github") or {}
    for key, expected in (("GITHUB_RUN_ID", PARENT_RUN_ID),
                          ("GITHUB_SHA", PARENT_LAUNCH_COMMIT),
                          ("GITHUB_REPOSITORY", PARENT_REPOSITORY)):
        if str(github.get(key)) != expected:
            raise ValueError(f"Parent runtime provenance differs: {key}")
    shared = {"github_run_id": PARENT_RUN_ID, "launch_commit": PARENT_LAUNCH_COMMIT,
              "seed": seed, "phase": "continuation",
              "source_manifest_sha256": PARENT_SOURCE_MANIFEST_SHA256,
              "data_manifest_sha256": data_hash}
    for name, value in (("github_execution", execution), ("completion", completion)):
        if any(value.get(key) != expected for key, expected in shared.items()):
            raise ValueError(f"Parent {name} provenance differs")
    if completion.get("status") != "complete" or completion.get("completed_updates") != PARENT_STEP:
        raise ValueError("Parent completion marker does not describe 100,000 updates")
    rows = [row for row in record.get("history", []) if row.get("step") == PARENT_STEP]
    if (len(rows) != 1 or rows[0].get("weights_sha256") != record.get("final_weights_sha256")
            or rows[0].get("spectral_weights_sha256") != record.get("final_weights_sha256")):
        raise ValueError("Parent 100k performance and spectra do not identify the same weights")
    if not record.get("continuity", {}).get("verified"):
        raise ValueError("Parent did not verify exact continuation from 20k")


def validate_parent_checkpoint(path, record, *, seed, config, data_hash):
    import torch
    from california_housing_benchmark import train as base
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("checkpoint_kind") != "full_resume" or state.get("completed_updates") != PARENT_STEP:
        raise ValueError("Parent must retain full optimizer/RNG state at exactly 100k")
    identity = state.get("identity", {})
    for key in ("optimizer", "seed", "phase", "config", "plan_sha256", "data_manifest_sha256",
                "source_bundle_sha256", "source_manifest_sha256", "numerical_runtime_sha256"):
        if identity.get(key) != record.get(key):
            raise ValueError(f"Parent checkpoint/result identity differs: {key}")
    if state.get("config") != config or state.get("result", {}).get("status") != "complete":
        raise ValueError("Parent full state has different settings or incomplete metrics")
    if (identity.get("seed") != seed or identity.get("data_manifest_sha256") != data_hash
            or base.model_hash(state["model"]) != record["final_weights_sha256"]
            or state.get("minibatch_sha256") != record.get("minibatch_sha256")):
        raise ValueError("Parent full state does not match its final measured model")
    # This checks required sampler/momentum/RNG fields, not merely model weights.
    from california_housing_sgdm_search.long_train import checkpoint_fingerprints
    return checkpoint_fingerprints(state)


def prepare(root):
    root = Path(root)
    verify_source()
    cohort = claim_cohort(root)
    source_plan = read_json(PACKAGE / "plan.json")
    validate_parent_source(source_plan)
    write_immutable(root / "source_plan.json", (PACKAGE / "plan.json").read_bytes())
    write_immutable(root / "SOURCE_SHA256.json", (PACKAGE / "SOURCE_SHA256.json").read_bytes())
    snapshot, files = restore_parent_snapshot(root, source_plan)
    data = read_json(root / "data" / "manifest.json")
    data_hash = sha256(root / "data" / "manifest.json")
    original = read_json(REPOSITORY / "california_housing_sgdm_search/PARENT_MANIFEST.json")
    if data_hash != original["data_manifest_sha256"]:
        raise ValueError("100k parent uses different prepared data")
    if sha256(root / "data" / "prepared.npz") != data["prepared_npz_sha256"]:
        raise ValueError("Prepared array archive does not match its original data manifest")
    write_immutable(root / "data_manifest.json", (root / "data" / "manifest.json").read_bytes())
    # The original loader checks raw hashes even when using prepared arrays.
    from california_housing_benchmark.data import ensure_raw_data
    ensure_raw_data(root / "data")
    for name, expected in data["raw_file_sha256"].items():
        if Path(name).name != name or sha256(root / "data" / "raw" / name) != expected:
            raise ValueError(f"Original raw input differs: {name}")
    if sha256(REPOSITORY / "california_housing_benchmark/data.py") != data["preparation_source_sha256"]:
        raise ValueError("Original data preparation source differs")
    rows, checkpoints = {}, {}
    for seed in source_plan["continuation_seeds"]:
        directory = root / "parent" / f"seed_{seed}"
        result, execution, completion = (read_json(directory / name) for name in
                                         ("result.json", "github_execution.json", "completion.json"))
        validate_parent_record(result, execution, completion, seed=seed,
                               config=source_plan["continuation_config"], data_hash=data_hash)
        if completion.get("result_sha256") != sha256(directory / "result.json"):
            raise ValueError("Parent completed result bytes differ from completion marker")
        fingerprints = validate_parent_checkpoint(directory / "checkpoint100k.pt", result,
                      seed=seed, config=source_plan["continuation_config"], data_hash=data_hash)
        row = {"seed": seed, "weights_sha256": result["final_weights_sha256"],
               "state_fingerprints": fingerprints}
        for filename, prefix in (("result.json", "result"), ("checkpoint100k.pt", "checkpoint"),
                                 ("github_execution.json", "github_execution"), ("completion.json", "completion")):
            metadata = files[f"parent/seed_{seed}/{filename}"]
            row.update({f"{prefix}_{key}": value for key, value in metadata.items()})
        rows[str(seed)] = row
        checkpoints[str(seed)] = {key: row[f"checkpoint_{key}"]
                                   for key in ("sha256", "git_blob_sha", "repository_path")}
    parent = {"schema_version": 1, **snapshot,
              "checkpoint_step": PARENT_STEP, "data_manifest_sha256": data_hash,
              "prepared_npz_sha256": data["prepared_npz_sha256"],
              "parent_source_manifest_sha256": PARENT_SOURCE_MANIFEST_SHA256, "seeds": rows}
    write_immutable(root / "parent_manifest.json", json_bytes(parent))
    plan = deepcopy(source_plan)
    plan["source_plan_sha256"] = sha256(PACKAGE / "plan.json")
    plan["continuation"] = {"repository": PARENT_REPOSITORY, "run_id": PARENT_RUN_ID,
                            "launch_commit": PARENT_LAUNCH_COMMIT,
                            "archive_commit": snapshot["archive_commit"],
                            "checkpoint_step": PARENT_STEP, "checkpoints": checkpoints}
    validate_effective_plan(plan, source_plan, plan["source_plan_sha256"])
    write_immutable(root / "plan.json", json_bytes(plan))
    prepared = {"schema_version": 1, "complete": True,
                "github_run_id": cohort["github_run_id"], "launch_commit": cohort["launch_commit"],
                "source_plan_sha256": plan["source_plan_sha256"],
                "effective_plan_sha256": sha256(root / "plan.json"),
                "source_manifest_sha256": cohort["source_manifest_sha256"],
                "parent_manifest_sha256": sha256(root / "parent_manifest.json"),
                "data_manifest_sha256": data_hash,
                "prepared_npz_sha256": data["prepared_npz_sha256"],
                "parent_archive_commit": snapshot["archive_commit"],
                "parent_seeds": sorted(source_plan["continuation_seeds"]),
                "preprocessing": "Restored original prepared arrays and raw hashes; no preprocessing refit.",
                "continuation": "Verified all five complete 100k full states before authorizing any new update."}
    write_immutable(root / "preparation.json", json_bytes(prepared))
    print(json.dumps(prepared, indent=2), flush=True)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root)


if __name__ == "__main__":
    main()

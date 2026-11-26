"""Reuse all 64 measured 20k grid prefixes without launching development training.

Original records remain byte-identical and retain their actual longer source
budgets. Only the analysis window changes. Failed spectral fits remain measured
failures; they never trigger replacement training or exclusion from RMSE tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from california_housing_benchmark.publish import auth_env, git
from california_housing_sgdm_search.prepare import git_blob_sha, write_immutable
from .publish import _safe_git_error
from .verify import (CAP, PACKAGE, REPOSITORY, REPOSITORY_NAME, claim_cohort,
                     read_json, sha256, validate_plan, verify_source)


PARENT_TRAIN_SHA256 = "93d8c691bb78b7c8d7d98c9606a93c806299141436e3303846619494f94938e5"
PARENT_SOURCE_MANIFEST_SHA256 = "72c3aaf260b8b06bca40d6dcf61486b09791283f3d65dea4402dc3b085c7908d"


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def finite_rmse(row):
    value = (row.get("validation") or {}).get("rmse")
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def stage_archive(root, source, required, optional=None, inspect_only=None):
    """Fetch one immutable commit and copy selected blobs, never full checkpoints."""
    root, optional, inspect_only = Path(root), optional or {}, inspect_only or []
    if source["repository"] != REPOSITORY_NAME:
        raise ValueError("Unexpected source repository")
    commit = source["archive_commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Input archive must be an immutable commit")
    env = auth_env()
    with tempfile.TemporaryDirectory(prefix="sgdm20k-input-") as temporary:
        work = Path(temporary)
        try:
            git(["init", "-q"], cwd=work)
            git(["remote", "add", "origin", f"https://github.com/{REPOSITORY_NAME}.git"], cwd=work)
            git(["fetch", "--depth=1", "--filter=blob:none", "origin", commit], cwd=work, env=env)
            if git(["rev-parse", "FETCH_HEAD"], cwd=work).stdout.decode().strip() != commit:
                raise ValueError("Fetched input commit differs")
            listing = git(["ls-tree", "-r", "--long", "FETCH_HEAD", "--", source["results_root"]], cwd=work)
            tree = {}
            for line in listing.stdout.decode().splitlines():
                header, path = line.split("\t", 1)
                mode, kind, blob, size = header.split()
                if kind == "blob":
                    tree[path] = {"mode": mode, "git_blob_sha": blob, "bytes": int(size)}
            missing = sorted(set(required) - set(tree))
            if missing:
                raise ValueError("Pinned source records are missing; restore retained files without retraining: " + ", ".join(missing))
            selected = {**required, **{path: target for path, target in optional.items() if path in tree}}
            git(["sparse-checkout", "init", "--no-cone"], cwd=work)
            git(["sparse-checkout", "set", "--no-cone", *["/" + path for path in selected]], cwd=work)
            git(["checkout", "--detach", "FETCH_HEAD"], cwd=work, env=env)
            metadata = {}
            for repository_path, local_path in selected.items():
                if tree[repository_path]["mode"] not in ("100644", "100755"):
                    raise ValueError("Refusing a non-regular archive input")
                path = work / repository_path
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Archive input is not a regular file")
                payload = path.read_bytes()
                entry = tree[repository_path]
                if len(payload) != entry["bytes"] or git_blob_sha(payload) != entry["git_blob_sha"]:
                    raise ValueError("Checked-out input bytes differ from their Git object")
                write_immutable(root / local_path, payload)
                metadata[local_path] = {
                    "local_path": local_path, "repository_path": repository_path,
                    "repository": REPOSITORY_NAME, "archive_commit": commit,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "git_blob_sha": entry["git_blob_sha"], "bytes": len(payload)}
            availability = {path: {"repository_path": path,
                "available_in_parent_archive": path in tree,
                "git_blob_sha": tree.get(path, {}).get("git_blob_sha"),
                "bytes": tree.get(path, {}).get("bytes")}
                for path in inspect_only}
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Cannot restore frozen capped-study inputs: {_safe_git_error(exc)}") from exc
    return metadata, availability


def validate_development(record, execution, *, candidate, seed, source, data_hash):
    """Validate the available prefix without relabeling a 50k source as 20k."""
    from california_housing_benchmark import train as base
    from california_housing_benchmark.spectral import aggregate_matrices
    from california_housing_sgdm_search.long_train import BASE_HASHES
    from california_housing_sgdm_search.search import has_test_measurement
    old_plan = read_json(REPOSITORY / "california_housing_sgdm_search/plan.json")
    required = {"optimizer": "sgdm", "seed": seed, "phase": "development",
                "config": candidate["config"], "total_updates": 50000,
                "plan_sha256": base.canonical_sha256(old_plan),
                "data_manifest_sha256": data_hash,
                "source_manifest_sha256": PARENT_SOURCE_MANIFEST_SHA256,
                "source_bundle_sha256": base.canonical_sha256({**BASE_HASHES, "long_train.py": PARENT_TRAIN_SHA256})}
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(f"Original development identity differs: {candidate['id']}/{seed}, {key}")
    if (not isinstance(record.get("completed_updates"), int)
            or not CAP <= record["completed_updates"] <= 50000):
        raise ValueError("Missing measured 20k prefix; development training fallback is forbidden")
    if record.get("candidate_id", candidate["id"]) != candidate["id"]:
        raise ValueError("Original development candidate differs")
    if has_test_measurement(record):
        raise ValueError("Development input contains test measurements")
    for item in [record.get("execution", {})] + record.get("resume_executions", []):
        github = item.get("github") or {}
        for key, expected in (("GITHUB_RUN_ID", source["run_id"]),
                              ("GITHUB_SHA", source["launch_commit"]),
                              ("GITHUB_REPOSITORY", source["repository"])):
            if str(github.get(key)) != str(expected):
                raise ValueError(f"Original development execution differs: {key}")
    shared = {"github_run_id": source["run_id"], "launch_commit": source["launch_commit"],
              "candidate_id": candidate["id"], "seed": seed, "phase": "development",
              "source_manifest_sha256": PARENT_SOURCE_MANIFEST_SHA256,
              "data_manifest_sha256": data_hash}
    if any(execution.get(key) != expected for key, expected in shared.items()):
        raise ValueError("Original per-job execution record differs")
    history = [row for row in record.get("history", []) if row.get("step", CAP + 1) <= CAP]
    if [row.get("step") for row in history] != [0, 2000, 5000, 10000, 15000, CAP]:
        raise ValueError("Original fixed measurement prefix is incomplete")
    if any(not finite_rmse(row) for row in history):
        raise ValueError("Original 20k prefix contains a nonfinite validation error")
    epochs = [row for row in record.get("epoch_history", []) if row.get("step", CAP + 1) <= CAP]
    epoch_size = record.get("epoch_size")
    if epoch_size != 52 or [row.get("step") for row in epochs] != list(range(52, CAP + 1, 52)):
        raise ValueError("Original validation epoch coverage through 20k is incomplete")
    if any(not finite_rmse(row) for row in epochs):
        raise ValueError("Original validation epochs contain a nonfinite error")
    endpoint = history[-1]
    weights = endpoint.get("weights_sha256")
    if not weights or endpoint.get("spectral_weights_sha256") != weights:
        raise ValueError("20k performance and spectra identify different model weights")
    spectral = endpoint.get("spectral") or {}
    calculated = aggregate_matrices(spectral.get("matrices", []), spectral.get("eligible_matrices"))
    for key in ("n_valid", "complete", "confidence_tail_alpha"):
        if spectral.get(key) != calculated[key]:
            raise ValueError("Stored 20k spectral average differs from its original raw fits")
    # A failed fit is a valid measured outcome. Its alpha remains None.
    return {"source_completed_updates": record["completed_updates"],
            "source_status": record.get("status"), "analysis_max_step": CAP, "new_updates": 0,
            "n_validation_observations_through_cap": len(epochs) + 2,
            "n_epoch_observations_through_cap": len(epochs),
            "spectral_valid_at_cap": calculated["complete"],
            "alpha_at_cap": calculated["confidence_tail_alpha"]}


def validate_control(record, *, seed, optimizer, source, data_hash):
    from california_housing_benchmark import train as base
    from california_housing_sgdm_search.long_train import BASE_HASHES
    if (record.get("seed") != seed or record.get("optimizer") != optimizer
            or record.get("status") != "complete" or record.get("completed_updates") != CAP
            or record.get("data_manifest_sha256") != data_hash
            or record.get("source_bundle_sha256") != base.canonical_sha256(BASE_HASHES)):
        raise ValueError("Original 20k control identity differs")
    github = record.get("execution", {}).get("github") or {}
    for key, expected in (("GITHUB_RUN_ID", source["run_id"]),
                          ("GITHUB_SHA", source["launch_commit"]),
                          ("GITHUB_REPOSITORY", source["repository"])):
        if str(github.get(key)) != str(expected):
            raise ValueError("Original 20k control provenance differs")
    endpoints = [row for row in record.get("history", []) if row.get("step") == CAP]
    if len(endpoints) != 1 or not finite_rmse(endpoints[0]):
        raise ValueError("Original control lacks its 20k endpoint")


def prepare(root):
    root = Path(root)
    verify_source()
    cohort = claim_cohort(root)
    plan = read_json(PACKAGE / "plan.json")
    validate_plan(plan)
    write_immutable(root / "plan.json", (PACKAGE / "plan.json").read_bytes())
    write_immutable(root / "SOURCE_SHA256.json", (PACKAGE / "SOURCE_SHA256.json").read_bytes())
    source = plan["parent_source"]
    prefix = source["results_root"]
    required = {f"{prefix}/data/manifest.json": "data/manifest.json",
                f"{prefix}/data/prepared.npz": "data/prepared.npz"}
    optional, checkpoints = {}, []
    for candidate in plan["candidates"]:
        for seed in plan["development_seeds"]:
            old = f"{prefix}/development/{candidate['id']}/seed_{seed}"
            local = f"inputs/development/{candidate['id']}/seed_{seed}"
            for name in ("result.json", "github_execution.json"):
                required[f"{old}/{name}"] = f"{local}/{name}"
            for name in ("completion.json", "trial.json"):
                optional[f"{old}/{name}"] = f"{local}/{name}"
            checkpoints.append(f"{old}/checkpoint_step_20000.pt")
    files, availability = stage_archive(root, source, required, optional, checkpoints)
    data_hash = sha256(root / "data/manifest.json")
    original = read_json(REPOSITORY / "california_housing_sgdm_search/PARENT_MANIFEST.json")
    if data_hash != original["data_manifest_sha256"]:
        raise ValueError("Capped study does not reuse the original prepared data")
    data = read_json(root / "data/manifest.json")
    if sha256(root / "data/prepared.npz") != data["prepared_npz_sha256"]:
        raise ValueError("Prepared arrays differ from the original manifest")
    write_immutable(root / "data_manifest.json", (root / "data/manifest.json").read_bytes())
    from california_housing_benchmark.data import ensure_raw_data
    ensure_raw_data(root / "data")
    for name, expected in data["raw_file_sha256"].items():
        if Path(name).name != name or sha256(root / "data/raw" / name) != expected:
            raise ValueError("Original raw data identity differs")
    if sha256(REPOSITORY / "california_housing_benchmark/data.py") != data["preparation_source_sha256"]:
        raise ValueError("Original data preparation source differs")
    development = []
    for candidate in plan["candidates"]:
        for seed in plan["development_seeds"]:
            local = f"inputs/development/{candidate['id']}/seed_{seed}"
            result = read_json(root / local / "result.json")
            execution = read_json(root / local / "github_execution.json")
            checked = validate_development(result, execution, candidate=candidate, seed=seed,
                                            source=source, data_hash=data_hash)
            checkpoint_path = f"{prefix}/development/{candidate['id']}/seed_{seed}/checkpoint_step_20000.pt"
            development.append({"candidate_id": candidate["id"], "seed": seed,
                "action": "reuse_20k_history", **checked,
                "result": files[f"{local}/result.json"],
                "github_execution": files[f"{local}/github_execution.json"],
                "completion": files.get(f"{local}/completion.json"),
                "trial": files.get(f"{local}/trial.json"),
                "checkpoint_20000": availability[checkpoint_path]})
    controls_source = plan["controls_source"]
    control_paths = {}
    for seed in range(2340, 2345):
        for optimizer in ("sgdm", "adamw", "modified_adamw"):
            path = f"{controls_source['results_root']}/runs/seed_{seed}/{optimizer}/result.json"
            control_paths[path] = f"inputs/controls/seed_{seed}/{optimizer}/result.json"
    controls_files, _ = stage_archive(root, controls_source, control_paths)
    controls = []
    for seed in range(2340, 2345):
        for optimizer in ("sgdm", "adamw", "modified_adamw"):
            path = f"inputs/controls/seed_{seed}/{optimizer}/result.json"
            record = read_json(root / path)
            validate_control(record, seed=seed, optimizer=optimizer, source=controls_source, data_hash=data_hash)
            controls.append({"seed": seed, "optimizer": optimizer, "result": controls_files[path]})
    inventory = {"schema_version": 1, "parent_source": source,
        "controls_source": controls_source, "development": development, "controls": controls,
        "n_development_records": len(development), "all_have_20000_measurement": True,
        "additional_development_updates": 0, "analysis_max_step": CAP,
        "n_valid_20000_spectral_fits": sum(row["spectral_valid_at_cap"] for row in development),
        "n_failed_20000_spectral_fits": sum(not row["spectral_valid_at_cap"] for row in development),
        "n_retained_full_20000_checkpoints": sum(row["checkpoint_20000"]["available_in_parent_archive"] for row in development),
        "interpretation": "A post hoc restriction of previously authorized longer trajectories to observations at or before 20,000 updates. Original source completion counts and failures remain unchanged. No development updates are repeated."}
    write_immutable(root / "source_inventory.json", json_bytes(inventory))
    prepared = {"schema_version": 1, "complete": True,
        "github_run_id": cohort["github_run_id"], "launch_commit": cohort["launch_commit"],
        "plan_file_sha256": cohort["plan_file_sha256"],
        "source_manifest_sha256": cohort["source_manifest_sha256"],
        "data_manifest_sha256": data_hash, "prepared_npz_sha256": data["prepared_npz_sha256"],
        "source_inventory_sha256": sha256(root / "source_inventory.json"),
        "n_development_records": len(development), "n_controls": len(controls),
        "additional_development_updates": 0, "maximum_new_training_updates": CAP,
        "parent_archive_commit": source["archive_commit"],
        "controls_archive_commit": controls_source["archive_commit"]}
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

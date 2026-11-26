"""Verify the frozen 200k continuation and one GitHub cohort identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
REPOSITORY_NAME = "CalculatedContent/rg_optimizers"
REQUIRED = {
    "california_housing_sgdm_200k/plan.json",
    "california_housing_sgdm_200k/long_train.py",
    "california_housing_sgdm_200k/run_one.py",
    "california_housing_sgdm_200k/prepare.py",
    "california_housing_sgdm_200k/verify.py",
    "california_housing_sgdm_200k/publish.py",
    "california_housing_sgdm_search/SOURCE_SHA256.json",
    *(f"california_housing_benchmark/{name}" for name in
      ("data.py", "train.py", "optimizer.py", "spectral.py")),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_source():
    # The previous source bundle remains unchanged, including its original plan.
    from california_housing_sgdm_search.verify import verify_source as verify_parent
    verify_parent()
    manifest = read_json(PACKAGE / "SOURCE_SHA256.json")
    files = manifest.get("files", {})
    if not REQUIRED.issubset(files):
        raise RuntimeError("200k source manifest omits required files")
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe frozen source path")
        if sha256(REPOSITORY / relative) != expected:
            raise RuntimeError(f"200k source hash mismatch: {name}")
    print(f"Verified {len(files)} immutable 200k continuation files.", flush=True)
    return manifest


def cohort_identity():
    if not os.environ.get("GITHUB_RUN_ID") or not os.environ.get("GITHUB_SHA"):
        raise RuntimeError("GitHub cohort initialization requires the workflow environment")
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY_NAME:
        raise RuntimeError("Unexpected GitHub repository")
    return {
        "experiment_id": read_json(PACKAGE / "plan.json")["experiment_id"],
        "repository": REPOSITORY_NAME,
        "github_run_id": os.environ["GITHUB_RUN_ID"],
        "launch_commit": os.environ["GITHUB_SHA"],
        "source_plan_sha256": sha256(PACKAGE / "plan.json"),
        "source_manifest_sha256": sha256(PACKAGE / "SOURCE_SHA256.json"),
    }


def claim_cohort(root, initialize=False):
    path = Path(root) / "cohort_origin.json"
    expected = cohort_identity()
    if path.exists():
        if read_json(path) != expected:
            raise RuntimeError("Existing 200k cohort belongs to another workflow/source; resume its failed jobs")
    elif initialize:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    else:
        raise RuntimeError("Missing initialized 200k cohort identity")
    return expected


def validate_effective_plan(plan, source_plan, source_plan_sha256):
    """Only the hashes of predetermined parent inputs are resolved at launch."""
    if plan.get("source_plan_sha256") != source_plan_sha256:
        raise ValueError("Effective plan has a different source plan")
    if {key: value for key, value in plan.items()
            if key not in ("continuation", "source_plan_sha256")} != source_plan:
        raise ValueError("Effective plan changes a frozen scientific setting")
    contract = plan.get("continuation", {})
    parent = source_plan["parent_source"]
    for key in ("repository", "run_id", "launch_commit"):
        if str(contract.get(key)) != str(parent[key]):
            raise ValueError(f"Effective continuation provenance differs: {key}")
    if contract.get("checkpoint_step") != 100000:
        raise ValueError("The 200k phase must continue exactly 100,000 updates")
    commit = contract.get("archive_commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Parent archive is not an immutable commit")
    if sorted(map(int, contract.get("checkpoints", {}))) != sorted(source_plan["continuation_seeds"]):
        raise ValueError("Effective plan changes the predetermined parent seeds")
    for seed in source_plan["continuation_seeds"]:
        row = contract["checkpoints"][str(seed)]
        expected_path = (f"{parent['results_root']}/continuations/seed_{seed}/"
                         "checkpoint_step_100000.pt")
        if set(row) != {"sha256", "git_blob_sha", "repository_path"} or row["repository_path"] != expected_path:
            raise ValueError("Effective plan changes the predetermined parent checkpoint path")
        for key, size in (("sha256", 64), ("git_blob_sha", 40)):
            value = row[key]
            if not isinstance(value, str) or len(value) != size or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid parent checkpoint content hash")


def load_context(root):
    root = Path(root)
    verify_source()
    cohort = claim_cohort(root)
    source_plan_hash = sha256(PACKAGE / "plan.json")
    if sha256(root / "source_plan.json") != source_plan_hash:
        raise ValueError("Archived source plan differs")
    if sha256(root / "SOURCE_SHA256.json") != sha256(PACKAGE / "SOURCE_SHA256.json"):
        raise ValueError("Archived source manifest differs")
    plan = read_json(root / "plan.json")
    validate_effective_plan(plan, read_json(PACKAGE / "plan.json"), source_plan_hash)
    plan_hash = sha256(root / "plan.json")
    data_hash = sha256(root / "data" / "manifest.json")
    if sha256(root / "data_manifest.json") != data_hash:
        raise ValueError("Prepared and archived data manifests differ")
    prepared = read_json(root / "preparation.json")
    required = {
        "complete": True,
        "github_run_id": cohort["github_run_id"],
        "launch_commit": cohort["launch_commit"],
        "source_plan_sha256": source_plan_hash,
        "effective_plan_sha256": plan_hash,
        "source_manifest_sha256": cohort["source_manifest_sha256"],
        "parent_manifest_sha256": sha256(root / "parent_manifest.json"),
        "data_manifest_sha256": data_hash,
        "parent_archive_commit": plan["continuation"]["archive_commit"],
        "parent_seeds": sorted(plan["continuation_seeds"]),
    }
    if any(prepared.get(key) != value for key, value in required.items()):
        raise ValueError("200k inputs lack complete matching preparation")
    return root, plan, {
        "source_manifest_sha256": cohort["source_manifest_sha256"],
        "data_manifest_sha256": data_hash,
        "plan_file_sha256": plan_hash,
        "effective_plan_sha256": plan_hash,
        "source_plan_sha256": source_plan_hash,
        "github_run_id": cohort["github_run_id"],
        "launch_commit": cohort["launch_commit"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    verify_source()
    if args.root is not None:
        claim_cohort(args.root, args.initialize)
    elif args.initialize:
        parser.error("--initialize requires --root")


if __name__ == "__main__":
    main()

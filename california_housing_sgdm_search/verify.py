"""Verify immutable source and claim one GitHub cohort without duplicating it."""
import argparse
import hashlib
import json
import os
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    from california_housing_benchmark.verify import main as verify_parent
    verify_parent()
    manifest = json.loads((PACKAGE / "SOURCE_SHA256.json").read_text())
    parent_manifest = REPOSITORY / "california_housing_benchmark/SOURCE_SHA256.json"
    if sha256(parent_manifest) != manifest["frozen_parent_source_manifest_sha256"]:
        raise RuntimeError("Parent source manifest changed")
    for name, expected in manifest["files"].items():
        if sha256(REPOSITORY / name) != expected:
            raise RuntimeError(f"Extension source hash mismatch: {name}")
    print(f'Verified {len(manifest["files"])} immutable SGDM extension files.', flush=True)


def cohort_identity():
    if not os.environ.get("GITHUB_RUN_ID") or not os.environ.get("GITHUB_SHA"):
        raise RuntimeError("GitHub cohort initialization requires the actual workflow environment")
    return {
        "experiment_id": "2026-11-26_sgdm_long_search",
        "repository": os.environ["GITHUB_REPOSITORY"],
        "github_run_id": os.environ["GITHUB_RUN_ID"],
        "launch_commit": os.environ["GITHUB_SHA"],
        "plan_file_sha256": sha256(PACKAGE / "plan.json"),
        "source_manifest_sha256": sha256(PACKAGE / "SOURCE_SHA256.json"),
        "parent_manifest_sha256": sha256(PACKAGE / "PARENT_MANIFEST.json"),
    }


def claim_cohort(root, initialize=False):
    path = Path(root) / "cohort_origin.json"
    expected = cohort_identity()
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError("Existing cohort belongs to different source or workflow; resume its failed jobs only")
    elif initialize:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    else:
        raise RuntimeError("Missing initialized cohort identity")
    print("Verified the workflow's unique cohort identity.", flush=True)


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

"""Verify the 20k cap, frozen inputs, and one new confirmation cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
REPOSITORY_NAME = "CalculatedContent/rg_optimizers"
CAP = 20000
REQUIRED = {
    "california_housing_sgdm_20k/plan.json",
    "california_housing_sgdm_20k/run_one.py",
    "california_housing_sgdm_20k/prepare.py",
    "california_housing_sgdm_20k/verify.py",
    "california_housing_sgdm_20k/publish.py",
    "california_housing_sgdm_search/SOURCE_SHA256.json",
    "california_housing_sgdm_search/long_train.py",
    *(f"california_housing_benchmark/{name}" for name in
      ("data.py", "train.py", "optimizer.py", "spectral.py")),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_plan(plan):
    old = read_json(REPOSITORY / "california_housing_sgdm_search/plan.json")
    if plan.get("development_updates") != CAP or plan.get("total_updates") != CAP:
        raise ValueError("All new SGDM training must stop at exactly 20,000 updates")
    if plan.get("checkpoint_updates") != [2000, CAP] or plan.get("eval_every") != 5000:
        raise ValueError("Capped checkpoint/evaluation cadence changed")
    for key in ("candidates", "development_seeds", "confirmation_seeds"):
        if plan.get(key) != old[key]:
            raise ValueError(f"The original full grid and held-out seed lists must remain unchanged: {key}")
    source = plan.get("parent_source", {})
    required = {"repository": REPOSITORY_NAME, "run_id": "45758895218",
                "launch_commit": "f89e5e00a3b9ad09c906b272fbc1d627ccecd714",
                "source_manifest_sha256": "72c3aaf260b8b06bca40d6dcf61486b09791283f3d65dea4402dc3b085c7908d"}
    if any(str(source.get(key)) != str(value) for key, value in required.items()):
        raise ValueError("Frozen development source provenance differs")
    commit = source.get("archive_commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Development inputs must use an immutable archive commit")


def verify_source():
    from california_housing_sgdm_search.verify import verify_source as verify_parent
    verify_parent()
    manifest = read_json(PACKAGE / "SOURCE_SHA256.json")
    files = manifest.get("files", {})
    if not REQUIRED.issubset(files):
        raise RuntimeError("20k source manifest omits required files")
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe frozen source path")
        if sha256(REPOSITORY / relative) != expected:
            raise RuntimeError(f"20k orchestration source hash mismatch: {name}")
    validate_plan(read_json(PACKAGE / "plan.json"))
    print(f"Verified {len(files)} immutable capped-study files and the 20,000-update limit.", flush=True)
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
        "plan_file_sha256": sha256(PACKAGE / "plan.json"),
        "source_manifest_sha256": sha256(PACKAGE / "SOURCE_SHA256.json"),
    }


def claim_cohort(root, initialize=False):
    path = Path(root) / "cohort_origin.json"
    expected = cohort_identity()
    if path.exists():
        if read_json(path) != expected:
            raise RuntimeError("Existing capped cohort belongs to another workflow/source; resume its failed jobs")
    elif initialize:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    else:
        raise RuntimeError("Missing initialized capped cohort identity")
    return expected


def load_context(root):
    root = Path(root)
    verify_source()
    cohort = claim_cohort(root)
    if sha256(root / "plan.json") != cohort["plan_file_sha256"]:
        raise ValueError("Archived 20k plan differs from frozen source")
    if sha256(root / "SOURCE_SHA256.json") != cohort["source_manifest_sha256"]:
        raise ValueError("Archived orchestration manifest differs")
    plan = read_json(root / "plan.json")
    data_hash = sha256(root / "data" / "manifest.json")
    if sha256(root / "data_manifest.json") != data_hash:
        raise ValueError("Prepared and archived data manifests differ")
    inventory_hash = sha256(root / "source_inventory.json")
    inventory = read_json(root / "source_inventory.json")
    if (inventory.get("n_development_records") != 64
            or inventory.get("all_have_20000_measurement") is not True
            or inventory.get("additional_development_updates") != 0
            or inventory.get("parent_source") != plan["parent_source"]):
        raise ValueError("Complete reused 20k development inputs are required; no development training fallback")
    rows = inventory.get("development", [])
    expected_pairs = {(candidate["id"], seed) for candidate in plan["candidates"]
                      for seed in plan["development_seeds"]}
    if (len(rows) != 64 or {(row.get("candidate_id"), row.get("seed")) for row in rows} != expected_pairs
            or any(row.get("action") != "reuse_20k_history" or row.get("new_updates") != 0
                   or row.get("analysis_max_step") != CAP for row in rows)):
        raise ValueError("Reused development inventory has missing, duplicated, or changed entries")
    controls = inventory.get("controls", [])
    if (inventory.get("controls_source") != plan["controls_source"] or len(controls) != 15
            or {(row.get("seed"), row.get("optimizer")) for row in controls}
            != {(seed, optimizer) for seed in range(2340, 2345)
                for optimizer in ("sgdm", "adamw", "modified_adamw")}):
        raise ValueError("The original three five-seed controls are incomplete")
    prepared = read_json(root / "preparation.json")
    required = {"complete": True, "github_run_id": cohort["github_run_id"],
                "launch_commit": cohort["launch_commit"],
                "plan_file_sha256": cohort["plan_file_sha256"],
                "source_manifest_sha256": cohort["source_manifest_sha256"],
                "data_manifest_sha256": data_hash, "source_inventory_sha256": inventory_hash,
                "n_development_records": 64, "n_controls": 15, "additional_development_updates": 0}
    if any(prepared.get(key) != value for key, value in required.items()):
        raise ValueError("Capped inputs lack complete matching preparation")
    return root, plan, {
        "source_manifest_sha256": cohort["source_manifest_sha256"],
        "orchestration_source_manifest_sha256": cohort["source_manifest_sha256"],
        "training_source_manifest_sha256": sha256(REPOSITORY / "california_housing_sgdm_search/SOURCE_SHA256.json"),
        "data_manifest_sha256": data_hash,
        "plan_file_sha256": cohort["plan_file_sha256"],
        "source_inventory_sha256": inventory_hash,
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

"""Run only fresh SGDM confirmation seeds, with an enforced 20,000-update cap.

All development observations are reused from their untouched source records.
This module has no development/recovery training path. The numerical trainer
is the unchanged frozen long_train module; its source identity remains distinct
from this orchestration package's manifest.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from california_housing_benchmark import train as base
from california_housing_sgdm_search.long_train import train_long, checkpoint_fingerprints
from california_housing_sgdm_search.search import retain_weights as retain_exports

CAP = 20000
SEEDS = tuple(range(3340, 3345))
REPOSITORY = "CalculatedContent/rg_optimizers"
PACKAGE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_manifest_sha256():
    return sha256(PACKAGE.parent / "california_housing_sgdm_search/SOURCE_SHA256.json")


def check_identity(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Archived confirmation identity differs: {key}")


def check_cap(plan, seed):
    if seed not in SEEDS or plan.get("confirmation_seeds") != list(SEEDS):
        raise ValueError("Only the five declared fresh confirmation seeds may train")
    if plan.get("total_updates") != CAP or plan.get("development_updates") != CAP:
        raise ValueError("The hard training cap is exactly 20,000 total updates")
    if plan.get("eval_every") != 5000:
        raise ValueError("Confirmation evaluation cadence must remain 5,000 updates")
    checkpoints = plan.get("checkpoint_updates", [])
    if (not {2000, CAP}.issubset(checkpoints)
            or any(not isinstance(value, int) or not 0 < value <= CAP for value in checkpoints)):
        raise ValueError("Checkpoint budgets must include 2k/20k and never exceed 20k")


@contextmanager
def exclusive(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".wrapper.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another worker owns this confirmation; duplicate refused") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def execution_record(directory, identity, seed, frozen_hash):
    expected = {**identity, "seed": seed, "phase": "confirmation", "total_updates": CAP,
                "training_source_manifest_sha256": training_manifest_sha256(),
                "orchestration_source_manifest_sha256": identity["source_manifest_sha256"],
                "frozen_selection_sha256": frozen_hash}
    path = directory / "github_execution.json"
    if path.exists():
        check_identity(read_json(path), expected)
    base.atomic_json(path, {**expected, "repository": os.getenv("GITHUB_REPOSITORY"),
                            "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
                            "github_job": os.getenv("GITHUB_JOB")})


def validate_result(record, plan, identity, config, seed):
    check_cap(plan, seed)
    check_identity(record, {
        "optimizer": "sgdm", "seed": seed, "phase": "confirmation", "config": config,
        "plan_sha256": base.canonical_sha256(plan),
        "source_manifest_sha256": training_manifest_sha256(),
        "data_manifest_sha256": identity["data_manifest_sha256"],
        "start_step": 0, "total_updates": CAP, "completed_updates": CAP,
        "new_updates": CAP, "status": "complete", "parent_checkpoint_sha256": None,
    })
    expected_sources = {name: sha256(PACKAGE.parent / "california_housing_benchmark" / name)
                        for name in ("data.py", "train.py", "optimizer.py", "spectral.py")}
    expected_sources["long_train.py"] = sha256(PACKAGE.parent / "california_housing_sgdm_search/long_train.py")
    if record.get("source_bundle_sha256") != base.canonical_sha256(expected_sources):
        raise ValueError("Numerical trainer differs from its frozen source")
    for execution in [record.get("execution", {}), *record.get("resume_executions", [])]:
        github = execution.get("github", {})
        for key, expected in (("GITHUB_RUN_ID", identity["github_run_id"]),
                              ("GITHUB_SHA", identity["launch_commit"]),
                              ("GITHUB_REPOSITORY", REPOSITORY)):
            if str(github.get(key)) != str(expected):
                raise ValueError(f"Training execution is outside this cohort: {key}")
        check_identity(execution, {"source_sha256": expected_sources,
                                   "source_manifest_sha256": training_manifest_sha256(),
                                   "precision": "float32", "torch_num_threads": 1,
                                   "torch_num_interop_threads": 1})
    history = record.get("history", [])
    if any(not isinstance(row.get("step"), int) or not 0 <= row["step"] <= CAP for row in history):
        raise ValueError("Confirmation history exceeds the 20k cap")
    if len(history) != len({row["step"] for row in history}):
        raise ValueError("Confirmation history contains duplicate measurements")
    if {row["step"] for row in history} != {0, 2000, 5000, 10000, 15000, CAP}:
        raise ValueError("Confirmation history does not contain the declared measurement budgets")
    if any(not isinstance(row.get("step"), int) or not 0 < row["step"] <= CAP
           for row in record.get("epoch_history", [])):
        raise ValueError("Epoch history exceeds the20k cap")
    endpoint = record.get("primary_endpoint") or {}
    selected = record.get("best_validation") or {}
    if endpoint.get("step") != CAP or endpoint.get("weights_sha256") != record.get("final_weights_sha256"):
        raise ValueError("Fixed20k endpoint is not bound to final weights")
    window = {"start": 0, "end": CAP, "observed_through": CAP, "complete": True}
    if record.get("selection_window") != window or selected.get("selection_window") != window:
        raise ValueError("Validation selection escapes the 20k budget")
    if not isinstance(selected.get("step"), int) or not 0 <= selected["step"] <= CAP:
        raise ValueError("Selected checkpoint exceeds the cap")
    for row in [*history, selected]:
        for split in ("train", "validation", "test"):
            value = (row.get(split) or {}).get("rmse")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid measured {split} RMSE")
        if row.get("spectral_weights_sha256") != row.get("weights_sha256") or not row.get("weights_sha256"):
            raise ValueError("Performance and spectral observations refer to different weights")
    return record


def retain_checkpoints(directory, record, required_steps):
    """Validate full states and exports without replaying completed training."""
    required_steps = set(required_steps)
    required = {"resume.pt", *[f"checkpoint_step_{step}.pt" for step in required_steps]}
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        raise ValueError("Restore missing retained checkpoints; completed training must not rerun: " + ", ".join(missing))
    history = {row["step"]: row for row in record["history"]}
    retained, final = {}, None
    for step in sorted(required_steps):
        path = directory / f"checkpoint_step_{step}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("checkpoint_kind") != "full_resume" or state.get("completed_updates") != step or step > CAP:
            raise ValueError("Retained full checkpoint has the wrong update budget")
        check_identity(record, state["identity"])
        if base.model_hash(state["model"]) != history.get(step, {}).get("weights_sha256"):
            raise ValueError("Retained checkpoint weights differ from their measured result")
        retained[path.name] = sha256(path)
        if step == CAP:
            final = state
    if final is None:
        raise ValueError("Missing final20k checkpoint")
    resume = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    if resume.get("identity") != final["identity"] or checkpoint_fingerprints(resume) != checkpoint_fingerprints(final):
        raise ValueError("Resume state differs from the retained final20k state")
    retained["resume.pt"] = sha256(directory / "resume.pt")
    retained.update(retain_exports(directory, record, require_full_steps=required_steps))
    return retained


def run_one(root, seed):
    from .verify import load_context
    from .analysis import validate_frozen_selection
    root, plan, identity = load_context(root)
    check_cap(plan, seed)
    frozen = validate_frozen_selection(root, plan, identity)
    config = frozen["selected"]
    if config.get("optimizer") != "sgdm" or config not in [row["config"] for row in plan["candidates"]]:
        raise ValueError("Frozen winner is not a declared SGDM candidate")
    frozen_hash = sha256(root / "frozen_selection.json")
    directory = root / "runs" / f"seed_{seed}"
    with exclusive(directory):
        result_path = directory / "result.json"
        record = read_json(result_path) if result_path.exists() else None
        if record is not None and record.get("completed_updates", 0) > CAP:
            raise ValueError("An existing run above20k cannot be reused or relabeled")
        resume_path = directory / "resume.pt"
        if resume_path.exists():
            state = torch.load(resume_path, map_location="cpu", weights_only=False)
            if state.get("completed_updates", 0) > CAP:
                raise ValueError("A retained state above20k cannot resume in this experiment")
        if record is None or record.get("status") != "complete":
            execution_record(directory, identity, seed, frozen_hash)
            record = train_long(plan, config, root / "data", directory, seed,
                                phase="confirmation", parent_checkpoint=None, resume=True)
        else:
            check_identity(read_json(directory / "github_execution.json"),
                           {**identity, "seed": seed, "phase": "confirmation",
                            "frozen_selection_sha256": frozen_hash})
        validate_result(record, plan, identity, config, seed)
        additions = {"plan_file_sha256": identity["plan_file_sha256"],
                     "source_inventory_sha256": identity["source_inventory_sha256"],
                     "training_source_manifest_sha256": training_manifest_sha256(),
                     "orchestration_source_manifest_sha256": identity["source_manifest_sha256"],
                     "frozen_selection_sha256": frozen_hash,
                     "github_run_id": identity["github_run_id"], "launch_commit": identity["launch_commit"],
                     "report_primary_estimand": "fixed_20000_updates"}
        for key, value in additions.items():
            if key in record and record[key] != value:
                raise ValueError(f"Archived confirmation metadata differs: {key}")
        if any(key not in record for key in additions):
            record = {**record, **additions}
            base.atomic_json(result_path, record)
        retained = retain_checkpoints(directory, record, plan["checkpoint_updates"])
        completion = {**identity, "seed": seed, "phase": "confirmation", "status": "complete",
                      "completed_updates": CAP, "new_updates": CAP,
                      "training_source_manifest_sha256": training_manifest_sha256(),
                      "orchestration_source_manifest_sha256": identity["source_manifest_sha256"],
                      "result_sha256": sha256(result_path), "retained_files": retained,
                      "frozen_selection_sha256": frozen_hash}
        path = directory / "completion.json"
        if path.exists():
            if read_json(path) != completion:
                raise ValueError("Completion marker differs from the retained run")
        else:
            base.atomic_json(path, completion)
        print(f"Completed or reused SGDM confirmation seed={seed}; hard cap={CAP}", flush=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    run_one(args.root, args.seed)


if __name__ == "__main__":
    main()

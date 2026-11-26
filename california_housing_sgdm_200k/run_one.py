"""Resume one pinned SGDM seed from 100k to the fixed 200k endpoint."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path

import torch

from california_housing_benchmark import train as base
from .long_train import (PARENT_REPOSITORY, checkpoint_fingerprints,
                         file_sha256, train_long)


def read_json(path):
    return json.loads(Path(path).read_text())


def check_identity(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Archived continuation identity differs: {key}")


@contextmanager
def exclusive(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".wrapper.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {directory}; duplicate refused") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def execution_record(directory, identity, seed):
    expected = {**identity, "seed": seed, "phase": "continuation"}
    path = directory / "github_execution.json"
    if path.exists():
        check_identity(read_json(path), expected)
    record = {**expected, "repository": os.getenv("GITHUB_REPOSITORY"),
              "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
              "github_job": os.getenv("GITHUB_JOB")}
    base.atomic_json(path, record)


def validate_result(record, plan, identity, config, seed):
    start, total = int(plan["continuation"]["checkpoint_step"]), int(plan["total_updates"])
    parent_hash = plan["continuation"]["checkpoints"][str(seed)]["sha256"]
    check_identity(record, {
        "optimizer": "sgdm", "seed": seed, "phase": "continuation", "config": config,
        "plan_sha256": base.canonical_sha256(plan),
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "data_manifest_sha256": identity["data_manifest_sha256"],
        "parent_checkpoint_sha256": parent_hash,
        "start_step": start, "total_updates": total, "completed_updates": total,
        "new_updates": total - start, "primary_estimand": "fixed_updates", "status": "complete",
    })
    for execution in [record.get("execution", {}), *record.get("resume_executions", [])]:
        github = execution.get("github", {})
        for key, expected in (("GITHUB_RUN_ID", identity["github_run_id"]),
                              ("GITHUB_SHA", identity["launch_commit"]),
                              ("GITHUB_REPOSITORY", PARENT_REPOSITORY)):
            if str(github.get(key)) != str(expected):
                raise ValueError(f"Training execution is outside the frozen continuation cohort: {key}")
    parent = record.get("parent", {})
    check_identity(parent, {"checkpoint_sha256": parent_hash,
                            **{key: plan["continuation"][key] for key in (
                                "repository", "run_id", "launch_commit", "archive_commit", "checkpoint_step")}})
    continuity = record.get("continuity", {})
    if (not continuity.get("verified") or continuity.get("parent") != continuity.get("start")
            or continuity.get("start", {}).get("completed_updates") != start
            or continuity.get("start", {}).get("model_sha256") != parent.get("weights_sha256")):
        raise ValueError("Completed continuation lacks exact parent-state continuity")
    endpoint, selected = record.get("primary_endpoint") or {}, record.get("best_validation") or {}
    if endpoint.get("step") != total or endpoint.get("weights_sha256") != record.get("final_weights_sha256"):
        raise ValueError("Fixed-budget endpoint differs from the retained final model")
    window = {"start": start, "end": total, "observed_through": total, "complete": True}
    if record.get("selection_window") != window or selected.get("selection_window") != window:
        raise ValueError("Optional validation selection does not cover exactly the continuation window")
    if not isinstance(selected.get("step"), int) or not start <= selected["step"] <= total:
        raise ValueError("Optional validation selection escapes the continuation window")
    for row in (endpoint, selected):
        for split in ("train", "validation", "test"):
            value = (row.get(split) or {}).get("rmse")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise FloatingPointError(f"Non-finite measured {split} RMSE")
        if plan.get("measure_spectra", True) and row.get("spectral_weights_sha256") != row.get("weights_sha256"):
            raise ValueError("Performance and spectral measurements refer to different weights")
    return record


def retain_weights(directory, record, required_steps):
    """Validate every full checkpoint and produce recoverable compact exports."""
    directory = Path(directory)
    required_steps = set(required_steps)
    missing = [name for name in ["resume.pt", *[f"checkpoint_step_{step}.pt" for step in sorted(required_steps)]]
               if not (directory / name).is_file()]
    if missing:
        raise ValueError("Completed run lacks retained full checkpoints; restore its archive, do not retrain: "
                         + ", ".join(missing))
    history = {row["step"]: row for row in record["history"]}
    final = None
    retained = {}
    for step in sorted(required_steps):
        path = directory / f"checkpoint_step_{step}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("checkpoint_kind") != "full_resume" or state.get("completed_updates") != step:
            raise ValueError(f"Retained checkpoint has wrong kind or update count: {path.name}")
        check_identity(record, state["identity"])
        actual_hash = base.model_hash(state["model"])
        if step not in history or actual_hash != history[step].get("weights_sha256"):
            raise ValueError(f"Retained weights differ from their measured checkpoint: {path.name}")
        if step == record["start_step"] and checkpoint_fingerprints(state) != record["continuity"]["start"]:
            raise ValueError("Retained continuation start differs from the exact parent state")
        if step == record["completed_updates"]:
            final = state
        retained[path.name] = file_sha256(path)
    if final is None:
        raise ValueError("Required full checkpoints omit the final endpoint")
    resume = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    if (resume.get("identity") != final["identity"]
            or checkpoint_fingerprints(resume) != checkpoint_fingerprints(final)):
        raise ValueError("Resume state differs from the retained final checkpoint")
    retained["resume.pt"] = file_sha256(directory / "resume.pt")
    exports = (
        ("best_checkpoint.pt", final["best_weights"], record["best_weights_sha256"],
         {"selected_step": record["best_validation"]["step"],
          "selected_epoch": record["best_validation"]["epoch"],
          "metrics": record["best_validation"], "selection_window": record["selection_window"]}),
        ("final_weights.pt", final["model"], record["final_weights_sha256"],
         {"completed_updates": record["completed_updates"]}),
    )
    for filename, weights, expected_hash, extra in exports:
        if base.model_hash(weights) != expected_hash:
            raise ValueError(f"Full checkpoint does not contain the measured weights: {filename}")
        path = directory / filename
        if not path.exists():
            base.atomic_checkpoint(path, {"checkpoint_kind": "weights_only", "identity": final["identity"],
                                         "model": weights, "weights_sha256": expected_hash, **extra})
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved.get("identity") != final["identity"] or saved.get("weights_sha256") != expected_hash
                or base.model_hash(saved["model"]) != expected_hash
                or any(saved.get(key) != value for key, value in extra.items())):
            raise ValueError(f"Retained export differs from its measured model: {filename}")
        retained[filename] = file_sha256(path)
    return retained


def run_one(root, seed):
    from .verify import load_context
    root, plan, identity = load_context(root)
    if seed not in plan["continuation_seeds"]:
        raise ValueError("Unexpected continuation seed")
    config = plan["continuation_config"]
    parent = root / "parent" / f"seed_{seed}" / "checkpoint100k.pt"
    expected = plan["continuation"]["checkpoints"][str(seed)]["sha256"]
    if not parent.is_file() or file_sha256(parent) != expected:
        raise ValueError("Pinned 100k checkpoint is unavailable or differs from its archive")
    directory = root / "continuations" / f"seed_{seed}"
    with exclusive(directory):
        result_path = directory / "result.json"
        record = read_json(result_path) if result_path.exists() else None
        if record is None or record.get("status") != "complete":
            execution_record(directory, identity, seed)
            record = train_long(plan, config, root / "data", directory, seed,
                                phase="continuation", parent_checkpoint=parent, resume=True)
        else:
            check_identity(read_json(directory / "github_execution.json"),
                           {**identity, "seed": seed, "phase": "continuation"})
        validate_result(record, plan, identity, config, seed)
        additions = {key: identity[key] for key in (
            "plan_file_sha256", "effective_plan_sha256", "source_plan_sha256", "github_run_id", "launch_commit")}
        for key, value in additions.items():
            if key in record and record[key] != value:
                raise ValueError(f"Retained result has changed metadata: {key}")
        if any(key not in record for key in additions):
            record = {**record, **additions}
            base.atomic_json(result_path, record)
        start, total = plan["continuation"]["checkpoint_step"], plan["total_updates"]
        required = {int(step) for step in plan["checkpoint_updates"] if start <= int(step) <= total} | {start, total}
        retained = retain_weights(directory, record, required)
        completion = {**identity, "seed": seed, "phase": "continuation", "status": "complete",
                      "completed_updates": total, "start_step": start, "new_updates": total - start,
                      "result_sha256": file_sha256(result_path), "retained_files": retained,
                      "parent_checkpoint_sha256": expected}
        existing = directory / "completion.json"
        if existing.exists():
            if read_json(existing) != completion:
                raise ValueError("Completed continuation archive differs from its retained completion record")
        else:
            base.atomic_json(existing, completion)
        print(f"Completed or reused continuation seed={seed}, updates={total}", flush=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    result = run_one(args.root, args.seed)
    print(json.dumps({key: result[key] for key in ("seed", "phase", "status", "completed_updates")}))


if __name__ == "__main__":
    main()

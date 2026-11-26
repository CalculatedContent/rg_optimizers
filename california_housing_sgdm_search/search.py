"""Run the complete frozen development grid and select by validation RMSE."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import traceback

import torch

from california_housing_benchmark import train as base
from .long_train import train_long

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCE_FILES = {
    "california_housing_sgdm_search/plan.json",
    "california_housing_sgdm_search/long_train.py",
    "california_housing_sgdm_search/search.py",
    "california_housing_sgdm_search/run_one.py",
    *(f"california_housing_benchmark/{name}" for name in ("data.py", "train.py", "optimizer.py", "spectral.py")),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(path, value)


@contextmanager
def exclusive(directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def check_identity(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Archived protocol/cohort identity differs: {key}")


def load_context(root):
    root = Path(root)
    plan = read_json(root / "plan.json")
    source = read_json(root / "SOURCE_SHA256.json")
    files = source.get("files", {})
    if not REQUIRED_SOURCE_FILES.issubset(files):
        raise ValueError("Frozen source manifest omits required training/selection files")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid frozen source path")
        if digest(PACKAGE_ROOT / relative) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    plan_hash = digest(root / "plan.json")
    if plan_hash != files["california_housing_sgdm_search/plan.json"]:
        raise ValueError("Archived plan differs from the frozen source plan")
    data_hash = digest(root / "data" / "manifest.json")
    if digest(root / "data_manifest.json") != data_hash:
        raise ValueError("Prepared and archived data manifests differ")
    identity = {
        "source_manifest_sha256": digest(root / "SOURCE_SHA256.json"),
        "data_manifest_sha256": data_hash, "plan_file_sha256": plan_hash,
        "github_run_id": os.getenv("GITHUB_RUN_ID"), "launch_commit": os.getenv("GITHUB_SHA"),
    }
    return root, plan, identity


def execution_record(directory, identity, seed, phase, candidate_id=None):
    path = Path(directory) / "github_execution.json"
    record = {**identity, "seed": seed, "phase": phase, "candidate_id": candidate_id,
              "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
              "github_job": os.getenv("GITHUB_JOB")}
    if path.exists():
        previous = read_json(path)
        check_identity(previous, {**identity, "seed": seed, "phase": phase,
                                  "candidate_id": candidate_id})
    write_json(path, record)


def has_test_measurement(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "test" and child not in (None, {}):
                return True
            if key.startswith("test_") and isinstance(child, (int, float)):
                return True
            if has_test_measurement(child):
                return True
    elif isinstance(value, list):
        return any(has_test_measurement(child) for child in value)
    return False


def validate_result(record, plan, identity, config, seed, phase, complete=True):
    expected = {"seed": seed, "phase": phase, "config": config,
                "plan_sha256": base.canonical_sha256(plan),
                "source_manifest_sha256": identity["source_manifest_sha256"],
                "data_manifest_sha256": identity["data_manifest_sha256"]}
    check_identity(record, expected)
    for execution in [record.get("execution", {})] + record.get("resume_executions", []):
        github = execution.get("github", {})
        if (github.get("GITHUB_RUN_ID") != identity["github_run_id"]
                or github.get("GITHUB_SHA") != identity["launch_commit"]):
            raise ValueError("Training execution is outside this frozen cohort")
    if phase == "development" and has_test_measurement(record):
        raise ValueError("Development record contains test measurements")
    if complete:
        budget = int(plan["development_updates"] if phase == "development" else plan["total_updates"])
        if record.get("status") != "complete" or record.get("completed_updates") != budget:
            raise ValueError("Training did not complete its declared fixed budget")
        value = record.get("best_validation", {}).get("validation", {}).get("rmse")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise FloatingPointError("Non-finite best validation RMSE")
    return record


def augment_result(directory, record, identity, **extra):
    additions = {"plan_file_sha256": identity["plan_file_sha256"], **extra}
    for key, value in additions.items():
        if key in record and record[key] != value:
            raise ValueError(f"Archived result metadata differs: {key}")
    if any(key not in record for key in additions):
        record = {**record, **additions}
        write_json(Path(directory) / "result.json", record)
    return record


def _checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def retain_weights(directory, record, *, require_full_steps=()):
    """Validate compact completion or repair missing exports without training."""
    directory = Path(directory)
    total = record["completed_updates"]
    if require_full_steps:
        required = {"resume.pt"} | {f"checkpoint_step_{step}.pt" for step in require_full_steps}
        missing = sorted(name for name in required if not (directory / name).is_file())
        if missing:
            raise ValueError("Completed run lacks retained full checkpoints; restore its archive, do not retrain: "
                             + ", ".join(missing))
    state = None
    def full_state():
        nonlocal state
        if state is None:
            path = next((directory / name for name in ("resume.pt", f"checkpoint_step_{total}.pt")
                         if (directory / name).is_file()), None)
            if path is None:
                raise ValueError("Missing weights export and completed full state; restore the retained artifact")
            state = _checkpoint(path)
            if (state.get("completed_updates") != total
                    or base.model_hash(state["model"]) != record["final_weights_sha256"]):
                raise ValueError("Retained full state disagrees with completed result")
        return state
    best_path = directory / "best_checkpoint.pt"
    if not best_path.exists():
        saved = full_state()
        best_weights = saved["best_weights"]
        if base.model_hash(best_weights) != record["best_weights_sha256"]:
            raise ValueError("Retained selected weights disagree with the completed result")
        base.atomic_checkpoint(best_path, {
            "checkpoint_kind": "weights_only", "identity": saved["identity"], "model": best_weights,
            "selected_step": record["best_validation"]["step"],
            "selected_epoch": record["best_validation"]["epoch"],
            "metrics": record["best_validation"], "weights_sha256": record["best_weights_sha256"],
            "selection_window": record["selection_window"]})
    final_path = directory / "final_weights.pt"
    if not final_path.exists():
        saved = full_state()
        base.atomic_checkpoint(final_path, {
            "checkpoint_kind": "weights_only", "identity": saved["identity"],
            "model": saved["model"], "weights_sha256": record["final_weights_sha256"],
            "completed_updates": total})
    for path, expected, step_key, step in (
        (best_path, record["best_weights_sha256"], "selected_step", record["best_validation"]["step"]),
        (final_path, record["final_weights_sha256"], "completed_updates", total),
    ):
        checkpoint = _checkpoint(path)
        if (checkpoint.get("weights_sha256") != expected or base.model_hash(checkpoint["model"]) != expected
                or checkpoint.get(step_key) != step):
            raise ValueError(f"Retained weights do not match their measured checkpoint: {path.name}")
    return {path.name: digest(path) for path in (best_path, final_path)}


def validate_trial(directory, trial, plan, identity, candidate, seed):
    directory = Path(directory)
    check_identity(trial, {**identity, "candidate_id": candidate["id"],
                           "config": candidate["config"], "seed": seed})
    if trial.get("result_sha256") != digest(directory / "result.json"):
        raise ValueError("Development result differs from its retained trial fingerprint")
    record = read_json(directory / "result.json")
    status = trial.get("status")
    validate_result(record, plan, identity, candidate["config"], seed, "development",
                    complete=status == "complete")
    if status == "complete":
        retained = retain_weights(directory, record)
        if trial.get("retained_files") != retained:
            raise ValueError("Development retained weights differ from their trial fingerprints")
        if trial.get("best_validation_rmse") != record["best_validation"]["validation"]["rmse"]:
            raise ValueError("Trial score differs from its validation-selected result")
    elif status == "numerical_failure":
        if not str(trial.get("error", "")).startswith("FloatingPointError:"):
            raise ValueError("Numerical-failure trial lacks an explicit numerical error")
        if record.get("status") != "failed":
            raise ValueError("Numerical-failure trial disagrees with its retained record")
    else:
        raise ValueError("Development trial is not terminal")
    return record


def validate_frozen_selection(frozen, plan, identity):
    """Verify the complete declared grid and deterministic validation-only winner."""
    check_identity(frozen, identity)
    candidates = {candidate["id"]: candidate for candidate in plan["candidates"]}
    trials = frozen.get("trials", [])
    if (frozen.get("calibration_complete") is not True
            or frozen.get("development_seeds") != plan["development_seeds"]
            or frozen.get("confirmation_seeds") != plan["confirmation_seeds"]
            or len(candidates) != 32 or len(trials) != 32
            or {trial.get("candidate_id") for trial in trials} != set(candidates)):
        raise ValueError("Frozen selection does not contain the completed full grid")
    eligible = []
    for trial in trials:
        candidate = candidates[trial["candidate_id"]]
        check_identity(trial, {**identity, "config": candidate["config"]})
        rows = trial.get("seed_results", [])
        if [row.get("seed") for row in rows] != plan["development_seeds"]:
            raise ValueError("Frozen candidate does not contain both development seeds")
        for row in rows:
            check_identity(row, {**identity, "candidate_id": candidate["id"], "config": candidate["config"]})
            if row.get("status") == "complete":
                value = row.get("best_validation_rmse")
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("Frozen candidate has a non-finite validation score")
            elif row.get("status") != "numerical_failure" or not str(row.get("error", "")).startswith("FloatingPointError:"):
                raise ValueError("Frozen candidate contains an unfinished execution")
        valid = all(row["status"] == "complete" for row in rows)
        mean = statistics.fmean(row["best_validation_rmse"] for row in rows) if valid else None
        if trial.get("eligible") != valid or trial.get("mean_best_validation_rmse") != mean:
            raise ValueError("Frozen candidate mean or eligibility is inconsistent")
        if valid:
            eligible.append((mean, candidate["id"], candidate["config"]))
    if not eligible:
        raise ValueError("Frozen grid has no candidate with both finite completed seeds")
    winner = min(eligible, key=lambda item: (item[0], item[1]))
    if (frozen.get("selected") != winner[2] or frozen.get("selected_candidate_id") != winner[1]
            or frozen.get("selected_mean_best_validation_rmse") != winner[0]):
        raise ValueError("Frozen winner differs from validation means and fixed ID tie break")
    return frozen


def run(root, candidate_id, seed):
    root, plan, identity = load_context(root)
    candidates = {candidate["id"]: candidate for candidate in plan["candidates"]}
    if candidate_id not in candidates or seed not in plan["development_seeds"]:
        raise ValueError("Unknown development candidate or seed")
    candidate = candidates[candidate_id]
    directory = root / "development" / candidate_id / f"seed_{seed}"
    with exclusive(directory, ".wrapper.lock"):
        trial_path, result_path = directory / "trial.json", directory / "result.json"
        if trial_path.exists():
            previous = read_json(trial_path)
            check_identity(previous, {**identity, "candidate_id": candidate_id,
                                      "config": candidate["config"], "seed": seed})
            if previous.get("status") in ("complete", "numerical_failure"):
                validate_trial(directory, previous, plan, identity, candidate, seed)
                print(f"Reusing terminal development trial {candidate_id} seed={seed}", flush=True)
                return previous
        execution_record(directory, identity, seed, "development", candidate_id)
        trial = {**identity, "schema_version": 1, "candidate_id": candidate_id,
                 "config": candidate["config"], "seed": seed, "status": "running"}
        write_json(trial_path, trial)
        try:
            existing = read_json(result_path) if result_path.exists() else None
            if existing is not None and existing.get("status") == "complete":
                # Archive/export recovery: never send completed training back to the backend.
                record = existing
            else:
                record = train_long(plan, candidate["config"], root / "data", directory, seed,
                                    phase="development", resume=True)
            validate_result(record, plan, identity, candidate["config"], seed, "development")
            record = augment_result(directory, record, identity, candidate_id=candidate_id)
            retained = retain_weights(directory, record)
            trial.update(status="complete", completed_updates=record["completed_updates"],
                         best_validation_rmse=record["best_validation"]["validation"]["rmse"],
                         best_validation_step=record["best_validation"]["step"],
                         result_sha256=digest(result_path), retained_files=retained,
                         best_weights_sha256=record["best_weights_sha256"],
                         final_weights_sha256=record["final_weights_sha256"])
            write_json(trial_path, trial)
            return trial
        except Exception as exc:
            numerical = isinstance(exc, FloatingPointError)
            trial.update(status="numerical_failure" if numerical else "execution_failure",
                         error=f"{type(exc).__name__}: {exc}",
                         result_sha256=digest(result_path) if result_path.exists() else None)
            if result_path.exists():
                record = read_json(result_path)
                trial["completed_updates"] = record.get("completed_updates")
            (directory / "error.log").write_text(traceback.format_exc())
            write_json(trial_path, trial)
            if numerical:
                if not result_path.exists():
                    raise RuntimeError("Numerical failure lacks a retained result; inspect the error before freezing") from exc
                validate_trial(directory, trial, plan, identity, candidate, seed)
                return trial
            raise


def freeze(root):
    root, plan, identity = load_context(root)
    with exclusive(root, ".freeze.lock"):
        path = root / "frozen_selection.json"
        if path.exists():
            previous = read_json(path)
            validate_frozen_selection(previous, plan, identity)
            print("Reusing the completed frozen selection.", flush=True)
            return previous
        candidates = plan["candidates"]
        if len(candidates) != 32 or len({candidate["id"] for candidate in candidates}) != 32:
            raise ValueError("The frozen grid must contain all 32 unique candidates")
        if len(plan["development_seeds"]) != 2 or len(set(plan["development_seeds"])) != 2:
            raise ValueError("Selection requires both distinct development seeds")
        trials = []
        for candidate in candidates:
            seed_results = []
            for seed in plan["development_seeds"]:
                directory = root / "development" / candidate["id"] / f"seed_{seed}"
                trial = read_json(directory / "trial.json")
                validate_trial(directory, trial, plan, identity, candidate, seed)
                seed_results.append(trial)
            eligible = all(trial["status"] == "complete" for trial in seed_results)
            aggregate = {**identity, "candidate_id": candidate["id"], "config": candidate["config"],
                         "status": "complete", "eligible": eligible, "seed_results": seed_results,
                         "mean_best_validation_rmse": statistics.fmean(
                             trial["best_validation_rmse"] for trial in seed_results) if eligible else None}
            write_json(root / "development" / candidate["id"] / "trial.json", aggregate)
            trials.append(aggregate)
        eligible = [trial for trial in trials if trial["eligible"]]
        if not eligible:
            raise RuntimeError("No candidate completed both development seeds with finite validation RMSE")
        winner = min(eligible, key=lambda trial: (trial["mean_best_validation_rmse"], trial["candidate_id"]))
        frozen = {**identity, "schema_version": 1, "experiment_id": plan["experiment_id"],
                  "calibration_complete": True, "development_seeds": plan["development_seeds"],
                  "confirmation_seeds": plan["confirmation_seeds"], "selected": winner["config"],
                  "selected_candidate_id": winner["candidate_id"],
                  "selected_mean_best_validation_rmse": winner["mean_best_validation_rmse"],
                  "selection": plan["selection"], "trials": trials}
        write_json(path, frozen)
        print(json.dumps({"selected_candidate_id": winner["candidate_id"], "selected": winner["config"]}), flush=True)
        return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "freeze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.command == "run":
        if args.candidate is None or args.seed is None:
            parser.error("run requires --candidate and --seed")
        result = run(args.root, args.candidate, args.seed)
        print(json.dumps({key: result[key] for key in ("candidate_id", "seed", "status")}))
    else:
        freeze(args.root)


if __name__ == "__main__":
    main()

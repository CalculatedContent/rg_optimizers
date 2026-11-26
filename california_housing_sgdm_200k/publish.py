"""Restore/archive only the SGDM 200,000-update continuation, with concurrent-writer retries.

Git transport avoids the repository's Actions REST quota. A separate clean
checkout receives each commit; local training outputs are never reset. Parent
cohort files and training source are outside the allowed publication prefix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import time

from california_housing_benchmark.publish import auth_env, git

REPOSITORY = "CalculatedContent/rg_optimizers"
PREFIX = "california_housing_sgdm_200k/results/2026-11-26_sgdm_200k"
MAX_ATTEMPTS = 24
IMMUTABLE_ROOT_FILES = {"plan.json", "SOURCE_SHA256.json", "data_manifest.json",
                        "parent_manifest.json", "preparation.json", "frozen_selection.json",
                        "cohort_origin.json", "source_plan.json", "parent_snapshot.json"}


def valid_paths(paths):
    """Accept repository-relative paths inside this experiment, and nowhere else."""
    result = []
    for name in paths:
        path = Path(name)
        normalized = path.as_posix()
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or not (normalized == PREFIX or normalized.startswith(PREFIX + "/"))):
            raise ValueError(f"Publication path outside the new experiment: {name}")
        if normalized not in result:
            result.append(normalized)
    if not result:
        raise ValueError("At least one publication path is required")
    return result


def refuse_symlinks(root, name):
    path = root / name
    for ancestor in [path, *path.parents]:
        if ancestor == root.parent:
            break
        if ancestor.is_symlink():
            raise ValueError(f"Refusing a symlink archive path: {ancestor}")
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"Refusing a symlink archive input: {child}")


def _json(path):
    return json.loads(Path(path).read_text())


def _cohort(record):
    github = (record.get("execution") or {}).get("github") or {}
    return (str(record.get("github_run_id") or github.get("GITHUB_RUN_ID") or ""),
            record.get("launch_commit") or github.get("GITHUB_SHA"))


def check_record_monotonic(incoming, archived, label):
    """Reject changed identities, progress rollback, or changed completed metrics."""
    fields = ("optimizer", "seed", "phase", "config", "plan_sha256",
              "data_manifest_sha256", "source_bundle_sha256", "source_manifest_sha256",
              "frozen_selection_sha256", "candidate_id", "parent_checkpoint_sha256",
              "parent_result_sha256", "parent_manifest_sha256")
    for key in fields:
        if archived.get(key) is not None and incoming.get(key) != archived[key]:
            raise RuntimeError(f"Archive identity differs for {label}: {key}")
    old_cohort, new_cohort = _cohort(archived), _cohort(incoming)
    if old_cohort[0] and old_cohort != new_cohort:
        raise RuntimeError(f"Archive GitHub cohort differs for {label}")
    if incoming.get("completed_updates", 0) < archived.get("completed_updates", 0):
        raise RuntimeError(f"Refusing older archived progress for {label}")
    if archived.get("status") == "complete":
        if incoming.get("status") != "complete":
            raise RuntimeError(f"Refusing to replace a completed result for {label}")
        for key in ("history", "best_validation", "primary_best_validation",
                    "final_weights_sha256", "best_weights_sha256"):
            if archived.get(key) is not None and incoming.get(key) != archived[key]:
                raise RuntimeError(f"Completed measurement differs for {label}: {key}")


def check_monotonic(incoming, archived):
    if not incoming.exists() or not archived.exists():
        return
    names = {"result.json", "trial.json", "github_execution.json"}
    candidates = ([incoming] if incoming.is_file() else
                  [path for path in incoming.rglob("*.json") if path.name in names])
    for path in candidates:
        if path.name not in names:
            continue
        other = archived if incoming.is_file() else archived / path.relative_to(incoming)
        if other.is_file():
            check_record_monotonic(_json(path), _json(other), str(path))


def check_immutable(source_root, incoming, archived):
    candidates = [incoming] if incoming.is_file() else list(incoming.rglob("*"))
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        suffix = relative.removeprefix(PREFIX + "/")
        immutable = (suffix in IMMUTABLE_ROOT_FILES or suffix.startswith("data/")
                     or suffix.startswith("parent/"))
        other = archived if incoming.is_file() else archived / path.relative_to(incoming)
        if immutable and other.is_file() and path.read_bytes() != other.read_bytes():
            raise RuntimeError(f"Frozen input differs from the repository: {relative}")


def ignore_archived_files(directory, names):
    # Every full checkpoint remains available. Never delete existing archive files.
    return set(shutil.ignore_patterns(".run.lock", ".wrapper.lock", "*.tmp", "__pycache__")(directory, names))


def _safe_git_error(error):
    text = (error.stderr or b"").decode("utf-8", errors="replace")
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            text = text.replace(token, "***")
    text = re.sub(r"(?i)(authorization\s*:)[^\r\n]*", r"\1 ***", text)
    return text[-1200:].strip()


def restore_or_publish(operation, branch, paths, message=None, source=None,
                       max_attempts=MAX_ATTEMPTS):
    paths = valid_paths(paths)
    source = Path(source or Path.cwd()).resolve()
    for name in paths:
        refuse_symlinks(source, name)
    env = auth_env()
    if operation in ("publish", "archive") and not (env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")):
        raise RuntimeError("Publication requires the workflow token")
    if operation not in ("restore", "publish", "archive"):
        raise ValueError("Unknown archive operation")
    if not 1 <= max_attempts <= 50:
        raise ValueError("Invalid retry limit")
    message = message or "Archive SGDM 200k results and full continuation checkpoints [skip ci]"
    with tempfile.TemporaryDirectory(prefix="sgdm200k-archive-") as temporary:
        work = Path(temporary)
        git(["init", "-q"], cwd=work)
        git(["check-ref-format", "--branch", branch], cwd=work)
        git(["remote", "add", "origin", f"https://github.com/{REPOSITORY}.git"], cwd=work)
        git(["config", "user.name", "github-actions[bot]"], cwd=work)
        git(["config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], cwd=work)
        for attempt in range(max_attempts):
            try:
                git(["fetch", "--depth=1", "--filter=blob:none", "origin", branch], cwd=work, env=env)
                git(["sparse-checkout", "init", "--no-cone"], cwd=work)
                git(["sparse-checkout", "set", "--no-cone", *["/" + name for name in paths]], cwd=work)
                git(["checkout", "-B", "archive", "FETCH_HEAD"], cwd=work, env=env)
            except subprocess.CalledProcessError as error:
                if attempt + 1 == max_attempts:
                    raise RuntimeError(f"Cannot restore archive head: {_safe_git_error(error)}") from error
                time.sleep(min(15, attempt + 1 + random.uniform(.25, 2)))
                continue
            if operation == "restore":
                for name in paths:
                    incoming, destination = work / name, source / name
                    if not incoming.exists():
                        continue
                    refuse_symlinks(work, name)
                    check_monotonic(incoming, destination)
                    check_immutable(work, incoming, destination)
                    if incoming.is_dir():
                        shutil.copytree(incoming, destination, dirs_exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(incoming, destination)
                print("Restored matching new-experiment paths when present.", flush=True)
                return None
            existing = []
            for name in paths:
                incoming, destination = source / name, work / name
                if not incoming.exists():
                    continue
                check_monotonic(incoming, destination)
                check_immutable(source, incoming, destination)
                if incoming.is_dir():
                    shutil.copytree(incoming, destination, dirs_exist_ok=True,
                                    ignore=ignore_archived_files)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(incoming, destination)
                existing.append(name)
            if not existing:
                print("No new-experiment outputs to archive.", flush=True)
                return None
            git(["add", "-A", "-f", "--", *existing], cwd=work)
            if git(["diff", "--cached", "--quiet"], cwd=work, check=False).returncode == 0:
                print("New-experiment outputs already archived.", flush=True)
                return None
            git(["commit", "-q", "-m", message], cwd=work)
            pushed = git(["push", "origin", f"HEAD:refs/heads/{branch}"],
                         cwd=work, env=env, check=False)
            if pushed.returncode == 0:
                commit = git(["rev-parse", "HEAD"], cwd=work).stdout.decode().strip()
                print(json.dumps({"archived_commit": commit, "paths": existing}), flush=True)
                return commit
            if attempt + 1 == max_attempts:
                raise RuntimeError(f"Archive push did not succeed after {max_attempts} concurrent-head retries; "
                                   "local outputs and the failure artifact remain recoverable")
            # Rebuild from the next fetched head: distinct worker paths merge,
            # and conflicting scientific records are checked again. No force push.
            git(["reset", "--hard", "HEAD"], cwd=work)
            time.sleep(min(15, attempt + 1 + random.uniform(.25, 2)))
    raise RuntimeError("Archive retry limit exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("restore", "publish", "archive"))
    parser.add_argument("--branch", "--ref", dest="branch", required=True)
    parser.add_argument("--paths", nargs="+")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--message")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    args = parser.parse_args()
    paths = (args.paths or []) + args.path
    if not paths:
        parser.error("--paths or --path is required")
    restore_or_publish(args.operation, args.branch, paths, args.message,
                       max_attempts=args.max_attempts)


if __name__ == "__main__":
    main()

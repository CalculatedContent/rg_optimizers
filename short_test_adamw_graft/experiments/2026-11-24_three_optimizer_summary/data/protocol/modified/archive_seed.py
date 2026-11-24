"""Preserve a completed GitHub seed or the final five-seed analysis in Git.

No training, recipe mutation, or historical compatibility gate occurs here.
Git pushes are restricted to the dedicated experiment branch and never forced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
BRANCH = "shakespeare-graft-five-seeds-20261124"
SEEDS = (1340, 1341, 1342, 1343, 1344)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def github_identity():
    keys = ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "GITHUB_SHA",
            "GITHUB_WORKFLOW", "GITHUB_REPOSITORY", "GITHUB_REF_NAME",
            "RUNNER_OS", "RUNNER_ARCH", "RUNNER_NAME", "ImageOS", "ImageVersion")
    return {key: os.environ.get(key) for key in keys}


def execution_environment():
    import torch
    cpu = subprocess.run(["lscpu", "--json"], capture_output=True, text=True)
    return dict(
        platform=platform.platform(), machine=platform.machine(),
        processor=platform.processor(), python=platform.python_version(),
        cpu_description=json.loads(cpu.stdout) if cpu.returncode == 0 else None,
        torch_version=torch.__version__, torch_build=torch.__config__.show(),
        torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
        worker_numerical_threads=1,
        note="Host/build metadata are recorded; identical kernels across hosts are not assumed.")


def require_github():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("This archive identifies a GitHub execution cohort; run it in Actions")
    if os.environ.get("GITHUB_REF_NAME") != BRANCH:
        raise RuntimeError("Refusing to archive or publish from another branch")


def restore_shared_preflight():
    # Preflight is already tracked in the experiment snapshot. Preserve its new
    # per-seed receipt first, then restore only this shared file before rebasing.
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", "preflight.json"],
                             cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode == 0:
        subprocess.run(["git", "restore", "--worktree", "--", "preflight.json"],
                       cwd=ROOT, check=True)


def archive_seed(seed):
    import torch
    folder = ROOT / "runs" / f"seed_{seed}"
    job = folder / "confirmation/rate_0.4/graft" / f"seed_{seed}"
    result = read(job / "result.json")
    if (not result.get("completed") or result["seed"] != seed
            or result["endpoint"]["step"] != 20000
            or [row["step"] for row in result["history"]] != list(range(0, 20001, 500))):
        raise ValueError("Expected a completed 20000-update seed and its 41 measurements")
    if any(row["alpha"]["successful_fits"] != 18 for row in result["history"]):
        raise ValueError("Every recorded step must have 18 successful matrix fits")
    design_path = folder / "confirmation/rate_0.4/design.json"
    design = read(design_path)
    if (design["study_seed"] != seed
            or design["recipes"]["graft"]["reference_lr_ratio"] != 400.0
            or design["recipes"]["graft"]["reference"]["peak_lr"] != 0.4
            or sha(design_path) != result["design_sha256"]):
        raise ValueError("Incorrect seed, tuned recipe, or design identity")

    checkpoint_paths = [job / f"checkpoint_{step}.pt" for step in (2000, 20000)]
    for step, path in zip((2000, 20000), checkpoint_paths):
        # These checkpoints were created by this trusted, frozen local worker.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint["progress"]["step"] != step
                or checkpoint["identity"] != result["identity"]
                or checkpoint["optimizer"]["reference_lr_ratio"] != 400.0):
            raise ValueError("Checkpoint step, seed/design identity, or graft ratio differs")
        del checkpoint
    archives = ROOT / "archives"
    archives.mkdir(exist_ok=True)
    archive = archives / f"seed_{seed}_checkpoints.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in checkpoint_paths:
            bundle.write(path, path.name)
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Checkpoint ZIP integrity check failed")
    if temporary.stat().st_size >= 95 * 1024 * 1024:
        raise ValueError("Checkpoint archive exceeds the conservative Git blob size limit")
    temporary.replace(archive)

    receipt = folder / "preflight.json"
    shutil.copyfile(ROOT / "preflight.json", receipt)
    if read(receipt).get("status") != "passed":
        raise ValueError("The local preflight did not pass")
    provenance = folder / "github_execution.json"
    write(provenance, dict(
        schema=1, seed=seed, archived_at=utc(), execution_origin="github_actions",
        github=github_identity(), environment=execution_environment(),
        result_sha256=sha(job / "result.json"), design_sha256=sha(design_path),
        checkpoint_archive=str(archive.relative_to(ROOT)), archive_sha256=sha(archive),
        checkpoints={path.name: dict(sha256=sha(path), bytes=path.stat().st_size)
                     for path in checkpoint_paths},
        primary_cohort="Five GitHub Actions seeds 1340–1344",
        excluded="Historical seed 1340 and local partial runs are retained separately; neither enters this cohort"))
    restore_shared_preflight()
    paths = sorted(folder.rglob("*.json")) + [archive]
    paths += [path for path in (ROOT / "logs" / f"seed_{seed}.log",
                               ROOT / "logs" / f"preflight_{seed}.log") if path.is_file()]
    return paths, f"Archive modified AdamW seed {seed} at 20000 updates"


def archive_analysis():
    summary = read(ROOT / "analysis/summary.json")
    if not summary.get("complete_five_seeds") or summary.get("n_completed") != 5:
        raise ValueError("Final analysis requires all five current results")
    executions = {}
    for seed in SEEDS:
        path = ROOT / "runs" / f"seed_{seed}" / "github_execution.json"
        metadata = read(path)
        if metadata["execution_origin"] != "github_actions":
            raise ValueError("A result is outside the GitHub primary cohort")
        result_path = (path.parent / "confirmation/rate_0.4/graft"
                       / f"seed_{seed}" / "result.json")
        if sha(result_path) != metadata["result_sha256"]:
            raise ValueError("A result differs from its archived execution record")
        executions[str(seed)] = metadata
    execution_path = ROOT / "analysis/execution_cohort.json"
    write(execution_path, dict(
        cohort="github_actions_primary_five_seeds", seeds=list(SEEDS),
        recorded_at=utc(), analysis_execution=github_identity(), executions=executions,
        local_partial_runs="Retained separately in the original workspace and excluded",
        historical_seed1340="Retained in existing_seed1340 and excluded",
        historical_controls="Native AdamW and SGDM remain historical-runtime comparators"))
    status_path = ROOT / "cohort_status.json"
    write(status_path, dict(
        status="complete", finished_at=utc(), target_updates=20000,
        cohort="github_actions_primary_five_seeds", seeds=list(SEEDS), completed_seeds=5,
        complete_five_seeds=True, analysis_summary="analysis/summary.json",
        execution_provenance="analysis/execution_cohort.json",
        github=github_identity(),
        excluded="Historical seed 1340 and local partial runs are not members of the primary cohort"))
    paths = [path for path in (ROOT / "analysis").iterdir()
             if path.is_file() and path.suffix in (".json", ".csv", ".md")]
    return sorted(paths) + [status_path], "Archive five-seed modified AdamW analysis"


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=check,
                          capture_output=True, text=True)


def publish(paths, message):
    if git("branch", "--show-current").stdout.strip() != BRANCH:
        raise RuntimeError("Refusing to publish outside the experiment branch")
    repository = Path(git("rev-parse", "--show-toplevel").stdout.strip())
    relative = [str(path.resolve().relative_to(repository)) for path in paths]
    git("config", "user.name", "github-actions[bot]")
    git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    # Experiment outputs may match repository-wide runs/ or archive ignores.
    # Force-add only this explicit allowlist; never force-push a branch.
    subprocess.run(["git", "add", "-f", "--", *relative], cwd=repository, check=True)
    if git("diff", "--name-only").stdout.strip():
        raise RuntimeError("Unstaged tracked changes remain; refusing to stash or overwrite them")
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("The selected archive is already committed", flush=True)
        return
    git("commit", "-m", message)
    for attempt in range(1, 9):
        git("fetch", "origin", BRANCH)
        rebase = git("rebase", f"origin/{BRANCH}", check=False)
        if rebase.returncode:
            git("rebase", "--abort", check=False)
            raise RuntimeError("Archive rebase conflicted; local results are retained for failure artifacts")
        pushed = git("push", "origin", f"HEAD:refs/heads/{BRANCH}", check=False)
        if pushed.returncode == 0:
            print(f"Published archive to {BRANCH}", flush=True)
            return
        if attempt < 8:
            time.sleep(min(15, 2 * attempt))
    raise RuntimeError("Push retries exhausted; results remain available for failure artifacts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int, choices=SEEDS)
    mode.add_argument("--analysis", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    require_github()
    paths, message = archive_analysis() if args.analysis else archive_seed(args.seed)
    if args.publish:
        publish(paths, message)
    print(json.dumps(dict(files=[str(path.relative_to(ROOT)) for path in paths],
                          published=args.publish), indent=2), flush=True)


if __name__ == "__main__":
    main()


"""One entry point: prepare an isolated environment and run the whole campaign.

Run on the existing TPU VM. No cloud allocation, authentication changes, or
training occurs for the default plan action. Production starts require existing
GitHub credentials and a persistent output mount; the campaign checks both.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "baseline.experiments.stock_gpt2_adamw_sgdm_tpu"

from .config import OPTIMIZER
from . import tune

HERE = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = Path("/mnt/disks/rg-data/gpt2small/stock_gpt2_adamw_sgdm_tpu")
DEFAULT_REPO = "CalculatedContent/rg_optimizers"
DEFAULT_BRANCH = "adamw-sgdm-tpu-results"


def isolated_python_environment():
    """Keep device/auth settings, but do not import packages from the caller."""
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def choose_base_python():
    candidates = [sys.executable]
    candidates.extend(shutil.which(name) for name in ("python3.12", "python3.11", "python3.10"))
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result = subprocess.run([candidate, "-c", "import json,sys;print(json.dumps(list(sys.version_info[:2])))"],
                                capture_output=True, text=True, check=False, env=isolated_python_environment())
        if result.returncode == 0 and tuple(json.loads(result.stdout)) in ((3, 10), (3, 11), (3, 12)):
            return os.path.abspath(candidate)
    raise RuntimeError("The pinned Torch/XLA environment needs Python3.10,3.11 or3.12 on this VM")


def requirements_hash(smoke):
    names = ("requirements-cpu.txt",) if smoke else ("requirements-cpu.txt", "requirements-tpu.txt")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def ensure_environment(work_dir, *, smoke=False, runner=subprocess.run):
    """Install only into our own venv; never modify system/site packages."""
    work_dir = Path(work_dir)
    environment = work_dir / ("venv-cpu" if smoke else "venv-tpu")
    interpreter = environment / "bin/python"
    marker = environment / "ADAMW_SGDM_ENVIRONMENT.json"
    desired = {"schema_version": 1, "mode": "smoke" if smoke else "production",
               "requirements_sha256": requirements_hash(smoke)}
    if environment.exists() and not (environment / "pyvenv.cfg").is_file():
        raise RuntimeError("Environment destination is occupied by a non-venv directory: " + str(environment))
    if not environment.exists():
        runner([choose_base_python(), "-m", "venv", str(environment)], check=True,
               env=isolated_python_environment())
    if not interpreter.exists():
        raise RuntimeError("Virtual environment has no usable Python: " + str(interpreter))
    if read_json(marker) != desired:
        requirements = HERE / ("requirements-cpu.txt" if smoke else "requirements-tpu.txt")
        print("Installing the pinned experiment environment", flush=True)
        runner([str(interpreter), "-m", "pip", "install", "--disable-pip-version-check",
                "-r", str(requirements)], check=True, env=isolated_python_environment())
        atomic_json(marker, desired)
    # Do not resolve this symlink: resolving venv/bin/python exits the venv.
    return str(interpreter.absolute())


def choose_campaign(args):
    work = Path(args.work_dir)
    if not work.is_absolute():
        raise ValueError("--work-dir must be an absolute path on the persistent volume")
    work = work.resolve()
    pointer = read_json(work / "LATEST_CAMPAIGN.json")
    if args.action in {"resume", "status"} and not args.campaign_id:
        if not pointer:
            raise FileNotFoundError("No saved campaign under this work directory")
        campaign_id = pointer["campaign_id"]
    else:
        campaign_id = args.campaign_id or ("adamw-sgdm-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", campaign_id):
        raise ValueError("Use a filename-safe campaign ID, at most80 characters")
    root = work / "campaigns" / campaign_id
    return work, root, campaign_id


def campaign_command(args, root, campaign_id, interpreter):
    command = [interpreter, "-u", str(HERE / "campaign.py"), args.action,
               "--campaign-dir", str(root)]
    if args.action in {"resume", "status"}:
        return command
    command += ["--data-dir", str(root.parents[1] / "fineweb"), "--campaign-id", campaign_id,
                "--device", "cpu" if args.smoke else "tpu", "--github-repo", args.github_repo,
                "--github-branch", args.github_branch,
                "--max-full-budget-multiplier", str(args.max_full_budget_multiplier),
                "--tracking-threads", str(args.tracking_threads), "--min-free-gib", str(min(30, args.min_free_gib))]
    if args.gcs_uri:
        command += ["--gcs-uri", args.gcs_uri]
    if args.source_sha:
        command += ["--source-sha", args.source_sha]
    if args.smoke:
        command.append("--smoke")
    if args.no_publish:
        command.append("--no-publish")
    return command


def validate_resume_options(args, manifest):
    """Reject explicit changes; parser defaults defer to the saved campaign."""
    if manifest.get("optimizer") != OPTIMIZER:
        raise ValueError("Saved campaign belongs to a different optimizer experiment")
    supplied = getattr(args, "_provided", set())
    for flag, field in {
        "--github-repo": "github_repo", "--github-branch": "github_branch",
        "--gcs-uri": "gcs_uri", "--source-sha": "source_sha",
        "--max-full-budget-multiplier": "max_full_budget_multiplier",
        "--tracking-threads": "tracking_threads",
    }.items():
        if flag in supplied and getattr(args, field) != manifest.get(field):
            raise ValueError(flag + " differs from the saved campaign; use a new campaign ID to change it")
    if "--smoke" in supplied and manifest["mode"] != "smoke":
        raise ValueError("--smoke differs from the saved campaign mode")
    if "--no-publish" in supplied and manifest["publication_required"]:
        raise ValueError("--no-publish differs from the saved publication policy")
    if "--min-free-gib" in supplied and args.min_free_gib < manifest["min_free_gib"]:
        raise ValueError("--min-free-gib cannot lower the saved campaign's free-space minimum")


def preview(args):
    work, root, campaign_id = choose_campaign(args)
    interpreter = work / ("venv-cpu" if args.smoke else "venv-tpu") / "bin/python"
    launch_args = argparse.Namespace(**{**vars(args), "action": "start"})
    return {"action": "plan", "executes": False, "installs_packages": False,
            "allocates_tpu": False, "campaign_id": campaign_id, "work_dir": str(work),
            "campaign_dir": str(root), "mode": "smoke" if args.smoke else "production",
            "optimizer": OPTIMIZER,
            "future_command": campaign_command(launch_args, root, campaign_id, str(interpreter)),
            "github_repository": args.github_repo, "github_results_branch": args.github_branch,
            "result_prefix": f"baseline/experiments/stock_gpt2_adamw_sgdm_tpu/results/{campaign_id}",
            "endpoint": {"loss_nats_per_token": 3.28, "perplexity": math.exp(3.28),
                         "official_validation_tokens": 10485760},
            "maximum_planned_training_tokens": 256 if args.smoke else
                tune.maximum_training_tokens(args.max_full_budget_multiplier),
            "automatic_steps": (["isolated dependencies", "synthetic CPU preflight", "2-candidate screen",
                                 "one selected synthetic finalist", "incremental full layer publication",
                                 "final plots and verification"] if args.smoke else
                                ["isolated dependencies", "GitHub write/read proof", "verified FineWeb data",
                                 "TPU hardware/optimizer preflight", "9-candidate screen", "3-candidate refinement",
                                 "full finalist to target", "one longer finalist if needed and enabled",
                                 "incremental full layer publication", "final plots and verification"]),
            "prerequisites": (["CPU host", "GitHub contents-write authentication unless --no-publish"] if args.smoke else
                              ["existing single-host eight-chip TPU VM", "persistent mounted disk",
                               "existing GitHub contents-write authentication"]),
            "optimality_guaranteed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("plan", "start", "resume", "status"), default="plan")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--campaign-id")
    parser.add_argument("--github-repo", default=DEFAULT_REPO)
    parser.add_argument("--github-branch", default=DEFAULT_BRANCH)
    parser.add_argument("--gcs-uri", help="Optional checkpoint backup prefix; each run receives its own child prefix")
    parser.add_argument("--max-full-budget-multiplier", type=int, choices=(1, 2), default=2)
    parser.add_argument("--tracking-threads", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=150)
    parser.add_argument("--smoke", action="store_true", help="Tiny synthetic CPU campaign, never a benchmark result")
    parser.add_argument("--no-publish", action="store_true", help="Synthetic smoke only")
    parser.add_argument("--source-sha", help="Explicit full commit SHA; a Git checkout is required in production")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args._provided = {argument.split("=", 1)[0] for argument in raw_argv if argument.startswith("--")}
    if args.no_publish and not args.smoke and args.action != "resume":
        parser.error("Production requires incremental GitHub publication")
    if args.tracking_threads < 1:
        parser.error("tracking-threads must be positive")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        parser.error("min-free-gib must be finite and nonnegative")
    if args.github_branch in {"main", "master"}:
        parser.error("Metric results require a dedicated non-main branch")
    if args.source_sha and not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        parser.error("source-sha must be a full40-character lowercase Git SHA")
    if args.action == "plan":
        print(json.dumps(preview(args), indent=2))
        return 0
    work, root, campaign_id = choose_campaign(args)
    if args.action == "status":
        manifest = read_json(root / "CAMPAIGN.json")
        if not manifest:
            raise FileNotFoundError("This campaign has not been created")
        print(json.dumps({"campaign_id": campaign_id, "campaign_dir": str(root),
                          "status": read_json(root / "CAMPAIGN_STATUS.json"),
                          "publication": read_json(root / "PUBLISH_STATUS.json"),
                          "final_report": read_json(root / "FINAL_REPORT.json")}, indent=2))
        return 0
    if args.action == "resume":
        manifest = read_json(root / "CAMPAIGN.json")
        if not manifest:
            raise FileNotFoundError("Resume requires the existing immutable campaign manifest")
        validate_resume_options(args, manifest)
        args.smoke = manifest["mode"] == "smoke"
        if "--min-free-gib" not in args._provided:
            args.min_free_gib = manifest["min_free_gib"]
    elif root.exists():
        raise FileExistsError("Campaign path already exists; use resume or a fresh campaign ID")
    # This local mount/free-space check happens before downloads or installation.
    from .preflight import check_output
    check_output(work, device="cpu" if args.smoke else "tpu", min_free_gib=args.min_free_gib)
    work.mkdir(parents=True, exist_ok=True)
    interpreter = ensure_environment(work, smoke=args.smoke)
    pointer = {"campaign_id": campaign_id, "campaign_dir": str(root),
               "mode": "smoke" if args.smoke else "production"}
    atomic_json(work / "LATEST_CAMPAIGN.json", pointer)
    command = campaign_command(args, root, campaign_id, interpreter)
    print("Starting the automated AdamW/SGDM graft campaign" if args.action == "start" else "Resuming the saved AdamW/SGDM graft campaign", flush=True)
    result = subprocess.run(command, check=False, env=isolated_python_environment())
    if result.returncode:
        print("Campaign startup did not succeed; no successful benchmark is being reported. See the campaign status above.", file=sys.stderr)
        return result.returncode
    print(json.dumps({"campaign_dir": str(root), "status_command": [sys.executable, str(Path(__file__).absolute()),
          "status", "--work-dir", str(work)], "note": "Training, selection, incremental metric publication and final verification run automatically."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

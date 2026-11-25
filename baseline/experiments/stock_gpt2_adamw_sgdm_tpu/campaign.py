"""Automatic AdamW/SGDM graft search, benchmark confirmation and publication.

This module coordinates existing training/measurement code. It does not allocate
TPUs, claim globally optimal hyperparameters, or report success from a partial
validation or an unpublished set of layer measurements.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "baseline.experiments.stock_gpt2_adamw_sgdm_tpu"

from .config import (BENCHMARK_VALIDATION_TOKENS, FULL_BUDGET_TOKENS,
                     OPTIMIZER, GRAFT_SCOPE, TrainingConfig, scientific_hash, smoke_config)
from . import launch, tune

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_REPO = "CalculatedContent/rg_optimizers"
DEFAULT_BRANCH = "adamw-sgdm-tpu-results"
RESULTS_PREFIX = "baseline/experiments/stock_gpt2_adamw_sgdm_tpu/results"
SUCCESSFUL_HORIZONS = {"budget_complete_target_met", "budget_complete_target_not_met"}
DIVERGED = {"diverged", "failed_nonfinite"}


class CampaignPaused(RuntimeError):
    pass


class CampaignFailure(RuntimeError):
    pass


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, payload):
    launch.atomic_json(path, payload)


@contextmanager
def campaign_lock(root, *, wait_seconds=0):
    with (Path(root) / "CAMPAIGN.lock").open("a") as handle:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise CampaignFailure("This campaign already has an active supervisor") from exc
                time.sleep(0.1)
        yield


def source_hashes():
    paths = sorted(HERE.glob("*.py")) + sorted(HERE.glob("requirements-*.txt"))
    paths += [REPO / "baseline/gpt2_small/muon_speedrun" / name
              for name in ("runtime.py", "stock_model.py", "stock_config.py")]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def source_commit(explicit=None, *, smoke=False):
    result = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    observed = result.stdout.strip() if result.returncode == 0 else None
    if explicit is not None and not re.fullmatch(r"[0-9a-f]{40}", explicit):
        raise ValueError("--source-sha must be a full 40-character Git commit SHA")
    if observed and explicit and observed != explicit:
        raise ValueError("--source-sha differs from the checked-out source commit")
    if not observed and not (smoke and explicit):
        raise ValueError("Use a complete Git checkout; only synthetic smoke may supply a SHA without Git metadata")
    if observed:
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no", "--",
                                str(HERE.relative_to(REPO)), "baseline/gpt2_small/muon_speedrun",
                                "baseline/gpt2_small/speedrun30"], capture_output=True, text=True, check=True)
        if dirty.stdout.strip():
            raise ValueError("Commit or restore changed experiment/dependency files before starting a reproducible campaign")
        # Untracked Python/requirements files can affect execution without
        # appearing in the tracked-dirty check. Every hashed source must be
        # reproducible from the commit that the campaign records.
        hashed_paths = set(source_hashes())
        tree = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--full-tree", "--name-only", "-z",
                               observed, "--", *sorted(hashed_paths)],
                              capture_output=True, text=True, check=True)
        committed_paths = set(filter(None, tree.stdout.split("\0")))
        missing = sorted(hashed_paths - committed_paths)
        if missing:
            raise ValueError("Hashed source files are not in the pinned HEAD commit: " + ", ".join(missing))
    return explicit or observed


def maximum_training_tokens(multiplier=2):
    return tune.maximum_training_tokens(multiplier)


def _smoke_configurations():
    base = replace(smoke_config(), n_layer=1, n_head=2, n_embd=24, vocab_size=64,
                   global_batch_sequences=1, microbatch_sequences=1, budget_tokens=64,
                   validation_tokens=64, eval_interval_tokens=64, spectral_interval_tokens=64,
                   checkpoint_interval_tokens=64, stage_id="screen")
    return [replace(base, peak_lr=lr, candidate_id=name, run_name=f"{name}-screen")
            for name, lr in (("graft-smoke-lr0001", 0.001), ("graft-smoke-lr0003", 0.003))]


def _write_smoke_stage(directory, configs, *, final=False):
    entries = []
    for config in configs:
        relative = Path("configs") / (config.candidate_id + ".json")
        config.write_json(Path(directory) / relative)
        entries.append({"candidate_id": config.candidate_id, "config": str(relative),
                        "config_sha256": config.config_sha256})
    return tune.write_immutable_json(Path(directory) / "stage.json", {
        "schema_version": 1, "smoke_only": True, "stage": {"stage_id": "full" if final else "screen"},
        "candidates": entries, "remaining_stages": [] if final else [{"stage_id": "full"}],
        "production_selection": False, "benchmark_target_eligible": False})


def check_existing_options(args, manifest):
    """Only explicitly supplied flags may conflict with a saved campaign.

    Parser defaults must not silently change the immutable original settings,
    and an explicit request to change them must not silently be ignored.
    """
    if manifest.get("optimizer") != OPTIMIZER or manifest.get("graft_scope") != GRAFT_SCOPE:
        raise ValueError("Saved campaign belongs to a different optimizer experiment")
    if str(Path(args.campaign_dir).resolve()) != manifest["campaign_dir"]:
        raise ValueError("Campaign directory differs from its immutable manifest")
    provided = getattr(args, "_provided", set())
    for field in ("campaign_id", "data_dir", "device", "source_sha", "seed", "github_repo", "github_branch",
                  "github_prefix", "gcs_uri", "max_full_budget_multiplier", "max_run_restarts",
                  "publication_retries", "publication_retry_seconds", "publish_interval_seconds",
                  "poll_seconds", "tracking_threads", "drain_timeout", "min_free_gib", "probe_timeout"):
        flag = "--" + field.replace("_", "-")
        if flag not in provided:
            continue
        value = getattr(args, field)
        if field == "data_dir":
            value = str(Path(value).resolve())
        elif field == "gcs_uri":
            value = launch.validate_gcs_uri(value)
        if value != manifest[field]:
            raise ValueError(flag + " differs from the immutable campaign; use a new campaign ID to change it")
    if "--smoke" in provided and manifest["mode"] != "smoke":
        raise ValueError("--smoke differs from the immutable campaign mode")
    if "--no-publish" in provided and manifest["publication_required"]:
        raise ValueError("--no-publish differs from the immutable campaign publication policy")


def create_plan(args):
    root = Path(args.campaign_dir)
    if not root.is_absolute():
        raise ValueError("--campaign-dir must be an absolute persistent path")
    root = root.resolve()
    existing = root / "CAMPAIGN.json"
    if existing.exists():
        manifest = tune.read_immutable_json(existing)
        check_existing_options(args, manifest)
        if manifest["source_hashes"] != source_hashes():
            raise ValueError("Campaign source changed; restore its original checkout or create a new campaign")
        return manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Campaign directory is occupied and has no campaign manifest")
    if not args.data_dir:
        raise ValueError("A new campaign requires --data-dir")
    campaign_id = args.campaign_id or root.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", campaign_id):
        raise ValueError("Use a filename-safe unique campaign ID, at most80 characters")
    smoke = bool(args.smoke)
    if args.no_publish and not smoke:
        raise ValueError("Production campaigns require incremental GitHub publication")
    device = args.device or ("cpu" if smoke else "tpu")
    if (smoke and device != "cpu") or (not smoke and device != "tpu"):
        raise ValueError("Synthetic smoke uses CPU; the production campaign uses the prepared TPU")
    branch = args.github_branch or DEFAULT_BRANCH
    if branch in {"main", "master"}:
        raise ValueError("Results must use a dedicated non-main branch")
    prefix = args.github_prefix or f"{RESULTS_PREFIX}/{campaign_id}"
    if not prefix.endswith("/results/" + campaign_id):
        raise ValueError("The GitHub prefix must end in /results/<campaign-id>")
    root.mkdir(parents=True, exist_ok=True)
    stages = [dict(stage) for stage in tune.DEFAULT_STAGES]
    stages[-1]["stop_at_target"] = True
    manifest = {"schema_version": 1, "campaign_id": campaign_id, "mode": "smoke" if smoke else "production",
                "optimizer": OPTIMIZER, "graft_scope": GRAFT_SCOPE,
                "source_sha": source_commit(args.source_sha, smoke=smoke), "source_hashes": source_hashes(),
                "campaign_dir": str(root), "data_dir": str(Path(args.data_dir).resolve()), "device": device,
                "seed": args.seed, "github_repo": args.github_repo or DEFAULT_REPO,
                "github_branch": branch, "github_prefix": prefix, "publication_required": not args.no_publish,
                "gcs_uri": launch.validate_gcs_uri(args.gcs_uri), "max_full_budget_multiplier": args.max_full_budget_multiplier,
                "maximum_planned_training_tokens": 256 if smoke else maximum_training_tokens(args.max_full_budget_multiplier),
                "stages": ["screen", "full"] if smoke else ["screen", "refine", "full"],
                "production_stage_plan": stages if not smoke else None,
                "max_run_restarts": args.max_run_restarts, "publication_retries": args.publication_retries,
                "publication_retry_seconds": args.publication_retry_seconds,
                "poll_seconds": args.poll_seconds, "publish_interval_seconds": args.publish_interval_seconds,
                "tracking_threads": args.tracking_threads, "drain_timeout": args.drain_timeout,
                "min_free_gib": args.min_free_gib, "probe_timeout": args.probe_timeout,
                "created_unix": time.time(), "automatic_allocation": False, "optimality_claim": False,
                "selection_scope": "best fully measured candidate in the declared finite single-seed search",
                "replay_note": "Token cap is the sum of planned horizons; lost/replayed work after a hard interruption can add compute."}
    tune.write_immutable_json(existing, manifest)
    manifest = tune.read_immutable_json(existing)
    search = root / "search"
    if smoke:
        _write_smoke_stage(search / "screen", _smoke_configurations())
    else:
        candidates = [replace(config, seed=args.seed, run_name=f"{config.candidate_id}-candidate-seed{args.seed}")
                      for config in tune.candidate_configs()]
        tune.write_stage(search / "screen", candidates, stages)
        tune.write_immutable_json(search / "plan.json", {"schema_version": 1, "stages": stages,
            "campaign_manifest_sha256": manifest["manifest_sha256"], "seed": args.seed,
            "maximum_planned_training_tokens": manifest["maximum_planned_training_tokens"],
            "full_confirmation_stops_at_official_target": True,
            "fallback": "one fresh twice-budget finalist if the first full attempt misses" if args.max_full_budget_multiplier == 2 else "none"})
    atomic_json(root / "CAMPAIGN_STATUS.json", {"status": "planned", "successful": False,
        "workflow_complete": False, "campaign_id": campaign_id, "mode": manifest["mode"],
        "run_records": {}, "attempts": {}, "updated_unix": time.time()})
    return manifest


def _publisher(manifest):
    if not manifest["publication_required"]:
        return None
    from .publish_metrics import Publisher
    return Publisher(repo=manifest["github_repo"], branch=manifest["github_branch"],
                     prefix=manifest["github_prefix"], source_sha=manifest["source_sha"],
                     campaign_id=manifest["campaign_id"])


def verify_result(config, status, tracking, *, full=False):
    """Audit an actual endpoint, including early official target completion."""
    for key in ("config_sha256", "candidate_id", "stage_id", "seed"):
        if status.get(key) != getattr(config, key):
            raise CampaignFailure("Training identity mismatch: " + key)
    if status.get("terminal") is not True:
        raise CampaignFailure("Training has not reached a terminal process state")
    step = status.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= config.max_steps:
        raise CampaignFailure("Invalid training step")
    if status.get("tokens_seen") != step * config.tokens_per_step:
        raise CampaignFailure("Training step/token mismatch")
    if status.get("status") in DIVERGED:
        return {"outcome": "diverged", "target_met": False, "validation": status.get("validation")}
    if status.get("status") == "paused":
        return {"outcome": "paused", "target_met": False, "validation": status.get("validation")}
    if not tracking or not tracking.get("complete") or tracking.get("failed") != 0 or tracking.get("pending") != 0:
        raise CampaignFailure("Required layer diagnostics are incomplete or failed")
    required = {0, step, *range(config.spectral_interval_tokens // config.tokens_per_step, step,
                               config.spectral_interval_tokens // config.tokens_per_step)}
    completed = tracking.get("completed_steps", [])
    declared = status.get("expected_snapshot_steps", [])
    if (any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= step for value in completed)
            or len(completed) != len(set(completed)) or not required.issubset(set(completed))
            or set(declared) != set(completed) or tracking.get("completed") != len(completed)
            or tracking.get("missing_expected_steps")):
        raise CampaignFailure("Required/declared layer snapshot coverage does not match this endpoint")
    if tracking.get("config_sha256") != config.config_sha256 or tracking.get("run_id") != status.get("run_id"):
        raise CampaignFailure("Layer diagnostics belong to another run/configuration")
    validation = status.get("validation") or {}
    nll = validation.get("val_nll")
    if (not isinstance(nll, (int, float)) or isinstance(nll, bool) or not math.isfinite(nll) or nll < 0
            or validation.get("step") != step or validation.get("tokens_seen") != status["tokens_seen"]
            or validation.get("evaluation_tokens") != config.validation_tokens
            or validation.get("validation_offset") != config.validation_offset
            or validation.get("validation_kind") != config.validation_kind):
        raise CampaignFailure("Validation is incomplete, nonfinite or not paired to this endpoint")
    target = bool(config.mode == "production" and config.validation_kind == "benchmark"
                  and config.validation_offset == 0 and config.validation_tokens == BENCHMARK_VALIDATION_TOKENS
                  and validation.get("full_benchmark_evaluation") is True and nll <= config.target_val_nll)
    if target:
        if status.get("status") not in SUCCESSFUL_HORIZONS | {"target_reached"}:
            raise CampaignFailure("A failed process cannot qualify from a stale target flag")
        return {"outcome": "target_reached", "target_met": True, "validation": validation}
    errors = tune.check_result(config, status, tracking)
    if errors:
        raise CampaignFailure("Incomplete training horizon: " + "; ".join(errors))
    return {"outcome": "budget_complete", "target_met": False, "validation": validation}


def _launch_spec(manifest, config_path, run_dir, *, resume=False):
    args = launch.parser().parse_args(["start", "--config", str(config_path), "--data-dir", manifest["data_dir"],
        "--run-dir", str(run_dir), "--device", manifest["device"],
        "--tracking-threads", str(manifest["tracking_threads"]), "--drain-timeout", str(manifest["drain_timeout"]),
        "--min-free-gib", str(manifest["min_free_gib"]), "--probe-timeout", str(manifest["probe_timeout"])])
    args.resume = resume
    if manifest.get("gcs_uri"):
        args.gcs_uri = manifest["gcs_uri"] + "/" + str(Path(run_dir).relative_to(manifest["campaign_dir"]))
    return launch.new_spec(args)


def _reap(pid):
    if isinstance(pid, int):
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def _stop_supervisor(run_dir):
    record = read_json(Path(run_dir) / "LAUNCH.json", {})
    pid = record.get("supervisor_pid")
    if run_supervisor_alive(run_dir):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _process_matches(pid, script, option, value):
    """Never treat a recycled PID as one of this campaign's supervisors."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    expected = [os.fsencode(str(script)), b"_supervise", os.fsencode(option), os.fsencode(str(value))]
    return any(command[index:index + len(expected)] == expected for index in range(len(command)))


def run_supervisor_alive(run_dir):
    root = Path(run_dir)
    record = read_json(root / "LAUNCH.json", {})
    launch_id = record.get("launch_id")
    if not isinstance(launch_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", launch_id):
        return False
    return _process_matches(record.get("supervisor_pid"), HERE / "launch.py", "--spec",
                            root / "launches" / (launch_id + ".json"))


def campaign_supervisor_alive(root):
    root = Path(root)
    record = read_json(root / "CAMPAIGN_LAUNCH.json", {})
    return _process_matches(record.get("supervisor_pid"), HERE / "campaign.py", "--campaign-dir", root)


def current_launcher_state(run_dir):
    """A resumed launch must not consume its previous launch's final status."""
    root = Path(run_dir)
    state = read_json(root / "LAUNCHER_STATUS.json", {})
    launched = read_json(root / "LAUNCH.json", {})
    if state.get("supervision_finished") and state.get("finished_unix", -1) < launched.get("started_unix", 0):
        return {"status": "starting_resumed_supervisor", "supervision_finished": False}
    return state


class Campaign:
    def __init__(self, manifest, *, publisher=None, run_starter=None, data_preparer=None,
                 sleep=time.sleep, clock=time.monotonic):
        self.manifest = manifest
        self.root = Path(manifest["campaign_dir"])
        self.publisher = _publisher(manifest) if publisher is None and manifest["publication_required"] else publisher
        self.run_starter = run_starter or launch.start_local
        self.data_preparer = data_preparer
        self.sleep, self.clock = sleep, clock
        self.stop_requested = False
        self.active_run = None
        self.state = read_json(self.root / "CAMPAIGN_STATUS.json", {})
        self.state.setdefault("run_records", {})
        self.state.setdefault("attempts", {})

    def update(self, status, **extra):
        self.state.update(status=status, campaign_id=self.manifest["campaign_id"], mode=self.manifest["mode"],
                          source_sha=self.manifest["source_sha"], updated_unix=time.time(), **extra)
        atomic_json(self.root / "CAMPAIGN_STATUS.json", self.state)

    def publish(self, operation, *, active_run=None):
        if self.publisher is None:
            return {"status": "explicitly_disabled_for_smoke", "complete": False, "commit_sha": None}
        last = None
        for attempt in range(self.manifest["publication_retries"]):
            try:
                receipt = operation()
                if not isinstance(receipt, dict) or not receipt.get("commit_sha"):
                    raise CampaignFailure("Publisher did not return a confirmed Git commit")
                return receipt
            except Exception as exc:
                last = str(exc)
                if active_run is not None:
                    _stop_supervisor(active_run)
                self.update("publication_retry", successful=False, publication_error=last,
                            publication_attempt=attempt + 1)
                if attempt + 1 < self.manifest["publication_retries"]:
                    self.sleep(self.manifest["publication_retry_seconds"])
        raise CampaignFailure("GitHub publication failed after bounded retries: " + str(last))

    def prepare_data(self, budget):
        if self.manifest["mode"] == "smoke":
            return
        self.update("preparing_data", preparing_budget_tokens=budget)
        if self.data_preparer:
            self.data_preparer(self.manifest["data_dir"], budget)
        else:
            from .dataset import prepare
            prepare(self.manifest["data_dir"], budget, download=True)

    def _retry_tracking(self, run_dir):
        tracking = read_json(run_dir / "TRACKING_STATUS.json", {})
        if tracking.get("complete") or not (run_dir / "TRAINING_DONE").exists():
            return
        self.update("retrying_diagnostics", active_run=str(run_dir))
        environment = dict(os.environ)
        environment.update(PJRT_DEVICE="CPU", JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                           OMP_NUM_THREADS=str(self.manifest["tracking_threads"]),
                           MKL_NUM_THREADS=str(self.manifest["tracking_threads"]),
                           OPENBLAS_NUM_THREADS=str(self.manifest["tracking_threads"]))
        with (run_dir / "logs/campaign_tracking_retry.log").open("a") as handle:
            result = subprocess.run([sys.executable, str(HERE / "metrics.py"), "--run-dir", str(run_dir),
                                     "--once", "--retry-failed", "--threads", str(self.manifest["tracking_threads"])],
                                    env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=self.manifest["drain_timeout"], check=False)
        atomic_json(run_dir / "DIAGNOSTICS_RECOVERY.json", {"exit_code": result.returncode, "attempted_unix": time.time()})

    def execute_run(self, stage, config_path):
        config = TrainingConfig.from_json(config_path)
        run_key = stage + "/" + config.candidate_id
        run_dir = self.root / "runs" / run_key
        self.active_run = run_dir
        attempts = self.state["attempts"].get(run_key, 0)
        last_publish, published_steps = -float("inf"), None
        while True:
            if self.stop_requested:
                raise CampaignPaused("Campaign pause requested")
            status = read_json(run_dir / "TRAINING_STATUS.json", {})
            launcher = current_launcher_state(run_dir)
            tracking = read_json(run_dir / "TRACKING_STATUS.json", {})
            finished = launcher.get("supervision_finished") is True
            if finished:
                _reap(read_json(run_dir / "LAUNCH.json", {}).get("supervisor_pid"))
                self._retry_tracking(run_dir)
                tracking = read_json(run_dir / "TRACKING_STATUS.json", {})
                if status.get("status") != "failed":
                    outcome = verify_result(config, status, tracking, full=stage in {"full", "extended"})
                    if outcome["outcome"] != "paused":
                        break
            if status.get("status") == "failed" and attempts > self.manifest["max_run_restarts"]:
                raise CampaignFailure("Run restart limit reached: " + run_key)
            previous = read_json(run_dir / "LAUNCH.json", {})
            running = not finished and run_supervisor_alive(run_dir)
            if not running:
                resume = (run_dir / "checkpoints/latest.pt").exists()
                if run_dir.exists() and not resume:
                    raise CampaignFailure("Interrupted run has no resumable checkpoint; preserving its directory: " + run_key)
                if attempts >= 1 + self.manifest["max_run_restarts"]:
                    raise CampaignFailure("Run restart limit reached: " + run_key)
                spec = _launch_spec(self.manifest, config_path, run_dir, resume=resume)
                if self.manifest["source_hashes"] != source_hashes():
                    raise CampaignFailure("Campaign source changed before launching " + run_key +
                                          "; restore the original checkout before resuming")
                attempts += 1
                self.state["attempts"][run_key] = attempts
                self.update("running", active_run=str(run_dir), active_stage=stage,
                            active_candidate=config.candidate_id, successful=False)
                self.run_starter(spec)
            while True:
                launcher = current_launcher_state(run_dir)
                tracking = read_json(run_dir / "TRACKING_STATUS.json", {})
                completed = tuple(tracking.get("completed_steps", []))
                if self.publisher and completed and (completed != published_steps or
                        self.clock() - last_publish >= self.manifest["publish_interval_seconds"]):
                    self.publish(lambda: self.publisher.publish_run(run_dir, run_key=run_key, final=False),
                                 active_run=run_dir)
                    published_steps, last_publish = completed, self.clock()
                    self.publish(lambda: self.publisher.publish_campaign(self.root, final=False), active_run=run_dir)
                if self.stop_requested:
                    _stop_supervisor(run_dir)
                if launcher.get("supervision_finished"):
                    break
                previous = read_json(run_dir / "LAUNCH.json", {})
                pid = previous.get("supervisor_pid")
                _reap(pid)
                if pid and not run_supervisor_alive(run_dir):
                    raise CampaignFailure("Run supervisor exited without a terminal status: " + run_key)
                self.sleep(self.manifest["poll_seconds"])
            if self.stop_requested:
                if self.publisher:
                    self.publish(lambda: self.publisher.publish_run(run_dir, run_key=run_key, final=False))
                raise CampaignPaused("Campaign paused after the active run drained")
        from .report import build_report, make_plots
        report, results = build_report(run_dir)
        if results:
            make_plots(run_dir, results)
        receipt = self.publish(lambda: self.publisher.publish_run(run_dir, run_key=run_key, final=True)) if self.publisher else {
            "status": "explicitly_disabled_for_smoke", "complete": False, "commit_sha": None}
        if self.publisher and receipt.get("complete") is not True:
            raise CampaignFailure("Final run publication is incomplete: " + run_key)
        record = {"run_key": run_key, "stage": stage, "candidate_id": config.candidate_id,
                  "config_sha256": config.config_sha256, "configuration": config.as_dict(),
                  "run_dir": str(run_dir), "outcome": outcome["outcome"], "target_met": outcome["target_met"],
                  "step": status["step"], "tokens_seen": status["tokens_seen"],
                  "validation": outcome["validation"], "best_validation": status.get("best_validation"),
                  "first_target_validation": status.get("first_target_validation"),
                  "tracking": tracking, "publication": receipt,
                  "final_layer_summary": report.get("final_layer_summary"),
                  "final_layer_coverage": report.get("final_layer_coverage")}
        self.state["run_records"][run_key] = record
        self.update("run_complete", active_run=None, last_completed_run=run_key)
        if self.publisher:
            self.publish(lambda: self.publisher.publish_campaign(self.root, final=False))
        self.active_run = None
        return record

    def select_smoke(self):
        stage, configs = tune._load_stage(self.root / "search/screen")
        records = tune._read_records(self.root / "runs/screen", configs)
        ranking = []
        for config in configs:
            status, tracking = records[config.candidate_id]
            errors = tune.check_result(config, status, tracking)
            if errors:
                raise CampaignFailure("Smoke selection failed: " + "; ".join(errors))
            ranking.append((status["validation"]["val_nll"], config.candidate_id, config))
        _, _, selected = min(ranking, key=lambda value: (value[0], value[1]))
        winner = replace(selected, budget_tokens=128, stage_id="full", run_name=selected.candidate_id + "-full",
                         eval_interval_tokens=128, spectral_interval_tokens=128, checkpoint_interval_tokens=128)
        destination = self.root / "search/full"
        tune.write_immutable_json(destination / "selection.json", {"smoke_only": True, "production_selection": False,
            "selected_candidate_ids": [selected.candidate_id],
            "ranking": [{"candidate_id": candidate, "val_nll": loss} for loss, candidate, _ in sorted(ranking, key=lambda x: (x[0], x[1]))]})
        _write_smoke_stage(destination, [winner], final=True)
        winner.write_json(destination / "winner.json")

    def finish(self, final_record):
        production = self.manifest["mode"] == "production"
        benchmark_met = production and final_record["target_met"]
        all_published = self.publisher is not None and all(
            record["publication"].get("complete") is True and record["publication"].get("commit_sha")
            for record in self.state["run_records"].values())
        if production and not all_published:
            raise CampaignFailure("A production campaign cannot finish without all run publications")
        full_records = [record for record in self.state["run_records"].values() if record["stage"] in {"full", "extended"}]
        observed = [record for record in full_records if record["outcome"] in {"budget_complete", "target_reached"} and record.get("validation") and
                    isinstance(record["validation"].get("val_nll"), (int, float)) and
                    math.isfinite(record["validation"]["val_nll"])]
        best = min(observed, key=lambda record: record["validation"]["val_nll"]) if observed else None
        best_validations = [{"run_key": record["run_key"], "configuration": record["configuration"],
                             "validation": record["best_validation"]} for record in observed
                            if record.get("best_validation") and
                            isinstance(record["best_validation"].get("val_nll"), (int, float)) and
                            math.isfinite(record["best_validation"]["val_nll"])]
        best_observed = min(best_validations, key=lambda item: item["validation"]["val_nll"]) if best_validations else None
        outcome = "target_reached" if benchmark_met else "budget_exhausted_target_not_met" if production else "smoke_completed_not_a_benchmark"
        final = {"schema_version": 1, "campaign_id": self.manifest["campaign_id"], "mode": self.manifest["mode"],
                 "outcome": outcome, "benchmark_target_met": benchmark_met,
                 "optimizer": OPTIMIZER, "graft_scope": GRAFT_SCOPE,
                 "target_val_nll": 3.28, "target_perplexity": math.exp(3.28),
                 "maximum_planned_training_tokens": self.manifest["maximum_planned_training_tokens"],
                 "recorded_training_tokens": sum(record.get("tokens_seen", 0) for record in self.state["run_records"].values()),
                 "all_runs_published": all_published, "best_full_endpoint": best,
                 "best_observed_benchmark_validation": best_observed,
                 "final_run": final_record, "run_records": self.state["run_records"],
                 "optimality_claim": False, "interpretation": "Best tested configuration within a finite single-seed search; no global optimum is certified."}
        atomic_json(self.root / "FINAL_REPORT.json", final)
        self.update("publishing_final", successful=False, workflow_complete=False,
                    benchmark_target_met=benchmark_met, outcome=outcome)
        initial_receipt = self.publish(lambda: self.publisher.publish_campaign(self.root, final=True)) if self.publisher else None
        if self.publisher and initial_receipt.get("complete") is not True:
            raise CampaignFailure("Final campaign report was not confirmed remotely")
        self.update("succeeded" if benchmark_met else outcome, successful=bool(benchmark_met and all_published),
                    workflow_complete=True, publication_complete=all_published,
                    final_publication_commit=initial_receipt.get("commit_sha") if initial_receipt else None,
                    active_run=None, completed_unix=time.time())
        if self.publisher:
            final_receipt = self.publish(lambda: self.publisher.publish_campaign(self.root, final=True))
            atomic_json(self.root / "FINAL_PUBLICATION.json", final_receipt)
        return self.state

    def execute(self):
        if self.manifest["source_hashes"] != source_hashes():
            raise CampaignFailure("Campaign source changed since planning")
        # The startup parent holds the lock while it records this child PID.
        with campaign_lock(self.root, wait_seconds=10):
            prior = read_json(self.root / "CAMPAIGN_STATUS.json", {})
            if prior.get("workflow_complete") and (not self.manifest["publication_required"] or prior.get("publication_complete")):
                return prior
            def request_stop(signum, frame):
                self.stop_requested = True
                if self.active_run:
                    _stop_supervisor(self.active_run)
            old = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                self.update("publication_preflight", successful=False, workflow_complete=False,
                            supervisor_pid=os.getpid())
                if self.publisher:
                    self.publish(lambda: self.publisher.preflight(campaign_manifest=self.manifest))
                    self.publish(lambda: self.publisher.publish_campaign(self.root, final=False))
                last = None
                for stage in self.manifest["stages"]:
                    stage_dir = self.root / "search" / stage
                    manifest, configs = tune._load_stage(stage_dir)
                    self.prepare_data(max(config.budget_tokens for config in configs))
                    for entry in manifest["candidates"]:
                        last = self.execute_run(stage, stage_dir / entry["config"])
                    if stage == "screen" and self.manifest["mode"] == "smoke":
                        self.select_smoke()
                    elif stage in {"screen", "refine"}:
                        next_stage = "refine" if stage == "screen" else "full"
                        tune.select_stage(stage_dir, self.root / "runs" / stage, self.root / "search" / next_stage)
                    self.update("stage_complete", last_completed_stage=stage)
                    if self.publisher:
                        self.publish(lambda: self.publisher.publish_campaign(self.root, final=False))
                if (self.manifest["mode"] == "production" and not last["target_met"]
                        and self.manifest["max_full_budget_multiplier"] == 2):
                    base = TrainingConfig.from_json(self.root / "search/full/winner.json")
                    longer = replace(base, budget_tokens=2 * FULL_BUDGET_TOKENS, stage_id="extended",
                                     run_name=f"{base.candidate_id}-extended-seed{base.seed}", stop_at_target=True,
                                     proposal_status="selected_from_search")
                    path = self.root / "search/extended/winner.json"
                    longer.write_json(path)
                    self.prepare_data(longer.budget_tokens)
                    last = self.execute_run("extended", path)
                return self.finish(last)
            except CampaignPaused as exc:
                self.update("paused", successful=False, workflow_complete=False, error=str(exc))
                if self.publisher:
                    self.publish(lambda: self.publisher.publish_campaign(self.root, final=False))
                return self.state
            except Exception as exc:
                if self.active_run:
                    _stop_supervisor(self.active_run)
                self.update("failed", successful=False, workflow_complete=False,
                            error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)


def start_campaign(manifest, *, resume=False):
    root = Path(manifest["campaign_dir"])
    state = read_json(root / "CAMPAIGN_STATUS.json", {})
    if state.get("workflow_complete"):
        return state
    with campaign_lock(root):
        previous = read_json(root / "CAMPAIGN_LAUNCH.json", {})
        if campaign_supervisor_alive(root):
            raise CampaignFailure("Campaign supervisor is still active")
        if state.get("status") not in {"planned"} and not resume:
            raise CampaignFailure("Use campaign.py resume for an interrupted or failed campaign")
        # Fail authentication/version problems synchronously, before starting TPU work.
        from .preflight import check_versions, source_checks
        source_checks()
        check_versions(manifest["device"], skip_spectral=False)
        publisher = _publisher(manifest)
        if publisher:
            publisher.preflight(campaign_manifest=manifest)
            publisher.publish_campaign(root, final=False)
        (root / "logs").mkdir(exist_ok=True)
        with (root / "logs/campaign.log").open("a") as handle:
            child = subprocess.Popen([sys.executable, "-u", str(HERE / "campaign.py"), "_supervise",
                                      "--campaign-dir", str(root)], stdin=subprocess.DEVNULL, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        record = {"campaign_id": manifest["campaign_id"], "supervisor_pid": child.pid,
                  "campaign_dir": str(root), "started_unix": time.time(), "resume": resume}
        atomic_json(root / "CAMPAIGN_LAUNCH.json", record)
    return {"status": "started", **record, "log": str(root / "logs/campaign.log"),
            "allocates_resources": False, "maximum_planned_training_tokens": manifest["maximum_planned_training_tokens"]}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", nargs="?", default="plan", choices=("plan", "start", "resume", "status", "_supervise"))
    result.add_argument("--campaign-dir", type=Path, required=True)
    result.add_argument("--data-dir", type=Path)
    result.add_argument("--campaign-id")
    result.add_argument("--device", choices=("cpu", "tpu"))
    result.add_argument("--source-sha")
    result.add_argument("--github-repo")
    result.add_argument("--github-branch")
    result.add_argument("--github-prefix")
    result.add_argument("--gcs-uri")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--max-full-budget-multiplier", type=int, choices=(1, 2), default=2)
    result.add_argument("--max-run-restarts", type=int, default=2)
    result.add_argument("--publication-retries", type=int, default=5)
    result.add_argument("--publication-retry-seconds", type=float, default=10)
    result.add_argument("--publish-interval-seconds", type=float, default=30)
    result.add_argument("--poll-seconds", type=float, default=5)
    result.add_argument("--tracking-threads", type=int, default=2)
    result.add_argument("--drain-timeout", type=int, default=3600)
    result.add_argument("--min-free-gib", type=float, default=30)
    result.add_argument("--probe-timeout", type=int, default=1200)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--no-publish", action="store_true", help="Explicit synthetic smoke only; never production success")
    return result


def main(argv=None):
    arguments = parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = arguments.parse_args(raw_argv)
    args._provided = {argument.split("=", 1)[0] for argument in raw_argv if argument.startswith("--")}
    for field in ("publication_retries", "tracking_threads", "drain_timeout", "probe_timeout"):
        if getattr(args, field) < 1:
            arguments.error(field + " must be positive")
    for field in ("publication_retry_seconds", "publish_interval_seconds", "poll_seconds", "min_free_gib", "max_run_restarts"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0 or (field == "poll_seconds" and value == 0):
            arguments.error(field + " must be finite and nonnegative (poll interval must be positive)")
    root = args.campaign_dir.resolve()
    if args.action == "status":
        value = read_json(root / "CAMPAIGN_STATUS.json")
        if value is None:
            arguments.error("No campaign status exists")
    elif args.action in {"resume", "_supervise"}:
        manifest = tune.read_immutable_json(root / "CAMPAIGN.json")
        check_existing_options(args, manifest)
        if manifest["source_hashes"] != source_hashes():
            raise CampaignFailure("Restore this campaign's source checkout before resuming")
        value = Campaign(manifest).execute() if args.action == "_supervise" else start_campaign(manifest, resume=True)
    else:
        manifest = create_plan(args)
        value = {"status": "planned", "campaign": manifest, "executes_training": False} if args.action == "plan" else start_campaign(manifest)
    print(json.dumps(value, indent=2, allow_nan=False))
    return 1 if value.get("status") in {"failed", "budget_exhausted_target_not_met"} else 0


if __name__ == "__main__":
    raise SystemExit(main())

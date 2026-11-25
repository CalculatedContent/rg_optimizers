"""Plan, preflight, or explicitly start on a user-specified existing TPU.

With no action this only prints a plan. It never allocates resources, installs
packages, changes IAM, replaces a run, or stops another process.
"""
from __future__ import annotations

# Keep direct-script entry points inside this experiment's package namespace.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]))
    __package__ = "baseline.experiments.stock_gpt2_adamw_sgdm_tpu"

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from .preflight import HERE, REPO, load_config, run_preflight, sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def launch_lock(run_dir, *, blocking=False):
    with (Path(run_dir) / "LAUNCH.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError("This run already has an active supervisor; nothing started") from exc
        yield


def validate_gcs_uri(uri):
    if uri is None:
        return None
    uri = uri.rstrip("/")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+/[A-Za-z0-9._/-]+", uri):
        raise ValueError("--gcs-uri must name an explicit gs://bucket/unique-run-prefix without wildcards")
    if any(part in (".", "..", "") for part in uri[5:].split("/")):
        raise ValueError("Cloud run prefix must have ordinary nonempty path components")
    return uri


def commands(spec):
    common = [spec["python"], "-u", str(HERE / "train.py"),
              "--config", spec["config"], "--data-dir", spec["data_dir"],
              "--run-dir", spec["run_dir"], "--device", spec["device"]]
    if spec.get("resume"):
        common.append("--resume")
    if spec.get("skip_spectral"):
        common.append("--no-spectral")
    tracking = [spec["python"], "-u", str(HERE / "metrics.py"),
                "--run-dir", spec["run_dir"], "--watch", "--threads", str(spec["tracking_threads"])]
    return {"training": common, "tracking": None if spec.get("skip_spectral") else tracking}


def new_spec(args):
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.skip_spectral and config.mode != "smoke":
        raise ValueError("--skip-spectral is available for the CPU smoke test only")
    if not Path(args.run_dir).is_absolute():
        raise ValueError("Supply an absolute --run-dir on your persistent volume")
    if args.tracking_threads < 1 or args.backup_interval < 30 or args.drain_timeout < 1:
        raise ValueError("Threads must be positive; backup interval >=30s; drain timeout positive")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        raise ValueError("Minimum free space must be finite and nonnegative")
    interpreter = args.python or sys.executable
    if not Path(interpreter).is_absolute():
        interpreter = shutil.which(interpreter) or interpreter
    # Resolving a venv's Python symlink would discard the environment and run
    # its base interpreter without the dependencies just checked by preflight.
    interpreter = os.path.abspath(interpreter)
    return {"schema": 1, "launch_id": uuid.uuid4().hex, "python": interpreter,
            "config": str(config_path), "config_sha256": config.config_sha256,
            "data_dir": str(Path(args.data_dir).resolve()), "run_dir": str(Path(args.run_dir).resolve()),
            "device": args.device, "resume": args.resume, "mode": config.mode,
            "skip_spectral": args.skip_spectral, "tracking_threads": args.tracking_threads,
            "gcs_uri": validate_gcs_uri(args.gcs_uri), "backup_interval": args.backup_interval,
            "drain_timeout": args.drain_timeout, "min_free_gib": args.min_free_gib,
            "probe_timeout": args.probe_timeout,
            "planned_updates": config.max_steps, "planned_tokens": config.budget_tokens,
            "created_unix": time.time()}


def ensure_current_interpreter(spec):
    if os.path.abspath(spec["python"]) != os.path.abspath(sys.executable):
        raise RuntimeError("Local preflight/start must use the same Python as training; "
                           "invoke launch.py with the requested interpreter: " + spec["python"])


def ensure_run_available(spec):
    root = Path(spec["run_dir"])
    if spec.get("resume"):
        if not (root / "checkpoints/latest.pt").is_file() or not (root / "checkpoints/latest.json").is_file():
            raise RuntimeError("--resume requires checkpoints/latest.pt and latest.json in this run directory")
        previous = read_json(root / "LAUNCH.json", {})
        if previous.get("config_sha256") not in (None, spec["config_sha256"]):
            raise RuntimeError("Resume configuration differs from the launch record")
        sidecar = read_json(root / "checkpoints/latest.json", {})
        if sidecar.get("config_sha256") not in (None, spec["config_sha256"]):
            raise RuntimeError("Resume configuration differs from checkpoint metadata")
    elif root.exists():
        # Even an empty existing directory can be somebody else's intended run.
        raise RuntimeError("Run directory already exists; choose a new path, or explicitly --resume a checkpoint")
    previous = read_json(root / "LAUNCH.json", {})
    if process_alive(previous.get("supervisor_pid")):
        raise RuntimeError("Recorded supervisor is still running; duplicate launch refused")


def gcloud(command, *, timeout=120):
    result = subprocess.run(["gcloud", *command], capture_output=True, text=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError("gcloud failed: " + (result.stderr or result.stdout)[-3000:])
    return result.stdout


def gcs_probe(uri, spec):
    """Explicit start only. Require a fresh namespace, or the same resumed run."""
    result = subprocess.run(["gcloud", "storage", "ls", uri + "/**"],
                            capture_output=True, text=True, timeout=90, check=False)
    if result.returncode and not any(message in (result.stderr + result.stdout).lower()
                                     for message in ("matched no objects", "no urls matched", "no objects matched")):
        raise RuntimeError("Cannot inspect backup destination: " + (result.stderr or result.stdout)[-3000:])
    occupied = result.returncode == 0 and bool(result.stdout.strip())
    if occupied:
        if not spec.get("resume"):
            raise RuntimeError("Backup prefix already contains objects; choose a new unique prefix")
        saved = json.loads(gcloud(["storage", "cat", uri + "/LAUNCH.json"]))
        if saved.get("config_sha256") != spec["config_sha256"] or saved.get("run_dir") != spec["run_dir"]:
            raise RuntimeError("Backup prefix belongs to another run; no objects changed")
    probe_name = uri + "/__write_probe_" + uuid.uuid4().hex + ".json"
    payload = json.dumps({"purpose": "adamw-graft-start-write-probe", "launch_id": spec["launch_id"]}) + "\n"
    uploaded = False
    try:
        with tempfile.TemporaryDirectory(prefix="adamw-graft-cloud-probe-") as folder:
            local = Path(folder) / "probe.json"
            local.write_text(payload)
            gcloud(["storage", "cp", str(local), probe_name])
            uploaded = True
            if gcloud(["storage", "cat", probe_name]) != payload:
                raise RuntimeError("Cloud write/read probe did not round-trip exactly")
    finally:
        if uploaded:
            gcloud(["storage", "rm", probe_name])
    return {"status": "write_read_delete_verified", "uri": uri, "checked_unix": time.time()}


def selected_backup_files(root):
    """Only durable outputs; never caches, pending model snapshots, or arbitrary tensors."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in ("snapshots", "xla-cache", "__pycache__") for part in relative.parts):
            continue
        if path.name.endswith(".tmp") or path.name == "LAUNCH.lock":
            continue
        checkpoint = relative.parts[0] == "checkpoints" and path.name in ("latest.pt", "best.pt", "previous.pt", "first_target.pt")
        report = relative.parts[0] == "reports" and path.suffix.lower() in (".png", ".pdf")
        if checkpoint or report or path.name == "TRAINING_DONE" or path.suffix in (".json", ".jsonl", ".csv", ".log", ".txt"):
            yield path


def stage_backup(root, staging, config_sha256):
    """Stage paired checkpoint generations, skipping either half of a publication race."""
    root, staging = Path(root), Path(staging)
    count, skipped = 0, []
    checkpoint_names = {stem + suffix for stem in ("latest", "best", "previous", "first_target")
                        for suffix in (".pt", ".json")}
    for path in selected_backup_files(root):
        relative = path.relative_to(root)
        if relative.parts[0] == "checkpoints" and path.name in checkpoint_names:
            continue
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            content = path.read_bytes()
            if path.suffix == ".json":
                json.loads(content)
            destination.write_bytes(content)
            count += 1
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    for stem in ("latest", "best", "previous", "first_target"):
        source = root / "checkpoints" / (stem + ".pt")
        sidecar = source.with_suffix(".json")
        if not source.exists() and not sidecar.exists():
            continue
        destination = staging / "checkpoints" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            receipt_bytes = sidecar.read_bytes()
            receipt = json.loads(receipt_bytes)
            os.link(source, destination)
            if receipt.get("config_sha256") != config_sha256:
                raise ValueError("Checkpoint configuration hash mismatch")
            if receipt.get("file") != source.name or receipt.get("sha256") != sha256(destination):
                raise ValueError("Checkpoint/sidecar generation mismatch")
            destination.with_suffix(".json").write_bytes(receipt_bytes)
            count += 2
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            destination.unlink(missing_ok=True)
            destination.with_suffix(".json").unlink(missing_ok=True)
            skipped.append({"checkpoint": stem, "reason": str(exc),
                            "action": "kept existing cloud copy; retry on next backup"})
    return {"files_staged": count, "skipped_checkpoints": skipped}


def backup_once(spec):
    root = Path(spec["run_dir"])
    if not spec.get("gcs_uri"):
        return {"status": "not_requested", "updated_unix": time.time()}
    # Hardlink immutable, atomically replaced checkpoints before upload. The
    # staging inode remains stable while training publishes a newer latest.pt.
    with tempfile.TemporaryDirectory(prefix=".adamw-graft-backup-", dir=root.parent) as folder:
        staging = Path(folder)
        staged = stage_backup(root, staging, spec["config_sha256"])
        gcloud(["storage", "rsync", "--recursive", str(staging), spec["gcs_uri"]], timeout=1800)
    return {"status": "synced", **staged, "uri": spec["gcs_uri"],
            "updated_unix": time.time(), "deletes_remote_objects": False}


def record_backup(spec):
    try:
        result = backup_once(spec)
    except Exception as exc:
        result = {"status": "failed", "error": str(exc), "updated_unix": time.time()}
    atomic_json(Path(spec["run_dir"]) / "BACKUP_STATUS.json", result)
    return result


def final_status(training_exit, tracking_exit, backup_status, training_status):
    """Keep independent outcomes: a backup failure does not rewrite training success."""
    return {"supervision_finished": True, "training_exit_code": training_exit,
            "training_process_succeeded": training_exit == 0,
            "training_status": training_status,
            "tracking_exit_code": tracking_exit,
            "tracking_process_succeeded": None if tracking_exit is None else tracking_exit == 0,
            "backup_status": backup_status, "finished_unix": time.time()}


def supervise(spec):
    root = Path(spec["run_dir"])
    with launch_lock(root, blocking=True):
        command = commands(spec)
        env = dict(os.environ)
        if spec["device"] == "tpu":
            env["PJRT_DEVICE"] = "TPU"
            env.pop("JAX_PLATFORMS", None)
        else:
            env.update(PJRT_DEVICE="CPU", CUDA_VISIBLE_DEVICES="")
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        tracking_env = dict(env)
        tracking_env.update(PJRT_DEVICE="CPU", JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                            OMP_NUM_THREADS=str(spec["tracking_threads"]),
                            MKL_NUM_THREADS=str(spec["tracking_threads"]),
                            OPENBLAS_NUM_THREADS=str(spec["tracking_threads"]))
        tracker, trainer = None, None
        interrupted = [False]
        def stop_requested(signum, _frame):
            interrupted[0] = True
            if trainer is not None and trainer.poll() is None:
                trainer.send_signal(signum)
        signal.signal(signal.SIGTERM, stop_requested)
        signal.signal(signal.SIGINT, stop_requested)
        backup_process = None
        tracking_failed = False
        next_backup = time.monotonic() + spec["backup_interval"]
        spec_file = root / "launches" / (spec["launch_id"] + ".json")
        with (root / "logs/training.log").open("a") as train_log, \
             (root / "logs/tracking.log").open("a") as tracking_log, \
             (root / "logs/backup.log").open("a") as backup_log:
            try:
                if command["tracking"]:
                    tracker = subprocess.Popen(command["tracking"], env=tracking_env, stdout=tracking_log,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                trainer = subprocess.Popen(command["training"], env=env, stdout=train_log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(root / "LAUNCHER_STATUS.json", {
                    "status": "running", "supervisor_pid": os.getpid(), "training_pid": trainer.pid,
                    "tracking_pid": tracker.pid if tracker else None, "started_unix": time.time()})
                while trainer.poll() is None:
                    if (tracker is not None and tracker.poll() is not None and not tracking_failed
                            and (tracker.returncode != 0 or not (root / "TRAINING_DONE").exists())):
                        tracking_failed = True
                        atomic_json(root / "TRACKING_PROCESS_STATUS.json", {
                            "status": "exited_before_training", "exit_code": tracker.returncode,
                            "note": "Requested a graceful trainer pause; inspect tracking.log before resume."})
                        try:
                            trainer.send_signal(signal.SIGTERM)
                        except ProcessLookupError:
                            pass  # trainer exited between poll and the graceful-pause signal
                    if spec.get("gcs_uri") and time.monotonic() >= next_backup:
                        if backup_process is None or backup_process.poll() is not None:
                            backup_process = subprocess.Popen(
                                [spec["python"], str(HERE / "launch.py"), "_backup", "--spec", str(spec_file)],
                                stdout=backup_log, stderr=subprocess.STDOUT, start_new_session=True)
                            next_backup = time.monotonic() + spec["backup_interval"]
                    try:
                        trainer.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                train_exit = trainer.returncode
            except Exception as exc:
                atomic_json(root / "LAUNCHER_ERROR.json", {"error": str(exc), "unix": time.time()})
                if trainer is not None and trainer.poll() is None:
                    trainer.terminate()
                    trainer.wait(timeout=120)
                train_exit = trainer.returncode if trainer is not None else 1
                if not train_exit:
                    train_exit = 1
            finally:
                # On abnormal exits, let the worker drain immutable snapshots.
                # The marker is not a statement that training met its token/loss target.
                done = root / "TRAINING_DONE"
                done.parent.mkdir(parents=True, exist_ok=True)
                if not done.exists():
                    atomic_json(done, {"reason": "training_process_exited", "unix": time.time()})
                tracking_exit = None
                if tracker is not None:
                    try:
                        tracking_exit = tracker.wait(timeout=spec["drain_timeout"])
                    except subprocess.TimeoutExpired:
                        tracker.terminate()
                        try:
                            tracker.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            tracker.kill()
                            tracker.wait(timeout=30)
                        tracking_exit = tracker.returncode
                        atomic_json(root / "TRACKING_PROCESS_STATUS.json", {
                            "status": "drain_timed_out", "exit_code": tracking_exit,
                            "note": "Pending snapshots retained; rerun metrics.py --watch or --once."})
                if backup_process is not None and backup_process.poll() is None:
                    try:
                        backup_process.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        backup_process.terminate()
                        backup_process.wait(timeout=30)
        backup = record_backup(spec)
        training = read_json(root / "TRAINING_STATUS.json", read_json(root / "status.json", {}))
        state = final_status(train_exit, tracking_exit, backup, training)
        state["interrupted"] = interrupted[0]
        state["tracking_failed_before_training_exit"] = tracking_failed
        state["tracking_status"] = read_json(root / "TRACKING_STATUS.json", {})
        atomic_json(root / "LAUNCHER_STATUS.json", state)
        # Publish the final status without re-staging all checkpoints.
        if spec.get("gcs_uri"):
            try:
                for name in ("LAUNCHER_STATUS.json", "BACKUP_STATUS.json"):
                    gcloud(["storage", "cp", str(root / name), spec["gcs_uri"] + "/" + name])
            except Exception as exc:
                atomic_json(root / "BACKUP_STATUS.json", {
                    "status": "failed", "phase": "final_status_upload", "error": str(exc),
                    "updated_unix": time.time()})
        return 0 if train_exit == 0 else 1


def start_local(spec):
    ensure_current_interpreter(spec)
    ensure_run_available(spec)
    report = run_preflight(config_path=spec["config"], data_dir=spec["data_dir"],
                           run_dir=spec["run_dir"], device=spec["device"],
                           skip_spectral=spec["skip_spectral"], min_free_gib=spec["min_free_gib"],
                           probe_timeout=spec["probe_timeout"])
    if spec.get("gcs_uri"):
        report["backup_probe"] = gcs_probe(spec["gcs_uri"], spec)
    root = Path(spec["run_dir"])
    if not spec["resume"]:
        root.mkdir(parents=True, exist_ok=False)
    with launch_lock(root):
        ensure_run_available({**spec, "resume": True}) if spec["resume"] else None
        (root / "logs").mkdir(exist_ok=True)
        (root / "launches").mkdir(exist_ok=True)
        # These stop markers belong only to this explicitly resumed run.
        if spec["resume"]:
            (root / "TRAINING_DONE").unlink(missing_ok=True)
        spec_file = root / "launches" / (spec["launch_id"] + ".json")
        atomic_json(spec_file, spec)
        atomic_json(root / "PREFLIGHT.json", report)
        with (root / "logs/supervisor.log").open("a") as handle:
            child = subprocess.Popen([spec["python"], "-u", str(HERE / "launch.py"), "_supervise",
                                      "--spec", str(spec_file)], stdin=subprocess.DEVNULL, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        record = {**spec, "supervisor_pid": child.pid, "started_unix": time.time()}
        atomic_json(root / "LAUNCH.json", record)
    print(json.dumps({"status": "started", "supervisor_pid": child.pid, "run_dir": str(root),
                      "training_log": str(root / "logs/training.log"),
                      "tracking_log": str(root / "logs/tracking.log"),
                      "note": "Detached local supervision. No automatic restart or resource allocation."}, indent=2))
    return 0


def remote_command(args):
    if not all((args.node, args.project, args.zone, args.remote_repo)):
        raise ValueError("Remote launch requires --node, --project, --zone, and --remote-repo")
    repository = Path(args.remote_repo)
    if not repository.is_absolute():
        raise ValueError("--remote-repo must be an absolute existing checkout path on the TPU")
    for name in ("data_dir", "run_dir"):
        if not Path(getattr(args, name)).is_absolute():
            raise ValueError(f"--{name.replace('_', '-')} must be an absolute path on the TPU")
    config = Path(args.config)
    if not config.is_absolute():
        config = repository / config
    command = [args.python or "python3", str(repository / HERE.relative_to(REPO) / "launch.py")]
    command += [args.action, "--config", str(config), "--data-dir", args.data_dir,
                "--run-dir", args.run_dir, "--device", args.device,
                "--tracking-threads", str(args.tracking_threads),
                "--backup-interval", str(args.backup_interval), "--drain-timeout", str(args.drain_timeout),
                "--min-free-gib", str(args.min_free_gib), "--probe-timeout", str(args.probe_timeout)]
    if args.resume:
        command.append("--resume")
    if args.skip_spectral:
        command.append("--skip-spectral")
    if args.gcs_uri:
        command.extend(["--gcs-uri", validate_gcs_uri(args.gcs_uri)])
    return ["gcloud", "compute", "tpus", "tpu-vm", "ssh", args.node,
            "--project", args.project, "--zone", args.zone, "--worker=0",
            "--command", shlex.join(command)]


def check_remote_node(args):
    data = json.loads(gcloud(["compute", "tpus", "tpu-vm", "describe", args.node,
                              "--project", args.project, "--zone", args.zone, "--format=json"]))
    if data.get("state") != "READY":
        raise RuntimeError("The supplied TPU is not READY; this launcher will not start or allocate it")
    if data.get("acceleratorType") != "v5litepod-8":
        raise RuntimeError("This prepared recipe targets a single-host v5litepod-8; observed "
                           + str(data.get("acceleratorType")))
    if len(data.get("networkEndpoints", [])) != 1:
        raise RuntimeError("Require exactly one TPU VM host (worker0)")
    if data.get("health") not in (None, "HEALTHY"):
        raise RuntimeError("The supplied TPU reports unhealthy state")
    return {"name": data.get("name"), "state": data["state"],
            "accelerator_type": data["acceleratorType"], "checked_unix": time.time()}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", nargs="?", default="plan",
                        choices=("plan", "preflight", "start", "_supervise", "_backup"))
    result.add_argument("--config")
    result.add_argument("--data-dir")
    result.add_argument("--run-dir")
    result.add_argument("--device", choices=("cpu", "tpu"), default="tpu")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--skip-spectral", action="store_true", help="CPU smoke only; recorded explicitly")
    result.add_argument("--python", help="Interpreter in the prepared venv; remote default is python3")
    result.add_argument("--tracking-threads", type=int, default=2)
    result.add_argument("--gcs-uri", help="Optional unique gs://bucket/run-prefix; never inferred")
    result.add_argument("--backup-interval", type=int, default=600)
    result.add_argument("--drain-timeout", type=int, default=3600)
    result.add_argument("--min-free-gib", type=float, default=30.0)
    result.add_argument("--probe-timeout", type=int, default=1200)
    result.add_argument("--node", help="Existing TPU node; never created or started")
    result.add_argument("--project")
    result.add_argument("--zone")
    result.add_argument("--remote-repo", help="Existing absolute checkout path on the TPU VM")
    result.add_argument("--spec", type=Path, help=argparse.SUPPRESS)
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.action in ("_supervise", "_backup"):
        if args.spec is None:
            arguments.error("Internal worker requires --spec")
        spec = json.loads(args.spec.read_text())
        if args.action == "_backup":
            return 0 if record_backup(spec)["status"] == "synced" else 1
        return supervise(spec)
    if not all((args.config, args.data_dir, args.run_dir)):
        arguments.error("--config, --data-dir, and --run-dir are required")
    if any((args.node, args.project, args.zone, args.remote_repo)):
        command = remote_command(args)
        if args.action == "plan":
            print(json.dumps({"action": "plan", "executes": False,
                              "remote_command": command, "shell_display": shlex.join(command),
                              "note": "The future command uses only the supplied existing node/checkout."}, indent=2))
            return 0
        observation = check_remote_node(args)
        print(json.dumps({"observed_tpu": observation}), flush=True)
        return subprocess.run(command, check=False).returncode
    spec = new_spec(args)
    if args.action == "plan":
        print(json.dumps({"action": "plan", "executes": False, "spec": spec,
                          "commands": commands(spec), "automatic_restart": False,
                          "allocates_resources": False}, indent=2))
        return 0
    if args.action == "preflight":
        ensure_current_interpreter(spec)
        report = run_preflight(config_path=spec["config"], data_dir=spec["data_dir"],
                               run_dir=spec["run_dir"], device=spec["device"],
                               skip_spectral=spec["skip_spectral"], min_free_gib=spec["min_free_gib"],
                               probe_timeout=spec["probe_timeout"])
        report["cloud_write_probe"] = "deferred_to_explicit_start" if spec.get("gcs_uri") else "not_requested"
        print(json.dumps(report, indent=2))
        return 0
    return start_local(spec)


if __name__ == "__main__":
    raise SystemExit(main())

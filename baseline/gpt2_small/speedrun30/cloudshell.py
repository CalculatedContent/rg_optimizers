"""Stop the current MuonClip service and run one bounded GPT-2 reference job."""
import argparse
import datetime as dt
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

PROJECT = "tpu-builders-504820"
ZONE = "us-west4-a"
QUEUE = "ww-gpt2-validation-48h-20261004-s1337"
NODE = QUEUE + "-node"
BASE = Path("/mnt/disks/rg-data/gpt2small")
LATEST = BASE / "SPEEDRUN30_LATEST.json"


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def active(unit):
    p = subprocess.run(["systemctl", "show", unit, "--property=ActiveState", "--value"],
                       capture_output=True, text=True, timeout=10)
    return p.stdout.strip() in {"active", "activating", "deactivating", "reloading"}


def stop_current():
    pointer = BASE/"MUONCLIP_LATEST.json"
    if not pointer.exists():
        return
    record = json.loads(pointer.read_text())
    unit, root = record["unit"], Path(record["root"]).resolve()
    if root.parent != BASE or not re.fullmatch(r"rg-gpt2-muonclip-\d{8}-\d{6}\.service", unit):
        raise RuntimeError("Unexpected existing service identity; nothing stopped.")
    if not active(unit):
        return
    print("Requesting final save from current MuonClip run.", flush=True)
    (root/"muonclip/STOP").touch()
    end = time.monotonic()+90
    while active(unit) and time.monotonic() < end:
        time.sleep(1)
    if active(unit):
        print("Stopping the old service; retaining its latest saved checkpoint.", flush=True)
        subprocess.run(["systemctl", "stop", "--no-block", unit], check=True)
        end = time.monotonic()+40
        while active(unit) and time.monotonic() < end:
            time.sleep(1)
    if active(unit):
        raise RuntimeError("Old service has not stopped; new job was not launched.")
    print("Old run stopped. Checkpoints, logs and FineWeb-Edu retained.", flush=True)


def status_remote():
    if not LATEST.exists():
        print("No 30-minute reference run launched.")
        return
    record = json.loads(LATEST.read_text())
    print(json.dumps(record, indent=2), flush=True)
    subprocess.run(["systemctl", "--no-pager", "--full", "status", record["unit"]])
    root = Path(record["root"])
    for name in ("RUN_STATUS.json", "status.json", "latest_validation.json"):
        if (root/name).exists():
            print(name + "\n" + (root/name).read_text(), flush=True)
    subprocess.run(["tail", "-n", "30", str(root/"run.log")])


def start_remote(commit):
    if os.geteuid() != 0 or not os.path.ismount("/mnt/disks/rg-data"):
        raise RuntimeError("Requires the existing mounted data disk and root.")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected pinned commit")
    with (BASE/"port-check-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if LATEST.exists() and active(json.loads(LATEST.read_text())["unit"]):
            print("The 30-minute run is already active; no duplicate launched.")
            status_remote()
            return
        allocation = json.loads((BASE/QUEUE/"allocation.json").read_text())
        if float(allocation["validation_deadline_unix"])-time.time() < 2100:
            raise RuntimeError("Less than 35 minutes remain on this allocation.")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        root = BASE/("gpt2-speedrun30-"+stamp)
        root.mkdir()
        repo = root/"repo"
        repo.mkdir()
        run(["git", "-C", str(repo), "init", "-q"])
        run(["git", "-C", str(repo), "remote", "add", "origin",
             "https://github.com/CalculatedContent/rg_optimizers.git"])
        run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", commit], timeout=120)
        run(["git", "-C", str(repo), "checkout", "--detach", commit])
        stop_current()
        # Existing common guard blocks other trainers/replays on this same TPU.
        path = repo/"baseline/gpt2_small/scripts/run_muonclip.py"
        spec = importlib.util.spec_from_file_location("existing_launch", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.assert_idle()
        start = time.time()
        deadline = min(start+1800, float(allocation["validation_deadline_unix"])-30)
        if deadline-start < 1750:
            raise RuntimeError("Allocation time exhausted during cleanup; no job launched.")
        unit = "rg-gpt2-speedrun30-"+stamp+".service"
        base = repo/"baseline/gpt2_small"
        env = {"PYTHONPATH":str(base/"src")+":"+str(base.parent/"nanogpt_one_head/src"),
               "PJRT_DEVICE":"TPU", "TPU_ACCELERATOR_TYPE":"v5litepod-8",
               "OMP_NUM_THREADS":"4", "OPENBLAS_NUM_THREADS":"4", "MKL_NUM_THREADS":"4",
               "XLA_USE_SPMD":"1", "TOKENIZERS_PARALLELISM":"false"}
        record = {"root":str(root), "unit":unit, "commit":commit,
                  "started_unix":start, "deadline_unix":deadline, "limit_seconds":1800,
                  "cloud_uri":"gs://tpu-builders-504820-ww-continuous8/gpt2small/"+root.name}
        (root/"launch.json").write_text(json.dumps(record, indent=2))
        (root/"commit.txt").write_text(commit+"\n")
        command = ["systemd-run", "--unit="+unit, "--property=Type=exec",
                   "--property=Restart=no", "--property=RuntimeMaxSec="+str(int(deadline-time.time())-5),
                   "--property=TimeoutStopSec=5", "--property=KillMode=control-group",
                   "--property=StandardOutput=append:"+str(root/"run.log"),
                   "--property=StandardError=append:"+str(root/"run.log")]
        command += ["--setenv="+key+"="+value for key,value in env.items()]
        command += ["/mnt/disks/rg-data/continuous8/venv/bin/python", "-u",
                    str(base/"speedrun30/worker.py"), str(root), str(deadline)]
        run(command)
        tmp = LATEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(LATEST)
        print("GPT-2 reference run started:", unit, flush=True)
        print("Hard stop UTC:", dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat(), flush=True)
        print("Log:", root/"run.log", flush=True)
        print("30 minutes maximum including benchmark data, compilation, training and backup.", flush=True)
        print("No WeightWatcher, per-tensor checks, preflight, or automatic restart.", flush=True)
        print("The TPU allocation itself remains available after the job stops.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("start", "status"))
    p.add_argument("--on-tpu", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--commit", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.on_tpu:
        start_remote(a.commit) if a.action == "start" else status_remote()
        return 0
    remote = ["sudo", "python3", "-c", Path(__file__).read_text(), a.action, "--on-tpu"]
    if a.action == "start":
        repo = Path(__file__).resolve().parents[3]
        if run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True).stdout.strip():
            raise RuntimeError("Use the clean worktree from the launch block.")
        commit = run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
        remote += ["--commit", commit]
    return subprocess.run(["gcloud", "compute", "tpus", "tpu-vm", "ssh", NODE,
        "--project="+PROJECT, "--zone="+ZONE, "--worker=0", "--command="+shlex.join(remote)]).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("30-minute launch:", exc, file=sys.stderr)
        sys.exit(1)

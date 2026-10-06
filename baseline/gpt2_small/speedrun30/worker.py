"""One trainer, one final backup; all work shares the same 30-minute deadline."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2)+"\n")
    tmp.replace(path)


def bounded(command, seconds):
    if seconds <= 0:
        return {"exit_code":None, "timed_out":True}
    child = subprocess.Popen(command, start_new_session=True)
    try:
        return {"exit_code":child.wait(timeout=seconds), "timed_out":False}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)
        return {"exit_code":child.returncode, "timed_out":True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("deadline", type=float)
    a = p.parse_args()
    root, deadline = a.root, a.deadline
    no_save = os.environ.get("RG_SPEEDRUN_NO_SAVE") == "1"
    here = Path(__file__).resolve().parent
    run = {"status":"running", "deadline_unix":deadline,
           "automatic_restart":False, "limit_seconds":1800, "no_save":no_save}
    write(root/"RUN_STATUS.json", run)
    try:
        # Leave 80 seconds for final cloud backup; no work extends the deadline.
        train_deadline = deadline-(10 if no_save else 80)
        command = [sys.executable, "-u", str(here/"train.py"),
            "--root", str(root), "--deadline", str(train_deadline),
            "--cache", "/mnt/disks/rg-data/benchmark-fineweb10B-889765ea"]
        if no_save:
            command.append("--no-save")
        result = bounded(command, train_deadline-time.time())
        run.update(result)
        run["status"] = "time_limit" if result["timed_out"] else (
            "finished" if result["exit_code"] == 0 else "failed")
    except Exception as exc:
        run.update(status="failed", error=str(exc))
    write(root/"RUN_STATUS.json", run)
    print(json.dumps(run), flush=True)
    if no_save:
        print("Speedrun stopped; no checkpoint save or cloud backup requested.", flush=True)
        return 0 if run["status"] == "finished" else 1
    backup = bounded([sys.executable, str(here.parent/"scripts/backup.py"), str(root)],
                     deadline-time.time()-10)
    run["backup"] = backup
    write(root/"RUN_STATUS.json", run)
    print("30-minute job finished. Results remain on the persistent disk.", flush=True)
    return 0 if run["status"] == "finished" else 1


if __name__ == "__main__":
    sys.exit(main())

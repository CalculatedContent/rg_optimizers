"""Qualify four independent TPU chips with the FULL study's real training path.

This is an explicitly partial acceptance/throughput test, not an experimental
result. It runs the registered full-shape MuonClip workload to a finite atomic
checkpoint, restarts in fresh processes, and projects sweep time from that
second segment. Nothing is copied from or written to a production run root.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

FIRST_STOP = 500
LAST_STOP = 2500
EXPECTED_STEPS = 39063
MATRICES = {"L00_W_Q", "L00_W_K", "L00_W_V", "L00_W_O",
            "L00_W_MLP_IN", "L00_W_MLP_OUT"}


def project_hours(tasks: list[dict], seconds_by_chip: dict[int, float],
                  segment_steps: int = LAST_STOP - FIRST_STOP) -> dict:
    """Per-chip queue makespan; timings must be measured concurrently."""
    if segment_steps < 1 or not tasks:
        raise ValueError("positive segment_steps and a nonempty task grid required")
    counts = Counter(int(task["chip"]) for task in tasks)
    if set(counts) != set(seconds_by_chip):
        raise ValueError("one measured segment is required for every scheduled chip")
    for value in seconds_by_chip.values():
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timings must be positive and finite")
    queues = {str(chip): count * EXPECTED_STEPS * seconds_by_chip[chip]
              / segment_steps / 3600 for chip, count in counts.items()}
    hours = max(queues.values())
    return {"jobs_per_chip": {str(k): v for k, v in counts.items()},
            "projected_queue_hours": queues, "projected_sweep_hours": hours,
            "planning_hours_with_25pct_margin": 1.25 * hours,
            "basis": "Concurrent full-shape 500-to-2500-step segments; linear projection, not a guarantee",
            "excludes": ["initial corpus preparation", "TPU provisioning/queueing",
                         "interruptions and retries", "extra final test/report work"],
            "caveat": "Prefix timing includes cold-process/resume and compilation overhead; later LR-floor graphs may be faster. Different seeds/loads can differ."}


class CheckpointBoundaryReached(Exception):
    pass


def artifact_check(run: Path, stop: int) -> dict:
    checkpoint = run / "checkpoint_latest.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("finite restart checkpoint is missing")
    if (run / "run_complete.json").exists():
        raise RuntimeError("a qualification prefix must not be recorded as a completed run")
    with (run / "spectral/layers.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"step", "matrix_name", "alpha_raw", "alpha_clip_xmax"}.issubset(reader.fieldnames or []):
            raise RuntimeError("required raw/clipped spectral schema is missing")
        spectral = list(reader)
    required_steps = (0, 2441) if stop >= LAST_STOP else (0,)
    for step in required_steps:
        names = {row["matrix_name"] for row in spectral if int(row["step"]) == step}
        if not MATRICES.issubset(names):
            raise RuntimeError(f"six-matrix WeightWatcher output missing at step {step}")
    with (run / "random_canary_metrics.csv").open(newline="") as handle:
        canaries = list(csv.DictReader(handle))
    for step in required_steps:
        rows = [row for row in canaries if int(row["step"]) == step]
        if len(rows) != 40 or len({row["id"] for row in rows}) != 40:
            raise RuntimeError(f"complete 40-canary evaluation missing at step {step}")
        if {int(row["dose"]) for row in rows} != {0, 1, 4, 16, 64}:
            raise RuntimeError("canary dose inventory changed")
    return {"checkpoint_bytes": checkpoint.stat().st_size,
            "verified_spectral_steps": list(required_steps),
            "canaries_per_evaluation": 40}


def worker(plan_file: Path, task_index: int, stop: int) -> int:
    from . import tpu_memorization_sweep as sweep
    from . import train_loop, engine
    plan = sweep.json_read(plan_file)
    task = sweep.Task(**plan["tasks"][task_index])
    root = Path(plan["root"])
    run = root / f"load_{task.load}" / "results" / task.optimizer / f"seed_{task.seed}"
    had_checkpoint = (run / "checkpoint_latest.pt").is_file()
    if stop not in (FIRST_STOP, LAST_STOP):
        raise ValueError("unsupported acceptance-test checkpoint boundary")
    if stop == LAST_STOP and not had_checkpoint:
        raise RuntimeError("second phase must resume the first phase's checkpoint")
    original = train_loop.save_training_checkpoint
    original_load = engine.load_training_checkpoint_for_resume
    restored_steps = []

    def audited_load(*args, **kwargs):
        state = original_load(*args, **kwargs)
        restored_steps.append(int(state[0]))
        return state

    def save_and_stop(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if Path(path).name == "checkpoint_latest.pt" and int(kwargs["step"]) >= stop:
            raise CheckpointBoundaryReached()
        return result

    # Same production engine and finite checks; stop only AFTER atomic save.
    train_loop.save_training_checkpoint = save_and_stop
    engine.load_training_checkpoint_for_resume = audited_load
    started = time.perf_counter()
    reached = False
    try:
        sweep.run_worker(plan_file, task_index)
    except CheckpointBoundaryReached:
        reached = True
    finally:
        train_loop.save_training_checkpoint = original
        engine.load_training_checkpoint_for_resume = original_load
    if not reached:
        raise RuntimeError("expected finite-checkpoint acceptance boundary was not reached")
    if stop == LAST_STOP and restored_steps != [FIRST_STOP]:
        raise RuntimeError(f"resume did not restore step {FIRST_STOP}: {restored_steps}")
    if had_checkpoint and not any((root / "recovery" / f"task_{task_index:03d}").iterdir()):
        raise RuntimeError("pre-resume archival did not run")
    from .runtime import choose_device, runtime_metadata
    evidence = dict(task=plan["tasks"][task_index], stop_step=stop,
                    resumed_existing_checkpoint=had_checkpoint,
                    verified_resume_steps=restored_steps,
                    worker_seconds=time.perf_counter() - started,
                    artifacts=artifact_check(run, stop),
                    runtime=runtime_metadata(choose_device("tpu")),
                    recorded_at=sweep.utc(), qualification_only=True)
    sweep.atomic_json(root / "qualification" / f"task_{task_index:03d}_{stop}.json", evidence)
    print(f"[tpu-qualify] PASS chip={task.chip} stop={stop} resume={had_checkpoint}", flush=True)
    return 0


def qualify(args) -> int:
    from . import tpu_memorization_sweep as sweep
    from .data import validate_prepared_data
    code, data = args.code.resolve(), args.data_root.resolve()
    parent = args.output_parent.resolve()
    source = sweep.source_info(code)
    configs = sweep.load_configs(code)
    metadata = validate_prepared_data(data, configs["0pct"])
    root = parent / f"tpu-full-qualification-{uuid.uuid4().hex}"
    sweep.check_root(root, code, args.allow_ephemeral)
    if data == code or code in data.parents:
        raise ValueError("data must be outside the checkout")
    plan_args = argparse.Namespace(root=root, code=code, data_root=data,
        hardware_block=args.hardware_block, seeds=list(sweep.SEEDS),
        chips=[0, 1, 2, 3], loads=list(sweep.LOADS), optimizers=["muon_clip"],
        canary_batch_size=40, allow_ephemeral=args.allow_ephemeral)
    plan = sweep.make_plan(plan_args, configs, source, metadata)
    plan["qualification_only"] = {"stop_steps": [FIRST_STOP, LAST_STOP],
                                    "must_not_pool_with_full_experiments": True}
    # Highest extra-bank load on each chip, using the registered seed assignment.
    tasks = [next(t for t in plan["tasks"] if t["chip"] == chip and t["load"] == "10pct")
             for chip in plan["chips"]]
    root.mkdir(parents=True, exist_ok=False)
    plan_file = root / "tpu_sweep_plan.json"
    sweep.freeze_plan(plan_file, plan)
    children, mutex = {}, threading.Lock()
    timings = {}
    print(f"[tpu-qualify] OUTPUT={root}", flush=True)

    def launch(task, stop):
        key = (task["index"], stop)
        log = root / f"chip_{task['chip']}_through_{stop}.log"
        env = sweep.child_environment(os.environ, task["chip"], root, code, args.hardware_block)
        start = time.perf_counter()
        with log.open("x") as output:
            process = subprocess.Popen([sys.executable, "-u", "-m",
                "rg_nanogpt_one_head.tpu_qualify_full", "_worker", "--plan-file",
                str(plan_file), "--task", str(task["index"]), "--stop", str(stop)],
                env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            with mutex:
                children[key] = process
            try:
                rc = process.wait()
            finally:
                with mutex:
                    children.pop(key, None)
        seconds = time.perf_counter() - start
        if rc != 0:
            print(f"[tpu-qualify] FAIL chip={task['chip']} exit={rc}; log={log}", flush=True)
            return None
        evidence = sweep.json_read(root / "qualification" / f"task_{task['index']:03d}_{stop}.json")
        evidence["process_wall_seconds"] = seconds
        sweep.atomic_json(root / "qualification" / f"task_{task['index']:03d}_{stop}.json", evidence)
        print(f"[tpu-qualify] PASS chip={task['chip']} step={stop} wall={seconds:.1f}s log={log}", flush=True)
        return seconds

    pool = ThreadPoolExecutor(max_workers=4)
    locks = ExitStack()
    try:
        for chip in plan["chips"]:
            locks.enter_context(sweep.exclusive_lock(Path("/tmp") / f"rg-tpu-memorization-chip-{chip}.lock"))
        for stop in (FIRST_STOP, LAST_STOP):
            futures = [pool.submit(launch, task, stop) for task in tasks]
            values = [future.result() for future in futures]
            if any(value is None for value in values):
                raise RuntimeError(f"hardware qualification failed; all artifacts preserved at {root}")
            if stop == LAST_STOP:
                timings = {task["chip"]: value for task, value in zip(tasks, values)}
    except KeyboardInterrupt:
        with mutex:
            for process in children.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
        return 130
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        locks.close()
    report = dict(passed=True, source=source, chips=[0, 1, 2, 3],
                  resumed_in_fresh_processes=True, qualification_only=True,
                  seconds_500_to_2500_by_chip=timings,
                  forecast=project_hours(plan["tasks"], timings),
                  output_root=str(root), recorded_at=sweep.utc())
    sweep.atomic_json(root / "qualification_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[tpu-qualify] ALL FOUR CHIPS PASSED. Report: {root / 'qualification_report.json'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-parent", type=Path, required=True,
                   help="mounted durable parent; creates a new independent qualification directory")
    p.add_argument("--hardware-block", required=True)
    p.add_argument("--allow-ephemeral", action="store_true")
    p = sub.add_parser("_worker", help=argparse.SUPPRESS)
    p.add_argument("--plan-file", type=Path, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "_worker":
            return worker(args.plan_file.resolve(), args.task, args.stop)
        return qualify(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"[tpu-qualify] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

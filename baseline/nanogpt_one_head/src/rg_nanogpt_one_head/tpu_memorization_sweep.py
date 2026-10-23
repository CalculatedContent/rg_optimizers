"""Full FineWeb memorization sweep: four independent TPU-chip workers.

This is deliberately not the ordinary-Muon four-way smoke test. Each task is
one complete, independently resumable 80M-token MuonClip experiment. Existing
runners, optimizers, configs, injection schedules and checkpoints are reused.
Run `python -m rg_nanogpt_one_head.tpu_memorization_sweep --help`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import uuid

VERSION = "tpu-memorization-independent-chips-v1"
SEEDS = (1337, 2027, 4099, 31415, 271828)
LOADS = {
    "0pct": ("harmful_memorization_0pct.yaml", 0.0),
    "0.1pct": ("harmful_memorization_0p1pct.yaml", 0.001),
    "0.5pct": ("harmful_memorization_0p5pct.yaml", 0.005),
    "2pct": ("harmful_memorization_2pct.yaml", 0.02),
    "10pct": ("harmful_memorization_10pct.yaml", 0.10),
}
REVISION = "593b3a867298afb8ce42625a270ef20ddcad28f9"
EXPECTED_STEPS = 39063


@dataclass(frozen=True)
class Task:
    index: int
    load: str
    seed: int
    optimizer: str
    chip: int

    @property
    def key(self) -> str:
        return f"load_{self.load}/{self.optimizer}/seed_{self.seed}"


def build_tasks(seeds=SEEDS, chips=(0, 1, 2, 3), loads=tuple(LOADS),
                optimizers=("muon_clip",)) -> list[Task]:
    if not chips or len(set(chips)) != len(chips) or any(c not in range(4) for c in chips):
        raise ValueError("chips must be distinct v5litepod-4 chip IDs (0,1,2,3)")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s > 2**32 - 1 for s in seeds):
        raise ValueError("seeds must be distinct uint32 integers")
    if not loads or len(set(loads)) != len(loads) or any(k not in LOADS for k in loads):
        raise ValueError("unknown or duplicate load")
    if not optimizers or len(set(optimizers)) != len(optimizers) or any(
            k not in ("muon_clip", "adamw") for k in optimizers):
        raise ValueError("only MuonClip and optional AdamW are supported")
    tasks = []
    # Seed-major ordering exposes the load curve early; round-robin balances work.
    for seed in seeds:
        for load in loads:
            for optimizer in optimizers:
                i = len(tasks)
                tasks.append(Task(i, load, seed, optimizer, chips[i % len(chips)]))
    return tasks


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process holds {path}; nothing was stopped") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def child_environment(base: dict[str, str], chip: int, root: Path, code: Path,
                      hardware_block: str) -> dict[str, str]:
    """Configure chip isolation BEFORE the child imports/initializes XLA."""
    env = dict(base)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                 "PJRT_LOCAL_PROCESS_COUNT", "PJRT_LOCAL_PROCESS_RANK",
                 "TPU_PROCESS_ADDRESSES", "TPU_PROCESS_PORT"):
        env.pop(name, None)
    env.update(PJRT_DEVICE="TPU", TPU_VISIBLE_CHIPS=str(chip),
               TPU_PROCESS_BOUNDS="1,1,1", TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1",
               RG_NANOGPT_HARDWARE_BLOCK_ID=hardware_block,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = str(code / "baseline/nanogpt_one_head/src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["XLA_PERSISTENT_CACHE_PATH"] = str(root / "cache" / f"xla_chip_{chip}")
    # No hidden downcasting; retain the scientific FP32 protocol.
    for name in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16"):
        if env.get(name, "").lower() in ("1", "true", "yes", "on"):
            raise ValueError(f"unset {name}; this protocol is FP32")
    return env


def validate_study_config(cfg: dict, fraction: float) -> None:
    dataset = cfg["dataset"]
    expected_data = dict(name="HuggingFaceFW/fineweb-edu", config="sample-10BT",
                         revision=REVISION, tokenizer="gpt2", train_tokens=80000000,
                         val_tokens=1000000, test_tokens=1000000)
    for key, value in expected_data.items():
        if dataset.get(key) != value:
            raise ValueError(f"not the full FineWeb protocol: dataset.{key}")
    for key, value in dict(vocab_size=50257, n_layer=1, n_head=1,
                           n_embd=128, block_size=256, dropout=0.0, bias=False, tie_weights=True).items():
        if cfg["model"].get(key) != value:
            raise ValueError(f"model.{key} differs from the one-head study")
    for key, value in dict(batch_size=4, grad_accum_steps=8, target_epochs=4.0,
                           epoch_interval=0.25, eval_interval_steps=500,
                           checkpoint_interval_steps=500, eval_batches=64).items():
        if cfg["training"].get(key) != value:
            raise ValueError(f"training.{key} differs from the full study")
    spec = cfg["memorization"]
    for key, value in dict(enabled=True, data_seed=20260925, doses=[0, 1, 4, 16, 64],
                           canaries_per_dose=8, prefix_tokens=64, suffix_tokens=32,
                           acquisition_fraction=0.5, harmful_load_fraction=fraction,
                           harmful_load_dose=64).items():
        if spec.get(key) != value:
            raise ValueError(f"memorization.{key} differs from the selected load")
    for key, value in dict(enabled=True, ERG=True, randomize=True, strict=True,
                           fix_fingers="clip_xmax", max_fingers=10,
                           require_raw_alpha=True).items():
        if cfg["weightwatcher"].get(key) != value:
            raise ValueError(f"weightwatcher.{key} differs from the study")
    if cfg["optimizer_profiles"]["muon_clip"]["family"] != "muon_clip":
        raise ValueError("ordinary Muon must not replace MuonClip")


def source_info(code: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(code), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("source checkout is dirty; preserve it and use a separate clean checkout")
    if Path(git("rev-parse", "--show-toplevel")).resolve() != code.resolve():
        raise ValueError("--code must be the repository root")
    result = {"commit": git("rev-parse", "HEAD")}
    for name in ("tpu_memorization_sweep.py", "canary_eval_batched.py"):
        result[name] = sha256(code / "baseline/nanogpt_one_head/src/rg_nanogpt_one_head" / name)
    # Fail if an installed package rather than this checkout is executing.
    if Path(__file__).resolve() != (code / "baseline/nanogpt_one_head/src/rg_nanogpt_one_head/tpu_memorization_sweep.py").resolve():
        raise RuntimeError("PYTHONPATH points at a different package than --code")
    return result


def load_configs(code: Path) -> dict:
    from .muonclip import install_muonclip_extension
    install_muonclip_extension()
    from .config import load_config
    folder = code / "baseline/experiments/nanogpt_one_head_2026_08_21_baseline/configs"
    configs = {}
    reference = None
    for load, (name, fraction) in LOADS.items():
        cfg = load_config(folder / name)
        validate_study_config(cfg, fraction)
        common = deepcopy(cfg)
        common.pop("protocol", None)
        common["memorization"].pop("harmful_load_fraction")
        if reference is None:
            reference = common
        elif common != reference:
            raise ValueError("load configs differ in more than protocol metadata and harmful load")
        configs[load] = cfg
    return configs


def check_root(root: Path, code: Path, allow_ephemeral: bool) -> None:
    if root == code or code in root.parents:
        raise ValueError("outputs must be outside the source checkout")
    if root.exists() and not (root / "tpu_sweep_plan.json").exists():
        allowed = {"data", "cache", ".sweep.lock"}
        extra = {p.name for p in root.iterdir()} - allowed
        if extra:
            raise ValueError(f"refusing an existing non-TPU-sweep root: {sorted(extra)}")
    if not allow_ephemeral:
        # Do not mistake an explicitly mounted /tmp RAM disk for durable storage.
        mounts = [root, *root.parents]
        prefixes = (Path("/mnt/disks"), Path("/mnt/hyperdisk"), Path("/mnt/persistent"))
        if not any(p.is_mount() and any(prefix in p.parents for prefix in prefixes)
                   for p in mounts):
            raise ValueError("--root must be on a mounted durable volume; --allow-ephemeral is only for disposable validation")


def make_plan(args, configs: dict, source: dict, metadata: dict) -> dict:
    tasks = build_tasks(args.seeds, args.chips, args.loads, args.optimizers)
    return dict(schema_version=1, executor=VERSION, root=str(args.root),
                code=str(args.code), data_root=str(args.data_root), source=source,
                hardware_block=args.hardware_block, chips=args.chips,
                seeds=args.seeds, loads=args.loads, optimizers=args.optimizers,
                canary_batch_size=args.canary_batch_size, allow_ephemeral=args.allow_ephemeral,
                corpus=metadata, configs=configs, tasks=[asdict(t) for t in tasks])


def freeze_plan(path: Path, plan: dict) -> None:
    if path.exists():
        if json_read(path) != plan:
            raise ValueError("existing plan differs: use its original source/options or a NEW root; nothing overwritten")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())


def task_config(plan: dict, task: Task) -> dict:
    cfg = deepcopy(plan["configs"][task.load])
    cfg["runtime"]["tpu_memorization_execution"] = {
        "version": VERSION, "parallelism": "independent_replicas_per_chip",
        "precision": "float32", "canary_evaluator": "fixed-shape-batched-canaries-v1",
        "canary_batch_size": plan["canary_batch_size"], "source": plan["source"],
        "registered_seeds": plan["seeds"], "hardware_block": plan["hardware_block"],
    }
    return cfg


def run_worker(plan_file: Path, index: int) -> int:
    plan = json_read(plan_file)
    task = Task(**plan["tasks"][index])
    root, code = Path(plan["root"]), Path(plan["code"])
    if source_info(code) != plan["source"]:
        raise ValueError("source changed during sweep; resume with the exact pinned checkout")
    if os.environ.get("TPU_VISIBLE_CHIPS") != str(task.chip):
        raise ValueError("worker was not started with the registered chip isolation")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from .muonclip import install_muonclip_extension
    install_muonclip_extension()
    from .canary_eval_batched import install_batched_canary_evaluation
    install_batched_canary_evaluation(plan["canary_batch_size"])
    from .runtime import choose_device, configure_runtime, runtime_metadata
    from .training import run_one
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    cfg = task_config(plan, task)
    device = choose_device("tpu")
    configure_runtime(device, cfg)
    if len(xm.get_xla_supported_devices()) != 1 or int(xr.world_size()) != 1:
        raise RuntimeError("expected one isolated TPU device per independent worker")
    results = root / f"load_{task.load}" / "results"
    run = results / task.optimizer / f"seed_{task.seed}"
    lock = root / "locks" / f"task_{task.index:03d}.lock"
    with exclusive_lock(lock):
        # The reference engine rolls back diagnostics to the finite checkpoint.
        # Archive everything first so interrupted observations are never lost.
        if run.exists() and any(run.iterdir()) and not (run / "run_complete.json").exists():
            archive = root / "recovery" / f"task_{task.index:03d}" / uuid.uuid4().hex
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run, archive)
            print(f"[tpu-sweep] preserved pre-resume state: {archive}", flush=True)
        print(f"[tpu-sweep] chip={task.chip} {task.key} FULL 80M/1M/1M; 39063 steps", flush=True)
        actual = run_one(cfg=cfg, data_root=plan["data_root"], results_root=results,
                         optimizer_name=task.optimizer, seed=task.seed, device=device,
                         resume=True, overwrite=False, progress=True)
        completion = json_read(Path(actual) / "run_complete.json")
        if completion.get("completed") is not True or completion.get("optimizer_steps") != EXPECTED_STEPS:
            raise RuntimeError("run did not complete the full registered horizon")
        atomic_json(root / "receipts" / f"task_{task.index:03d}.json",
                    dict(task=asdict(task), verified_at=utc(), completion=completion,
                         runtime=runtime_metadata(device)))
    return 0


def run_sweep(plan_file: Path, retries: int) -> int:
    plan = json_read(plan_file)
    root, code = Path(plan["root"]), Path(plan["code"])
    tasks = [Task(**v) for v in plan["tasks"]]
    stopped = threading.Event()
    children, children_lock = {}, threading.Lock()
    def chip_queue(chip):
        with exclusive_lock(Path("/tmp") / f"rg-tpu-memorization-chip-{chip}.lock"):
            for task in (t for t in tasks if t.chip == chip):
                if stopped.is_set():
                    return False
                env = child_environment(os.environ, chip, root, code, plan["hardware_block"])
                task_ok = False
                for attempt in range(1, retries + 2):
                    log = root / "logs" / f"task_{task.index:03d}_{uuid.uuid4().hex}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    state = dict(task=asdict(task), started_at=utc(), attempt=attempt,
                                 state="running", log=str(log))
                    print(f"[tpu-sweep] START chip={chip} {task.key} log={log}", flush=True)
                    with log.open("x") as output:
                        process = subprocess.Popen([sys.executable, "-u", "-m",
                            "rg_nanogpt_one_head.tpu_memorization_sweep", "_worker",
                            "--plan-file", str(plan_file), "--task", str(task.index)],
                            env=env, stdout=output, stderr=subprocess.STDOUT,
                            start_new_session=True)
                        with children_lock:
                            children[chip] = process
                        state["pid"] = process.pid
                        atomic_json(root / "status" / f"task_{task.index:03d}.json", state)
                        rc = process.wait()
                        with children_lock:
                            children.pop(chip, None)
                    state.update(finished_at=utc(), exit_code=rc,
                                 state="complete" if rc == 0 else "failed")
                    atomic_json(root / "status" / f"task_{task.index:03d}.json", state)
                    print(f"[tpu-sweep] {state['state'].upper()} chip={chip} {task.key} exit={rc}", flush=True)
                    if rc == 0:
                        task_ok = True
                        break
                    if stopped.is_set():
                        break
                if not task_ok:
                    stopped.set()  # active tasks finish; no further tasks are started
                    return False
        return True
    executor = ThreadPoolExecutor(max_workers=len(plan["chips"]))
    results = []
    try:
        futures = [executor.submit(chip_queue, c) for c in plan["chips"]]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                stopped.set()
                raise
    except KeyboardInterrupt:
        stopped.set()
        with children_lock:
            for process in children.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)  # only our own children
        return 130
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return 0 if results and all(results) else 1


def show_status(root: Path) -> int:
    plan = json_read(root / "tpu_sweep_plan.json")
    complete = 0
    print("chip  load     seed       optimizer   state       step/39063")
    for value in plan["tasks"]:
        task = Task(**value)
        state_path = root / "status" / f"task_{task.index:03d}.json"
        state = json_read(state_path).get("state", "pending") if state_path.exists() else "pending"
        receipt = root / "receipts" / f"task_{task.index:03d}.json"
        if receipt.exists():
            state = "complete"
            complete += 1
        path = root / f"load_{task.load}" / "results" / task.optimizer / f"seed_{task.seed}" / "metrics.csv"
        step = "-"
        if path.exists():
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("step", "").isdigit() and row.get("val_loss"):
                        step = row["step"]
        print(f"{task.chip:4}  {task.load:7}  {task.seed:9}  {task.optimizer:10}  {state:10}  {step}")
    print(f"Verified completions: {complete}/{len(plan['tasks'])}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("prepare", "plan", "run"):
        q = sub.add_parser(command)
        q.add_argument("--code", type=Path, required=True, help="clean Git repository root")
        q.add_argument("--data-root", type=Path, required=True, help="shared, verified 80M/1M/1M corpus")
        if command != "prepare":
            q.add_argument("--root", type=Path, required=True, help="NEW TPU sweep root on a mounted durable volume")
            q.add_argument("--hardware-block", required=True, help="label for this homogeneous TPU block")
            q.add_argument("--seeds", type=lambda v: [int(x) for x in v.split(",")], default=list(SEEDS))
            q.add_argument("--chips", type=lambda v: [int(x) for x in v.split(",")], default=[0, 1, 2, 3])
            q.add_argument("--loads", type=lambda v: v.split(","), default=list(LOADS))
            q.add_argument("--optimizers", type=lambda v: v.split(","), default=["muon_clip"])
            q.add_argument("--canary-batch-size", type=int, default=40)
            q.add_argument("--allow-ephemeral", action="store_true")
            q.add_argument("--retries", type=int, default=2)
    q = sub.add_parser("status")
    q.add_argument("--root", type=Path, required=True)
    q = sub.add_parser("_worker", help=argparse.SUPPRESS)
    q.add_argument("--plan-file", type=Path, required=True)
    q.add_argument("--task", type=int, required=True)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "status":
            return show_status(args.root.resolve())
        if args.command == "_worker":
            return run_worker(args.plan_file.resolve(), args.task)
        args.code, args.data_root = args.code.resolve(), args.data_root.resolve()
        if args.data_root == args.code or args.code in args.data_root.parents:
            raise ValueError("data must be outside the source checkout")
        source = source_info(args.code)
        configs = load_configs(args.code)
        from .data import prepare_fineweb_edu, validate_prepared_data
        if args.command == "prepare":
            if args.data_root.exists() and any(args.data_root.iterdir()):
                validate_prepared_data(args.data_root, configs["0pct"])
                print(f"[tpu-sweep] reusing verified corpus: {args.data_root}")
            else:
                prepare_fineweb_edu(configs["0pct"], args.data_root)
            return 0
        if args.retries < 0 or args.canary_batch_size < 1:
            raise ValueError("invalid retries or canary batch size")
        metadata = validate_prepared_data(args.data_root, configs["0pct"])
        args.root = args.root.resolve()
        check_root(args.root, args.code, args.allow_ephemeral)
        plan = make_plan(args, configs, source, metadata)
        if args.command == "plan":
            print(json.dumps({k: v for k, v in plan.items() if k != "configs"}, indent=2))
            print(f"FULL STUDY: {len(plan['tasks'])} runs, {EXPECTED_STEPS} steps each; no training started")
            return 0
        args.root.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(args.root / ".sweep.lock"):
            plan_file = args.root / "tpu_sweep_plan.json"
            freeze_plan(plan_file, plan)
            return run_sweep(plan_file, args.retries)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"[tpu-sweep] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

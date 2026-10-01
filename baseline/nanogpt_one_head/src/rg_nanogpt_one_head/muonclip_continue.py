"""Checkpoint-based finite or open-ended MuonClip training series.

The supervisor stays on CPU. Every segment uses a fresh accelerator worker;
only full-state checkpoints connect segments. No cloud resources are created.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

import yaml

from .completion import validate_completed_run
from .continuation import make_continuation_config, pause_reason
from .muonclip_resilient import _atomic_json, _utc_now, run_resilient
from .run_utils import run_directory
from .provenance import source_fingerprint_payload, scientific_dependency_versions


def series_environment() -> dict:
    return {"source": source_fingerprint_payload(), "dependencies": scientific_dependency_versions()}


@contextmanager
def series_lock(root: Path):
    with (root / "driver.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"a continuation driver or worker already holds {root}/driver.lock") from exc
        # Keep the lock alive in the accelerator worker even if this supervisor
        # dies, preventing a second launcher from writing the same checkpoint.
        yield handle.fileno()


def prune_completed_segments(root: Path, state: dict) -> None:
    """Delete only named checkpoint files in older, validated, owned segments."""
    keep = int(state["keep_segments"])
    if keep < 2:
        raise ValueError("retain at least two completed segments")
    for record in state["completed_segments"][:-keep]:
        segment = (root / record["directory"]).resolve()
        if not segment.is_relative_to(root.resolve()) or segment.parent != (root / "segments").resolve() or not segment.name.startswith("segment_"):
            raise RuntimeError("refusing to prune outside owned segment directories")
        run_dir = run_directory(segment, "muon_clip", state["seed"])
        if run_dir.resolve() != segment / "muon_clip" / f"seed_{state['seed']}":
            raise RuntimeError("refusing to prune a symlinked run")
        complete = json.loads((run_dir / "run_complete.json").read_text())
        if complete["fingerprint"] != record["fingerprint"]:
            raise RuntimeError("refusing to prune a segment with changed completion identity")
        paths = [run_dir / f"checkpoint_{name}.pt" for name in ("initial", "latest", "best", "final")]
        paths.extend((run_dir / "epoch_checkpoints").glob("model_epoch_*.pt"))
        deleted = []
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(run_dir):
                raise RuntimeError("refusing to prune a checkpoint symlink")
            if path.is_file():
                path.unlink()
                deleted.append(str(path.relative_to(run_dir)))
        if deleted:
            _atomic_json(run_dir / "checkpoints_pruned.json", {
                "pruned_at_utc": _utc_now(), "files": deleted,
                "policy": "metrics, spectra, config, lineage and test results retained; full checkpoint validation occurred before archival",
            })


def _save_state(root: Path, state: dict) -> None:
    state["updated_at_utc"] = _utc_now()
    _atomic_json(root / "series.json", state)


def drive_series(root: Path, state: dict, lock_fd: int) -> int:
    if state["environment"] != series_environment():
        raise RuntimeError("series source/dependencies changed; restore the pinned environment or explicitly start a new series")
    stop_file = root / "STOP"
    def request_stop(signum, frame):
        del signum, frame
        stop_file.touch()
        print("[one-head-series] stop requested; waiting for the next saved checkpoint", flush=True)

    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        state["pid"] = os.getpid()
        state["status"] = "running"
        _save_state(root, state)
        while state["additional_steps"] is None or state["completed_steps"] < state["additional_steps"]:
            if state["environment"] != series_environment():
                raise RuntimeError("series source/dependencies changed between segments")
            if stop_file.exists():
                state["status"] = "paused"
                _save_state(root, state)
                return 75
            index = len(state["completed_segments"]) + 1
            relative = f"segments/segment_{index:06d}"
            segment = root / relative
            config_path = segment / "config.yaml"
            if not state.get("active_segment"):
                steps = state["segment_steps"]
                if state["additional_steps"] is not None:
                    steps = min(steps, state["additional_steps"] - state["completed_steps"])
                parent = Path(state["latest_checkpoint"])
                cfg = make_continuation_config(
                    parent, steps=steps, learning_rate=state["learning_rate"],
                    test_interval=state["test_interval_steps"], stop_file=stop_file,
                    min_free_disk_gb=state["min_free_disk_gb"],
                )
                segment.mkdir(parents=True, exist_ok=True)
                temporary = config_path.with_suffix(".yaml.tmp")
                temporary.write_text(yaml.safe_dump(cfg, sort_keys=False))
                temporary.replace(config_path)
                state["active_segment"] = relative
                _save_state(root, state)
            elif state["active_segment"] != relative:
                raise RuntimeError("active segment does not match series history")
            cfg = yaml.safe_load(config_path.read_text())
            reason = pause_reason(cfg, root)
            if reason:
                print(f"[one-head-series] {reason}", flush=True)
                state["status"] = "paused"
                state["pause_reason"] = reason
                _save_state(root, state)
                return 75
            print(f"[one-head-series] segment={index} global_start={cfg['continuation']['global_step_offset']} steps={cfg['training']['max_steps']}", flush=True)
            args = argparse.Namespace(
                config=str(config_path), seed=state["seed"], data_root=state["data_root"],
                results_root=str(segment), device=state["device"],
                max_no_progress_failures=state["max_no_progress_failures"],
                retry_delay_seconds=5.0, lock_fd=lock_fd,
                worker_log=str(segment / "launch.log"),
            )
            code = run_resilient(args)
            if code:
                state["status"] = "paused" if code == 75 else "failed"
                state["last_exit_code"] = code
                _save_state(root, state)
                return code
            run_dir = run_directory(segment, "muon_clip", state["seed"])
            # Independently audit artifacts before using a segment as a parent
            # or considering any older checkpoint for retention pruning.
            validate_completed_run(run_dir, verify_checkpoints=True)
            complete = json.loads((run_dir / "run_complete.json").read_text())
            state["completed_segments"].append({
                "directory": relative, "fingerprint": complete["fingerprint"],
                "steps": complete["optimizer_steps"], "global_step": complete["global_step"],
            })
            state["completed_steps"] += complete["optimizer_steps"]
            state["latest_checkpoint"] = str(run_dir / "checkpoint_final.pt")
            state["active_segment"] = None
            _save_state(root, state)
            prune_completed_segments(root, state)
        state["status"] = "completed"
        _save_state(root, state)
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        _save_state(root, state)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume", "stop", "status"))
    parser.add_argument("--series-root", type=Path, required=True)
    parser.add_argument("--from-checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", choices=("tpu", "xla", "cpu", "cuda", "mps"), default="tpu")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--forever", action="store_true")
    duration.add_argument("--additional-steps", type=int)
    parser.add_argument("--segment-steps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--test-interval-steps", type=int, default=10000)
    parser.add_argument("--keep-segments", type=int, default=3)
    parser.add_argument("--min-free-disk-gb", type=float, default=5.0)
    parser.add_argument("--max-no-progress-failures", type=int, default=3)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    root = args.series_root.expanduser().resolve()
    if args.action in ("stop", "status", "resume") and not (root / "series.json").is_file():
        parser.error("series.json is missing; choose an existing series")
    if args.action == "stop":
        (root / "STOP").touch()
        print("Stop requested. The worker will pause after its next atomic checkpoint.")
        return
    if args.action == "status":
        print((root / "series.json").read_text())
        print("Recorded status may be stale after host loss; inspect the checkpoint and worker log before resuming.")
        return
    if args.action == "start":
        if not args.from_checkpoint or not args.data_root or not (args.forever or args.additional_steps):
            parser.error("start requires --from-checkpoint, --data-root and --forever or --additional-steps")
        if args.segment_steps < 1 or args.test_interval_steps < 1 or args.keep_segments < 2 or args.max_no_progress_failures < 1:
            parser.error("steps/interval/retry budget must be positive; keep at least two segments")
        if args.additional_steps is not None and args.additional_steps < 1:
            parser.error("--additional-steps must be positive")
        if not math.isfinite(args.min_free_disk_gb) or args.min_free_disk_gb < 0:
            parser.error("--min-free-disk-gb must be finite and nonnegative")
        if root.exists() and any(path.name != "driver.lock" for path in root.iterdir()):
            parser.error("start requires an empty series directory; use resume for an existing series")
    root.mkdir(parents=True, exist_ok=True)
    if args.background:
        # Launch the same pinned interpreter and arguments; only the foreground
        # child creates state and acquires the writer lock.
        command = [sys.executable, "-u", "-m", "rg_nanogpt_one_head.muonclip_continue",
                   *[x for x in sys.argv[1:] if x != "--background"]]
        # Keep this log outside a new root, which must remain empty for start.
        log = root.parent / f"{root.name}.driver.log"
        with log.open("a") as output:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Started driver pid={child.pid}; log={log}")
        return
    from .muonclip import install_muonclip_extension
    install_muonclip_extension()
    with series_lock(root) as lock_fd:
        if args.action == "start":
            cfg = make_continuation_config(
                args.from_checkpoint, steps=args.segment_steps, learning_rate=args.learning_rate,
                test_interval=args.test_interval_steps, stop_file=root / "STOP", min_free_disk_gb=args.min_free_disk_gb,
            )
            state = {
                "schema_version": 1, "created_at_utc": _utc_now(), "seed": cfg["continuation"]["seed"],
                "origin_checkpoint": str(args.from_checkpoint.expanduser().resolve()),
                "origin_global_step": cfg["continuation"]["global_step_offset"],
                "latest_checkpoint": str(args.from_checkpoint.expanduser().resolve()),
                "data_root": str(args.data_root.expanduser().resolve()), "device": args.device,
                "additional_steps": args.additional_steps, "segment_steps": args.segment_steps,
                "learning_rate": cfg["optimizer_profiles"]["muon_clip"]["learning_rate"],
                "test_interval_steps": args.test_interval_steps, "keep_segments": args.keep_segments,
                "min_free_disk_gb": args.min_free_disk_gb, "max_no_progress_failures": args.max_no_progress_failures,
                "completed_steps": 0, "completed_segments": [], "active_segment": None,
                "environment": series_environment(),
            }
        else:
            state = json.loads((root / "series.json").read_text())
            (root / "STOP").unlink(missing_ok=True)
        code = drive_series(root, state, lock_fd)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

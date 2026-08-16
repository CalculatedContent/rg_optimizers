from __future__ import annotations

"""Progress-aware fresh-process recovery for long MuonClip MPS runs.

This launcher runs exactly one MuonClip worker process at a time. If Metal/MPS
terminates a worker, the next worker resumes from ``checkpoint_latest.pt``.

The retry budget counts only consecutive failures that do not advance the
verified checkpoint. Therefore intermittent Metal failures may be recovered
throughout a long run, while deterministic failures at one checkpoint stop
after a small bounded number of attempts.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import torch

from .run_utils import run_directory


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_step(path: str | Path) -> int | None:
    """Return the verified checkpoint step, or ``None`` if no file exists.

    An existing but unreadable checkpoint is a hard error: the supervisor must
    not discard or overwrite a file whose validity cannot be established.
    """

    checkpoint = Path(path)
    if not checkpoint.is_file():
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "step" not in payload:
        raise RuntimeError(f"checkpoint has no step field: {checkpoint}")
    step = int(payload["step"])
    if step < 0:
        raise RuntimeError(f"checkpoint has a negative step: {checkpoint}")
    return step


def _archive_without_checkpoint(run_dir: Path) -> Path | None:
    """Archive a partial run that never produced a restart checkpoint."""

    if not run_dir.exists():
        return None
    archive = run_dir.with_name(
        run_dir.name
        + ".failed-no-checkpoint."
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if archive.exists():
        shutil.rmtree(archive)
    run_dir.replace(archive)
    print(
        "[one-head-resilient] archived partial run with no restart checkpoint: "
        f"{archive}",
        flush=True,
    )
    return archive


def _worker_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "rg_nanogpt_one_head.muonclip",
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--optimizer",
        "muon_clip",
        "--seeds",
        str(int(args.seed)),
        "--data-root",
        str(Path(args.data_root).expanduser().resolve()),
        "--results-root",
        str(Path(args.results_root).expanduser().resolve()),
        "--device",
        "mps",
        "--mps-worker",
        "--mps-retries",
        "0",
    ]


def run_resilient(args: argparse.Namespace) -> int:
    if int(args.max_no_progress_failures) < 1:
        raise ValueError("max_no_progress_failures must be at least one")
    if float(args.retry_delay_seconds) < 0.0:
        raise ValueError("retry_delay_seconds must be nonnegative")

    results_root = Path(args.results_root).expanduser().resolve()
    run_dir = run_directory(results_root, "muon_clip", int(args.seed))
    latest_checkpoint = run_dir / "checkpoint_latest.pt"
    status_path = (
        results_root / f"_muonclip_resilient_seed_{int(args.seed)}.json"
    )

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    last_verified_step = _checkpoint_step(latest_checkpoint)
    no_progress_failures = 0
    attempt = 0
    command = _worker_command(args)

    while True:
        attempt += 1
        (run_dir / "run_failed.json").unlink(missing_ok=True)
        print(
            "[one-head-resilient] starting fresh MPS worker "
            f"attempt={attempt} checkpoint_step="
            f"{last_verified_step if last_verified_step is not None else 'none'} "
            f"no_progress_failures={no_progress_failures}/"
            f"{int(args.max_no_progress_failures)}",
            flush=True,
        )

        _atomic_json(
            status_path,
            {
                "schema_version": 1,
                "completed": False,
                "running": True,
                "attempt": int(attempt),
                "seed": int(args.seed),
                "run_dir": str(run_dir),
                "checkpoint_path": str(latest_checkpoint),
                "last_verified_checkpoint_step": last_verified_step,
                "no_progress_failures": int(no_progress_failures),
                "max_no_progress_failures": int(
                    args.max_no_progress_failures
                ),
                "updated_at_utc": _utc_now(),
            },
        )

        result = subprocess.run(
            command,
            env=environment,
            check=False,
        )
        return_code = int(result.returncode)

        if return_code == 0:
            final_step = _checkpoint_step(latest_checkpoint)
            _atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "completed": True,
                    "running": False,
                    "attempts": int(attempt),
                    "seed": int(args.seed),
                    "run_dir": str(run_dir),
                    "checkpoint_path": str(latest_checkpoint),
                    "last_verified_checkpoint_step": final_step,
                    "no_progress_failures": 0,
                    "updated_at_utc": _utc_now(),
                },
            )
            print(
                "[one-head-resilient] worker completed successfully "
                f"seed={int(args.seed)} attempts={attempt}",
                flush=True,
            )
            return 0

        checkpoint_exists = latest_checkpoint.is_file()
        try:
            current_step = _checkpoint_step(latest_checkpoint)
        except Exception as exc:
            _atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "completed": False,
                    "running": False,
                    "attempts": int(attempt),
                    "seed": int(args.seed),
                    "run_dir": str(run_dir),
                    "checkpoint_path": str(latest_checkpoint),
                    "checkpoint_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "last_exit_code": return_code,
                    "updated_at_utc": _utc_now(),
                },
            )
            print(
                "[one-head-resilient] refusing to continue because the "
                f"restart checkpoint cannot be verified: {exc}",
                flush=True,
            )
            return return_code or 1

        advanced = (
            current_step is not None
            and (
                last_verified_step is None
                or current_step > last_verified_step
            )
        )
        if advanced:
            last_verified_step = current_step
            no_progress_failures = 0
            print(
                "[one-head-resilient] worker failed after making progress; "
                f"verified checkpoint advanced to step={current_step}. "
                "The no-progress failure budget has been reset.",
                flush=True,
            )
        else:
            no_progress_failures += 1
            print(
                "[one-head-resilient] worker failed without advancing the "
                "verified checkpoint "
                f"(consecutive={no_progress_failures}/"
                f"{int(args.max_no_progress_failures)}).",
                flush=True,
            )

        if not checkpoint_exists:
            _archive_without_checkpoint(run_dir)

        _atomic_json(
            status_path,
            {
                "schema_version": 1,
                "completed": False,
                "running": False,
                "attempt": int(attempt),
                "seed": int(args.seed),
                "run_dir": str(run_dir),
                "checkpoint_path": str(latest_checkpoint),
                "last_verified_checkpoint_step": last_verified_step,
                "last_exit_code": return_code,
                "no_progress_failures": int(no_progress_failures),
                "max_no_progress_failures": int(
                    args.max_no_progress_failures
                ),
                "updated_at_utc": _utc_now(),
            },
        )

        if no_progress_failures >= int(args.max_no_progress_failures):
            print(
                "[one-head-resilient] stopping after repeated failures at "
                "the same checkpoint. This prevents an infinite retry loop "
                "for a deterministic software or configuration error.",
                flush=True,
            )
            return return_code or 1

        delay = float(args.retry_delay_seconds)
        if delay:
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a long MuonClip MPS experiment with progress-aware "
            "fresh-process checkpoint recovery"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--max-no-progress-failures",
        type=int,
        default=3,
        help=(
            "stop after this many consecutive worker failures that do not "
            "advance checkpoint_latest.pt; progress resets the count"
        ),
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=5.0,
    )
    args = parser.parse_args()
    raise SystemExit(run_resilient(args))


if __name__ == "__main__":
    main()

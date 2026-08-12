from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import torch

from .config import (
    SUPPORTED_OPTIMIZERS,
    canonical_seeds,
    load_config,
    roots,
)
from .data import prepare_fineweb_edu
from .engine import run_one
from .run_utils import run_directory, run_is_complete
from .runtime import choose_device


def _resolve_roots(
    *,
    data_root: str | Path | None,
    results_root: str | Path | None,
    device: str | torch.device,
) -> tuple[Path, Path, torch.device]:
    # Resolve the accelerator before data preparation. On a TPU VM this catches
    # a missing or incompatible torch_xla installation before downloading or
    # writing the corpus.
    resolved_device = choose_device(device)
    if data_root is not None and results_root is not None:
        return Path(data_root), Path(results_root), resolved_device
    resolved = roots(resolved_device)
    return (
        Path(data_root) if data_root is not None else resolved["data"],
        (
            Path(results_root)
            if results_root is not None
            else resolved["results"]
        ),
        resolved_device,
    )


def _release_accelerator(device: torch.device) -> None:
    """Release cached accelerator state after a replicate exits.

    A fresh subprocess is the primary MPS isolation boundary. This cleanup also
    protects programmatic callers that execute several replicates in one Python
    process.
    """

    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        return

    if device.type != "mps" or not hasattr(torch, "mps"):
        gc.collect()
        return

    try:
        torch.mps.synchronize()
    except Exception as exc:
        # A Metal GPU recovery can make synchronization itself fail. Cleanup is
        # still attempted, and the original training exception is allowed to
        # propagate from the caller.
        print(
            "[one-head-mps] warning: synchronize during cleanup failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception as exc:
                print(
                    "[one-head-mps] warning: empty_cache failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )


def run_optimizer_replicates(
    *,
    cfg: dict,
    config_path: str | Path,
    optimizer_name: str,
    seeds: Sequence[int] | None = None,
    data_root: str | Path | None = None,
    results_root: str | Path | None = None,
    device: str | torch.device = "auto",
    resume: bool = True,
    overwrite: bool = False,
    prepare_data: bool = True,
    progress: bool = True,
) -> list[Path]:
    del config_path
    data_path, results_path, resolved_device = _resolve_roots(
        data_root=data_root,
        results_root=results_root,
        device=device,
    )
    selected_seeds = tuple(
        int(seed) for seed in (seeds or canonical_seeds(cfg))
    )
    if prepare_data:
        prepare_fineweb_edu(cfg, data_path)
    run_dirs: list[Path] = []
    for seed in selected_seeds:
        try:
            run_dirs.append(
                run_one(
                    cfg=deepcopy(cfg),
                    data_root=data_path,
                    results_root=results_path,
                    optimizer_name=optimizer_name,
                    seed=seed,
                    device=resolved_device,
                    resume=resume,
                    overwrite=overwrite,
                    progress=progress,
                )
            )
        finally:
            _release_accelerator(resolved_device)
    return run_dirs


def run_all_replicates(
    *,
    cfg: dict,
    config_path: str | Path,
    seeds: Sequence[int] | None = None,
    data_root: str | Path | None = None,
    results_root: str | Path | None = None,
    device: str | torch.device = "auto",
    resume: bool = True,
    overwrite: bool = False,
    progress: bool = True,
) -> list[Path]:
    data_path, results_path, resolved_device = _resolve_roots(
        data_root=data_root,
        results_root=results_root,
        device=device,
    )
    prepare_fineweb_edu(cfg, data_path)
    outputs: list[Path] = []
    for optimizer_name in SUPPORTED_OPTIMIZERS:
        outputs.extend(
            run_optimizer_replicates(
                cfg=cfg,
                config_path=config_path,
                optimizer_name=optimizer_name,
                seeds=seeds,
                data_root=data_path,
                results_root=results_path,
                device=resolved_device,
                resume=resume,
                overwrite=overwrite,
                prepare_data=False,
                progress=progress,
            )
        )
    return outputs


def _mps_worker_module(optimizer_name: str) -> str:
    # The opt-in MuonClip launcher installs its extension before delegating to
    # training.main. A child process must repeat that installation.
    return (
        "rg_nanogpt_one_head.muonclip"
        if optimizer_name == "muon_clip"
        else "rg_nanogpt_one_head.training"
    )


def _mps_worker_command(
    *,
    args: argparse.Namespace,
    optimizer_name: str,
    seed: int,
    data_root: Path,
    results_root: Path,
    first_attempt: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        _mps_worker_module(optimizer_name),
        "--config",
        str(Path(args.config).resolve()),
        "--optimizer",
        optimizer_name,
        "--seeds",
        str(int(seed)),
        "--data-root",
        str(data_root.resolve()),
        "--results-root",
        str(results_root.resolve()),
        "--device",
        "mps",
        "--mps-worker",
        "--mps-retries",
        "0",
    ]
    if first_attempt and bool(args.overwrite):
        command.append("--overwrite")
    if first_attempt and bool(args.no_resume):
        command.append("--no-resume")
    return command


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


def _clear_failure_marker(run_dir: Path) -> None:
    (run_dir / "run_failed.json").unlink(missing_ok=True)


def _record_failed_replicate(
    *,
    run_dir: Path,
    optimizer_name: str,
    seed: int,
    attempts: int,
    exit_code: int,
    latest_checkpoint: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "failed": True,
        "completed": False,
        "optimizer": str(optimizer_name),
        "seed": int(seed),
        "attempts": int(attempts),
        "last_exit_code": int(exit_code),
        "last_verified_checkpoint": (
            str(latest_checkpoint) if latest_checkpoint.is_file() else None
        ),
        "failure_policy": "skip_replicate_and_continue_batch",
        "recorded_at_utc": _utc_now(),
    }
    _atomic_json(run_dir / "run_failed.json", record)
    return record


def _prepare_retry_without_checkpoint(run_dir: Path) -> None:
    """Restart cleanly when a failed worker never wrote a finite checkpoint."""

    if not run_dir.exists():
        return
    archive = run_dir.with_name(
        run_dir.name
        + ".failed-no-checkpoint."
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if archive.exists():
        shutil.rmtree(archive)
    run_dir.replace(archive)
    print(
        "[one-head-mps] no verified checkpoint exists; archived the partial "
        f"run at {archive} and will restart from initialization",
        flush=True,
    )


def _write_batch_status(
    *,
    results_root: Path,
    started_at_utc: str,
    optimizers: Sequence[str],
    seeds: Sequence[int],
    completed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> Path:
    status_path = results_root / "_batch_status.json"
    payload = {
        "schema_version": 1,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now(),
        "accelerator": "mps",
        "requested_optimizers": [str(value) for value in optimizers],
        "requested_seeds": [int(value) for value in seeds],
        "requested_replicates": int(len(optimizers) * len(seeds)),
        "completed_replicates": int(len(completed)),
        "failed_replicates": int(len(failed)),
        "all_completed": not failed,
        "completed": completed,
        "failed": failed,
    }
    _atomic_json(status_path, payload)
    return status_path


def _run_isolated_mps_workers(
    *,
    args: argparse.Namespace,
    cfg: dict,
    seeds: Sequence[int],
) -> dict[str, Any]:
    data_root, results_root, resolved_device = _resolve_roots(
        data_root=args.data_root,
        results_root=args.results_root,
        device="mps",
    )
    if resolved_device.type != "mps":
        raise RuntimeError(
            "internal MPS worker supervisor selected a non-MPS device"
        )

    prepare_fineweb_edu(cfg, data_root)
    optimizers = (
        tuple(SUPPORTED_OPTIMIZERS)
        if args.optimizer == "all"
        else (str(args.optimizer),)
    )
    max_attempts = 1 + int(args.mps_retries)
    fail_fast = bool(getattr(args, "fail_fast", False))
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    started_at_utc = _utc_now()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for optimizer_name in optimizers:
        for seed in seeds:
            run_dir = run_directory(results_root, optimizer_name, int(seed))
            succeeded = False
            last_return_code = 0
            for attempt in range(1, max_attempts + 1):
                first_attempt = attempt == 1
                command = _mps_worker_command(
                    args=args,
                    optimizer_name=optimizer_name,
                    seed=int(seed),
                    data_root=data_root,
                    results_root=results_root,
                    first_attempt=first_attempt,
                )
                print(
                    "[one-head-mps] starting isolated worker "
                    f"optimizer={optimizer_name} seed={seed} "
                    f"attempt={attempt}/{max_attempts}",
                    flush=True,
                )
                result = subprocess.run(
                    command,
                    env=environment,
                    check=False,
                )
                last_return_code = int(result.returncode)
                if result.returncode == 0:
                    _clear_failure_marker(run_dir)
                    completed.append(
                        {
                            "optimizer": str(optimizer_name),
                            "seed": int(seed),
                            "attempts": int(attempt),
                            "run_dir": str(run_dir),
                        }
                    )
                    print(
                        "[one-head-mps] worker complete "
                        f"optimizer={optimizer_name} seed={seed}",
                        flush=True,
                    )
                    succeeded = True
                    break

                latest = run_dir / "checkpoint_latest.pt"
                if attempt < max_attempts:
                    _release_accelerator(resolved_device)
                    time.sleep(2.0)
                    if latest.is_file():
                        print(
                            "[one-head-mps] worker failed; Metal will restart in "
                            "a fresh process and resume from the last finite "
                            f"atomic checkpoint: {latest}",
                            flush=True,
                        )
                    else:
                        _prepare_retry_without_checkpoint(run_dir)
                    continue

                failure = _record_failed_replicate(
                    run_dir=run_dir,
                    optimizer_name=optimizer_name,
                    seed=int(seed),
                    attempts=max_attempts,
                    exit_code=last_return_code,
                    latest_checkpoint=latest,
                )
                failed.append(failure)
                checkpoint_note = (
                    f"last verified checkpoint: {latest}"
                    if latest.is_file()
                    else "no verified checkpoint was written"
                )
                message = (
                    "isolated MPS worker exhausted its retries with exit code "
                    f"{last_return_code} for optimizer={optimizer_name} "
                    f"seed={seed}; {checkpoint_note}"
                )
                if fail_fast:
                    status_path = _write_batch_status(
                        results_root=results_root,
                        started_at_utc=started_at_utc,
                        optimizers=optimizers,
                        seeds=seeds,
                        completed=completed,
                        failed=failed,
                    )
                    raise RuntimeError(
                        message + f"; batch status: {status_path}"
                    )

                print(
                    "[one-head-mps] SKIPPING failed replicate and continuing "
                    f"to the next optimizer/seed: {message}",
                    flush=True,
                )
                break

            if not succeeded:
                _release_accelerator(resolved_device)

    status_path = _write_batch_status(
        results_root=results_root,
        started_at_utc=started_at_utc,
        optimizers=optimizers,
        seeds=seeds,
        completed=completed,
        failed=failed,
    )
    print(
        "[one-head-mps] batch finished: "
        f"completed={len(completed)} failed={len(failed)} "
        f"status={status_path}",
        flush=True,
    )
    if failed:
        print(
            "[one-head-mps] PARTIAL SUCCESS: failed replicates remain "
            "incomplete and are excluded from completed-run analysis; inspect "
            f"{status_path}",
            flush=True,
        )
    return {
        "completed": completed,
        "failed": failed,
        "status_path": status_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one-head FineWeb-Edu optimizer baselines"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--optimizer",
        choices=[*SUPPORTED_OPTIMIZERS, "all"],
        default="all",
    )
    parser.add_argument(
        "--seeds",
        help="comma-separated seeds; default comes from the config",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    parser.add_argument(
        "--device",
        choices=("auto", "tpu", "xla", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--mps-retries",
        type=int,
        default=1,
        help=(
            "fresh-process resume attempts after an MPS worker failure; "
            "default: 1"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "stop the MPS batch after a replicate exhausts its retries; by "
            "default the failed replicate is recorded and the next seed runs"
        ),
    )
    parser.add_argument(
        "--no-mps-isolation",
        action="store_true",
        help="run MPS replicates in the current process (debugging only)",
    )
    parser.add_argument(
        "--mps-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.mps_retries < 0:
        parser.error("--mps-retries must be nonnegative")
    if args.overwrite and args.no_resume:
        parser.error("--overwrite and --no-resume are mutually exclusive")

    cfg = load_config(args.config)
    seeds = (
        tuple(
            int(value.strip())
            for value in args.seeds.split(",")
            if value.strip()
        )
        if args.seeds
        else canonical_seeds(cfg)
    )
    if not seeds:
        parser.error("at least one seed is required")

    resolved_device = choose_device(args.device)
    if (
        resolved_device.type == "mps"
        and not args.mps_worker
        and not args.no_mps_isolation
    ):
        _run_isolated_mps_workers(
            args=args,
            cfg=cfg,
            seeds=seeds,
        )
        return

    common = dict(
        cfg=cfg,
        config_path=args.config,
        seeds=seeds,
        data_root=args.data_root,
        results_root=args.results_root,
        device=resolved_device,
        resume=not args.no_resume,
        overwrite=args.overwrite,
    )
    if args.optimizer == "all":
        run_all_replicates(**common)
    else:
        run_optimizer_replicates(
            **common,
            optimizer_name=args.optimizer,
        )


if __name__ == "__main__":
    main()

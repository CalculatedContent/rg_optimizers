#!/usr/bin/env python3
"""Visible, fail-fast Mac launch of the exact paired random-canary experiment.

Always uses a new output root. Prior data can be verified and copied; no old
results, checkpoints, or metadata are replaced. No dependencies are installed.
"""
from __future__ import annotations

import argparse
import codecs
from datetime import datetime
import os
from pathlib import Path
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

import run_experiment as campaign


CONFIG = campaign.EXPERIMENT_DIR / "configs" / "fineweb_random_canaries.yaml"
HEARTBEAT_SECONDS = 30.0


def _emit(log, text: str) -> None:
    print(text, flush=True)
    log.write(text + "\n")
    log.flush()


def _stop_group(process) -> None:
    # Only the process group created by this launch is signalled.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_stage(name, command, *, environment, log, heartbeat=HEARTBEAT_SECONDS):
    _emit(log, f"[{datetime.now().isoformat(timespec='seconds')}] START {name}: {shlex.join(command)}")
    started = last_output = time.monotonic()
    next_heartbeat = started + heartbeat
    with subprocess.Popen(
        command, cwd=campaign.EXPERIMENT_DIR, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map() or process.poll() is None:
                    for key, _ in selector.select(timeout=min(1.0, heartbeat)):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        text = decoder.decode(chunk)
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        log.write(text)
                        log.flush()
                        last_output = time.monotonic()
                    now = time.monotonic()
                    if now >= next_heartbeat and process.poll() is None:
                        _emit(
                            log,
                            f"[heartbeat] stage={name} pid={process.pid} "
                            f"elapsed_min={(now - started) / 60:.1f} "
                            f"last_output_sec={now - last_output:.0f}; "
                            "process alive; this is NOT proof of a completed training update",
                        )
                        next_heartbeat = now + heartbeat
                final_text = decoder.decode(b"", final=True)
                if final_text:
                    sys.stdout.write(final_text)
                    log.write(final_text)
            code = process.wait()
        except BaseException:
            _stop_group(process)
            raise
    _emit(log, f"END {name}: exit_code={code}")
    if code:
        raise campaign.CampaignError(f"{name} failed with exit_code={code}; later stages were NOT started")


def _reuse_data(source_root, paths, cfg, log):
    source_root = campaign.resolve_experiment_root(
        {campaign.EXPERIMENT_ROOT_ENV: str(source_root)}
    )
    if not source_root.is_dir():
        _emit(log, f"No existing data root at {source_root}; prepare will download the corpus")
        return
    # An old prepare process must not be writing while we read its corpus.
    lock = campaign._acquire_exclusive_lock(source_root / "logs" / "prepare.log.lock")
    try:
        source = source_root / "data"
        if not (source / "meta.json").is_file():
            _emit(log, "Old corpus has no completed metadata; preserving it and preparing fresh data")
            return
        campaign._verified_data_metadata({"data": source}, cfg)
        for name in ("train.bin", "val.bin", "test.bin", "meta.json"):
            shutil.copy2(source / name, paths["data"] / name)
        campaign._verified_data_metadata(paths, cfg)
        _emit(log, f"REUSED verified corpus from {source}; old results untouched")
    finally:
        campaign._release_exclusive_lock(lock)


def _workflow(root, paths, args, log):
    _emit(log, f"OUTPUT ROOT: {root}")
    _emit(log, f"CONSOLE LOG: {root / 'console.log'}")
    _emit(log, f"CONFIG: {CONFIG}; Python: {sys.executable}")
    _emit(log, "PLAN: verify setup -> doctor on MPS -> reuse/prepare corpus -> AdamW -> MuonClip -> verify completion")
    campaign._require_clean_git()
    cfg = campaign._validate_protocol_config(CONFIG)
    _emit(log, f"EXPECTED: 2 runs, seed=1337, {campaign._expected_total_steps(cfg)} optimizer updates each")
    environment = os.environ.copy()
    environment[campaign.EXPERIMENT_ROOT_ENV] = str(root)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["RG_NANOGPT_LOG_EVERY_STEP"] = "1"
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    base = [sys.executable, "-u", str(campaign.SCRIPT_PATH)]
    config = ["--config", str(CONFIG)]
    _run_stage("doctor", base + ["doctor", *config, "--device", "mps"], environment=environment, log=log)
    if args.reuse_root:
        _reuse_data(args.reuse_root, paths, cfg, log)
    _run_stage("prepare", base + ["prepare", *config], environment=environment, log=log)
    selection = ["--optimizers", "adamw,muon_clip", "--seeds", "1337"]
    _run_stage(
        "training", base + ["run", *config, *selection, "--device", "mps", "--stop-on-error", "--mps-retries", "0"],
        environment=environment, log=log,
    )
    _run_stage("verify", base + ["status", *config, *selection], environment=environment, log=log)
    _emit(log, "SUCCESS: both optimizer runs have verified completion artifacts")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="NEW /tmp output directory (must not already exist)")
    parser.add_argument("--reuse-root", help="old /tmp experiment root containing prepared data")
    args = parser.parse_args(argv)
    default_root = Path("/tmp") / (
        "rg-nanogpt-canaries-visible-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    try:
        root = campaign.resolve_experiment_root({campaign.EXPERIMENT_ROOT_ENV: str(args.root or default_root)})
        root.mkdir(parents=True, exist_ok=False)
    except (campaign.CampaignError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    paths = campaign._paths(root)
    campaign._create_runtime_directories(paths)
    with (root / "console.log").open("x", encoding="utf-8", buffering=1) as log:
        try:
            _workflow(root, paths, args, log)
        except KeyboardInterrupt:
            _emit(log, "INTERRUPTED: old data and any completed checkpoints are preserved")
            return 130
        except Exception as exc:
            _emit(log, f"FAILED: {type(exc).__name__}: {exc}")
            _emit(log, f"Inspect {root / 'console.log'}; this is NOT a completed experiment")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run, resume, inspect, and report the large MuonClip Apple-MPS experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]
PACKAGE_ROOT = REPOSITORY_ROOT / "baseline" / "nanogpt_one_head"
CONFIG_PATH = EXPERIMENT_DIR / "configs" / "muonclip_long_mps.yaml"
ROOT_ENV = "RG_NANOGPT_LARGE_EXPERIMENT_ROOT"
DEFAULT_ROOT = Path("/tmp/rg-nanogpt-muonclip-large-20260830")
OPTIMIZER = "muon_clip"
SEED = 20260830


def experiment_root() -> Path:
    value = Path(os.environ.get(ROOT_ENV, str(DEFAULT_ROOT))).expanduser()
    if not value.is_absolute():
        raise ValueError(f"{ROOT_ENV} must be an absolute path")
    return value.resolve(strict=False)


def paths() -> dict[str, Path]:
    root = experiment_root()
    result = {
        "root": root,
        "data": root / "data",
        "results": root / "results",
        "logs": root / "logs",
        "report": root / "live_report",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def run_dir() -> Path:
    return paths()["results"] / OPTIMIZER / f"seed_{SEED}"


def _install_and_load() -> tuple[dict[str, Any], Any, Any, Any]:
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension

    install_muonclip_extension()
    from rg_nanogpt_one_head.config import load_config, max_steps, tokens_per_step
    from rg_nanogpt_one_head.model import GPT, GPTConfig

    cfg = load_config(CONFIG_PATH)
    return cfg, GPT, GPTConfig, (max_steps, tokens_per_step)


def doctor(device_request: str, smoke_step: bool) -> None:
    import datasets  # noqa: F401
    import matplotlib  # noqa: F401
    import pandas  # noqa: F401
    import torch
    import weightwatcher
    from rg_nanogpt_one_head.runtime import choose_device

    cfg, GPT, GPTConfig, helpers = _install_and_load()
    max_steps, tokens_per_step = helpers
    model = GPT(GPTConfig(**cfg["model"]))
    parameter_count = model.parameter_count()
    step_tokens = tokens_per_step(cfg)
    total_steps = max_steps(cfg)
    processed_tokens = total_steps * step_tokens
    resolved = choose_device(device_request)
    payload = {
        "config": str(CONFIG_PATH),
        "experiment_root": str(experiment_root()),
        "device_request": device_request,
        "resolved_device": str(resolved),
        "torch_version": torch.__version__,
        "weightwatcher_version": weightwatcher.__version__,
        "parameter_count": parameter_count,
        "transformer_blocks": int(cfg["model"]["n_layer"]),
        "attention_heads_per_block": int(cfg["model"]["n_head"]),
        "transformer_matrix_count": 6 * int(cfg["model"]["n_layer"]),
        "train_tokens": int(cfg["dataset"]["train_tokens"]),
        "processed_tokens": processed_tokens,
        "tokens_per_parameter": processed_tokens / parameter_count,
        "tokens_per_optimizer_step": step_tokens,
        "optimizer_steps": total_steps,
        "warmup_steps": round(
            total_steps
            * float(cfg["optimizer_profiles"][OPTIMIZER]["warmup_fraction"])
        ),
        "optimizer": OPTIMIZER,
        "seed": SEED,
        "estimated_dataset_gib": (
            2
            * sum(
                int(cfg["dataset"][name])
                for name in ("train_tokens", "val_tokens", "test_tokens")
            )
            / 1024**3
        ),
    }
    if smoke_step:
        from rg_nanogpt_one_head.config import optimizer_profile
        from rg_nanogpt_one_head.optimizers import (
            make_optimizer_handles,
            optimizer_step,
            zero_grad,
        )
        from rg_nanogpt_one_head.runtime import synchronize

        model.to(resolved)
        model.train()
        handles = make_optimizer_handles(
            model,
            optimizer_profile(cfg, OPTIMIZER),
        )
        zero_grad(handles)
        x = torch.randint(
            0,
            int(cfg["model"]["vocab_size"]),
            (
                int(cfg["training"]["batch_size"]),
                int(cfg["model"]["block_size"]),
            ),
            device=resolved,
        )
        _, loss = model(x, x)
        if loss is None:
            raise RuntimeError("large-model smoke step did not return a loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        optimizer_step(handles)
        synchronize(resolved)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise FloatingPointError("large-model smoke loss is non-finite")
        payload["smoke_optimizer_step"] = "passed"
        payload["smoke_loss"] = loss_value
    print(json.dumps(payload, indent=2, sort_keys=True))


def prepare(force: bool) -> None:
    cfg, _, _, _ = _install_and_load()
    from rg_nanogpt_one_head.data import prepare_fineweb_edu

    prepare_fineweb_edu(cfg, paths()["data"], force=force)


def run_training(device: str, mps_retries: int, overwrite: bool) -> int:
    resolved = paths()
    command = [
        sys.executable,
        "-u",
        "-m",
        "rg_nanogpt_one_head.muonclip",
        "--config",
        str(CONFIG_PATH),
        "--optimizer",
        OPTIMIZER,
        "--seeds",
        str(SEED),
        "--data-root",
        str(resolved["data"]),
        "--results-root",
        str(resolved["results"]),
        "--device",
        device,
        "--mps-retries",
        str(int(mps_retries)),
        "--fail-fast",
    ]
    if overwrite:
        command.append("--overwrite")

    log_path = resolved["logs"] / "train.log"
    print("[large-muonclip] command:", " ".join(command), flush=True)
    print(f"[large-muonclip] log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=EXPERIMENT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
    return int(process.wait())


def _last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    for _ in range(2):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            return rows[-1] if rows else None
        except (csv.Error, OSError):
            continue
    return None


def _number(row: dict[str, str], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def status() -> None:
    cfg, _, _, helpers = _install_and_load()
    total_steps = helpers[0](cfg)
    directory = run_dir()
    print(f"RUN: {directory}")

    completion = directory / "run_complete.json"
    if completion.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        print("STATE: COMPLETE")
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("STATE: INCOMPLETE OR RUNNING")

    row = _last_csv_row(directory / "metrics.csv")
    if row is None:
        print("TRAINING: waiting for the first evaluation row")
    else:
        step = int(_number(row, "step"))
        elapsed = _number(row, "elapsed_sec")
        remaining = max(0, total_steps - step)
        eta = remaining * elapsed / step if step > 0 else float("nan")
        print(
            "TRAINING: "
            f"step={step:,}/{total_steps:,} ({100 * step / total_steps:.2f}%) "
            f"epoch={_number(row, 'epoch'):.4f} "
            f"train_loss={_number(row, 'train_loss'):.4f} "
            f"val_loss={_number(row, 'val_loss'):.4f} "
            f"val_ppl={_number(row, 'val_perplexity'):.2f} "
            f"val_acc={100 * _number(row, 'val_accuracy'):.2f}% "
            f"tokens_per_sec={_number(row, 'tokens_per_sec'):,.0f} "
            f"eta_hours={eta / 3600:.1f}"
        )

    spectral = _last_csv_row(directory / "spectral" / "summary.csv")
    if spectral is None:
        print("WEIGHTWATCHER: waiting for the first permanent state")
    else:
        print(
            "WEIGHTWATCHER: "
            f"step={int(_number(spectral, 'step')):,} "
            f"epoch={_number(spectral, 'epoch'):.4f} "
            f"matrices={int(_number(spectral, 'n_matrices'))} "
            f"alpha_raw_median={_number(spectral, 'alpha_raw_median'):.3f} "
            "alpha_clip_median="
            f"{_number(spectral, 'alpha_clip_xmax_median'):.3f} "
            f"ERG_gap_median={_number(spectral, 'ERG_gap_median'):.3f} "
            f"num_traps_mean={_number(spectral, 'num_traps_mean'):.2f}"
        )

    checkpoint = directory / "checkpoint_latest.pt"
    print(
        "CHECKPOINT: "
        + (
            f"present ({checkpoint.stat().st_size / 1024**2:.1f} MiB)"
            if checkpoint.is_file()
            else "not written yet"
        )
    )
    process = subprocess.run(
        ["pgrep", "-fl", "rg_nanogpt_one_head.muonclip"],
        text=True,
        capture_output=True,
        check=False,
    )
    print("PROCESS:")
    print(process.stdout.strip() or "no MuonClip process found")


def report(open_report: bool) -> None:
    output = paths()["report"] / "report.html"
    command = [
        sys.executable,
        str(EXPERIMENT_DIR / "scripts" / "build_live_report.py"),
        "--experiment-root",
        str(experiment_root()),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(f"REPORT: {output}")
    if open_report:
        subprocess.run(["open", str(output)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--device", default="auto")
    doctor_parser.add_argument("--smoke-step", action="store_true")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--mps-retries", type=int, default=20)
    run_parser.add_argument("--overwrite", action="store_true")

    subparsers.add_parser("status")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--open", action="store_true")

    args = parser.parse_args()
    if args.command == "doctor":
        doctor(args.device, args.smoke_step)
    elif args.command == "prepare":
        prepare(args.force)
    elif args.command == "run":
        if args.mps_retries < 0:
            parser.error("--mps-retries must be nonnegative")
        raise SystemExit(
            run_training(args.device, args.mps_retries, args.overwrite)
        )
    elif args.command == "status":
        status()
    elif args.command == "report":
        report(args.open)


if __name__ == "__main__":
    main()

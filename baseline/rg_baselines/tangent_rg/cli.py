"""Command-line entry point for restartable tangent-RG training.

Example::

    python -m rg_baselines.tangent_rg.cli train \
        --config configs/pilot_1000_epochs.yaml \
        --optimizer muon --output-root /persistent/rg-runs --seed 1337 --resume
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

from .config import SUPPORTED_OPTIMIZERS, load_config
from .protocol import (
    build_analysis_plan,
    make_run_layout,
    make_tail_checkpoint_layout,
    tail_checkpoint_epochs,
    validate_disjoint_checkpoint_layouts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rg_baselines.tangent_rg.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train or resume one optimizer/seed arm")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--device")
    train.add_argument("--data-dir", type=Path)
    train.add_argument(
        "--tail-checkpoint-root",
        type=Path,
        help=(
            "temporary cache root beneath /tmp "
            "(default: /tmp/rg-mnist-mlp3-tangent-checkpoints)"
        ),
    )
    train.add_argument("--resume", action="store_true")
    train.add_argument("--overwrite", action="store_true")
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve/print schedule and storage contract without loading MNIST",
    )
    train.add_argument("--quiet", action="store_true")
    return parser


def _dry_run_payload(config, output_root: Path) -> dict:
    optimization_examples = 60_000 - int(config.validation_size)
    steps_per_epoch = int(math.ceil(optimization_examples / config.batch_size))
    plan = build_analysis_plan(config, steps_per_epoch=steps_per_epoch)
    layout = make_run_layout(config)
    tail_layout = make_tail_checkpoint_layout(config)
    validate_disjoint_checkpoint_layouts(layout, tail_layout)
    matrix_elements = 512 * 784 + 512 * 512
    model_parameter_count = 669_706
    tail_count = len(tail_checkpoint_epochs(config.epochs))
    tail_model_tensor_bytes = tail_count * model_parameter_count * 4
    captured_tensor_count = 7  # before/after, gradient, source, polar, direction, delta
    dense_bytes_upper_bound = (
        len(plan.capture_completed_steps)
        * matrix_elements
        * captured_tensor_count
        * 4
    )
    return {
        "config": config.to_dict(),
        "analysis_plan": plan.to_dict(),
        "run_dir": str(layout.root),
        "tail_checkpoint_cache_dir": str(tail_layout.root),
        "tail_checkpoint_epochs": list(tail_checkpoint_epochs(config.epochs)),
        "tail_checkpoint_count": tail_count,
        "tail_checkpoint_fp32_model_tensor_payload_bytes": tail_model_tensor_bytes,
        "tail_checkpoint_fp32_model_tensor_payload_gib": (
            tail_model_tensor_bytes / 2**30
        ),
        "tail_checkpoint_fp32_model_tensor_payload_mib_per_state": (
            model_parameter_count * 4 / 2**20
        ),
        "output_root": str(output_root),
        "optimization_examples": optimization_examples,
        "estimated_dense_capture_gib_upper_bound": dense_bytes_upper_bound / 2**30,
        "note": (
            "estimate excludes one bounded full calibration state per burst and "
            "does not execute or download data; the tail checkpoint estimate is "
            "the exact fp32 model-tensor payload for 669706 MLP3 parameters and "
            "excludes Torch serialization/container overhead"
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "train":
        raise AssertionError(f"unhandled command: {args.command}")
    config = load_config(args.config).with_overrides(
        optimizer=args.optimizer,
        seed=args.seed,
        device=args.device,
        data_dir=None if args.data_dir is None else str(args.data_dir),
        run_root=str(args.output_root),
        tail_checkpoint_cache_root=(
            None
            if args.tail_checkpoint_root is None
            else str(args.tail_checkpoint_root)
        ),
    )
    if args.dry_run:
        print(json.dumps(_dry_run_payload(config, args.output_root), indent=2, sort_keys=True))
        return 0

    from .training import run_training

    result = run_training(
        config,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        progress=not bool(args.quiet),
    )
    print(json.dumps(result.completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

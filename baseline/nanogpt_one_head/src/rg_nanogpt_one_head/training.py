from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import torch

from .config import SUPPORTED_OPTIMIZERS, canonical_seeds, load_config, roots
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
    run_dirs = []
    for seed in selected_seeds:
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
    args = parser.parse_args()
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
    common = dict(
        cfg=cfg,
        config_path=args.config,
        seeds=seeds,
        data_root=args.data_root,
        results_root=args.results_root,
        device=args.device,
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

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Sequence

from .config import SUPPORTED_OPTIMIZERS, canonical_seeds, load_config, roots
from .data import prepare_fineweb_edu
from .engine import run_one
from .run_utils import run_directory, run_is_complete


def run_optimizer_replicates(
    *,
    cfg: dict,
    config_path: str | Path,
    optimizer_name: str,
    seeds: Sequence[int] | None = None,
    data_root: str | Path | None = None,
    results_root: str | Path | None = None,
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    prepare_data: bool = True,
    progress: bool = True,
) -> list[Path]:
    resolved = roots()
    data_root = Path(data_root or resolved["data"])
    results_root = Path(results_root or resolved["results"])
    selected_seeds = tuple(int(seed) for seed in (seeds or canonical_seeds(cfg)))
    if prepare_data:
        prepare_fineweb_edu(cfg, data_root)
    run_dirs = []
    for seed in selected_seeds:
        run_dirs.append(
            run_one(
                cfg=deepcopy(cfg),
                data_root=data_root,
                results_root=results_root,
                optimizer_name=optimizer_name,
                seed=seed,
                device=device,
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
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    progress: bool = True,
) -> list[Path]:
    resolved = roots()
    data_root = Path(data_root or resolved["data"])
    results_root = Path(results_root or resolved["results"])
    prepare_fineweb_edu(cfg, data_root)
    outputs: list[Path] = []
    for optimizer_name in SUPPORTED_OPTIMIZERS:
        outputs.extend(
            run_optimizer_replicates(
                cfg=cfg,
                config_path=config_path,
                optimizer_name=optimizer_name,
                seeds=seeds,
                data_root=data_root,
                results_root=results_root,
                device=device,
                resume=resume,
                overwrite=overwrite,
                prepare_data=False,
                progress=progress,
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-head FineWeb-Edu optimizer baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--optimizer", choices=[*SUPPORTED_OPTIMIZERS, "all"], default="all")
    parser.add_argument("--seeds", help="comma-separated seeds; default comes from the config")
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seeds = (
        tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        if args.seeds
        else canonical_seeds(cfg)
    )
    if args.optimizer == "all":
        run_all_replicates(
            cfg=cfg,
            config_path=args.config,
            seeds=seeds,
            data_root=args.data_root,
            results_root=args.results_root,
            device=args.device,
            resume=not args.no_resume,
            overwrite=args.overwrite,
        )
    else:
        run_optimizer_replicates(
            cfg=cfg,
            config_path=args.config,
            optimizer_name=args.optimizer,
            seeds=seeds,
            data_root=args.data_root,
            results_root=args.results_root,
            device=args.device,
            resume=not args.no_resume,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

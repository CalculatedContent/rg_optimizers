from __future__ import annotations

"""Transform construction and WeightWatcher tables for one quotient method."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rg_nanogpt_one_head.muonclip_angular_order_parameters import (
    build_transform,
    haar_susceptibility_mean,
    method_matched_null,
)

from config_io import ExperimentConfig
from weightwatcher_support import finite_quantiles, run_weightwatcher


def build_transforms(
    *,
    methods: tuple[str, ...],
    trajectory: dict[str, list[np.ndarray]],
    records: list[dict[str, Any]],
    output_dir: Path,
    config: ExperimentConfig,
) -> tuple[
    dict[str, dict[str, list[np.ndarray]]],
    pd.DataFrame,
    dict[tuple[int, int], np.ndarray],
]:
    matrix_names = sorted(trajectory)
    haar_means: dict[tuple[int, int], np.ndarray] = {}
    for matrix_name in matrix_names:
        shape = tuple(trajectory[matrix_name][0].shape)
        if shape not in haar_means:
            haar_means[shape] = haar_susceptibility_mean(
                shape,
                samples=config.haar_samples,
                seed=config.seed + 100003 * len(haar_means),
            )

    transforms = {
        method: {name: [] for name in matrix_names}
        for method in methods
    }
    metadata_rows: list[dict[str, Any]] = []
    transform_dir = output_dir / "transformed_matrices"
    transform_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        method_dir = transform_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(records):
            saved: dict[str, np.ndarray] = {}
            for matrix_name in matrix_names:
                sequence = trajectory[matrix_name]
                shape = tuple(sequence[index].shape)
                result = build_transform(
                    method,
                    sequence,
                    index,
                    haar_mean=haar_means.get(shape),
                    diffusion_mass=config.diffusion_mass,
                    temporal_max_block=config.temporal_max_block,
                    seed=config.seed + 1009 * index,
                )
                transforms[method][matrix_name].append(result.matrix)
                saved[matrix_name] = result.matrix
                metadata_rows.append(
                    {
                        "method": method,
                        "matrix_name": matrix_name,
                        "checkpoint_index": index,
                        "step": record["step"],
                        "epoch": record["epoch"],
                        **result.metadata,
                    }
                )
            np.savez_compressed(
                method_dir / f"step_{record['step']:07d}.npz",
                **saved,
            )

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(output_dir / "transform_metadata.csv", index=False)
    return transforms, metadata, haar_means


def actual_metrics(
    *,
    methods: tuple[str, ...],
    transforms: dict[str, dict[str, list[np.ndarray]]],
    matrix_names: list[str],
    records: list[dict[str, Any]],
    output_dir: Path,
    config: ExperimentConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method in methods:
        for index, record in enumerate(records):
            matrices = {
                name: transforms[method][name][index]
                for name in matrix_names
            }
            frames.append(
                run_weightwatcher(
                    matrices,
                    record=record,
                    method=method,
                    namespace="actual",
                    output_root=output_dir,
                    config=config,
                    make_plots=True,
                )
            )
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["alpha"] = pd.to_numeric(result.get("alpha"), errors="coerce")
    result["y_E"] = 2.0 - result["alpha"]
    result.to_csv(output_dir / "actual_weightwatcher_metrics.csv", index=False)
    return result


def null_metrics(
    *,
    methods: tuple[str, ...],
    trajectory: dict[str, list[np.ndarray]],
    matrix_names: list[str],
    records: list[dict[str, Any]],
    haar_means: dict[tuple[int, int], np.ndarray],
    output_dir: Path,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames: list[pd.DataFrame] = []
    for replicate in range(config.null_replicates):
        rng = np.random.default_rng(config.seed + 7_000_001 + replicate)
        for method in methods:
            for index, record in enumerate(records):
                matrices: dict[str, np.ndarray] = {}
                for matrix_name in matrix_names:
                    sequence = trajectory[matrix_name]
                    shape = tuple(sequence[index].shape)
                    result = method_matched_null(
                        method,
                        sequence,
                        index,
                        rng=rng,
                        haar_mean=haar_means.get(shape),
                        diffusion_mass=config.diffusion_mass,
                        temporal_max_block=config.temporal_max_block,
                        seed=config.seed + replicate,
                    )
                    matrices[matrix_name] = result.matrix
                frame = run_weightwatcher(
                    matrices,
                    record=record,
                    method=method,
                    namespace=f"null_rep_{replicate:03d}",
                    output_root=output_dir,
                    config=config,
                    make_plots=False,
                )
                frame["null_replicate"] = replicate
                frames.append(frame)

    result = pd.concat(frames, ignore_index=True, sort=False)
    result.to_csv(output_dir / "null_weightwatcher_metrics.csv", index=False)

    grouping = ["method", "matrix_name", "checkpoint_index", "step"]
    summaries: dict[str, pd.DataFrame] = {}
    for metric in ("alpha", "rand_distance", "D"):
        summary = (
            result.groupby(grouping, dropna=False)[metric]
            .apply(finite_quantiles)
            .unstack()
            .reset_index()
        )
        summary.to_csv(output_dir / f"null_{metric}_summary.csv", index=False)
        summaries[metric] = summary
    return result, summaries

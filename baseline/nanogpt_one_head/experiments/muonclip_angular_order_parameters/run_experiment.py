from __future__ import annotations

"""Run one MuonClip angular quotient experiment across saved checkpoints."""

from dataclasses import asdict
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rg_nanogpt_one_head.muonclip_angular_order_parameters import METHOD_ORDER

from analysis_core import actual_metrics, build_transforms, null_metrics
from config_io import (
    ExperimentConfig,
    alias_index,
    discover_checkpoints,
    load_trajectory,
)
from plot_support import contact_sheet, esd_overlay, metric_grid, write_manifest


def run_experiment(
    method: str,
    *,
    experiment_name: str,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Analyze the raw matrices and one explicit angular quotient field."""
    if method not in METHOD_ORDER or method == "raw":
        raise KeyError(
            f"Expected one non-raw method from {METHOD_ORDER}; got {method!r}"
        )

    config = config or ExperimentConfig.from_env()
    config.validate()

    run_dir = Path(config.run_dir).expanduser().resolve()
    output_dir = (
        run_dir
        / "diagnostics"
        / "experimental_muonclip_angular_order_parameters"
        / experiment_name
    )
    plot_dir = output_dir / "plots"
    contact_dir = output_dir / "contact_sheets"
    for directory in (output_dir, plot_dir, contact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    methods = ("raw", method)
    records = discover_checkpoints(run_dir)
    trajectory, model_config = load_trajectory(run_dir, records)
    matrix_names = sorted(trajectory)
    best_index = alias_index(records, "best")
    final_index = alias_index(records, "final")

    checkpoint_table = pd.DataFrame(
        {
            "checkpoint_index": [record["index"] for record in records],
            "step": [record["step"] for record in records],
            "epoch": [record["epoch"] for record in records],
            "aliases": [",".join(record["aliases"]) for record in records],
            "path": [str(record["path"]) for record in records],
        }
    )
    checkpoint_table.to_csv(output_dir / "checkpoint_table.csv", index=False)

    transforms, transform_metadata, haar_means = build_transforms(
        methods=methods,
        trajectory=trajectory,
        records=records,
        output_dir=output_dir,
        config=config,
    )
    actual = actual_metrics(
        methods=methods,
        transforms=transforms,
        matrix_names=matrix_names,
        records=records,
        output_dir=output_dir,
        config=config,
    )
    null, null_summaries = null_metrics(
        methods=methods,
        trajectory=trajectory,
        matrix_names=matrix_names,
        records=records,
        haar_means=haar_means,
        output_dir=output_dir,
        config=config,
    )

    plot_files: list[Path] = []
    for current_method in methods:
        plot_files.extend(
            [
                metric_grid(
                    method=current_method,
                    metric="alpha",
                    ylabel="WeightWatcher alpha",
                    records=records,
                    matrix_names=matrix_names,
                    actual=actual,
                    null_summary=null_summaries["alpha"],
                    path=plot_dir / f"{current_method}__alpha_vs_step.png",
                    reference=2.0,
                ),
                metric_grid(
                    method=current_method,
                    metric="y_E",
                    ylabel="RG exponent y_E = 2-alpha",
                    records=records,
                    matrix_names=matrix_names,
                    actual=actual,
                    null_summary=None,
                    path=plot_dir / f"{current_method}__yE_vs_step.png",
                    reference=0.0,
                ),
                metric_grid(
                    method=current_method,
                    metric="rand_distance",
                    ylabel="WeightWatcher RAND distance",
                    records=records,
                    matrix_names=matrix_names,
                    actual=actual,
                    null_summary=null_summaries["rand_distance"],
                    path=plot_dir / f"{current_method}__rand_vs_step.png",
                ),
                metric_grid(
                    method=current_method,
                    metric="D",
                    ylabel="power-law KS distance D",
                    records=records,
                    matrix_names=matrix_names,
                    actual=actual,
                    null_summary=null_summaries["D"],
                    path=plot_dir / f"{current_method}__D_vs_step.png",
                ),
            ]
        )
        for matrix_name in matrix_names:
            overlay = esd_overlay(
                method=current_method,
                matrix_name=matrix_name,
                matrices=transforms[current_method][matrix_name],
                records=records,
                path=plot_dir
                / f"{current_method}__{matrix_name}__esd_overlay.png",
            )
            if overlay is not None:
                plot_files.append(overlay)

    contact_sheets = [
        contact_sheet(
            method=current_method,
            records=records,
            matrix_names=matrix_names,
            metrics=actual,
            path=contact_dir
            / f"{current_method}__native_esd_contact_sheet.png",
        )
        for current_method in methods
    ]

    grouping = ["method", "matrix_name", "checkpoint_index", "step"]
    endpoint = actual[
        actual["checkpoint_index"].isin([best_index, final_index])
    ].copy()
    alpha_null = null_summaries["alpha"].rename(
        columns={
            "q025": "null_alpha_q025",
            "median": "null_alpha_median",
            "q975": "null_alpha_q975",
            "n": "null_alpha_n",
        }
    )
    rand_null = null_summaries["rand_distance"].rename(
        columns={
            "q025": "null_rand_q025",
            "median": "null_rand_median",
            "q975": "null_rand_q975",
            "n": "null_rand_n",
        }
    )
    endpoint = endpoint.merge(alpha_null, on=grouping, how="left").merge(
        rand_null, on=grouping, how="left"
    )
    endpoint["distance_to_alpha_2"] = np.abs(endpoint["alpha"] - 2.0)
    endpoint["alpha_outside_null"] = (
        (endpoint["alpha"] < endpoint["null_alpha_q025"])
        | (endpoint["alpha"] > endpoint["null_alpha_q975"])
    )
    endpoint["rand_above_null"] = (
        pd.to_numeric(endpoint.get("rand_distance"), errors="coerce")
        > endpoint["null_rand_q975"]
    )
    endpoint.to_csv(output_dir / "endpoint_summary.csv", index=False)

    manifest = {
        "config": asdict(config),
        "method": method,
        "methods": list(methods),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "matrix_names": matrix_names,
        "checkpoints": checkpoint_table.to_dict(orient="records"),
        "best_index": best_index,
        "final_index": final_index,
        "model_config": model_config,
        "plot_files": [str(path) for path in plot_files],
        "contact_sheets": [str(path) for path in contact_sheets],
        "weightwatcher_call": {
            "plot": True,
            "randomize": True,
            "ERG": False,
            "fix_fingers": "clip_xmax",
            "min_evals": config.min_evals,
            "max_fingers": config.max_fingers,
        },
    }
    write_manifest(output_dir / "manifest.json", manifest)

    del trajectory, transforms
    gc.collect()
    return {
        "config": config,
        "method": method,
        "records": records,
        "checkpoint_table": checkpoint_table,
        "matrix_names": matrix_names,
        "transform_metadata": transform_metadata,
        "actual_metrics": actual,
        "null_metrics": null,
        "null_summaries": null_summaries,
        "endpoint_summary": endpoint,
        "plot_files": plot_files,
        "contact_sheets": contact_sheets,
        "output_dir": output_dir,
        "manifest": manifest,
    }

from __future__ import annotations

"""Native WeightWatcher execution for experimental matrix inventories."""

import gc
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rg_nanogpt_one_head.spectral import _attach_matrix_metadata
from config_io import ExperimentConfig


class MatrixHolder(nn.Module):
    def __init__(self, matrices: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.matrix_metadata: list[dict[str, object]] = []
        for name, array in matrices.items():
            values = np.asarray(array, dtype=np.float32)
            layer = nn.Linear(values.shape[1], values.shape[0], bias=False)
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(values))
            layer.weight.requires_grad_(False)
            self.add_module(name, layer)
            self.matrix_metadata.append(
                {"matrix_name": name, "matrix_type": name, "block": 0}
            )


def native_esd_path(directory: Path, layer_id: Any) -> Path | None:
    text = str(layer_id)
    if text.endswith(".0"):
        text = text[:-2]
    expected = directory / f"ww.layer{text}.esd.png"
    if expected.is_file():
        return expected
    candidates = sorted(directory.glob("ww.layer*.esd.png"))
    return candidates[0] if len(candidates) == 1 else None


def run_weightwatcher(
    matrices: dict[str, np.ndarray],
    *,
    record: dict[str, Any],
    method: str,
    namespace: str,
    output_root: Path,
    config: ExperimentConfig,
    make_plots: bool,
) -> pd.DataFrame:
    import weightwatcher as ww

    directory = (
        output_root
        / "weightwatcher"
        / namespace
        / method
        / f"step_{record['step']:07d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.csv"
    if metrics_path.is_file() and not config.force:
        return pd.read_csv(metrics_path)

    usable = {
        name: np.asarray(matrix, dtype=np.float64)
        for name, matrix in matrices.items()
        if np.linalg.norm(matrix) > 1e-10
    }
    zero_names = sorted(set(matrices) - set(usable))
    rows: list[dict[str, Any]] = []

    if usable:
        holder = MatrixHolder(usable)
        diagnostic_seed = (
            config.seed
            + 1_000_003
            + int(record["step"])
            + sum(map(ord, method + namespace))
        )
        random.seed(diagnostic_seed)
        np.random.seed(diagnostic_seed % (2**32 - 1))
        torch.manual_seed(diagnostic_seed)
        old_show = plt.show
        plt.show = lambda *args, **kwargs: None
        try:
            watcher = ww.WeightWatcher(model=holder)
            kwargs = {
                "plot": bool(make_plots),
                "min_evals": config.min_evals,
                "randomize": True,
                "ERG": False,
                "fix_fingers": "clip_xmax",
                "max_fingers": config.max_fingers,
            }
            if make_plots:
                kwargs["savefig"] = str(directory)
            details = watcher.analyze(**kwargs)
        finally:
            plt.show = old_show
            plt.close("all")

        frame = _attach_matrix_metadata(
            pd.DataFrame(details), holder.matrix_metadata
        )
        for _, row in frame.iterrows():
            item = row.to_dict()
            item["status"] = "ok"
            image = (
                native_esd_path(directory, row.get("layer_id"))
                if make_plots
                else None
            )
            item["native_esd_file"] = str(image) if image else ""
            rows.append(item)
        del holder, watcher, details, frame
        gc.collect()

    for name in zero_names:
        rows.append(
            {
                "matrix_name": name,
                "matrix_type": name,
                "block": 0,
                "status": "zero_by_construction",
                "alpha": np.nan,
                "D": np.nan,
                "rand_distance": np.nan,
                "native_esd_file": "",
            }
        )

    result = pd.DataFrame(rows)
    result.insert(0, "namespace", namespace)
    result.insert(1, "method", method)
    result.insert(2, "checkpoint_index", int(record["index"]))
    result.insert(3, "step", int(record["step"]))
    result.insert(4, "epoch", float(record["epoch"]))
    result.insert(5, "aliases", ",".join(record["aliases"]))
    result.to_csv(metrics_path, index=False)
    return result


def finite_quantiles(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return pd.Series(
            {"q025": np.nan, "median": np.nan, "q975": np.nan, "n": 0}
        )
    q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
    return pd.Series(
        {
            "q025": float(q025),
            "median": float(median),
            "q975": float(q975),
            "n": int(values.size),
        }
    )

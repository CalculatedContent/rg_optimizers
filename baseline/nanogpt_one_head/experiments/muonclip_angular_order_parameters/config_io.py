from __future__ import annotations

"""Configuration and checkpoint I/O for MuonClip quotient experiments."""

from dataclasses import dataclass
import gc
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from rg_nanogpt_one_head.angular_weightwatcher_core import (
    _build_model,
    _load_payload,
    _model_config,
)
from rg_nanogpt_one_head.model import transformer_matrix_items


@dataclass(frozen=True)
class ExperimentConfig:
    run_dir: str
    seed: int = 4242
    min_evals: int = 20
    max_fingers: int = 10
    null_replicates: int = 8
    haar_samples: int = 64
    diffusion_mass: float = 0.05
    temporal_max_block: int = 8
    force: bool = False

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        run_dir = os.environ.get("RUN_DIR", "").strip()
        if not run_dir:
            raise EnvironmentError("Set RUN_DIR before running this notebook")
        return cls(
            run_dir=run_dir,
            seed=int(os.environ.get("TARGET_SEED", "4242")),
            min_evals=int(os.environ.get("WW_MIN_EVALS", "20")),
            max_fingers=int(os.environ.get("WW_MAX_FINGERS", "10")),
            null_replicates=int(os.environ.get("ANGULAR_QUOTIENT_NULLS", "8")),
            haar_samples=int(
                os.environ.get("ANGULAR_QUOTIENT_HAAR_SAMPLES", "64")
            ),
            diffusion_mass=float(
                os.environ.get("ANGULAR_QUOTIENT_DIFFUSION_MASS", "0.05")
            ),
            temporal_max_block=int(
                os.environ.get("ANGULAR_QUOTIENT_MAX_BLOCK", "8")
            ),
            force=os.environ.get("ANGULAR_QUOTIENT_FORCE", "0").lower()
            in {"1", "true", "yes", "on"},
        )

    def validate(self) -> None:
        run_dir = Path(self.run_dir).expanduser()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        if self.null_replicates < 3:
            raise ValueError("ANGULAR_QUOTIENT_NULLS must be at least 3")
        if self.haar_samples < 4:
            raise ValueError("ANGULAR_QUOTIENT_HAAR_SAMPLES must be at least 4")
        if self.diffusion_mass <= 0.0:
            raise ValueError("ANGULAR_QUOTIENT_DIFFUSION_MASS must be positive")
        if self.temporal_max_block < 1:
            raise ValueError("ANGULAR_QUOTIENT_MAX_BLOCK must be positive")


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return int(_load_payload(path).get("step", -1))


def discover_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    priorities = {
        "initial": 100,
        "final": 90,
        "best": 80,
        "epoch": 70,
        "latest": 10,
    }
    candidates: list[tuple[Path, str, int]] = []
    for filename, alias in (
        ("checkpoint_initial.pt", "initial"),
        ("checkpoint_best.pt", "best"),
        ("checkpoint_final.pt", "final"),
        ("checkpoint_latest.pt", "latest"),
    ):
        path = run_dir / filename
        if path.is_file():
            candidates.append((path, alias, priorities[alias]))
    for path in sorted((run_dir / "epoch_checkpoints").glob("*.pt")):
        candidates.append((path, "epoch", priorities["epoch"]))

    by_step: dict[int, list[tuple[Path, str, int]]] = {}
    for path, alias, priority in candidates:
        by_step.setdefault(checkpoint_step(path), []).append(
            (path, alias, priority)
        )
    if 0 not in by_step:
        raise FileNotFoundError("No real saved step-zero checkpoint was found")

    epoch_by_step: dict[int, float] = {}
    epoch_csv = run_dir / "epoch_metrics.csv"
    if epoch_csv.is_file():
        frame = pd.read_csv(epoch_csv)
        epoch_column = (
            "epoch"
            if "epoch" in frame.columns
            else "nominal_epoch"
            if "nominal_epoch" in frame.columns
            else None
        )
        if epoch_column:
            for _, row in frame.iterrows():
                try:
                    epoch_by_step[int(row["step"])] = float(row[epoch_column])
                except Exception:
                    pass

    records: list[dict[str, Any]] = []
    for step, items in sorted(by_step.items()):
        selected = max(items, key=lambda item: item[2])[0]
        aliases = tuple(
            sorted(
                {item[1] for item in items},
                key=lambda alias: -priorities[alias],
            )
        )
        payload = _load_payload(selected)
        records.append(
            {
                "index": len(records),
                "step": int(step),
                "epoch": float(
                    epoch_by_step.get(step, payload.get("epoch", np.nan))
                ),
                "aliases": aliases,
                "path": selected,
            }
        )
    return records


def alias_index(records: list[dict[str, Any]], alias: str) -> int:
    indices = [
        index
        for index, record in enumerate(records)
        if alias in record["aliases"]
    ]
    return indices[-1] if indices else len(records) - 1


def load_trajectory(
    run_dir: Path,
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    initial_payload = _load_payload(records[0]["path"])
    final_record = next(
        (record for record in records if "final" in record["aliases"]),
        records[-1],
    )
    final_payload = _load_payload(final_record["path"])
    model_config = _model_config(initial_payload, final_payload, run_dir)

    trajectory: dict[str, list[np.ndarray]] = {}
    for record in records:
        model = _build_model(_load_payload(record["path"]), model_config)
        matrices = {
            name: parameter.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
            for name, _, _, parameter in transformer_matrix_items(model)
        }
        if not trajectory:
            trajectory = {name: [] for name in matrices}
        for name, matrix in matrices.items():
            trajectory[name].append(matrix)
        del model, matrices
        gc.collect()
    return trajectory, model_config

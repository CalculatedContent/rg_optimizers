"""Hardened public runtime for the MNIST reference baselines.

The numerical training implementation lives in :mod:`rg_baselines.runner`.
This module enforces append-safe restart behavior before delegating to it:
partial artifacts without a matching latest checkpoint are rejected, tabular
and ESD histories are truncated to the checkpoint epoch, and completed runs
must retain their completion and final-state artifacts.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import BaselineConfig
from .runner import run_baseline as _run_baseline

_EPOCH_KEY = re.compile(r"^epoch_(\d{3})__")
_PROGRESS_TABLES = (
    "performance_by_epoch.csv",
    "spectral_metrics_by_epoch_and_layer.csv",
    "weightwatcher_details_by_epoch.csv",
    "optimizer_groups_by_epoch.csv",
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _truncate_table(path: Path, epoch: int) -> None:
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    if "epoch" not in frame.columns:
        raise RuntimeError(f"restart table has no epoch column: {path}")
    retained = frame[frame["epoch"].astype(int) <= int(epoch)].copy()
    _atomic_csv(retained, path)


def _truncate_esd_history(path: Path, epoch: int) -> None:
    if not path.is_file():
        return
    with np.load(path) as archive:
        retained: dict[str, np.ndarray] = {}
        for key in archive.files:
            match = _EPOCH_KEY.match(key)
            if match is None:
                raise RuntimeError(
                    f"unrecognized ESD history key {key!r}; refusing unsafe resume"
                )
            if int(match.group(1)) <= int(epoch):
                retained[key] = archive[key]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **retained)
    temporary.replace(path)


def _prepare_resume_directory(
    output_dir: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> None:
    if not output_dir.exists() or overwrite:
        return
    completion = output_dir / "run_complete.json"
    if completion.is_file():
        required = (
            output_dir / "final_state.pt",
            output_dir / "checkpoint_latest.pt",
            output_dir / "checkpoint_best.pt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "completed MNIST run is missing required artifacts: "
                + ", ".join(missing)
            )
        return

    nontrivial = [
        path for path in output_dir.iterdir() if path.name != "manifest.json"
    ]
    if not nontrivial:
        return
    if not resume:
        raise FileExistsError(
            f"incomplete MNIST run already exists: {output_dir}"
        )

    latest = output_dir / "checkpoint_latest.pt"
    if not latest.is_file():
        raise FileNotFoundError(
            "cannot resume an incomplete MNIST run without "
            f"{latest}; use a new output directory or explicit overwrite"
        )
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    if "epoch" not in payload:
        raise RuntimeError(f"latest checkpoint has no epoch: {latest}")
    checkpoint_epoch = int(payload["epoch"])
    for filename in _PROGRESS_TABLES:
        _truncate_table(output_dir / filename, checkpoint_epoch)
    _truncate_esd_history(output_dir / "esd_history.npz", checkpoint_epoch)

    # These are terminal products. They are regenerated only after training
    # reaches completion and must never survive a rollback to an earlier epoch.
    for filename in (
        "combined_metrics_by_epoch_and_layer.csv",
        "final_state.pt",
        "test_results.json",
    ):
        (output_dir / filename).unlink(missing_ok=True)


def run_baseline(
    config: BaselineConfig,
    *,
    data_dir: str | Path = "./data",
    device: torch.device | None = None,
    output_dir: str | Path | None = None,
    progress: bool = True,
    resume: bool = True,
    overwrite: bool = False,
):
    """Run the public MNIST baseline with strict restart validation."""

    if output_dir is not None:
        _prepare_resume_directory(
            Path(output_dir), resume=resume, overwrite=overwrite
        )
    result = _run_baseline(
        config,
        data_dir=data_dir,
        device=device,
        output_dir=output_dir,
        progress=progress,
        resume=resume,
        overwrite=overwrite,
    )
    if output_dir is not None:
        root = Path(output_dir)
        required = (
            root / "run_complete.json",
            root / "final_state.pt",
            root / "checkpoint_latest.pt",
            root / "checkpoint_best.pt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "MNIST runner returned without required terminal artifacts: "
                + ", ".join(missing)
            )
    return result

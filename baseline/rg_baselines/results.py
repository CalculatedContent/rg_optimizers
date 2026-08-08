"""Result persistence and strict completeness checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import BaselineConfig


@dataclass
class BaselineResult:
    config: BaselineConfig
    performance: pd.DataFrame
    spectral_metrics: pd.DataFrame
    weightwatcher_details: pd.DataFrame
    optimizer_groups: pd.DataFrame
    combined_metrics: pd.DataFrame
    esd_arrays: dict[str, np.ndarray]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.performance.to_csv(out / "performance_by_epoch.csv", index=False)
        self.spectral_metrics.to_csv(
            out / "spectral_metrics_by_epoch_and_layer.csv", index=False
        )
        self.weightwatcher_details.to_csv(
            out / "weightwatcher_details_by_epoch.csv", index=False
        )
        self.optimizer_groups.to_csv(
            out / "optimizer_groups_by_epoch.csv", index=False
        )
        self.combined_metrics.to_csv(
            out / "combined_metrics_by_epoch_and_layer.csv", index=False
        )
        np.savez_compressed(out / "esd_history.npz", **self.esd_arrays)
        (out / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2), encoding="utf-8"
        )
        temporary = out / "final_state.pt.tmp"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
            },
            temporary,
        )
        temporary.replace(out / "final_state.pt")


def validate_result(result: BaselineResult) -> None:
    epochs = set(range(result.config.epochs + 1))
    if set(result.performance.epoch.astype(int)) != epochs:
        raise RuntimeError("performance epochs are incomplete")

    performance = {
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "test_loss",
        "test_accuracy",
        "primary_lr",
        "test_monitoring_only",
    }
    if performance - set(result.performance):
        raise RuntimeError("required train/validation/test metrics are missing")
    numeric = performance - {"test_monitoring_only"}
    if result.performance[list(numeric)].isna().any().any():
        raise RuntimeError("required performance metric contains NaN")
    if not result.performance["test_monitoring_only"].astype(int).eq(1).all():
        raise RuntimeError("the official MNIST test set is not monitoring-only")
    if not result.performance["global_step"].astype(int).is_monotonic_increasing:
        raise RuntimeError("global_step is not monotonic")
    if (result.performance["primary_lr"].astype(float) <= 0).any():
        raise RuntimeError("primary learning rate must remain positive")

    spectral = {
        "alpha",
        "num_traps",
        "detX_num",
        "num_pl_spikes",
        "ERG_gap",
        "m_midpoint",
        "trace_log_midpoint_per_eval",
    }
    if spectral - set(result.spectral_metrics):
        raise RuntimeError("required spectral metrics are missing")
    valid = result.spectral_metrics[
        result.spectral_metrics.status.astype(str).eq("ok")
    ].copy()
    for epoch in epochs:
        layers = set(
            valid.loc[valid.epoch.astype(int).eq(epoch), "layer"].astype(str)
        )
        if not {"fc1", "fc2", "fc3"}.issubset(layers):
            raise RuntimeError(f"epoch {epoch} is missing WeightWatcher layers")
    if valid[list(spectral)].isna().any().any():
        raise RuntimeError("required spectral metric contains NaN")

    traps = valid.num_traps.to_numpy(dtype=float)
    if (traps < 0.0).any() or not np.allclose(traps, np.rint(traps)):
        raise RuntimeError("num_traps must contain non-negative integer counts")
    expected_gap = (
        valid.detX_num.astype(int) - valid.num_pl_spikes.astype(int)
    )
    if not np.array_equal(
        expected_gap.to_numpy(), valid.ERG_gap.astype(int).to_numpy()
    ):
        raise RuntimeError("ERG_gap audit failed")
    midpoint = np.floor(
        (
            valid.detX_num.astype(float)
            + valid.num_pl_spikes.astype(float)
        )
        / 2
    ).astype(int)
    if not np.array_equal(
        midpoint.to_numpy(), valid.m_midpoint.astype(int).to_numpy()
    ):
        raise RuntimeError("midpoint audit failed")

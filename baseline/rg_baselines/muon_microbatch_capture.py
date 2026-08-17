"""Microbatch weight capture and relative-flow spectra for MNIST MLP3 Muon.

The MNIST baseline has no gradient accumulation, so each DataLoader minibatch is
also one optimizer microbatch. This module saves only the three 2-D weight
matrices, keeping the per-step artifact substantially smaller than a full model
and optimizer checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

DEFAULT_MATRIX_NAMES = ("fc1.weight", "fc2.weight", "fc3.weight")
CAPTURE_DIRNAME = "microbatch_checkpoints"
INDEX_FILENAME = "checkpoint_index.csv"
MANIFEST_FILENAME = "manifest.json"

_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _atomic_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _append_index(path: Path, row: Mapping[str, Any]) -> None:
    fields = [
        "global_step",
        "epoch",
        "batch_index",
        "examples_seen",
        "training_loss",
        "primary_lr",
        "auxiliary_lr",
        "checkpoint_path",
    ]
    rows: list[dict[str, Any]] = []
    if path.is_file() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    step = int(row["global_step"])
    rows = [item for item in rows if int(item["global_step"]) != step]
    rows.append({key: row.get(key, "") for key in fields})
    rows.sort(key=lambda item: int(item["global_step"]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _named_matrix_parameters(
    model: torch.nn.Module,
    matrix_names: Iterable[str],
) -> dict[str, torch.nn.Parameter]:
    requested = tuple(str(name) for name in matrix_names)
    available = dict(model.named_parameters())
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"matrix parameters not found: {missing}")
    selected = {name: available[name] for name in requested}
    bad = [name for name, value in selected.items() if value.ndim != 2]
    if bad:
        raise ValueError(f"captured parameters must be matrices: {bad}")
    return selected


def estimated_capture_bytes(
    model: torch.nn.Module,
    *,
    matrix_names: Iterable[str] = DEFAULT_MATRIX_NAMES,
    dtype: str = "float32",
    checkpoint_count: int = 1,
) -> int:
    """Estimate raw tensor bytes, excluding zip/container overhead."""

    if dtype not in _DTYPE_BY_NAME:
        raise ValueError(f"unsupported checkpoint dtype: {dtype}")
    if checkpoint_count < 0:
        raise ValueError("checkpoint_count must be nonnegative")
    matrices = _named_matrix_parameters(model, matrix_names)
    element_size = torch.empty((), dtype=_DTYPE_BY_NAME[dtype]).element_size()
    values = sum(int(value.numel()) for value in matrices.values())
    return int(values * element_size * checkpoint_count)


class MuonMicrobatchCheckpointRecorder:
    """Append-safe recorder for MLP3 weight matrices after optimizer updates."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        model: torch.nn.Module,
        matrix_names: Iterable[str] = DEFAULT_MATRIX_NAMES,
        capture_every: int = 1,
        max_capture_step: int = 0,
        dtype: str = "float32",
    ) -> None:
        if int(capture_every) < 1:
            raise ValueError("capture_every must be positive")
        if int(max_capture_step) < 0:
            raise ValueError("max_capture_step must be nonnegative")
        if dtype not in _DTYPE_BY_NAME:
            raise ValueError(
                f"dtype must be one of {tuple(_DTYPE_BY_NAME)}, got {dtype!r}"
            )
        self.run_dir = Path(run_dir)
        self.model = model
        self.matrix_names = tuple(str(name) for name in matrix_names)
        self.capture_every = int(capture_every)
        self.max_capture_step = int(max_capture_step)
        self.dtype_name = str(dtype)
        self.dtype = _DTYPE_BY_NAME[self.dtype_name]
        self.parameters = _named_matrix_parameters(model, self.matrix_names)
        self.capture_dir = self.run_dir / CAPTURE_DIRNAME
        self.frame_dir = self.capture_dir / "frames"
        self.index_path = self.capture_dir / INDEX_FILENAME
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> None:
        shapes = {
            name: list(parameter.shape)
            for name, parameter in self.parameters.items()
        }
        per_checkpoint = estimated_capture_bytes(
            self.model,
            matrix_names=self.matrix_names,
            dtype=self.dtype_name,
        )
        _atomic_json(
            {
                "schema_version": 1,
                "purpose": "mnist_mlp3_muon_microbatch_weight_capture",
                "microbatch_semantics": (
                    "one MNIST DataLoader minibatch and one optimizer update; "
                    "the baseline does not use gradient accumulation"
                ),
                "matrix_names": list(self.matrix_names),
                "matrix_shapes": shapes,
                "checkpoint_dtype": self.dtype_name,
                "capture_every": self.capture_every,
                "max_capture_step": self.max_capture_step,
                "max_capture_step_semantics": "0 means no capture limit",
                "estimated_raw_bytes_per_checkpoint": per_checkpoint,
            },
            self.capture_dir / MANIFEST_FILENAME,
        )

    def should_capture(self, global_step: int) -> bool:
        step = int(global_step)
        if step < 0:
            raise ValueError("global_step must be nonnegative")
        if self.max_capture_step and step > self.max_capture_step:
            return False
        if step != 0 and step % self.capture_every != 0:
            return False
        return not self.checkpoint_path(step).is_file()

    def checkpoint_path(self, global_step: int) -> Path:
        return self.frame_dir / f"step_{int(global_step):07d}.pt"

    @torch.no_grad()
    def capture(
        self,
        *,
        global_step: int,
        epoch: int,
        batch_index: int,
        examples_seen: int,
        training_loss: float = float("nan"),
        learning_rates: Mapping[str, float] | None = None,
    ) -> Path | None:
        step = int(global_step)
        if not self.should_capture(step):
            return None
        rates = dict(learning_rates or {})
        matrices = {
            name: parameter.detach().to(device="cpu", dtype=self.dtype).clone()
            for name, parameter in self.parameters.items()
        }
        path = self.checkpoint_path(step)
        _atomic_torch_save(
            {
                "schema_version": 1,
                "purpose": "mnist_mlp3_muon_microbatch_weights",
                "global_step": step,
                "epoch": int(epoch),
                "batch_index": int(batch_index),
                "examples_seen": int(examples_seen),
                "training_loss": float(training_loss),
                "learning_rates": {
                    "primary": float(rates.get("primary", float("nan"))),
                    "auxiliary": float(rates.get("auxiliary", float("nan"))),
                },
                "matrix_names": list(self.matrix_names),
                "checkpoint_dtype": self.dtype_name,
                "matrices": matrices,
            },
            path,
        )
        _append_index(
            self.index_path,
            {
                "global_step": step,
                "epoch": int(epoch),
                "batch_index": int(batch_index),
                "examples_seen": int(examples_seen),
                "training_loss": float(training_loss),
                "primary_lr": float(rates.get("primary", float("nan"))),
                "auxiliary_lr": float(rates.get("auxiliary", float("nan"))),
                "checkpoint_path": str(path.relative_to(self.run_dir)),
            },
        )
        return path


def load_microbatch_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("purpose") != "mnist_mlp3_muon_microbatch_weights":
        raise ValueError(f"not an MLP3 Muon microbatch checkpoint: {path}")
    matrices = payload.get("matrices")
    if not isinstance(matrices, dict) or not matrices:
        raise ValueError(f"checkpoint contains no matrices: {path}")
    return payload


def load_microbatch_index(run_dir: str | Path):
    """Load the checkpoint index as a DataFrame without importing pandas early."""

    import pandas as pd

    path = Path(run_dir) / CAPTURE_DIRNAME / INDEX_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"microbatch checkpoint index is empty: {path}")
    root = Path(run_dir)
    frame["checkpoint_path"] = [
        str(value if Path(str(value)).is_absolute() else root / str(value))
        for value in frame["checkpoint_path"]
    ]
    return frame.sort_values("global_step").reset_index(drop=True)


def matrix_esd_eigenvalues(matrix: torch.Tensor) -> np.ndarray:
    """Return positive eigenvalues of W W^T (equivalently sigma(W)^2)."""

    value = matrix.detach().to(device="cpu", dtype=torch.float64)
    if value.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape={tuple(value.shape)}")
    singular_values = torch.linalg.svdvals(value)
    eigenvalues = singular_values.square().numpy()
    eigenvalues = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > 0.0)]
    return np.sort(eigenvalues)


def relative_flow_operator(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    pinv_rtol: float = 1e-6,
) -> tuple[torch.Tensor, str]:
    """Construct the supported square map between successive weight matrices.

    For a wide/full-row-rank matrix W, use J = W_t pinv(W_{t-1}) in output
    space. For a tall matrix, use J = pinv(W_{t-1}) W_t in input space. This
    chooses the smaller min(m, n)-dimensional supported space.
    """

    left = previous.detach().to(device="cpu", dtype=torch.float64)
    right = current.detach().to(device="cpu", dtype=torch.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError(
            "successive matrices must be 2-D with the same shape, got "
            f"{tuple(left.shape)} and {tuple(right.shape)}"
        )
    if not math.isfinite(float(pinv_rtol)) or pinv_rtol <= 0.0:
        raise ValueError("pinv_rtol must be positive and finite")
    pseudo = torch.linalg.pinv(left, rtol=float(pinv_rtol))
    rows, columns = left.shape
    if rows <= columns:
        return right @ pseudo, "output"
    return pseudo @ right, "input"


def relative_flow_esd_eigenvalues(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    pinv_rtol: float = 1e-6,
) -> tuple[np.ndarray, str]:
    operator, side = relative_flow_operator(
        previous, current, pinv_rtol=pinv_rtol
    )
    return matrix_esd_eigenvalues(operator), side


def log_flow_deviation(eigenvalues: np.ndarray, *, zero_tol: float = 1e-12) -> np.ndarray:
    """Return |log(lambda)|, dropping the trivial identity/orthogonal mode."""

    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    deviations = np.abs(np.log(values))
    return np.sort(deviations[deviations > float(zero_tol)])


__all__ = [
    "CAPTURE_DIRNAME",
    "DEFAULT_MATRIX_NAMES",
    "MuonMicrobatchCheckpointRecorder",
    "estimated_capture_bytes",
    "load_microbatch_checkpoint",
    "load_microbatch_index",
    "log_flow_deviation",
    "matrix_esd_eigenvalues",
    "relative_flow_esd_eigenvalues",
    "relative_flow_operator",
]

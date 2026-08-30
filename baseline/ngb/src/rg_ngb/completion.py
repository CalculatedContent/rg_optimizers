"""Validation for completed NGB v4 experiment directories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import torch

_REQUIRED_FILES = (
    "run_complete.json",
    "manifest.json",
    "metrics.csv",
    "epoch_metrics.csv",
    "checkpoint_latest.pt",
    "checkpoint_best.pt",
    "checkpoint_final.pt",
    "test_results.json",
    "spectral/layers.csv",
    "spectral/summary.csv",
)


class CompletedRunValidationError(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise CompletedRunValidationError(
        "completed NGB run is stale or inconsistent: "
        + message
        + ". Use the correct protocol results directory or rerun with explicit overwrite."
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"could not read valid JSON from {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{path} is not a JSON object")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        _fail(f"could not read {path}: {exc}")
    if frame.empty:
        _fail(f"{path} is empty")
    return frame


def _integer(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        _fail(f"{label} is not an integer: {value!r}")


def _expect(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        _fail(f"{label} mismatch; observed {observed!r}, expected {expected!r}")


def _unique_steps(frame: pd.DataFrame, label: str) -> tuple[int, ...]:
    if "step" not in frame.columns:
        _fail(f"{label} has no step column")
    values = pd.to_numeric(frame["step"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values, np.rint(values)):
        _fail(f"{label} contains invalid step values")
    steps = tuple(int(value) for value in np.rint(values))
    if len(steps) != len(set(steps)):
        _fail(f"{label} contains duplicate step rows")
    return steps


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        _fail(f"could not load checkpoint {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"checkpoint {path} is not a mapping")
    return payload


def validate_completed_run(
    run_dir: str | Path,
    *,
    expected_fingerprint: str | None = None,
    expected_optimizer: str | None = None,
    expected_seed: int | None = None,
    expected_total_steps: int | None = None,
    expected_matrix_count: int | None = None,
    verify_checkpoints: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir)
    missing = [str(root / relative) for relative in _REQUIRED_FILES if not (root / relative).is_file() or (root / relative).stat().st_size == 0]
    if missing:
        _fail("missing required artifacts: " + ", ".join(missing))

    completion = _read_json(root / "run_complete.json")
    manifest = _read_json(root / "manifest.json")
    test_results = _read_json(root / "test_results.json")
    if completion.get("completed") is not True:
        _fail("run_complete.json does not declare completed=true")

    recorded_fingerprint = str(completion.get("fingerprint", ""))
    fingerprint = str(expected_fingerprint) if expected_fingerprint is not None else recorded_fingerprint
    recorded_optimizer = str(completion.get("optimizer", ""))
    optimizer = str(expected_optimizer) if expected_optimizer is not None else recorded_optimizer
    recorded_seed = _integer(completion.get("seed"), "completion seed")
    seed = int(expected_seed) if expected_seed is not None else recorded_seed
    recorded_steps = _integer(completion.get("optimizer_steps"), "completion optimizer_steps")
    total_steps = int(expected_total_steps) if expected_total_steps is not None else recorded_steps
    best_step = _integer(completion.get("best_validation_step"), "completion best_validation_step")
    if not fingerprint or not optimizer:
        _fail("completion record lacks fingerprint or optimizer")

    model_cfg = manifest.get("model")
    if not isinstance(model_cfg, dict):
        _fail("manifest model section is missing")
    recorded_matrix_count = 6 * _integer(model_cfg.get("n_layer"), "manifest model.n_layer")
    matrix_count = int(expected_matrix_count) if expected_matrix_count is not None else recorded_matrix_count

    _expect(recorded_fingerprint, fingerprint, "completion fingerprint")
    _expect(recorded_optimizer, optimizer, "completion optimizer")
    _expect(recorded_seed, seed, "completion seed")
    _expect(recorded_steps, total_steps, "completion optimizer_steps")
    _expect(str(manifest.get("protocol_fingerprint", "")), fingerprint, "manifest fingerprint")
    _expect(str(manifest.get("optimizer", "")), optimizer, "manifest optimizer")
    _expect(_integer(manifest.get("seed"), "manifest seed"), seed, "manifest seed")
    _expect(_integer(manifest.get("max_steps"), "manifest max_steps"), total_steps, "manifest max_steps")
    _expect(recorded_matrix_count, matrix_count, "manifest transformer matrix count")

    final_test = test_results.get("final")
    selected_test = test_results.get("validation_selected")
    if not isinstance(final_test, dict) or not isinstance(selected_test, dict):
        _fail("test_results.json lacks final or validation_selected results")
    _expect(_integer(final_test.get("step"), "final test step"), total_steps, "final test step")
    _expect(_integer(selected_test.get("step"), "selected test step"), best_step, "selected test step")

    metrics = _read_csv(root / "metrics.csv")
    epoch_metrics = _read_csv(root / "epoch_metrics.csv")
    layers = _read_csv(root / "spectral" / "layers.csv")
    summary = _read_csv(root / "spectral" / "summary.csv")
    metric_steps = _unique_steps(metrics, "metrics.csv")
    epoch_steps = _unique_steps(epoch_metrics, "epoch_metrics.csv")
    summary_steps = _unique_steps(summary, "spectral/summary.csv")
    for label, steps in (("metrics.csv", metric_steps), ("epoch_metrics.csv", epoch_steps)):
        if 0 not in steps or total_steps not in steps or max(steps) != total_steps:
            _fail(f"{label} does not span step zero through {total_steps}")

    if "test_monitoring_only" not in epoch_metrics.columns:
        _fail("epoch_metrics.csv lacks test_monitoring_only")
    policy = pd.to_numeric(epoch_metrics["test_monitoring_only"], errors="coerce")
    if policy.isna().any() or not policy.astype(int).eq(1).all():
        _fail("epoch_metrics.csv violates the monitoring-only test policy")

    required_layer_columns = {"step", "matrix_name", "alpha", "ERG_gap", "num_traps"}
    missing_columns = required_layer_columns.difference(layers.columns)
    if missing_columns:
        _fail("spectral/layers.csv is missing columns " + ", ".join(sorted(missing_columns)))
    if layers.duplicated(["step", "matrix_name"]).any():
        _fail("spectral/layers.csv has duplicate step/matrix rows")
    layer_step_frame = layers[["step"]].drop_duplicates().sort_values("step")
    layer_steps = _unique_steps(layer_step_frame, "spectral/layers.csv")
    if set(summary_steps) != set(epoch_steps) or set(layer_steps) != set(epoch_steps):
        _fail("spectral steps do not match epoch_metrics.csv")
    if not layers.groupby("step")["matrix_name"].nunique().eq(matrix_count).all():
        _fail(f"spectral/layers.csv does not contain {matrix_count} matrices per epoch")
    if "n_matrices" not in summary.columns:
        _fail("spectral/summary.csv lacks n_matrices")
    counts = pd.to_numeric(summary["n_matrices"], errors="coerce")
    if counts.isna().any() or not counts.astype(int).eq(matrix_count).all():
        _fail(f"spectral/summary.csv does not report {matrix_count} matrices per epoch")

    if "checkpoint_path" not in epoch_metrics.columns:
        _fail("epoch_metrics.csv lacks checkpoint_path")
    resolved_paths: list[Path] = []
    for value in epoch_metrics["checkpoint_path"]:
        recorded = Path(str(value))
        candidate = recorded if recorded.is_file() else root / "epoch_checkpoints" / recorded.name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            _fail(f"missing epoch checkpoint: {recorded}")
        resolved_paths.append(candidate.resolve())
    if len(resolved_paths) != len(set(resolved_paths)) or len(resolved_paths) != len(set(epoch_steps)):
        _fail("epoch checkpoint inventory is duplicated or incomplete")

    if verify_checkpoints:
        try:
            best_loss = float(completion.get("best_validation_loss"))
        except (TypeError, ValueError):
            _fail("run_complete.json has invalid best_validation_loss")
        for filename, expected_step in (("checkpoint_latest.pt", total_steps), ("checkpoint_final.pt", total_steps), ("checkpoint_best.pt", best_step)):
            payload = _load_checkpoint(root / filename)
            _expect(str(payload.get("fingerprint", "")), fingerprint, f"{filename} fingerprint")
            _expect(str(payload.get("optimizer_name", "")), optimizer, f"{filename} optimizer")
            _expect(_integer(payload.get("seed"), f"{filename} seed"), seed, f"{filename} seed")
            _expect(_integer(payload.get("step"), f"{filename} step"), expected_step, f"{filename} step")
            _expect(_integer(payload.get("best_validation_step"), f"{filename} best step"), best_step, f"{filename} best step")
            try:
                stored_loss = float(payload.get("best_validation_loss"))
            except (TypeError, ValueError):
                _fail(f"{filename} has invalid best_validation_loss")
            if not math.isclose(stored_loss, best_loss, rel_tol=1e-12, abs_tol=1e-12):
                _fail(f"{filename} best_validation_loss does not match completion")

    return completion

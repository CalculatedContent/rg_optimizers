"""Validation for completed one-head nanoGPT experiment directories."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import torch

from .checkpoints import (
    model_state_sha256,
    optimizer_state_sha256,
    require_finite_checkpoint_state,
)

_REQUIRED_FILES = (
    "run_complete.json",
    "manifest.json",
    "metrics.csv",
    "epoch_metrics.csv",
    "checkpoint_initial.pt",
    "checkpoint_latest.pt",
    "checkpoint_best.pt",
    "checkpoint_final.pt",
    "test_results.json",
    "spectral/layers.csv",
    "spectral/summary.csv",
)
_TEST_METRIC_COMPLETION_KEYS = {
    "loss": "final_test_loss",
    "perplexity": "final_test_perplexity",
    "bits_per_token": "final_test_bits_per_token",
    "accuracy": "final_test_accuracy",
    "top5_accuracy": "final_test_top5_accuracy",
    "bleu": "final_test_bleu",
    "continuation_token_accuracy": (
        "final_test_continuation_token_accuracy"
    ),
    "continuation_exact_match": "final_test_continuation_exact_match",
}
_HELD_OUT_CURVE_COLUMNS = (
    "test_loss",
    "test_perplexity",
    "test_bits_per_token",
    "test_accuracy",
    "test_top5_accuracy",
    "test_bleu",
    "test_continuation_token_accuracy",
    "test_continuation_exact_match",
    "test_generalization_gap",
)


class CompletedRunValidationError(RuntimeError):
    """A nominally completed run is missing, stale, or inconsistent."""


def _fail(message: str) -> NoReturn:
    raise CompletedRunValidationError(
        "completed one-head nanoGPT run is stale or inconsistent: "
        + message
        + ". Use a new results directory or rerun with explicit overwrite."
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"could not read valid JSON from {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{path} does not contain a JSON object")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        _fail(f"could not read {path}: {exc}")
    if frame.empty:
        _fail(f" {path} is empty")
    return frame


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        _fail(f"{label} is not an integer: {value!r}")


def _as_finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        _fail(f"{label} is not numeric: {value!r}")
    if not math.isfinite(result):
        _fail(f"{label} is non-finite: {value!r}")
    return result


def _expect(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        _fail(
            f"{label} mismatch; observed {observed!r}, expected {expected!r}"
        )


def _step_tuple(frame: pd.DataFrame, label: str) -> tuple[int, ...]:
    if "step" not in frame.columns:
        _fail(f"{label} has no step column")
    values = pd.to_numeric(frame["step"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        _fail(f"{label} contains non-finite step values")
    rounded = np.rint(values)
    if not np.allclose(values, rounded):
        _fail(f"{label} contains non-integer step values")
    steps = tuple(int(value) for value in rounded)
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


def _validate_checkpoint_identity(
    payload: dict[str, Any],
    *,
    path: Path,
    fingerprint: str,
    optimizer: str,
    seed: int,
    step: int,
    schema_version: int,
) -> None:
    _expect(
        _as_int(payload.get("schema_version"), f"{path.name} schema_version"),
        schema_version,
        f"{path.name} schema_version",
    )
    _expect(
        str(payload.get("fingerprint", "")),
        fingerprint,
        f"{path.name} fingerprint",
    )
    _expect(
        str(payload.get("optimizer_name", "")),
        optimizer,
        f"{path.name} optimizer",
    )
    _expect(
        _as_int(payload.get("seed"), f"{path.name} seed"),
        seed,
        f"{path.name} seed",
    )
    _expect(
        _as_int(payload.get("step"), f"{path.name} step"),
        step,
        f"{path.name} step",
    )
    model_state = payload.get("model")
    if not isinstance(model_state, dict) or not model_state:
        _fail(f"{path.name} has no non-empty model state")


def _validate_model_tensor_inventory(
    state: dict[str, torch.Tensor],
    *,
    path: Path,
    expected: dict[str, tuple[tuple[int, ...], str]] | None = None,
) -> dict[str, tuple[tuple[int, ...], str]]:
    inventory: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            _fail(f"{path.name} model entry is not a tensor: {name}")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            _fail(f"{path.name} contains a non-finite model tensor: {name}")
        inventory[str(name)] = (tuple(value.shape), str(value.dtype))
    if expected is not None and inventory != expected:
        _fail(
            f"{path.name} model key/shape/dtype inventory differs from "
            "checkpoint_initial.pt"
        )
    return inventory


def _validate_muonclip_qk(
    root: Path,
    *,
    manifest: dict[str, Any],
    total_steps: int,
) -> None:
    path = root / "muonclip_qk.csv"
    if not path.is_file() or path.stat().st_size == 0:
        _fail(f"MuonClip run lacks required QK diagnostics: {path}")
    frame = _read_csv(path)
    required = (
        "step",
        "threshold",
        "steps_in_interval",
        "head_observations",
        "active_heads",
        "active_fraction",
        "mean_max_logit",
        "max_logit",
        "mean_gamma",
        "min_gamma",
    )
    missing = set(required).difference(frame.columns)
    if missing:
        _fail(
            "MuonClip QK diagnostics lack columns: "
            + ", ".join(sorted(missing))
        )
    numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        _fail("MuonClip QK diagnostics contain non-finite values")

    profile = manifest.get("optimizer_profile")
    if not isinstance(profile, dict):
        _fail("manifest optimizer_profile is not a mapping")
    diagnostic_interval = _as_int(
        profile.get("qk_diagnostics_interval"),
        "manifest MuonClip QK diagnostic interval",
    )
    if diagnostic_interval <= 0:
        _fail("manifest has a non-positive MuonClip QK diagnostic interval")
    expected_steps = list(
        range(diagnostic_interval, total_steps + 1, diagnostic_interval)
    )
    if not expected_steps or expected_steps[-1] != total_steps:
        expected_steps.append(total_steps)
    observed_steps = numeric["step"].to_numpy(dtype=float)
    if not np.allclose(observed_steps, np.rint(observed_steps)):
        _fail("MuonClip QK diagnostic steps are not integers")
    if tuple(int(value) for value in observed_steps) != tuple(expected_steps):
        _fail("MuonClip QK diagnostics do not cover the exact interval grid")

    expected_intervals = np.diff(np.asarray([0, *expected_steps], dtype=int))
    observed_intervals = numeric["steps_in_interval"].to_numpy(dtype=float)
    if not np.array_equal(observed_intervals, expected_intervals.astype(float)):
        _fail("MuonClip QK interval lengths do not match their recorded steps")
    if int(observed_intervals.sum()) != total_steps:
        _fail("MuonClip QK intervals do not cover the full training horizon")

    model = manifest.get("model")
    if not isinstance(model, dict):
        _fail("manifest model configuration is not a mapping")
    heads_per_step = _as_int(model.get("n_layer"), "manifest n_layer") * _as_int(
        model.get("n_head"), "manifest n_head"
    )
    if heads_per_step <= 0:
        _fail("manifest has a non-positive layer/head inventory")
    expected_observations = observed_intervals * heads_per_step
    observations = numeric["head_observations"].to_numpy(dtype=float)
    active_heads = numeric["active_heads"].to_numpy(dtype=float)
    active_fraction = numeric["active_fraction"].to_numpy(dtype=float)
    if not np.array_equal(observations, expected_observations):
        _fail("MuonClip QK head observations do not match one observation/head/step")
    if (
        (active_heads < 0.0).any()
        or (active_heads > observations).any()
        or not np.allclose(
            active_fraction,
            active_heads / observations,
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        _fail("MuonClip QK active-head counts/fractions are inconsistent")

    threshold = _as_finite_float(
        profile.get("qk_clip_threshold"),
        "manifest MuonClip QK threshold",
    )
    if not numeric["threshold"].eq(threshold).all():
        _fail("MuonClip QK thresholds differ from the optimizer profile")
    if (
        not numeric["active_fraction"].between(0.0, 1.0).all()
        or not numeric["mean_gamma"].between(0.0, 1.0).all()
        or not numeric["min_gamma"].between(0.0, 1.0).all()
        or (numeric["min_gamma"] > numeric["mean_gamma"]).any()
        or (numeric["mean_max_logit"] > numeric["max_logit"]).any()
    ):
        _fail("MuonClip QK diagnostics violate registered bounds")


def validate_completed_run(
    run_dir: str | Path,
    *,
    expected_fingerprint: str | None = None,
    expected_optimizer: str | None = None,
    expected_seed: int | None = None,
    expected_total_steps: int | None = None,
    verify_checkpoints: bool = True,
) -> dict[str, Any]:
    """Validate a completed run before it is skipped or analyzed.

    Expected values supplied by the current configuration make this a stale-run
    guard. Without them, the function still verifies internal consistency.
    """

    root = Path(run_dir)
    missing = [
        str(root / relative)
        for relative in _REQUIRED_FILES
        if not (root / relative).is_file()
        or (root / relative).stat().st_size == 0
    ]
    if missing:
        _fail("missing required artifacts: " + ", ".join(missing))

    completion = _read_json(root / "run_complete.json")
    manifest = _read_json(root / "manifest.json")
    test_results = _read_json(root / "test_results.json")
    if completion.get("completed") is not True:
        _fail("run_complete.json does not declare completed=true")

    recorded_fingerprint = str(completion.get("fingerprint", ""))
    fingerprint = (
        str(expected_fingerprint)
        if expected_fingerprint is not None
        else recorded_fingerprint
    )
    if not fingerprint or not recorded_fingerprint:
        _fail("the completion record has no protocol fingerprint")

    recorded_optimizer = str(completion.get("optimizer", ""))
    optimizer = (
        str(expected_optimizer)
        if expected_optimizer is not None
        else recorded_optimizer
    )
    if not optimizer or not recorded_optimizer:
        _fail("the completion record has no optimizer")

    recorded_seed = _as_int(completion.get("seed"), "completion seed")
    seed = int(expected_seed) if expected_seed is not None else recorded_seed
    recorded_steps = _as_int(
        completion.get("optimizer_steps"), "completion optimizer_steps"
    )
    total_steps = (
        int(expected_total_steps)
        if expected_total_steps is not None
        else recorded_steps
    )
    best_step = _as_int(
        completion.get("best_validation_step"),
        "completion best_validation_step",
    )

    _expect(recorded_fingerprint, fingerprint, "completion fingerprint")
    _expect(recorded_optimizer, optimizer, "completion optimizer")
    _expect(recorded_seed, seed, "completion seed")
    _expect(recorded_steps, total_steps, "completion optimizer_steps")
    _expect(
        str(manifest.get("protocol_fingerprint", "")),
        fingerprint,
        "manifest fingerprint",
    )
    _expect(str(manifest.get("optimizer", "")), optimizer, "manifest optimizer")
    _expect(_as_int(manifest.get("seed"), "manifest seed"), seed, "manifest seed")
    _expect(
        _as_int(manifest.get("max_steps"), "manifest max_steps"),
        total_steps,
        "manifest max_steps",
    )
    initial_model_hash = str(manifest.get("initial_model_sha256", ""))
    if (
        len(initial_model_hash) != 64
        or any(character not in "0123456789abcdef" for character in initial_model_hash)
    ):
        _fail("manifest has no valid initial-model tensor hash")

    final_test = test_results.get("final")
    selected_test = test_results.get("validation_selected")
    test_policy = str(test_results.get("policy", "")).lower()
    if (
        "held out" not in test_policy
        or "validation" not in test_policy
        or "never" not in test_policy
    ):
        _fail("test_results.json does not declare the held-out test policy")
    if not isinstance(final_test, dict) or not isinstance(selected_test, dict):
        _fail("test_results.json lacks final or validation_selected results")
    parsed_test_metrics: dict[str, dict[str, float]] = {}
    for label, values in (
        ("final", final_test),
        ("validation_selected", selected_test),
    ):
        missing_metrics = set(_TEST_METRIC_COMPLETION_KEYS).difference(values)
        if missing_metrics:
            _fail(
                f"test_results.json {label} lacks metrics: "
                + ", ".join(sorted(missing_metrics))
            )
        parsed = {
            metric: _as_finite_float(
                values[metric], f"{label} test {metric}"
            )
            for metric in _TEST_METRIC_COMPLETION_KEYS
        }
        for metric in (
            "accuracy",
            "top5_accuracy",
            "continuation_token_accuracy",
            "continuation_exact_match",
        ):
            if not 0.0 <= parsed[metric] <= 1.0:
                _fail(f"{label} test {metric} is outside [0, 1]")
        if parsed["top5_accuracy"] < parsed["accuracy"]:
            _fail(f"{label} test top5_accuracy is below top-1 accuracy")
        if (
            parsed["continuation_exact_match"]
            > parsed["continuation_token_accuracy"] + 1e-12
        ):
            _fail(
                f"{label} continuation exact-match exceeds token accuracy"
            )
        if parsed["loss"] < 0.0 or parsed["bits_per_token"] < 0.0:
            _fail(f"{label} test loss/bits_per_token must be nonnegative")
        if parsed["perplexity"] <= 0.0:
            _fail(f"{label} test perplexity must be positive")
        if not 0.0 <= parsed["bleu"] <= 100.0:
            _fail(f"{label} test BLEU is outside [0, 100]")
        if not math.isclose(
            math.log(parsed["perplexity"]),
            parsed["loss"],
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            _fail(f"{label} test perplexity is inconsistent with loss")
        if not math.isclose(
            parsed["bits_per_token"] * math.log(2.0),
            parsed["loss"],
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            _fail(f"{label} test bits_per_token is inconsistent with loss")
        parsed_test_metrics[label] = parsed

    for metric, completion_key in _TEST_METRIC_COMPLETION_KEYS.items():
        completion_value = _as_finite_float(
            completion.get(completion_key), f"completion {completion_key}"
        )
        if not math.isclose(
            completion_value,
            parsed_test_metrics["final"][metric],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            _fail(
                f"completion {completion_key} does not match final test {metric}"
            )
    _expect(
        _as_int(final_test.get("step"), "final test step"),
        total_steps,
        "final test step",
    )
    _expect(
        _as_int(selected_test.get("step"), "selected test step"),
        best_step,
        "selected test step",
    )

    metrics = _read_csv(root / "metrics.csv")
    epoch_metrics = _read_csv(root / "epoch_metrics.csv")
    layers = _read_csv(root / "spectral" / "layers.csv")
    summary = _read_csv(root / "spectral" / "summary.csv")
    metric_steps = _step_tuple(metrics, "metrics.csv")
    epoch_steps = _step_tuple(epoch_metrics, "epoch_metrics.csv")
    summary_steps = _step_tuple(summary, "spectral/summary.csv")
    weightwatcher = manifest.get("weightwatcher", {})
    if not isinstance(weightwatcher, dict):
        _fail("manifest weightwatcher configuration is not a mapping")
    clip_xmax = weightwatcher.get("fix_fingers") == "clip_xmax"

    for label, steps in (
        ("metrics.csv", metric_steps),
        ("epoch_metrics.csv", epoch_steps),
    ):
        if 0 not in steps or total_steps not in steps or max(steps) != total_steps:
            _fail(f"{label} does not span step zero through {total_steps}")

    for label, frame in (
        ("metrics.csv", metrics),
        ("epoch_metrics.csv", epoch_metrics),
    ):
        missing_held_out = set(_HELD_OUT_CURVE_COLUMNS).difference(
            frame.columns
        )
        if missing_held_out:
            _fail(
                f"{label} is missing held-out placeholder columns "
                + ", ".join(sorted(missing_held_out))
            )
        leaked = [
            column
            for column in _HELD_OUT_CURVE_COLUMNS
            if not frame[column].isna().all()
        ]
        if leaked:
            _fail(
                f"{label} leaks held-out test outcomes into training curves: "
                + ", ".join(leaked)
            )

    if "test_monitoring_only" not in epoch_metrics.columns:
        _fail("epoch_metrics.csv has no legacy test policy flag")
    policy = pd.to_numeric(epoch_metrics["test_monitoring_only"], errors="coerce")
    if policy.isna().any() or not policy.astype(int).eq(1).all():
        _fail("epoch_metrics.csv violates the held-out test policy")
    if "test_held_out" not in epoch_metrics.columns:
        _fail("epoch_metrics.csv has no test_held_out column")
    held_out = pd.to_numeric(epoch_metrics["test_held_out"], errors="coerce")
    if held_out.isna().any() or not held_out.astype(int).eq(1).all():
        _fail("epoch_metrics.csv does not mark every test curve as held out")

    required_layer_columns = {
        "step",
        "matrix_name",
        "alpha",
        "ERG_gap",
        "num_traps",
        "run_seed",
        "diagnostic_seed",
        "protocol_fingerprint",
        "model_state_sha256",
    }
    if clip_xmax:
        required_layer_columns.update(
            {
                "raw_alpha",
                "alpha_raw",
                "alpha_clip_xmax",
                "alpha_delta",
                "num_fingers",
                "finger_policy",
                "primary_alpha_variant",
                "weightwatcher_analysis_calls",
            }
        )
    missing_columns = required_layer_columns.difference(layers.columns)
    if missing_columns:
        _fail(
            "spectral/layers.csv is missing columns "
            + ", ".join(sorted(missing_columns))
        )
    if layers.duplicated(["step", "matrix_name"]).any():
        _fail("spectral/layers.csv has duplicate step/matrix rows")
    layer_steps = _step_tuple(
        layers[["step"]].drop_duplicates().sort_values("step"),
        "spectral/layers.csv",
    )
    if set(summary_steps) != set(epoch_steps) or set(layer_steps) != set(epoch_steps):
        _fail("spectral steps do not match epoch_metrics.csv")
    if not layers.groupby("step")["matrix_name"].nunique().eq(6).all():
        _fail("spectral/layers.csv does not contain six matrices per epoch")
    if not layers["protocol_fingerprint"].astype(str).eq(fingerprint).all():
        _fail("spectral/layers.csv fingerprint does not match the run")
    layer_seeds = pd.to_numeric(layers["run_seed"], errors="coerce")
    if layer_seeds.isna().any() or not layer_seeds.astype(int).eq(seed).all():
        _fail("spectral/layers.csv seed does not match the run")
    layer_diagnostic_seeds = pd.to_numeric(
        layers["diagnostic_seed"], errors="coerce"
    )
    expected_diagnostic_seeds = (
        pd.to_numeric(layers["step"], errors="coerce").astype(int)
        + int(seed)
        + 1_000_003
    )
    if (
        layer_diagnostic_seeds.isna().any()
        or not layer_diagnostic_seeds.astype(int).equals(
            expected_diagnostic_seeds
        )
    ):
        _fail("spectral/layers.csv diagnostic seed binding is invalid")
    spectral_model_hash_by_step: dict[int, str] = {}
    for step_value, group in layers.groupby("step"):
        hashes = group["model_state_sha256"].astype(str).unique().tolist()
        if (
            len(hashes) != 1
            or len(hashes[0]) != 64
            or any(character not in "0123456789abcdef" for character in hashes[0].lower())
        ):
            _fail("spectral/layers.csv has an invalid model-state hash")
        spectral_model_hash_by_step[int(step_value)] = hashes[0]

        raw_path = (
            root
            / "spectral"
            / "raw"
            / f"weightwatcher_step_{int(step_value):07d}.csv"
        )
        status_path = (
            root / "spectral" / f"status_step_{int(step_value):07d}.json"
        )
        if not raw_path.is_file() or not status_path.is_file():
            _fail(
                f"spectral step {int(step_value)} lacks raw CSV/integrity status"
            )
        status = _read_json(status_path)
        expected_diagnostic_seed = int(seed) + 1_000_003 + int(step_value)
        if (
            status.get("completed") is not True
            or str(status.get("raw_csv_sha256", "")) != _file_sha256(raw_path)
            or str(status.get("protocol_fingerprint", "")) != fingerprint
            or str(status.get("model_state_sha256", "")) != hashes[0]
            or _as_int(status.get("run_seed"), "spectral status seed") != seed
            or _as_int(
                status.get("diagnostic_seed"),
                "spectral status diagnostic seed",
            )
            != expected_diagnostic_seed
        ):
            _fail(f"spectral step {int(step_value)} integrity status is invalid")
        raw = _read_csv(raw_path)
        if (
            "matrix_name" not in raw.columns
            or len(raw) != 6
            or raw["matrix_name"].astype(str).nunique() != 6
        ):
            _fail(f"spectral raw step {int(step_value)} lacks six matrices")
        for identity, expected in (
            ("protocol_fingerprint", fingerprint),
            ("model_state_sha256", hashes[0]),
            ("run_seed", seed),
            ("diagnostic_seed", expected_diagnostic_seed),
        ):
            if identity not in raw.columns or not raw[identity].astype(str).eq(
                str(expected)
            ).all():
                _fail(
                    f"spectral raw step {int(step_value)} has invalid {identity}"
                )
        raw_sorted = raw.sort_values("matrix_name").reset_index(drop=True)
        layer_sorted = group.sort_values("matrix_name").reset_index(drop=True)
        if not raw_sorted["matrix_name"].astype(str).equals(
            layer_sorted["matrix_name"].astype(str)
        ):
            _fail(
                f"spectral raw/layer matrix identity differs at step {int(step_value)}"
            )
        for metric in ("alpha", "ERG_gap", "num_traps", "rand_distance"):
            if metric not in raw_sorted.columns or metric not in layer_sorted.columns:
                _fail(f"spectral raw/layer data lacks {metric}")
            raw_values = pd.to_numeric(raw_sorted[metric], errors="coerce")
            layer_values = pd.to_numeric(layer_sorted[metric], errors="coerce")
            if not np.allclose(
                raw_values.to_numpy(dtype=float),
                layer_values.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ):
                _fail(
                    f"spectral raw/layer {metric} differs at step {int(step_value)}"
                )
    if clip_xmax:
        numeric_columns = (
            "raw_alpha",
            "alpha_raw",
            "alpha_clip_xmax",
            "alpha_delta",
            "num_fingers",
            "weightwatcher_analysis_calls",
        )
        numeric = layers[list(numeric_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            _fail("clip-Xmax/raw-alpha spectral values are non-finite")
        alpha = pd.to_numeric(layers["alpha"], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.allclose(
            alpha,
            numeric["alpha_clip_xmax"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            _fail("alpha and alpha_clip_xmax aliases disagree")
        if not np.allclose(
            numeric["raw_alpha"].to_numpy(dtype=float),
            numeric["alpha_raw"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            _fail("raw_alpha and alpha_raw aliases disagree")
        if not np.allclose(
            numeric["alpha_delta"].to_numpy(dtype=float),
            numeric["alpha_raw"].to_numpy(dtype=float)
            - numeric["alpha_clip_xmax"].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        ):
            _fail("alpha_delta does not equal raw minus clip-Xmax alpha")
        if not numeric["weightwatcher_analysis_calls"].eq(1).all():
            _fail("WeightWatcher was not called exactly once per checkpoint")
        if not layers["finger_policy"].astype(str).eq(
            "fix_fingers=clip_xmax"
        ).all():
            _fail("spectral rows do not declare fix_fingers=clip_xmax")
        if not layers["primary_alpha_variant"].astype(str).eq(
            "clip_xmax"
        ).all():
            _fail("spectral rows do not declare clipped alpha as primary")
        if len(epoch_steps) < 10:
            _fail("clip-Xmax campaign has fewer than ten permanent states")
        for column in (
            "alpha_raw_n",
            "alpha_raw_median",
            "alpha_clip_xmax_n",
            "alpha_clip_xmax_median",
        ):
            if column not in summary.columns:
                _fail(f"spectral/summary.csv has no {column} column")
        status_paths = [
            root / "spectral" / f"status_step_{step:07d}.json"
            for step in epoch_steps
        ]
        missing_status = [str(path) for path in status_paths if not path.is_file()]
        if missing_status:
            _fail(
                "missing WeightWatcher completion records: "
                + ", ".join(missing_status)
            )
        for path in status_paths:
            status = _read_json(path)
            if status.get("completed") is not True:
                _fail(f"WeightWatcher status is incomplete: {path}")
            _expect(
                _as_int(
                    status.get("weightwatcher_analysis_calls"),
                    f"{path.name} analysis call count",
                ),
                1,
                f"{path.name} analysis call count",
            )
    if "n_matrices" not in summary.columns:
        _fail("spectral/summary.csv has no n_matrices column")
    matrix_counts = pd.to_numeric(summary["n_matrices"], errors="coerce")
    if matrix_counts.isna().any() or not matrix_counts.astype(int).eq(6).all():
        _fail("spectral/summary.csv does not report six matrices per epoch")

    if "checkpoint_path" not in epoch_metrics.columns:
        _fail("epoch_metrics.csv has no checkpoint_path column")
    recorded_checkpoint_paths = [
        Path(str(value))
        for value in epoch_metrics["checkpoint_path"]
    ]
    if len(recorded_checkpoint_paths) != len(set(epoch_steps)):
        _fail(
            "epoch checkpoint inventory does not match "
            "epoch_metrics.csv"
        )
    resolved_checkpoint_paths: list[Path] = []
    for recorded in recorded_checkpoint_paths:
        candidate = recorded
        if not candidate.is_file():
            candidate = root / "epoch_checkpoints" / recorded.name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            _fail(
                f"missing epoch checkpoint recorded by "
                f"epoch_metrics.csv: {recorded}"
            )
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            _fail(
                "epoch_metrics.csv references a checkpoint outside the run "
                f"directory: {recorded}"
            )
        resolved_checkpoint_paths.append(resolved)
    if len(resolved_checkpoint_paths) != len(
        set(resolved_checkpoint_paths)
    ):
        _fail(
            "epoch_metrics.csv references duplicate epoch "
            "checkpoints"
        )

    if optimizer == "muon_clip":
        _validate_muonclip_qk(
            root,
            manifest=manifest,
            total_steps=total_steps,
        )

    if verify_checkpoints:
        try:
            best_loss = float(completion.get("best_validation_loss"))
        except (TypeError, ValueError):
            _fail("run_complete.json has invalid best_validation_loss")
        initial_inventory: dict[str, tuple[tuple[int, ...], str]] | None = None
        for filename, expected_step in (
            ("checkpoint_initial.pt", 0),
            ("checkpoint_latest.pt", total_steps),
            ("checkpoint_final.pt", total_steps),
            ("checkpoint_best.pt", best_step),
        ):
            payload = _load_checkpoint(root / filename)
            _validate_checkpoint_identity(
                payload,
                path=root / filename,
                fingerprint=fingerprint,
                optimizer=optimizer,
                seed=seed,
                step=expected_step,
                schema_version=5,
            )
            computed_model_hash = model_state_sha256(payload["model"])
            if str(payload.get("model_state_sha256", "")) != computed_model_hash:
                _fail(f"{filename} model-state SHA-256 does not match")
            optimizer_states = payload.get("optimizers")
            expected_optimizer_count = (
                1 if optimizer in {"adam", "adamw"} else 2
            )
            if (
                not isinstance(optimizer_states, list)
                or len(optimizer_states) != expected_optimizer_count
                or not all(isinstance(state, dict) for state in optimizer_states)
            ):
                _fail(f"{filename} optimizer-state inventory is invalid")
            try:
                computed_optimizer_hash = optimizer_state_sha256(
                    optimizer_states
                )
            except (TypeError, ValueError) as exc:
                _fail(f"{filename} optimizer-state hashing failed: {exc}")
            if str(payload.get("optimizer_state_sha256", "")) != (
                computed_optimizer_hash
            ):
                _fail(f"{filename} optimizer-state SHA-256 does not match")
            try:
                require_finite_checkpoint_state(
                    model_state=payload["model"],
                    optimizer_states=optimizer_states,
                    step=expected_step,
                )
            except FloatingPointError as exc:
                _fail(f"{filename} contains non-finite checkpoint state: {exc}")
            observed_inventory = _validate_model_tensor_inventory(
                payload["model"],
                path=root / filename,
                expected=initial_inventory,
            )
            if filename == "checkpoint_initial.pt":
                initial_inventory = observed_inventory
                if model_state_sha256(payload["model"]) != initial_model_hash:
                    _fail(
                        "checkpoint_initial.pt model tensors do not match the "
                        "manifest initial-model hash"
                    )
                continue
            _expect(
                _as_int(
                    payload.get("best_validation_step"),
                    f"{filename} best_validation_step",
                ),
                best_step,
                f"{filename} best_validation_step",
            )
            try:
                stored_loss = float(payload.get("best_validation_loss"))
            except (TypeError, ValueError):
                _fail(f"{filename} has invalid best_validation_loss")
            if not math.isclose(
                stored_loss, best_loss, rel_tol=1e-12, abs_tol=1e-12
            ):
                _fail(f"{filename} best_validation_loss does not match completion")

        required_epoch_columns = {"step", "epoch", "nominal_epoch"}
        missing_epoch_columns = required_epoch_columns.difference(
            epoch_metrics.columns
        )
        if missing_epoch_columns:
            _fail(
                "epoch_metrics.csv lacks checkpoint identity columns: "
                + ", ".join(sorted(missing_epoch_columns))
            )
        for (_, row), path in zip(
            epoch_metrics.iterrows(),
            resolved_checkpoint_paths,
            strict=True,
        ):
            expected_step = _as_int(row["step"], "epoch checkpoint step")
            nominal_epoch = _as_finite_float(
                row["nominal_epoch"], "epoch checkpoint nominal_epoch"
            )
            actual_epoch = _as_finite_float(
                row["epoch"], "epoch checkpoint actual_epoch"
            )
            payload = _load_checkpoint(path)
            _validate_checkpoint_identity(
                payload,
                path=path,
                fingerprint=fingerprint,
                optimizer=optimizer,
                seed=seed,
                step=expected_step,
                schema_version=3,
            )
            if str(payload.get("model_state_sha256", "")) != (
                model_state_sha256(payload["model"])
            ):
                _fail(f"{path.name} model-state SHA-256 does not match")
            if str(payload.get("model_state_sha256", "")) != (
                spectral_model_hash_by_step.get(expected_step)
            ):
                _fail(
                    f"{path.name} model tensors do not match WeightWatcher state"
                )
            if initial_inventory is None:  # pragma: no cover - initial loads first
                _fail("initial model tensor inventory is unavailable")
            _validate_model_tensor_inventory(
                payload["model"],
                path=path,
                expected=initial_inventory,
            )
            if payload.get("purpose") != (
                "per_epoch_model_only_analysis_checkpoint"
            ):
                _fail(f"{path.name} has the wrong checkpoint purpose")
            stored_nominal = _as_finite_float(
                payload.get("nominal_epoch"), f"{path.name} nominal_epoch"
            )
            stored_actual = _as_finite_float(
                payload.get("actual_epoch"), f"{path.name} actual_epoch"
            )
            if not math.isclose(
                stored_nominal,
                nominal_epoch,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                _fail(f"{path.name} nominal_epoch does not match epoch_metrics.csv")
            if not math.isclose(
                stored_actual,
                actual_epoch,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                _fail(f"{path.name} actual_epoch does not match epoch_metrics.csv")

    return completion

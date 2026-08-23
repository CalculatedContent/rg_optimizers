#!/usr/bin/env python3
"""Build the audited report for the dated one-head nanoGPT campaign.

The report builder is intentionally stricter than a generic plotting script.
By default it requires exactly the preregistered 2 x 5 campaign:

    optimizers: adamw, muon_clip
    seeds:      1337, 2027, 4099, 31415, 271828

It validates completion and matched campaign invariants, preserves the seeded
run as the unit of replication, computes paired seed differences, aggregates
WeightWatcher alphas only after taking the six-matrix median within each run,
and uses validation loss alone for the saturation diagnostic.

All generated artifacts are written below an explicitly supplied /tmp output
directory.  The source results are read-only inputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# Importing Matplotlib can create a configuration directory immediately. Keep
# even ``--help`` read-only, then bind its cache to the validated /tmp report
# tree before the first plotting import.
plt: Any | None = None


def _initialize_matplotlib(output_root: Path) -> None:
    global plt
    if plt is not None:
        return
    cache = output_root.parent / "cache" / "matplotlib-report"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    plt = pyplot


OPTIMIZERS: tuple[str, ...] = ("adamw", "muon_clip")
SEEDS: tuple[int, ...] = (1337, 2027, 4099, 31415, 271828)
MATRIX_TYPES: tuple[str, ...] = (
    "W_Q",
    "W_K",
    "W_V",
    "W_O",
    "W_MLP_IN",
    "W_MLP_OUT",
)
OPTIMIZER_LABELS: Mapping[str, str] = {
    "adamw": "AdamW",
    "muon_clip": "MuonClip + auxiliary AdamW",
}
OPTIMIZER_COLORS: Mapping[str, str] = {
    "adamw": "#D55E00",
    "muon_clip": "#CC79A7",
}
SPLIT_COLORS: Mapping[str, str] = {
    "train": "#0072B2",
    "val": "#E69F00",
    "test": "#009E73",
    "other": "#6B7280",
}
T_975_DF1 = 12.7062047364
T_975_DF2 = 4.3026527297
T_975_DF3 = 3.1824463053
T_975_DF4 = 2.7764451052
SATURATION_DELTA_NATS = 0.01
MIN_PERMANENT_CHECKPOINTS = 10
EXPECTED_TOTAL_STEPS = 39_063
QK_DIAGNOSTIC_INTERVAL = 500
EXPECTED_QK_STEPS: tuple[int, ...] = (
    *range(QK_DIAGNOSTIC_INTERVAL, EXPECTED_TOTAL_STEPS + 1, QK_DIAGNOSTIC_INTERVAL),
    EXPECTED_TOTAL_STEPS,
)
EXPECTED_PERMANENT_STEPS: tuple[int, ...] = (
    0,
    2_441,
    4_883,
    7_324,
    9_766,
    12_207,
    14_648,
    17_090,
    19_531,
    21_973,
    24_414,
    26_855,
    29_297,
    31_738,
    34_180,
    36_621,
    39_063,
)
FROZEN_CONFIG_SHA256 = (
    "ebbbdfa30efe96b0b0c1c68ae4fc81909361502d89ad336d1181d00fcb85876a"
)
PINNED_WEIGHTWATCHER = "0.7.7"
PINNED_PACKAGE_VERSION = "0.5.1"
SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]
FROZEN_CONFIG = EXPERIMENT_DIR / "configs" / "baseline.yaml"
MANIFEST_PACKAGES: tuple[str, ...] = (
    "python",
    "rg-nanogpt-one-head",
    "torch",
    "torch-xla",
    "numpy",
    "pandas",
    "scipy",
    "PyYAML",
    "datasets",
    "tiktoken",
    "sacrebleu",
    "weightwatcher",
    "powerlaw",
    "papermill",
    "packaging",
)

TEST_RESULT_METRICS: tuple[str, ...] = (
    "test_loss",
    "test_perplexity",
    "test_bits_per_token",
    "test_accuracy",
    "test_top5_accuracy",
    "test_bleu",
    "test_continuation_token_accuracy",
    "test_continuation_exact_match",
)

FINAL_EPOCH_METRICS: tuple[str, ...] = (
    "train_loss",
    "val_loss",
    "test_loss",
    "train_perplexity",
    "val_perplexity",
    "test_perplexity",
    "train_bits_per_token",
    "val_bits_per_token",
    "test_bits_per_token",
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
    "train_top5_accuracy",
    "val_top5_accuracy",
    "test_top5_accuracy",
    "test_bleu",
    "test_continuation_token_accuracy",
    "test_continuation_exact_match",
)

FINITE_EPOCH_METRICS: tuple[str, ...] = (
    "train_loss",
    "val_loss",
    "train_perplexity",
    "val_perplexity",
    "train_bits_per_token",
    "val_bits_per_token",
    "train_accuracy",
    "val_accuracy",
    "train_top5_accuracy",
    "val_top5_accuracy",
)

HELD_OUT_CURVE_COLUMNS: tuple[str, ...] = (
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

REQUIRED_RUN_FILES: tuple[str, ...] = (
    "manifest.json",
    "run_complete.json",
    "metrics.csv",
    "epoch_metrics.csv",
    "spectral/layers.csv",
    "spectral/summary.csv",
    "test_results.json",
)

CSV_ARTIFACTS: tuple[str, ...] = (
    "campaign_runs.csv",
    "metrics_all.csv",
    "epoch_metrics_all.csv",
    "spectral_layers_all.csv",
    "spectral_summary_all.csv",
    "test_results_all.csv",
    "qk_diagnostics_all.csv",
    "qk_summary.csv",
    "performance_summary.csv",
    "paired_seed_differences.csv",
    "alpha_run_medians.csv",
    "alpha_across_seed_summary.csv",
    "saturation_diagnostics.csv",
    "saturation_integer_epoch_validation.csv",
    "saturation_across_seed_summary.csv",
    "checkpoint_sha256.csv",
)


class CampaignValidationError(RuntimeError):
    """Raised when the input cannot support the requested scientific report."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _requires_complete(args: argparse.Namespace) -> bool:
    """Resolve strictness for the CLI and programmatic legacy callers."""

    if hasattr(args, "require_complete"):
        return bool(args.require_complete)
    return not bool(getattr(args, "allow_incomplete", False))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _frozen_campaign_config() -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - launcher dependency gate
        raise CampaignValidationError("PyYAML is required to verify the frozen config") from exc
    try:
        payload = yaml.safe_load(FROZEN_CONFIG.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignValidationError(
            f"could not read the frozen campaign config {FROZEN_CONFIG}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CampaignValidationError("the frozen campaign config is not a mapping")
    observed = _canonical_sha256(payload)
    if observed != FROZEN_CONFIG_SHA256:
        raise CampaignValidationError(
            "the checked-out campaign config differs from its frozen contract: "
            f"observed={observed}, expected={FROZEN_CONFIG_SHA256}"
        )
    return payload


def _repository_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CampaignValidationError(f"could not resolve the source Git commit: {exc}") from exc
    commit = completed.stdout.strip()
    if not commit:
        raise CampaignValidationError("Git returned an empty source commit")
    return commit


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            _json_ready(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(
        temporary,
        dpi=180,
        bbox_inches="tight",
        format=path.suffix.lstrip("."),
    )
    temporary.replace(path)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _is_strictly_within(path: Path, parent: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_parent = parent.resolve(strict=False)
    return resolved_path != resolved_parent and _is_within(
        resolved_path, resolved_parent
    )


def _validate_paths(results_root: Path, output_root: Path) -> None:
    allowed_tmp_roots = {
        Path("/tmp").resolve(strict=False),
        Path("/private/tmp").resolve(strict=False),
    }
    for label, path in (
        ("results root", results_root),
        ("output root", output_root),
    ):
        if not any(
            _is_strictly_within(path, tmp_root)
            for tmp_root in allowed_tmp_roots
        ):
            raise CampaignValidationError(
                f"{label} must be strictly below resolved /tmp or "
                f"/private/tmp; observed {path}"
            )
    experiment_root_value = os.environ.get("RG_NANOGPT_EXPERIMENT_ROOT", "")
    if not experiment_root_value.strip():
        raise CampaignValidationError(
            "RG_NANOGPT_EXPERIMENT_ROOT is required for report generation"
        )
    experiment_root = Path(experiment_root_value)
    if not experiment_root.is_absolute() or "~" in experiment_root_value:
        raise CampaignValidationError(
            "RG_NANOGPT_EXPERIMENT_ROOT must be an absolute non-tilde path"
        )
    experiment_root = experiment_root.resolve(strict=False)
    if not any(
        _is_strictly_within(experiment_root, tmp_root)
        for tmp_root in allowed_tmp_roots
    ):
        raise CampaignValidationError(
            "RG_NANOGPT_EXPERIMENT_ROOT must resolve strictly below /tmp or "
            "/private/tmp"
        )
    for label, path in (
        ("results root", results_root),
        ("output root", output_root),
    ):
        if not _is_strictly_within(path, experiment_root):
            raise CampaignValidationError(
                f"{label} must be strictly below RG_NANOGPT_EXPERIMENT_ROOT: {path}"
            )
    home_value = os.environ.get("HOME")
    if home_value:
        home = Path(home_value).resolve(strict=False)
        for label, path in (
            ("results root", results_root),
            ("output root", output_root),
        ):
            if _is_within(path, home):
                raise CampaignValidationError(
                    f"{label} must never be HOME or below HOME: {path}"
                )
    if not results_root.is_dir():
        raise CampaignValidationError(
            f"results root does not exist or is not a directory: {results_root}"
        )
    if _is_within(output_root, results_root) or _is_within(results_root, output_root):
        raise CampaignValidationError(
            "results root and report output root must not contain one another"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignValidationError(f"could not read valid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignValidationError(f"JSON artifact is not an object: {path}")
    return payload


def _read_csv(path: Path, *, allow_empty: bool = False) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise CampaignValidationError(f"could not read CSV {path}: {exc}") from exc
    if frame.empty and not allow_empty:
        raise CampaignValidationError(f"required CSV is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise CampaignValidationError(f"{label} is missing columns: {missing}")


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _with_identity(
    frame: pd.DataFrame,
    *,
    optimizer: str,
    seed: int,
    source_path: Path,
) -> pd.DataFrame:
    result = frame.copy()
    if "optimizer" in result.columns:
        observed = set(result["optimizer"].dropna().astype(str))
        if observed and observed != {optimizer}:
            raise CampaignValidationError(
                f"optimizer identity mismatch in {source_path}: {sorted(observed)}"
            )
        result = result.drop(columns=["optimizer"])
    if "seed" in result.columns:
        observed_seed = set(pd.to_numeric(result["seed"], errors="coerce").dropna().astype(int))
        if observed_seed and observed_seed != {int(seed)}:
            raise CampaignValidationError(
                f"seed identity mismatch in {source_path}: {sorted(observed_seed)}"
            )
        result = result.drop(columns=["seed"])
    result.insert(0, "optimizer", optimizer)
    result.insert(1, "optimizer_label", OPTIMIZER_LABELS[optimizer])
    result.insert(2, "seed", int(seed))
    result.insert(3, "source_file", str(source_path))
    return result


def _matrix_type(value: Any) -> str:
    text = str(value).upper()
    for matrix in sorted(MATRIX_TYPES, key=len, reverse=True):
        if text == matrix or text.endswith("_" + matrix) or matrix in text:
            return matrix
    return text


def _first_existing_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _normalize_spectral_layers(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    _require_columns(frame, ("step", "epoch"), label)
    result = _numeric(frame, ("step", "tokens_seen", "epoch"))
    if "matrix_type" not in result.columns:
        if "matrix_name" not in result.columns:
            raise CampaignValidationError(
                f"{label} contains neither matrix_type nor matrix_name"
            )
        result["matrix_type"] = result["matrix_name"].map(_matrix_type)
    else:
        result["matrix_type"] = result["matrix_type"].map(_matrix_type)

    if "matrix_name" not in result.columns:
        result["matrix_name"] = result["matrix_type"]

    clipped_column = _first_existing_column(
        result,
        ("alpha_clip_xmax", "clip_xmax_alpha", "fixed_alpha", "alpha"),
    )
    raw_column = _first_existing_column(
        result,
        ("alpha_raw", "raw_alpha", "alpha_before_finger_clip"),
    )
    if clipped_column is None or raw_column is None:
        raise CampaignValidationError(
            f"{label} must contain both clipped and raw WeightWatcher alpha; "
            f"observed columns={sorted(result.columns)}"
        )
    result["alpha_clip_xmax"] = pd.to_numeric(
        result[clipped_column], errors="coerce"
    )
    result["alpha_raw"] = pd.to_numeric(result[raw_column], errors="coerce")
    result["alpha_clip_minus_raw"] = (
        result["alpha_clip_xmax"] - result["alpha_raw"]
    )
    return result


def _expected_run_directory(results_root: Path, optimizer: str, seed: int) -> Path:
    return results_root / optimizer / f"seed_{int(seed)}"


def _unexpected_seed_directories(results_root: Path) -> list[str]:
    unexpected: list[str] = []
    pattern = re.compile(r"^seed_(-?\d+)$")
    expected = set(SEEDS)
    for optimizer in OPTIMIZERS:
        optimizer_root = results_root / optimizer
        if not optimizer_root.is_dir():
            continue
        for child in optimizer_root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match and int(match.group(1)) not in expected:
                unexpected.append(str(child))
    for child in results_root.iterdir():
        if not child.is_dir() or child.name in OPTIMIZERS:
            continue
        if any(
            grandchild.is_dir() and pattern.match(grandchild.name)
            for grandchild in child.iterdir()
        ):
            unexpected.append(str(child))
    return sorted(unexpected)


def _validate_step_span(
    frame: pd.DataFrame,
    *,
    total_steps: int,
    label: str,
) -> None:
    _require_columns(frame, ("step",), label)
    steps = pd.to_numeric(frame["step"], errors="coerce")
    if steps.isna().any():
        raise CampaignValidationError(f"{label} contains nonnumeric steps")
    integer_steps = steps.astype(int)
    if not np.allclose(steps.to_numpy(dtype=float), integer_steps.to_numpy(dtype=float)):
        raise CampaignValidationError(f"{label} contains noninteger steps")
    if 0 not in set(integer_steps) or total_steps not in set(integer_steps):
        raise CampaignValidationError(
            f"{label} does not span step zero through {total_steps}"
        )
    if int(integer_steps.max()) != int(total_steps):
        raise CampaignValidationError(
            f"{label} extends to {integer_steps.max()}, expected {total_steps}"
        )


def _validate_spectral_inventory(
    layers: pd.DataFrame,
    epoch_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    label: str,
) -> None:
    _require_columns(
        layers,
        (
            "step",
            "matrix_type",
            "alpha_raw",
            "alpha_clip_xmax",
            "alpha",
            "raw_alpha",
            "alpha_delta",
            "num_fingers",
            "finger_policy",
            "primary_alpha_variant",
            "weightwatcher_analysis_calls",
        ),
        f"{label} spectral/layers.csv",
    )
    alpha_values = layers[["alpha_raw", "alpha_clip_xmax"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(alpha_values.to_numpy(dtype=float)).all():
        raise CampaignValidationError(
            f"{label} has non-finite raw or clip_xmax alpha values"
        )
    aliases = layers[
        ["alpha", "raw_alpha", "alpha_delta", "num_fingers"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(aliases.to_numpy(dtype=float)).all():
        raise CampaignValidationError(f"{label} has non-finite alpha alias values")
    clipped = pd.to_numeric(layers["alpha_clip_xmax"], errors="coerce")
    raw = pd.to_numeric(layers["alpha_raw"], errors="coerce")
    if not np.allclose(aliases["alpha"], clipped, rtol=0.0, atol=0.0):
        raise CampaignValidationError(f"{label} alpha alias differs from alpha_clip_xmax")
    if not np.allclose(aliases["raw_alpha"], raw, rtol=0.0, atol=0.0):
        raise CampaignValidationError(f"{label} raw_alpha alias differs from alpha_raw")
    if not np.allclose(
        aliases["alpha_delta"], raw - clipped, rtol=1e-12, atol=1e-12
    ) or (aliases["num_fingers"] < 0).any():
        raise CampaignValidationError(
            f"{label} has an invalid alpha_delta or negative num_fingers"
        )
    analysis_calls = pd.to_numeric(
        layers["weightwatcher_analysis_calls"], errors="coerce"
    )
    if analysis_calls.isna().any() or not analysis_calls.eq(1).all():
        raise CampaignValidationError(
            f"{label} does not record exactly one WeightWatcher analysis call "
            "per checkpoint row"
        )
    if not layers["finger_policy"].astype(str).eq(
        "fix_fingers=clip_xmax"
    ).all():
        raise CampaignValidationError(
            f"{label} does not consistently declare fix_fingers=clip_xmax"
        )
    if not layers["primary_alpha_variant"].astype(str).eq(
        "clip_xmax"
    ).all():
        raise CampaignValidationError(
            f"{label} does not consistently declare clip_xmax as primary alpha"
        )

    expected_matrices = set(MATRIX_TYPES)
    epoch_steps = set(pd.to_numeric(epoch_metrics["step"], errors="coerce").astype(int))
    layer_steps = set(pd.to_numeric(layers["step"], errors="coerce").astype(int))
    summary_steps = set(pd.to_numeric(summary["step"], errors="coerce").astype(int))
    if layer_steps != epoch_steps or summary_steps != epoch_steps:
        raise CampaignValidationError(
            f"{label} spectral steps do not exactly match permanent checkpoint steps"
        )
    if summary["step"].duplicated().any():
        raise CampaignValidationError(
            f"{label} spectral/summary.csv contains duplicate steps"
        )
    _require_columns(summary, ("n_matrices",), f"{label} spectral/summary.csv")
    matrix_counts = pd.to_numeric(summary["n_matrices"], errors="coerce")
    if matrix_counts.isna().any() or not matrix_counts.eq(len(MATRIX_TYPES)).all():
        raise CampaignValidationError(
            f"{label} spectral summaries do not report exactly six matrices"
        )
    duplicated = layers.duplicated(["step", "matrix_type"], keep=False)
    if duplicated.any():
        rows = layers.loc[duplicated, ["step", "matrix_type"]].to_dict("records")
        raise CampaignValidationError(
            f"{label} contains duplicate step/matrix spectral rows: {rows[:12]}"
        )
    for step, group in layers.groupby("step", sort=True):
        observed = set(group["matrix_type"].astype(str))
        if observed != expected_matrices or len(group) != len(MATRIX_TYPES):
            raise CampaignValidationError(
                f"{label} step={int(step)} has matrices={sorted(observed)}, "
                f"expected={sorted(expected_matrices)}"
            )


def _normalize_test_results(
    payload: Mapping[str, Any],
    *,
    optimizer: str,
    seed: int,
    source_path: Path,
) -> pd.DataFrame:
    policy = str(payload.get("policy", "")).lower()
    if "held out" not in policy or "validation" not in policy or "never" not in policy:
        raise CampaignValidationError(
            f"{source_path} does not declare the held-out post-training test policy"
        )
    rows: list[dict[str, Any]] = []
    aliases = {
        "final": ("final",),
        "validation_selected": ("validation_selected", "best", "selected"),
    }
    for canonical, candidates in aliases.items():
        key = next((name for name in candidates if name in payload), None)
        if key is None or not isinstance(payload[key], Mapping):
            raise CampaignValidationError(
                f"{source_path} lacks required test result {canonical!r}"
            )
        values = payload[key]
        row = {
            "optimizer": optimizer,
            "optimizer_label": OPTIMIZER_LABELS[optimizer],
            "seed": int(seed),
            "source_file": str(source_path),
            "checkpoint": canonical,
        }
        for output, source_names in {
            "step": ("step",),
            "test_loss": ("loss", "test_loss"),
            "test_perplexity": ("perplexity", "test_perplexity"),
            "test_bits_per_token": (
                "bits_per_token",
                "test_bits_per_token",
            ),
            "test_accuracy": ("accuracy", "test_accuracy"),
            "test_top5_accuracy": (
                "top5_accuracy",
                "test_top5_accuracy",
            ),
            "test_bleu": ("bleu", "test_bleu"),
            "test_continuation_token_accuracy": (
                "continuation_token_accuracy",
                "test_continuation_token_accuracy",
            ),
            "test_continuation_exact_match": (
                "continuation_exact_match",
                "test_continuation_exact_match",
            ),
        }.items():
            source = next((name for name in source_names if name in values), None)
            if source is None:
                raise CampaignValidationError(
                    f"{source_path} {canonical} lacks {output}"
                )
            row[output] = values[source]
        rows.append(row)
    result = _numeric(pd.DataFrame(rows), ("step", *TEST_RESULT_METRICS))
    values = result[list(TEST_RESULT_METRICS)]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise CampaignValidationError(f"{source_path} has non-finite test metrics")
    for _, row in result.iterrows():
        label = str(row["checkpoint"])
        loss = float(row["test_loss"])
        perplexity = float(row["test_perplexity"])
        bits = float(row["test_bits_per_token"])
        bounded = (
            float(row["test_accuracy"]),
            float(row["test_top5_accuracy"]),
            float(row["test_continuation_token_accuracy"]),
            float(row["test_continuation_exact_match"]),
        )
        if (
            loss < 0.0
            or bits < 0.0
            or perplexity <= 0.0
            or not math.isclose(math.log(perplexity), loss, rel_tol=1e-10, abs_tol=1e-10)
            or not math.isclose(bits * math.log(2.0), loss, rel_tol=1e-10, abs_tol=1e-10)
            or any(not 0.0 <= value <= 1.0 for value in bounded)
            or float(row["test_top5_accuracy"]) < float(row["test_accuracy"])
            or float(row["test_continuation_exact_match"])
            > float(row["test_continuation_token_accuracy"]) + 1e-12
            or not 0.0 <= float(row["test_bleu"]) <= 100.0
        ):
            raise CampaignValidationError(
                f"{source_path} {label} has inconsistent or out-of-range test metrics"
            )
    return result


def _campaign_invariants(manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime = manifest.get("runtime_environment", {})
    runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
    source = manifest.get(
        "source_repository",
        manifest.get("source", {}),
    )
    source_mapping = source if isinstance(source, Mapping) else {}
    return {
        "protocol": manifest.get("protocol"),
        "config_sha256": manifest.get("config_sha256"),
        "model": manifest.get("model"),
        "training": manifest.get("training"),
        "evaluation": manifest.get("evaluation"),
        "weightwatcher": manifest.get("weightwatcher"),
        "data_metadata": manifest.get("data_metadata"),
        "tokens_per_step": manifest.get("tokens_per_step"),
        "max_steps": manifest.get("max_steps"),
        "package_versions": manifest.get("package_versions"),
        "runtime_environment": _runtime_block_identity(runtime_mapping),
        "accelerator": runtime_mapping.get("accelerator"),
        "torch_version": manifest.get("torch_version", runtime_mapping.get("torch_version")),
        "git_available": source_mapping.get("available"),
        "git_commit": manifest.get(
            "git_commit",
            manifest.get("source_commit", source_mapping.get("commit")),
        ),
        "git_dirty": manifest.get(
            "git_dirty",
            source_mapping.get("dirty"),
        ),
    }


def _runtime_block_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(runtime)
    if str(identity.get("hardware_block_id_source", "")) == "user":
        for field in (
            "python_executable",
            "processor",
            "cuda_device_uuid",
            "cuda_device_count",
            "xla_process_index",
        ):
            identity.pop(field, None)
    return identity


def _validate_matched_campaign(
    manifests: Sequence[Mapping[str, Any]],
    *,
    allow_mixed_runtime: bool,
) -> None:
    if not manifests:
        raise CampaignValidationError("no complete run manifests were loaded")
    frozen = _frozen_campaign_config()
    current_commit = _repository_head()
    reference = _campaign_invariants(manifests[0])
    scientific_keys = (
        "protocol",
        "config_sha256",
        "model",
        "training",
        "evaluation",
        "weightwatcher",
        "data_metadata",
        "tokens_per_step",
        "max_steps",
        "package_versions",
    )
    for index, manifest in enumerate(manifests):
        observed = _campaign_invariants(manifest)
        if index:
            for key in scientific_keys:
                if _canonical_json(observed[key]) != _canonical_json(reference[key]):
                    raise CampaignValidationError(
                        f"campaign invariant {key!r} differs in manifest index {index}"
                    )
            for key in ("git_available", "git_commit", "git_dirty"):
                if _canonical_json(observed[key]) != _canonical_json(reference[key]):
                    raise CampaignValidationError(
                        f"source invariant {key!r} differs across runs"
                    )
            if not allow_mixed_runtime and _canonical_json(
                observed["runtime_environment"]
            ) != _canonical_json(reference["runtime_environment"]):
                raise CampaignValidationError(
                    "runtime/hardware-block identity differs across runs; "
                    "create separate reports for separate hardware blocks"
                )
        if observed.get("git_available") is not True:
            raise CampaignValidationError(
                f"manifest index {index} has no readable Git source identity"
            )
        if observed.get("git_dirty") is not False:
            raise CampaignValidationError(
                "run manifests must identify a clean source tree; "
                f"manifest index {index} records dirty={observed.get('git_dirty')!r}"
            )
        commit = str(observed.get("git_commit", ""))
        if not commit or commit == "unknown":
            raise CampaignValidationError(
                f"manifest index {index} has no exact Git commit"
            )
        if commit != current_commit:
            raise CampaignValidationError(
                f"manifest index {index} was produced by {commit}, but the "
                f"checked-out report source is {current_commit}"
            )

        if str(manifest.get("config_sha256", "")) != FROZEN_CONFIG_SHA256:
            raise CampaignValidationError(
                f"manifest index {index} does not use the frozen campaign config"
            )
        initial_model_hash = str(manifest.get("initial_model_sha256", ""))
        if (
            len(initial_model_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in initial_model_hash.lower()
            )
        ):
            raise CampaignValidationError(
                f"manifest index {index} has no initial-model tensor hash"
            )
        for key in ("protocol", "model", "training", "evaluation", "weightwatcher"):
            if _canonical_json(manifest.get(key)) != _canonical_json(frozen[key]):
                raise CampaignValidationError(
                    f"manifest index {index} {key} differs from the frozen config"
                )
        optimizer = str(manifest.get("optimizer", ""))
        expected_profile = dict(frozen["optimizer_profiles"].get(optimizer, {}))
        expected_profile["name"] = optimizer
        if _canonical_json(manifest.get("optimizer_profile")) != _canonical_json(
            expected_profile
        ):
            raise CampaignValidationError(
                f"manifest index {index} optimizer profile differs from the frozen config"
            )
        if int(manifest.get("max_steps", -1)) != EXPECTED_TOTAL_STEPS:
            raise CampaignValidationError(
                f"manifest index {index} has the wrong optimizer-step horizon"
            )
        expected_tokens_per_step = (
            int(frozen["training"]["batch_size"])
            * int(frozen["training"]["grad_accum_steps"])
            * int(frozen["model"]["block_size"])
        )
        if int(manifest.get("tokens_per_step", -1)) != expected_tokens_per_step:
            raise CampaignValidationError(
                f"manifest index {index} has the wrong tokens_per_step"
            )
        packages = manifest.get("package_versions")
        if not isinstance(packages, Mapping):
            raise CampaignValidationError(
                f"manifest index {index} has no package_versions mapping"
            )
        missing_packages = [name for name in MANIFEST_PACKAGES if not packages.get(name)]
        if missing_packages:
            raise CampaignValidationError(
                f"manifest index {index} lacks dependency versions: {missing_packages}"
            )
        if str(packages.get("weightwatcher")) != PINNED_WEIGHTWATCHER:
            raise CampaignValidationError(
                f"manifest index {index} did not use WeightWatcher {PINNED_WEIGHTWATCHER}"
            )
        if str(packages.get("rg-nanogpt-one-head")) != PINNED_PACKAGE_VERSION:
            raise CampaignValidationError(
                f"manifest index {index} did not use campaign package "
                f"{PINNED_PACKAGE_VERSION}"
            )
        data = manifest.get("data_metadata")
        expected_dataset_fields = {
            "schema_version": 2,
            "dataset_name": frozen["dataset"]["name"],
            "dataset_config": frozen["dataset"]["config"],
            "dataset_split": frozen["dataset"]["split"],
            "dataset_revision": frozen["dataset"]["revision"],
            "tokenizer": frozen["dataset"]["tokenizer"],
            "vocab_size": frozen["model"]["vocab_size"],
            "eot_token": 50_256,
            "dtype": "uint16",
            "document_disjoint_splits": True,
            "splits": {
                "train": frozen["dataset"]["train_tokens"],
                "val": frozen["dataset"]["val_tokens"],
                "test": frozen["dataset"]["test_tokens"],
            },
        }
        if not isinstance(data, Mapping) or any(
            data.get(key) != value for key, value in expected_dataset_fields.items()
        ):
            raise CampaignValidationError(
                f"manifest index {index} data metadata differs from the frozen corpus"
            )
        files = data.get("files")
        if not isinstance(files, Mapping) or any(
            not isinstance(files.get(split), Mapping)
            or files[split].get("path") != f"{split}.bin"
            or len(str(files[split].get("sha256", ""))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(files[split].get("sha256", "")).lower()
            )
            or int(files[split].get("bytes", -1)) != int(tokens) * 2
            for split, tokens in expected_dataset_fields["splits"].items()
        ):
            raise CampaignValidationError(
                f"manifest index {index} has no complete corpus hash inventory"
            )

        runtime = manifest.get("runtime_environment", {})
        if not isinstance(runtime, Mapping):
            raise CampaignValidationError(
                f"manifest index {index} has no runtime_environment mapping"
            )
        if runtime.get("float32_matmul_precision") != "highest":
            raise CampaignValidationError(
                f"manifest index {index} did not use matmul_precision=highest"
            )
        if runtime.get("deterministic_algorithms") is not True:
            raise CampaignValidationError(
                f"manifest index {index} did not enable deterministic algorithms"
            )
        if runtime.get("deterministic_warn_only") is not False:
            raise CampaignValidationError(
                f"manifest index {index} used deterministic warn-only mode"
            )
        if runtime.get("accelerator") not in {"cpu", "mps", "cuda", "tpu"}:
            raise CampaignValidationError(
                f"manifest index {index} records an unsupported accelerator: "
                f"{runtime.get('accelerator')!r}"
            )
        if not str(runtime.get("hardware_block_id", "")).strip() or not str(
            runtime.get("hardware_block_id_source", "")
        ).strip():
            raise CampaignValidationError(
                f"manifest index {index} has no complete hardware-block identity"
            )
        if runtime.get("accelerator") == "cuda" and (
            runtime.get("cuda_matmul_allow_tf32") is not False
            or runtime.get("cudnn_allow_tf32") is not False
        ):
            raise CampaignValidationError(
                f"manifest index {index} enabled CUDA TF32"
            )

    if len(manifests) == len(OPTIMIZERS) * len(SEEDS):
        initial_hashes_by_seed: dict[int, dict[str, str]] = defaultdict(dict)
        for manifest in manifests:
            initial_hashes_by_seed[int(manifest["seed"])][
                str(manifest["optimizer"])
            ] = str(manifest["initial_model_sha256"])
        for seed in SEEDS:
            hashes = initial_hashes_by_seed.get(seed, {})
            if set(hashes) != set(OPTIMIZERS) or len(set(hashes.values())) != 1:
                raise CampaignValidationError(
                    "optimizer arms do not share identical step-zero tensors for "
                    f"seed {seed}: {hashes}"
                )

    model = reference.get("model")
    training = reference.get("training")
    weightwatcher = reference.get("weightwatcher")
    protocol = reference.get("protocol")
    if not isinstance(model, Mapping) or (
        int(model.get("n_layer", -1)) != 1
        or int(model.get("n_head", -1)) != 1
    ):
        raise CampaignValidationError(
            "the dated baseline requires a one-layer, one-head model"
        )
    if not isinstance(training, Mapping) or (
        tuple(int(value) for value in training.get("seeds", ())) != SEEDS
        or not math.isclose(float(training.get("target_epochs", -1.0)), 4.0)
        or not math.isclose(float(training.get("epoch_interval", -1.0)), 0.25)
    ):
        raise CampaignValidationError(
            "the dated baseline requires seeds "
            "1337/2027/4099/31415/271828, four epochs, "
            "and quarter-epoch permanent states"
        )
    if not isinstance(weightwatcher, Mapping) or (
        weightwatcher.get("enabled") is not True
        or weightwatcher.get("strict") is not True
        or weightwatcher.get("fix_fingers") != "clip_xmax"
        or int(weightwatcher.get("max_fingers", -1)) != 10
        or weightwatcher.get("require_raw_alpha") is not True
    ):
        raise CampaignValidationError(
            "the dated baseline requires strict one-call WeightWatcher with "
            "fix_fingers=clip_xmax, max_fingers=10, and raw alpha"
        )
    if not isinstance(protocol, Mapping) or protocol.get("name") != (
        "nanogpt_one_head_2026_08_21_ww_baseline"
    ):
        raise CampaignValidationError(
            "run manifests do not identify the dated baseline protocol"
        )


def _load_campaign(
    results_root: Path,
    *,
    require_complete: bool,
    allow_extra_runs: bool,
    allow_mixed_runtime: bool,
) -> dict[str, Any]:
    unexpected = _unexpected_seed_directories(results_root)
    if unexpected and not allow_extra_runs:
        raise CampaignValidationError(
            "unexpected seed directories make this more than the exact 2 x 5 campaign: "
            + ", ".join(unexpected)
        )

    run_rows: list[dict[str, Any]] = []
    metrics_frames: list[pd.DataFrame] = []
    epoch_frames: list[pd.DataFrame] = []
    layer_frames: list[pd.DataFrame] = []
    spectral_summary_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    qk_frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    completions: dict[tuple[str, int], dict[str, Any]] = {}
    run_dirs: dict[tuple[str, int], Path] = {}
    errors: list[str] = []

    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            run_dir = _expected_run_directory(results_root, optimizer, seed)
            run_dirs[(optimizer, seed)] = run_dir
            missing = [
                relative
                for relative in REQUIRED_RUN_FILES
                if not (run_dir / relative).is_file()
                or (run_dir / relative).stat().st_size == 0
            ]
            if missing:
                message = (
                    f"optimizer={optimizer} seed={seed} missing required artifacts: {missing}"
                )
                errors.append(message)
                run_rows.append({
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS[optimizer],
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "complete": False,
                    "validation_error": message,
                })
                continue

            try:
                manifest = _read_json(run_dir / "manifest.json")
                completion = _read_json(run_dir / "run_complete.json")
                manifest_optimizer = str(manifest.get("optimizer", ""))
                completion_optimizer = str(completion.get("optimizer", ""))
                manifest_seed = int(manifest.get("seed"))
                completion_seed = int(completion.get("seed"))
                if manifest_optimizer != optimizer or completion_optimizer != optimizer:
                    raise CampaignValidationError(
                        f"optimizer identity mismatch in {run_dir}"
                    )
                if manifest_seed != seed or completion_seed != seed:
                    raise CampaignValidationError(f"seed identity mismatch in {run_dir}")
                if completion.get("completed") is not True:
                    raise CampaignValidationError(
                        f"run_complete.json does not declare completed=true: {run_dir}"
                    )
                if str(completion.get("fingerprint", "")) != str(
                    manifest.get("protocol_fingerprint", "")
                ):
                    raise CampaignValidationError(
                        f"manifest/completion protocol fingerprint mismatch in {run_dir}"
                    )
                optimizer_profile = manifest.get("optimizer_profile", {})
                if not isinstance(optimizer_profile, Mapping) or str(
                    optimizer_profile.get("family", "")
                ) != optimizer:
                    raise CampaignValidationError(
                        f"optimizer profile family mismatch in {run_dir}"
                    )

                total_steps = int(completion["optimizer_steps"])
                if int(manifest.get("max_steps", total_steps)) != total_steps:
                    raise CampaignValidationError(
                        f"manifest/completion total-step mismatch in {run_dir}"
                    )
                if total_steps != EXPECTED_TOTAL_STEPS:
                    raise CampaignValidationError(
                        f"optimizer-step horizon is {total_steps}, expected "
                        f"{EXPECTED_TOTAL_STEPS}: {run_dir}"
                    )

                metrics_path = run_dir / "metrics.csv"
                metrics = _numeric(
                    _read_csv(metrics_path),
                    (
                        "step",
                        "tokens_seen",
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "test_loss",
                        "train_perplexity",
                        "val_perplexity",
                        "test_perplexity",
                        "train_bits_per_token",
                        "val_bits_per_token",
                        "test_bits_per_token",
                        "train_accuracy",
                        "val_accuracy",
                        "test_accuracy",
                        "train_top5_accuracy",
                        "val_top5_accuracy",
                        "test_top5_accuracy",
                        "test_bleu",
                        "test_continuation_token_accuracy",
                        "test_continuation_exact_match",
                        "val_generalization_gap",
                        "test_generalization_gap",
                        "tokens_per_sec",
                    ),
                )
                _require_columns(
                    metrics,
                    ("step", "epoch", *FINAL_EPOCH_METRICS),
                    str(metrics_path),
                )
                _validate_step_span(metrics, total_steps=total_steps, label=str(metrics_path))
                always_finite = metrics[[
                    "train_loss",
                    "val_loss",
                    "train_perplexity",
                    "val_perplexity",
                    "train_bits_per_token",
                    "val_bits_per_token",
                    "train_accuracy",
                    "val_accuracy",
                    "train_top5_accuracy",
                    "val_top5_accuracy",
                ]].apply(pd.to_numeric, errors="coerce")
                if not np.isfinite(always_finite.to_numpy(dtype=float)).all():
                    raise CampaignValidationError(
                        f"{metrics_path} contains non-finite train/validation metrics"
                    )
                if metrics[list(HELD_OUT_CURVE_COLUMNS)].notna().any().any():
                    raise CampaignValidationError(
                        f"{metrics_path} leaks held-out test outcomes into training curves"
                    )
                validation_minimum_index = metrics["val_loss"].idxmin()
                validation_minimum = metrics.loc[validation_minimum_index]
                if int(validation_minimum["step"]) != int(
                    completion["best_validation_step"]
                ) or not math.isclose(
                    float(validation_minimum["val_loss"]),
                    float(completion["best_validation_loss"]),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise CampaignValidationError(
                        f"validation-selected checkpoint metadata disagrees with {metrics_path}"
                    )

                epoch_path = run_dir / "epoch_metrics.csv"
                epoch_metrics = _numeric(
                    _read_csv(epoch_path),
                    (
                        "step",
                        "tokens_seen",
                        "epoch",
                        "nominal_epoch",
                        "tokens_per_sec",
                        "val_generalization_gap",
                        "test_generalization_gap",
                        *FINAL_EPOCH_METRICS,
                    ),
                )
                _require_columns(
                    epoch_metrics,
                    (
                        "step",
                        "epoch",
                        "nominal_epoch",
                        "checkpoint_path",
                        "val_loss",
                        "test_monitoring_only",
                        "test_held_out",
                    ),
                    str(epoch_path),
                )
                _validate_step_span(
                    epoch_metrics, total_steps=total_steps, label=str(epoch_path)
                )
                if epoch_metrics["step"].duplicated().any():
                    raise CampaignValidationError(
                        f"duplicate permanent-checkpoint steps in {epoch_path}"
                    )
                observed_permanent_steps = tuple(
                    sorted(pd.to_numeric(epoch_metrics["step"], errors="raise").astype(int))
                )
                if observed_permanent_steps != EXPECTED_PERMANENT_STEPS:
                    raise CampaignValidationError(
                        f"{epoch_path} does not contain the exact frozen 17-state grid"
                    )
                monitoring_policy = pd.to_numeric(
                    epoch_metrics["test_monitoring_only"], errors="coerce"
                )
                if monitoring_policy.isna().any() or not monitoring_policy.eq(1).all():
                    raise CampaignValidationError(
                        f"{epoch_path} violates the held-out test policy"
                    )
                held_out_policy = pd.to_numeric(
                    epoch_metrics["test_held_out"], errors="coerce"
                )
                if held_out_policy.isna().any() or not held_out_policy.eq(1).all():
                    raise CampaignValidationError(
                        f"{epoch_path} does not mark every test curve as held out"
                    )
                epoch_outcomes = epoch_metrics[list(FINITE_EPOCH_METRICS)].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                if not np.isfinite(epoch_outcomes.to_numpy(dtype=float)).all():
                    raise CampaignValidationError(
                        f"{epoch_path} contains non-finite permanent-state outcomes"
                    )
                if epoch_metrics[list(HELD_OUT_CURVE_COLUMNS)].notna().any().any():
                    raise CampaignValidationError(
                        f"{epoch_path} leaks held-out test outcomes into permanent-state curves"
                    )
                if len(epoch_metrics) < MIN_PERMANENT_CHECKPOINTS:
                    raise CampaignValidationError(
                        f"{epoch_path} contains {len(epoch_metrics)} permanent checkpoints; "
                        f"at least {MIN_PERMANENT_CHECKPOINTS} are required"
                    )

                layers_path = run_dir / "spectral" / "layers.csv"
                layers = _normalize_spectral_layers(
                    _read_csv(layers_path), str(layers_path)
                )
                spectral_summary_path = run_dir / "spectral" / "summary.csv"
                spectral_summary = _numeric(
                    _read_csv(spectral_summary_path),
                    ("step", "tokens_seen", "epoch"),
                )
                _require_columns(spectral_summary, ("step", "epoch"), str(spectral_summary_path))
                _validate_spectral_inventory(
                    layers,
                    epoch_metrics,
                    spectral_summary,
                    label=f"optimizer={optimizer} seed={seed}",
                )
                for step_value in sorted(epoch_metrics["step"].astype(int)):
                    status_path = (
                        run_dir
                        / "spectral"
                        / f"status_step_{step_value:07d}.json"
                    )
                    if not status_path.is_file() or status_path.stat().st_size == 0:
                        raise CampaignValidationError(
                            f"missing WeightWatcher completion record: {status_path}"
                        )
                    status = _read_json(status_path)
                    if status.get("completed") is not True:
                        raise CampaignValidationError(
                            f"WeightWatcher completion record is incomplete: {status_path}"
                        )
                    if int(status.get("weightwatcher_analysis_calls", -1)) != 1:
                        raise CampaignValidationError(
                            f"WeightWatcher was not called exactly once: {status_path}"
                        )
                    if status.get("finger_policy") != "fix_fingers=clip_xmax":
                        raise CampaignValidationError(
                            f"WeightWatcher finger policy mismatch: {status_path}"
                        )
                    for count_key in (
                        "alpha_raw_valid_matrices",
                        "alpha_clip_xmax_valid_matrices",
                    ):
                        if int(status.get(count_key, -1)) != len(MATRIX_TYPES):
                            raise CampaignValidationError(
                                f"{status_path} does not record six valid "
                                f"matrices for {count_key}"
                            )

                test_path = run_dir / "test_results.json"
                test_results = _normalize_test_results(
                    _read_json(test_path),
                    optimizer=optimizer,
                    seed=seed,
                    source_path=test_path,
                )
                final_step = int(
                    test_results.loc[
                        test_results["checkpoint"].eq("final"), "step"
                    ].iloc[0]
                )
                selected_step = int(
                    test_results.loc[
                        test_results["checkpoint"].eq("validation_selected"), "step"
                    ].iloc[0]
                )
                if final_step != total_steps:
                    raise CampaignValidationError(
                        f"final test step={final_step}, expected {total_steps}: {test_path}"
                    )
                if selected_step != int(completion["best_validation_step"]):
                    raise CampaignValidationError(
                        f"validation-selected test step mismatch: {test_path}"
                    )
                final_test_row = test_results[
                    test_results["checkpoint"].eq("final")
                ].iloc[0]
                for metric, completion_key in {
                    "test_loss": "final_test_loss",
                    "test_perplexity": "final_test_perplexity",
                    "test_bits_per_token": "final_test_bits_per_token",
                    "test_accuracy": "final_test_accuracy",
                    "test_top5_accuracy": "final_test_top5_accuracy",
                    "test_bleu": "final_test_bleu",
                    "test_continuation_token_accuracy": (
                        "final_test_continuation_token_accuracy"
                    ),
                    "test_continuation_exact_match": (
                        "final_test_continuation_exact_match"
                    ),
                }.items():
                    observed = float(final_test_row[metric])
                    recorded = float(completion[completion_key])
                    if not math.isfinite(observed) or not math.isclose(
                        observed,
                        recorded,
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    ):
                        raise CampaignValidationError(
                            f"completion/test result mismatch for {metric}: {test_path}"
                        )

                qk_path = run_dir / "muonclip_qk.csv"
                if optimizer == "muon_clip":
                    if not qk_path.is_file() or qk_path.stat().st_size == 0:
                        raise CampaignValidationError(
                            f"MuonClip run lacks required QK diagnostics: {qk_path}"
                        )
                    qk = _numeric(
                        _read_csv(qk_path),
                        (
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
                        ),
                    )
                    _require_columns(
                        qk,
                        (
                            "step",
                            "threshold",
                            "head_observations",
                            "active_fraction",
                            "mean_max_logit",
                            "max_logit",
                            "mean_gamma",
                            "min_gamma",
                        ),
                        str(qk_path),
                    )
                    qk_required = qk[[
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
                    ]].apply(pd.to_numeric, errors="coerce")
                    if not np.isfinite(qk_required.to_numpy(dtype=float)).all():
                        raise CampaignValidationError(
                            f"{qk_path} contains non-finite QK diagnostics"
                        )
                    observed_steps = qk_required["step"].to_numpy(dtype=float)
                    if (
                        not np.allclose(observed_steps, np.rint(observed_steps))
                        or tuple(int(value) for value in observed_steps)
                        != EXPECTED_QK_STEPS
                    ):
                        raise CampaignValidationError(
                            f"{qk_path} does not cover the exact 500-step QK grid"
                        )
                    expected_intervals = np.diff(
                        np.asarray([0, *EXPECTED_QK_STEPS], dtype=int)
                    )
                    observed_intervals = qk_required[
                        "steps_in_interval"
                    ].to_numpy(dtype=float)
                    if (
                        not np.array_equal(
                            observed_intervals,
                            expected_intervals.astype(float),
                        )
                        or int(observed_intervals.sum()) != EXPECTED_TOTAL_STEPS
                    ):
                        raise CampaignValidationError(
                            f"{qk_path} QK intervals do not cover the training horizon"
                        )
                    observations = qk_required[
                        "head_observations"
                    ].to_numpy(dtype=float)
                    active_heads = qk_required["active_heads"].to_numpy(
                        dtype=float
                    )
                    active_fraction = qk_required[
                        "active_fraction"
                    ].to_numpy(dtype=float)
                    if (
                        not np.array_equal(observations, observed_intervals)
                        or (active_heads < 0.0).any()
                        or (active_heads > observations).any()
                        or not np.allclose(
                            active_fraction,
                            active_heads / observations,
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    ):
                        raise CampaignValidationError(
                            f"{qk_path} QK head counts/fractions are inconsistent"
                        )
                    if (
                        not qk_required["threshold"].eq(100.0).all()
                        or not qk_required["active_fraction"].between(0.0, 1.0).all()
                        or not qk_required["mean_gamma"].between(0.0, 1.0).all()
                        or not qk_required["min_gamma"].between(0.0, 1.0).all()
                        or (qk_required["min_gamma"] > qk_required["mean_gamma"]).any()
                        or (
                            qk_required["mean_max_logit"]
                            > qk_required["max_logit"]
                        ).any()
                    ):
                        raise CampaignValidationError(
                            f"{qk_path} violates the registered QK diagnostic bounds"
                        )
                    qk_frames.append(
                        _with_identity(
                            qk,
                            optimizer=optimizer,
                            seed=seed,
                            source_path=qk_path,
                        )
                    )

                metrics_frames.append(
                    _with_identity(
                        metrics,
                        optimizer=optimizer,
                        seed=seed,
                        source_path=metrics_path,
                    )
                )
                epoch_frames.append(
                    _with_identity(
                        epoch_metrics,
                        optimizer=optimizer,
                        seed=seed,
                        source_path=epoch_path,
                    )
                )
                layer_frames.append(
                    _with_identity(
                        layers,
                        optimizer=optimizer,
                        seed=seed,
                        source_path=layers_path,
                    )
                )
                spectral_summary_frames.append(
                    _with_identity(
                        spectral_summary,
                        optimizer=optimizer,
                        seed=seed,
                        source_path=spectral_summary_path,
                    )
                )
                test_frames.append(test_results)
                manifests.append(manifest)
                completions[(optimizer, seed)] = completion

                runtime = manifest.get("runtime_environment", {})
                runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
                run_rows.append({
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS[optimizer],
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "complete": True,
                    "validation_error": "",
                    "optimizer_steps": total_steps,
                    "train_epochs": completion.get("train_epochs"),
                    "elapsed_seconds": completion.get("elapsed_seconds"),
                    "best_validation_step": completion.get("best_validation_step"),
                    "best_validation_loss": completion.get("best_validation_loss"),
                    "final_test_loss": completion.get("final_test_loss"),
                    "final_test_perplexity": completion.get("final_test_perplexity"),
                    "final_test_accuracy": completion.get("final_test_accuracy"),
                    "final_test_bleu": completion.get("final_test_bleu"),
                    "protocol_fingerprint": completion.get(
                        "fingerprint", manifest.get("protocol_fingerprint")
                    ),
                    "accelerator": runtime_mapping.get("accelerator"),
                    "device": manifest.get("device", runtime_mapping.get("device")),
                    "torch_version": manifest.get(
                        "torch_version", runtime_mapping.get("torch_version")
                    ),
                    "git_commit": _campaign_invariants(manifest).get("git_commit"),
                    "git_dirty": _campaign_invariants(manifest).get("git_dirty"),
                })
            except (CampaignValidationError, KeyError, TypeError, ValueError) as exc:
                message = f"optimizer={optimizer} seed={seed}: {exc}"
                errors.append(message)
                run_rows.append({
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS[optimizer],
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "complete": False,
                    "validation_error": message,
                })

    if errors and require_complete:
        raise CampaignValidationError(
            "the exact 2 x 5 campaign is incomplete or invalid:\n- "
            + "\n- ".join(errors)
        )
    if require_complete and len(manifests) != len(OPTIMIZERS) * len(SEEDS):
        raise CampaignValidationError(
            f"loaded {len(manifests)} valid runs, expected exactly "
            f"{len(OPTIMIZERS) * len(SEEDS)}"
        )
    _validate_matched_campaign(manifests, allow_mixed_runtime=allow_mixed_runtime)

    def combine(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
        return (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )

    return {
        "campaign_runs": pd.DataFrame(run_rows),
        "metrics": combine(metrics_frames),
        "epoch_metrics": combine(epoch_frames),
        "spectral_layers": combine(layer_frames),
        "spectral_summary": combine(spectral_summary_frames),
        "test_results": combine(test_frames),
        "qk": combine(qk_frames),
        "manifests": manifests,
        "completions": completions,
        "run_dirs": run_dirs,
        "validation_errors": errors,
        "unexpected_runs": unexpected,
    }


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    mean = float(array.mean())
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "sd": np.nan,
            "sem": np.nan,
            "ci95_half_width": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    sd = float(array.std(ddof=1))
    sem = sd / math.sqrt(n)
    critical_values = {
        2: T_975_DF1,
        3: T_975_DF2,
        4: T_975_DF3,
        5: T_975_DF4,
    }
    if n not in critical_values:
        raise CampaignValidationError(
            f"Student-t summary expected at most five seeds, observed n={n}"
        )
    critical = critical_values[n]
    half = critical * sem
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_half_width": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _summarize_performance(
    test_results: pd.DataFrame,
    epoch_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not test_results.empty:
        for (optimizer, checkpoint), group in test_results.groupby(
            ["optimizer", "checkpoint"], sort=True
        ):
            for metric in TEST_RESULT_METRICS:
                statistics = _stats(group[metric])
                rows.append({
                    "source": "test_results",
                    "optimizer": optimizer,
                    "optimizer_label": OPTIMIZER_LABELS[optimizer],
                    "checkpoint": checkpoint,
                    "metric": metric,
                    "valid_exact_n5": bool(statistics["n"] == 5),
                    **statistics,
                })

    if not epoch_metrics.empty:
        final_rows = (
            epoch_metrics.sort_values(["optimizer", "seed", "step"])
            .groupby(["optimizer", "seed"], as_index=False, sort=True)
            .tail(1)
        )
        for optimizer, group in final_rows.groupby("optimizer", sort=True):
            for metric in FINITE_EPOCH_METRICS:
                if metric in group.columns:
                    statistics = _stats(group[metric])
                    rows.append({
                        "source": "final_permanent_checkpoint",
                        "optimizer": optimizer,
                        "optimizer_label": OPTIMIZER_LABELS[optimizer],
                        "checkpoint": "final",
                        "metric": metric,
                        "valid_exact_n5": bool(statistics["n"] == 5),
                        **statistics,
                    })
    return pd.DataFrame(rows)


def _paired_seed_differences(
    test_results: pd.DataFrame,
    epoch_metrics: pd.DataFrame,
    *,
    require_complete: bool,
) -> pd.DataFrame:
    sources: list[pd.DataFrame] = []
    if not test_results.empty:
        melted = test_results.melt(
            id_vars=["optimizer", "seed", "checkpoint"],
            value_vars=list(TEST_RESULT_METRICS),
            var_name="metric",
            value_name="value",
        )
        melted["source"] = "test_results"
        sources.append(melted)

    if not epoch_metrics.empty:
        final_rows = (
            epoch_metrics.sort_values(["optimizer", "seed", "step"])
            .groupby(["optimizer", "seed"], as_index=False, sort=True)
            .tail(1)
        )
        value_columns = [
            name for name in FINITE_EPOCH_METRICS if name in final_rows.columns
        ]
        melted = final_rows.melt(
            id_vars=["optimizer", "seed"],
            value_vars=value_columns,
            var_name="metric",
            value_name="value",
        )
        melted["checkpoint"] = "final"
        melted["source"] = "final_permanent_checkpoint"
        sources.append(melted)

    if not sources:
        return pd.DataFrame()
    values = pd.concat(sources, ignore_index=True, sort=False)
    contrasts = tuple(combinations(OPTIMIZERS, 2))
    rows: list[dict[str, Any]] = []
    for (source, checkpoint, metric), group in values.groupby(
        ["source", "checkpoint", "metric"], sort=True
    ):
        for optimizer_a, optimizer_b in contrasts:
            left = group[group["optimizer"].eq(optimizer_a)][["seed", "value"]]
            right = group[group["optimizer"].eq(optimizer_b)][["seed", "value"]]
            paired = left.merge(
                right,
                on="seed",
                how="inner",
                suffixes=("_a", "_b"),
            ).sort_values("seed")
            paired["difference_b_minus_a"] = paired["value_b"] - paired["value_a"]
            finite = paired[np.isfinite(pd.to_numeric(
                paired["difference_b_minus_a"], errors="coerce"
            ))]
            if require_complete and tuple(finite["seed"].astype(int)) != SEEDS:
                raise CampaignValidationError(
                    f"paired contrast {optimizer_b}-{optimizer_a}, source={source}, "
                    f"checkpoint={checkpoint}, metric={metric} lacks exact seeds {SEEDS}"
                )
            statistics = _stats(finite["difference_b_minus_a"])
            row: dict[str, Any] = {
                "source": source,
                "checkpoint": checkpoint,
                "metric": metric,
                "optimizer_a": optimizer_a,
                "optimizer_b": optimizer_b,
                "contrast": f"{optimizer_b} minus {optimizer_a}",
                "difference_definition": "optimizer_b - optimizer_a",
                "paired_seeds": ",".join(str(value) for value in finite["seed"]),
                "valid_exact_n5": bool(statistics["n"] == 5),
                **statistics,
            }
            for _, paired_row in finite.iterrows():
                seed = int(paired_row["seed"])
                row[f"seed_{seed}_a"] = float(paired_row["value_a"])
                row[f"seed_{seed}_b"] = float(paired_row["value_b"])
                row[f"seed_{seed}_difference"] = float(
                    paired_row["difference_b_minus_a"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _alpha_run_medians(layers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    keys = ["optimizer", "seed", "step", "epoch"]
    for values, group in layers.groupby(keys, sort=True):
        optimizer, seed, step, epoch = values
        observed = set(group["matrix_type"].astype(str))
        if observed != set(MATRIX_TYPES) or len(group) != len(MATRIX_TYPES):
            raise CampaignValidationError(
                f"cannot form six-matrix run median for {values}: {sorted(observed)}"
            )
        raw = pd.to_numeric(group["alpha_raw"], errors="coerce")
        clipped = pd.to_numeric(group["alpha_clip_xmax"], errors="coerce")
        rows.append({
            "optimizer": optimizer,
            "optimizer_label": OPTIMIZER_LABELS[optimizer],
            "seed": int(seed),
            "step": int(step),
            "epoch": float(epoch),
            "matrix_count": int(len(group)),
            "alpha_raw_finite_count": int(np.isfinite(raw).sum()),
            "alpha_clip_xmax_finite_count": int(np.isfinite(clipped).sum()),
            "alpha_raw_six_matrix_median": float(raw.median(skipna=True)),
            "alpha_clip_xmax_six_matrix_median": float(clipped.median(skipna=True)),
            "alpha_clip_minus_raw_six_matrix_median": float(
                clipped.median(skipna=True) - raw.median(skipna=True)
            ),
        })
    run_medians = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    value_columns = (
        "alpha_raw_six_matrix_median",
        "alpha_clip_xmax_six_matrix_median",
        "alpha_clip_minus_raw_six_matrix_median",
    )
    for (optimizer, step, epoch), group in run_medians.groupby(
        ["optimizer", "step", "epoch"], sort=True
    ):
        for metric in value_columns:
            statistics = _stats(group[metric])
            summary_rows.append({
                "optimizer": optimizer,
                "optimizer_label": OPTIMIZER_LABELS[optimizer],
                "step": int(step),
                "epoch": float(epoch),
                "metric": metric,
                "aggregation_order": (
                    "median across six matrices within each seeded run, "
                    "then Student-t summary across seeds"
                ),
                "valid_exact_n5": bool(statistics["n"] == 5),
                **statistics,
            })
    return run_medians, pd.DataFrame(summary_rows)


def _validation_saturation(
    epoch_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Diagnose plateaus from permanent-state validation loss only."""

    run_rows: list[dict[str, Any]] = []
    integer_rows: list[dict[str, Any]] = []
    for (optimizer, seed), group in epoch_metrics.groupby(
        ["optimizer", "seed"], sort=True
    ):
        observations = group[["step", "epoch", "val_loss"]].copy()
        observations = _numeric(observations, ("step", "epoch", "val_loss"))
        observations = observations.dropna(subset=["epoch", "val_loss"])
        observations = observations[np.isfinite(observations["val_loss"])]
        observations = observations.sort_values(["epoch", "step"])
        if observations.empty:
            raise CampaignValidationError(
                f"no finite validation loss for optimizer={optimizer} seed={seed}"
            )
        maximum_epoch = float(observations["epoch"].max())
        targets = range(0, int(math.floor(maximum_epoch + 1e-9)) + 1)
        selected: list[dict[str, Any]] = []
        used_steps: set[int] = set()
        for target in targets:
            distance = (observations["epoch"] - float(target)).abs()
            index = distance.idxmin()
            row = observations.loc[index]
            if float(distance.loc[index]) > 0.15:
                continue
            step = int(row["step"])
            if step in used_steps:
                continue
            used_steps.add(step)
            selected.append({
                "optimizer": optimizer,
                "optimizer_label": OPTIMIZER_LABELS[optimizer],
                "seed": int(seed),
                "target_epoch": int(target),
                "actual_epoch": float(row["epoch"]),
                "step": step,
                "val_loss": float(row["val_loss"]),
            })
        selected_frame = pd.DataFrame(selected).sort_values("target_epoch")
        if selected_frame.empty:
            raise CampaignValidationError(
                f"could not select integer-epoch validation points for {optimizer}/{seed}"
            )
        selected_frame["validation_improvement_nats"] = (
            selected_frame["val_loss"].shift(1) - selected_frame["val_loss"]
        )
        selected_frame["near_flat_interval"] = (
            selected_frame["validation_improvement_nats"].abs()
            <= SATURATION_DELTA_NATS
        )
        selected_frame["degrading_interval"] = (
            selected_frame["validation_improvement_nats"]
            < -SATURATION_DELTA_NATS
        )
        selected_frame["two_consecutive_near_flat"] = (
            selected_frame["near_flat_interval"]
            & selected_frame["near_flat_interval"].shift(1, fill_value=False)
        )
        selected_frame["two_consecutive_degrading"] = (
            selected_frame["degrading_interval"]
            & selected_frame["degrading_interval"].shift(1, fill_value=False)
        )
        integer_rows.extend(selected_frame.to_dict("records"))

        final_two_intervals_flat = bool(
            len(selected_frame) >= 3
            and selected_frame["near_flat_interval"].iloc[-2:].all()
        )
        degradation_rows = selected_frame[selected_frame["two_consecutive_degrading"]]
        best_index = observations["val_loss"].idxmin()
        best = observations.loc[best_index]
        final = observations.iloc[-1]
        run_rows.append({
            "optimizer": optimizer,
            "optimizer_label": OPTIMIZER_LABELS[optimizer],
            "seed": int(seed),
            "criterion": (
                "validation-only: abs(one-epoch NLL improvement) <= "
                f"{SATURATION_DELTA_NATS:.3f} nat/token for each of the final "
                "two complete intervals"
            ),
            "plateau_detected": final_two_intervals_flat,
            "plateau_assessment_end_epoch": (
                float(selected_frame.iloc[-1]["target_epoch"])
                if final_two_intervals_flat
                else np.nan
            ),
            "degradation_detected": bool(not degradation_rows.empty),
            "first_degradation_end_epoch": (
                float(degradation_rows.iloc[0]["target_epoch"])
                if not degradation_rows.empty
                else np.nan
            ),
            "best_validation_step": int(best["step"]),
            "best_validation_epoch": float(best["epoch"]),
            "best_validation_loss": float(best["val_loss"]),
            "final_validation_step": int(final["step"]),
            "final_validation_epoch": float(final["epoch"]),
            "final_validation_loss": float(final["val_loss"]),
            "validation_observations": int(len(observations)),
            "integer_epoch_observations": int(len(selected_frame)),
            "test_metrics_used": False,
        })

    diagnostics = pd.DataFrame(run_rows)
    integer_diagnostics = pd.DataFrame(integer_rows)
    summary_rows: list[dict[str, Any]] = []
    for optimizer, group in diagnostics.groupby("optimizer", sort=True):
        for metric in (
            "plateau_assessment_end_epoch",
            "first_degradation_end_epoch",
            "best_validation_epoch",
            "best_validation_loss",
            "final_validation_loss",
        ):
            summary_rows.append({
                "optimizer": optimizer,
                "optimizer_label": OPTIMIZER_LABELS[optimizer],
                "metric": metric,
                **_stats(group[metric]),
            })
    return diagnostics, integer_diagnostics, pd.DataFrame(summary_rows)


def _qk_summary(qk: pd.DataFrame) -> pd.DataFrame:
    if qk.empty:
        return pd.DataFrame(columns=(
            "optimizer",
            "seed",
            "diagnostic_rows",
            "last_step",
            "active_fraction_weighted",
            "max_logit_observed",
            "min_gamma_observed",
        ))
    rows: list[dict[str, Any]] = []
    for (optimizer, seed), group in qk.groupby(["optimizer", "seed"], sort=True):
        weights = (
            pd.to_numeric(group["head_observations"], errors="coerce")
            if "head_observations" in group.columns
            else pd.Series(1.0, index=group.index)
        ).fillna(0.0).clip(lower=0.0)
        active = pd.to_numeric(group["active_fraction"], errors="coerce")
        finite = np.isfinite(active) & np.isfinite(weights)
        weighted_active = (
            float(np.average(active[finite], weights=weights[finite]))
            if finite.any() and float(weights[finite].sum()) > 0.0
            else float(active[finite].mean()) if finite.any() else np.nan
        )
        rows.append({
            "optimizer": optimizer,
            "optimizer_label": OPTIMIZER_LABELS[optimizer],
            "seed": int(seed),
            "diagnostic_rows": int(len(group)),
            "last_step": float(pd.to_numeric(group["step"], errors="coerce").max()),
            "active_fraction_weighted": weighted_active,
            "max_logit_observed": float(
                pd.to_numeric(group["max_logit"], errors="coerce").max()
            ),
            "min_gamma_observed": float(
                pd.to_numeric(group["min_gamma"], errors="coerce").min()
            ),
        })
    return pd.DataFrame(rows)


def _resolve_checkpoint(recorded: Any, run_dir: Path) -> Path:
    path = Path(str(recorded))
    candidates = (
        path,
        run_dir / "epoch_checkpoints" / path.name,
        run_dir / path.name,
    )
    for candidate in candidates:
        if candidate.is_file() and _is_within(candidate, run_dir):
            return candidate.resolve()
    raise CampaignValidationError(
        f"could not resolve recorded checkpoint {recorded!r} below {run_dir}"
    )


def _checkpoint_index(
    results_root: Path,
    epoch_metrics: pd.DataFrame,
    run_dirs: Mapping[tuple[str, int], Path],
    completions: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    require_complete: bool,
) -> pd.DataFrame:
    inventory: dict[tuple[str, int, Path], dict[str, Any]] = {}

    def register(
        optimizer: str,
        seed: int,
        path: Path,
        role: str,
        *,
        step: int | None = None,
        epoch: float | None = None,
    ) -> None:
        resolved = path.resolve(strict=False)
        key = (optimizer, int(seed), resolved)
        record = inventory.setdefault(key, {
            "optimizer": optimizer,
            "optimizer_label": OPTIMIZER_LABELS[optimizer],
            "seed": int(seed),
            "checkpoint_path": str(resolved),
            "roles": set(),
            "steps": set(),
            "epochs": set(),
        })
        record["roles"].add(role)
        if step is not None:
            record["steps"].add(int(step))
        if epoch is not None and math.isfinite(float(epoch)):
            record["epochs"].add(float(epoch))

    for (optimizer, seed), run_dir in run_dirs.items():
        if not run_dir.is_dir():
            continue
        completion = completions.get((optimizer, seed), {})
        standard = {
            "initial": run_dir / "checkpoint_initial.pt",
            "latest": run_dir / "checkpoint_latest.pt",
            "best": run_dir / "checkpoint_best.pt",
            "final": run_dir / "checkpoint_final.pt",
        }
        for role, path in standard.items():
            if not path.is_file():
                if require_complete:
                    raise CampaignValidationError(
                        f"missing required {role} checkpoint: {path}"
                    )
                continue
            step: int | None = None
            if role in {"latest", "final"} and "optimizer_steps" in completion:
                step = int(completion["optimizer_steps"])
            elif role == "best" and "best_validation_step" in completion:
                step = int(completion["best_validation_step"])
            elif role == "initial":
                step = 0
            register(optimizer, seed, path, role, step=step)

    for _, row in epoch_metrics.iterrows():
        optimizer = str(row["optimizer"])
        seed = int(row["seed"])
        run_dir = run_dirs[(optimizer, seed)]
        try:
            path = _resolve_checkpoint(row["checkpoint_path"], run_dir)
        except CampaignValidationError:
            if require_complete:
                raise
            continue
        register(
            optimizer,
            seed,
            path,
            "permanent_epoch",
            step=int(row["step"]),
            epoch=float(row.get("nominal_epoch", row.get("epoch", np.nan))),
        )

    permanent_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for record in inventory.values():
        path = Path(record["checkpoint_path"])
        if not path.is_file():
            if require_complete:
                raise CampaignValidationError(f"checkpoint disappeared: {path}")
            continue
        roles = sorted(record.pop("roles"))
        steps = sorted(record.pop("steps"))
        epochs = sorted(record.pop("epochs"))
        if "permanent_epoch" in roles:
            permanent_counts[(record["optimizer"], record["seed"])] += 1
        try:
            relative = str(path.resolve().relative_to(results_root.resolve()))
        except ValueError:
            relative = str(path.resolve())
        rows.append({
            **record,
            "checkpoint_relative_path": relative,
            "roles": ",".join(roles),
            "steps": ",".join(str(value) for value in steps),
            "epochs": ",".join(f"{value:.12g}" for value in epochs),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        })
    if require_complete:
        for optimizer in OPTIMIZERS:
            for seed in SEEDS:
                count = permanent_counts[(optimizer, seed)]
                if count < MIN_PERMANENT_CHECKPOINTS:
                    raise CampaignValidationError(
                        f"checkpoint index found {count} permanent checkpoints for "
                        f"{optimizer}/{seed}, expected at least {MIN_PERMANENT_CHECKPOINTS}"
                    )
    return pd.DataFrame(rows).sort_values(
        ["optimizer", "seed", "checkpoint_relative_path"]
    ).reset_index(drop=True)


def _curve_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = frame[["epoch", "seed", metric]].copy()
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected[np.isfinite(selected[metric])]
    for epoch, group in selected.groupby("epoch", sort=True):
        rows.append({"epoch": float(epoch), **_stats(group[metric])})
    return pd.DataFrame(rows)


def _plot_metric_curve(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    *,
    label: str,
    color: str,
    scale: float = 1.0,
    linestyle: str = "-",
) -> None:
    if metric not in frame.columns:
        return
    finite = frame[np.isfinite(pd.to_numeric(frame[metric], errors="coerce"))].copy()
    if finite.empty:
        return
    finite[metric] = pd.to_numeric(finite[metric], errors="coerce") * float(scale)
    for _, seed_frame in finite.groupby("seed", sort=True):
        seed_frame = seed_frame.sort_values("epoch")
        axis.plot(
            seed_frame["epoch"].to_numpy(dtype=float),
            seed_frame[metric].to_numpy(dtype=float),
            color=color,
            alpha=0.20,
            linewidth=0.8,
            linestyle=linestyle,
        )
    summary = _curve_summary(finite, metric)
    if summary.empty:
        return
    x = summary["epoch"].to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    low = summary["ci95_low"].to_numpy(dtype=float)
    high = summary["ci95_high"].to_numpy(dtype=float)
    axis.plot(
        x,
        mean,
        color=color,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
    )
    valid = np.isfinite(low) & np.isfinite(high)
    if valid.any():
        axis.fill_between(x[valid], low[valid], high[valid], color=color, alpha=0.12)


def _plot_posthoc_test_metric(
    axis: plt.Axes,
    test_results: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
) -> None:
    order = ("validation_selected", "final")
    labels = ("validation-selected", "final")
    for index, checkpoint in enumerate(order):
        values = pd.to_numeric(
            test_results.loc[
                test_results["checkpoint"].eq(checkpoint), metric
            ],
            errors="coerce",
        )
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        x_values = np.full(len(values), float(index))
        axis.scatter(
            x_values,
            values.to_numpy(dtype=float),
            color="#6B7280",
            alpha=0.65,
            s=24,
            zorder=3,
        )
        statistics = _stats(values)
        axis.errorbar(
            [index],
            [statistics["mean"]],
            yerr=[[statistics["ci95_half_width"]], [statistics["ci95_half_width"]]],
            color="#D55E00",
            marker="o",
            capsize=4,
            linewidth=1.5,
            zorder=4,
        )
    axis.set_xticks(range(len(order)), labels)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)


def _optimizer_performance_plot(
    metrics: pd.DataFrame,
    test_results: pd.DataFrame,
    *,
    optimizer: str,
    path: Path,
) -> None:
    frame = metrics[metrics["optimizer"].eq(optimizer)].copy()
    if frame.empty:
        return
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    flat_axes = axes.ravel()
    panels = (
        (
            "Cross-entropy loss",
            (
                ("train_loss", "train", SPLIT_COLORS["train"], 1.0, "-"),
                ("val_loss", "validation", SPLIT_COLORS["val"], 1.0, "-"),
            ),
            "NLL (nats/token)",
        ),
        (
            "Perplexity",
            (
                ("train_perplexity", "train", SPLIT_COLORS["train"], 1.0, "-"),
                ("val_perplexity", "validation", SPLIT_COLORS["val"], 1.0, "-"),
            ),
            "perplexity",
        ),
        (
            "Next-token top-1 accuracy",
            (
                ("train_accuracy", "train", SPLIT_COLORS["train"], 100.0, "-"),
                ("val_accuracy", "validation", SPLIT_COLORS["val"], 100.0, "-"),
            ),
            "token accuracy (%)",
        ),
        (
            "Generalization gaps",
            (
                ("val_generalization_gap", "validation - train", SPLIT_COLORS["val"], 1.0, "-"),
            ),
            "loss gap (nats/token)",
        ),
    )
    for axis, (title, curves, ylabel) in zip(flat_axes[:4], panels, strict=True):
        for metric, label, color, scale, linestyle in curves:
            _plot_metric_curve(
                axis,
                frame,
                metric,
                label=label,
                color=color,
                scale=scale,
                linestyle=linestyle,
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("corpus-equivalent epoch")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(frameon=False, fontsize=8)
    optimizer_tests = test_results[test_results["optimizer"].eq(optimizer)]
    _plot_posthoc_test_metric(
        flat_axes[4],
        optimizer_tests,
        "test_loss",
        title="Held-out post-training test NLL",
        ylabel="NLL (nats/token)",
    )
    _plot_posthoc_test_metric(
        flat_axes[5],
        optimizer_tests,
        "test_bleu",
        title="Held-out greedy-continuation BLEU",
        ylabel="corpus BLEU",
    )
    figure.suptitle(
        f"{OPTIMIZER_LABELS[optimizer]}: seeded runs (mean and 95% Student-t CI)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _atomic_figure(path, figure)
    plt.close(figure)


def _alpha_variant_summary(
    frame: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    return _curve_summary(frame, metric)


def _optimizer_alpha_plot(
    layers: pd.DataFrame,
    *,
    optimizer: str,
    path: Path,
) -> None:
    selected = layers[layers["optimizer"].eq(optimizer)].copy()
    if selected.empty:
        return
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    variants = (
        ("alpha_raw", "raw alpha", "#6B7280", "--"),
        ("alpha_clip_xmax", "clip_xmax alpha", OPTIMIZER_COLORS[optimizer], "-"),
    )
    for axis, matrix in zip(axes.flat, MATRIX_TYPES, strict=True):
        matrix_frame = selected[selected["matrix_type"].eq(matrix)].copy()
        for metric, label, color, linestyle in variants:
            for _, seed_frame in matrix_frame.groupby("seed", sort=True):
                seed_frame = seed_frame.sort_values("epoch")
                axis.plot(
                    seed_frame["epoch"].to_numpy(dtype=float),
                    seed_frame[metric].to_numpy(dtype=float),
                    color=color,
                    alpha=0.20,
                    linewidth=0.8,
                    linestyle=linestyle,
                )
            summary = _alpha_variant_summary(matrix_frame, metric)
            if summary.empty:
                continue
            x = summary["epoch"].to_numpy(dtype=float)
            mean = summary["mean"].to_numpy(dtype=float)
            low = summary["ci95_low"].to_numpy(dtype=float)
            high = summary["ci95_high"].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
                label=label,
            )
            valid = np.isfinite(low) & np.isfinite(high)
            if valid.any():
                axis.fill_between(
                    x[valid], low[valid], high[valid], color=color, alpha=0.12
                )
        axis.axhline(2.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(matrix)
        axis.set_ylabel("WeightWatcher alpha")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("corpus-equivalent epoch")
    figure.suptitle(
        f"{OPTIMIZER_LABELS[optimizer]}: raw versus clip_xmax alpha by matrix",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _atomic_figure(path, figure)
    plt.close(figure)


def _optimizer_erg_plot(
    layers: pd.DataFrame,
    *,
    optimizer: str,
    path: Path,
) -> None:
    selected = layers[layers["optimizer"].eq(optimizer)].copy()
    if selected.empty:
        return
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    diagnostics = (
        ("ERG_gap", "ERG gap", OPTIMIZER_COLORS[optimizer], "-"),
        ("num_traps", "correlation traps", "#6B7280", "--"),
    )
    for axis, matrix in zip(axes.flat, MATRIX_TYPES, strict=True):
        matrix_frame = selected[selected["matrix_type"].eq(matrix)].copy()
        for metric, label, color, linestyle in diagnostics:
            _plot_metric_curve(
                axis,
                matrix_frame,
                metric,
                label=label,
                color=color,
                linestyle=linestyle,
            )
        axis.set_title(matrix)
        axis.set_ylabel("WeightWatcher diagnostic")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(frameon=False, fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("corpus-equivalent epoch")
    figure.suptitle(
        f"{OPTIMIZER_LABELS[optimizer]}: ERG gap and correlation traps by matrix",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _atomic_figure(path, figure)
    plt.close(figure)


def _format_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    if abs(numeric) >= 1_000 or (0 < abs(numeric) < 1e-3):
        return f"{numeric:.4g}"
    return f"{numeric:.5f}"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in selected.iterrows():
        values = [
            str(_format_float(row[column])).replace("|", "\\|")
            for column in headers
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _table_html(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if frame.empty:
        return "<p><em>No rows available.</em></p>"
    selected = frame if columns is None else frame[
        [column for column in columns if column in frame.columns]
    ]
    return selected.to_html(
        index=False,
        border=0,
        classes="dataframe",
        na_rep="",
        float_format=lambda value: _format_float(value),
        escape=True,
    )


def _latest_alpha_summary(alpha_summary: pd.DataFrame) -> pd.DataFrame:
    if alpha_summary.empty:
        return alpha_summary
    latest_rows: list[pd.DataFrame] = []
    for optimizer, group in alpha_summary.groupby("optimizer", sort=True):
        latest_epoch = group["epoch"].max()
        latest_rows.append(group[group["epoch"].eq(latest_epoch)])
    return pd.concat(latest_rows, ignore_index=True, sort=False)


def _write_summary(
    output_root: Path,
    *,
    campaign_runs: pd.DataFrame,
    performance: pd.DataFrame,
    paired: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    saturation: pd.DataFrame,
    qk_summary: pd.DataFrame,
    warnings: Sequence[str],
) -> Path:
    primary_performance = performance[
        performance["source"].eq("test_results")
        & performance["checkpoint"].eq("validation_selected")
    ].copy()
    primary_performance = primary_performance[
        primary_performance["metric"].isin(TEST_RESULT_METRICS)
    ]
    primary_pairs = paired[
        paired["source"].eq("test_results")
        & paired["checkpoint"].eq("validation_selected")
        & paired["metric"].isin(("test_loss", "test_accuracy"))
    ].copy()
    latest_alpha = _latest_alpha_summary(alpha_summary)
    lines = [
        "# One-head nanoGPT baseline — 2026-08-21",
        "",
        "This report validates the exact AdamW/MuonClip × five-seed "
        "campaign. The complete seeded run is the unit of replication. Test "
        "results are reported only as post-run outcomes; the saturation "
        "diagnostic uses validation loss exclusively.",
        "",
        "Metric definitions: accuracy is fixed-probe next-token top-1 "
        "accuracy (not classification or whole-sequence accuracy); top-5 "
        "accuracy is reported separately. Perplexity is exp(mean token NLL) "
        "and bits/token is NLL/log(2). BLEU is fixed greedy held-out "
        "continuation BLEU, a secondary language-model diagnostic—not "
        "translation BLEU. Continuation token accuracy and exact match are "
        "reported alongside it.",
        "",
        "## Campaign status",
        "",
        _markdown_table(
            campaign_runs,
            (
                "optimizer",
                "seed",
                "complete",
                "optimizer_steps",
                "train_epochs",
                "best_validation_loss",
                "final_test_loss",
                "accelerator",
                "git_commit",
            ),
        ),
        "",
        "## Validation-selected test performance",
        "",
        _markdown_table(
            primary_performance,
            (
                "optimizer",
                "metric",
                "n",
                "mean",
                "sd",
                "ci95_low",
                "ci95_high",
            ),
        ),
        "",
        "## Paired seed differences",
        "",
        "Differences are always `optimizer_b - optimizer_a`; every valid row "
        "uses the same five seeds and a df=4 Student-t interval.",
        "",
        _markdown_table(
            primary_pairs,
            (
                "metric",
                "contrast",
                "n",
                "mean",
                "sd",
                "ci95_low",
                "ci95_high",
            ),
        ),
        "",
        "## Validation-only saturation",
        "",
        _markdown_table(
            saturation,
            (
                "optimizer",
                "seed",
                "plateau_detected",
                "plateau_assessment_end_epoch",
                "degradation_detected",
                "first_degradation_end_epoch",
                "best_validation_epoch",
                "best_validation_loss",
                "final_validation_loss",
                "test_metrics_used",
            ),
        ),
        "",
        "## Final six-matrix alpha summaries",
        "",
        "Each seeded value is first the median over exactly six transformer "
        "matrices. The displayed interval is then computed across the five "
        "seeded run medians.",
        "",
        _markdown_table(
            latest_alpha,
            (
                "optimizer",
                "epoch",
                "metric",
                "n",
                "mean",
                "sd",
                "ci95_low",
                "ci95_high",
            ),
        ),
        "",
        "## MuonClip QK diagnostics",
        "",
        _markdown_table(
            qk_summary,
            (
                "optimizer",
                "seed",
                "active_fraction_weighted",
                "max_logit_observed",
                "min_gamma_observed",
            ),
        ),
        "",
        "## Artifacts",
        "",
    ]
    for filename in CSV_ARTIFACTS:
        lines.append(f"- `{filename}`")
    lines.extend([
        "- `plots/`",
        "- `report.html`",
        "- `results_manifest.json`",
    ])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    destination = output_root / "SUMMARY.md"
    _atomic_write_text(destination, "\n".join(lines) + "\n")
    return destination


def _write_html_report(
    output_root: Path,
    *,
    campaign_runs: pd.DataFrame,
    performance: pd.DataFrame,
    paired: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    saturation: pd.DataFrame,
    qk_summary: pd.DataFrame,
    checkpoint_index: pd.DataFrame,
    warnings: Sequence[str],
) -> Path:
    selected_performance = performance[
        performance["source"].eq("test_results")
    ]
    selected_pairs = paired[
        paired["source"].eq("test_results")
        & paired["metric"].isin(("test_loss", "test_accuracy"))
    ]
    latest_alpha = _latest_alpha_summary(alpha_summary)
    artifact_links = "\n".join(
        f'<li><a href="{html.escape(filename)}">{html.escape(filename)}</a></li>'
        for filename in CSV_ARTIFACTS
    )
    image_sections: list[str] = []
    for optimizer in OPTIMIZERS:
        label = html.escape(OPTIMIZER_LABELS[optimizer])
        performance_path = f"plots/{optimizer}_performance.png"
        alpha_path = f"plots/{optimizer}_alpha_raw_vs_clip_xmax.png"
        erg_path = f"plots/{optimizer}_erg_gap_num_traps.png"
        image_sections.append(
            f"<h3>{label}</h3>"
            f'<img src="{performance_path}" alt="{label} performance trajectories">'
            f'<img src="{alpha_path}" alt="{label} raw versus clipped alpha">'
            f'<img src="{erg_path}" alt="{label} ERG gap and correlation traps">'
        )
    warning_html = (
        "<h2>Warnings</h2><ul>"
        + "".join(f"<li>{html.escape(value)}</li>" for value in warnings)
        + "</ul>"
        if warnings
        else ""
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>One-head nanoGPT baseline — 2026-08-21</title>
<style>
body {{ max-width: 1500px; margin: 0 auto; padding: 2rem; font-family: system-ui, sans-serif; color: #1f2937; }}
h1, h2, h3 {{ color: #111827; }}
.note {{ background: #f3f4f6; border-left: 4px solid #4b5563; padding: 0.8rem 1rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; margin-bottom: 1.5rem; }}
th, td {{ border: 1px solid #d1d5db; padding: 0.35rem 0.5rem; text-align: right; }}
th {{ background: #f3f4f6; position: sticky; top: 0; }}
td:first-child, th:first-child {{ text-align: left; }}
img {{ display: block; width: 100%; height: auto; margin: 0.75rem 0 2rem; border: 1px solid #e5e7eb; }}
code {{ background: #f3f4f6; padding: 0.1rem 0.25rem; }}
</style>
</head>
<body>
<h1>One-head nanoGPT baseline — 2026-08-21</h1>
<p class="note">Exact campaign: AdamW and MuonClip; seeds 1337, 2027, 4099, 31415, and 271828. The run is the replicate. Test quantities never enter the saturation calculation. A 95% interval with n=5 uses Student-t critical value 2.77645.</p>
<h2>Campaign validation</h2>
{_table_html(campaign_runs, ("optimizer", "seed", "complete", "optimizer_steps", "train_epochs", "best_validation_loss", "final_test_loss", "accelerator", "torch_version", "git_commit"))}
<h2>Test performance</h2>
<p>Accuracy is next-token top-1 accuracy on a fixed held-out probe, not a classification or sequence-level accuracy. Top-5 accuracy is separate. Perplexity is exp(mean token NLL), and bits/token is NLL/log(2). BLEU is a secondary fixed greedy held-out continuation diagnostic—not translation BLEU—and is accompanied by continuation token accuracy and exact match.</p>
{_table_html(selected_performance, ("optimizer", "checkpoint", "metric", "n", "mean", "sd", "ci95_low", "ci95_high"))}
<h2>Paired seeded differences</h2>
<p>Every difference is <code>optimizer_b - optimizer_a</code>. Positive and negative values must be interpreted according to whether higher or lower is preferable for the metric.</p>
{_table_html(selected_pairs, ("checkpoint", "metric", "contrast", "n", "mean", "sd", "ci95_low", "ci95_high", "valid_exact_n5"))}
<h2>Validation-only saturation diagnostic</h2>
<p>Plateau means |one-epoch validation-NLL improvement| ≤ {SATURATION_DELTA_NATS:.3f} nat/token for two consecutive intervals. Runs are not stopped by this diagnostic.</p>
{_table_html(saturation, ("optimizer", "seed", "plateau_detected", "plateau_assessment_end_epoch", "degradation_detected", "first_degradation_end_epoch", "best_validation_epoch", "best_validation_loss", "final_validation_loss", "test_metrics_used"))}
<h2>Final WeightWatcher alpha</h2>
<p>The six layer values are reduced to one median inside each seeded run before any across-seed mean or confidence interval is computed.</p>
{_table_html(latest_alpha, ("optimizer", "epoch", "metric", "n", "mean", "sd", "ci95_low", "ci95_high", "valid_exact_n5"))}
<h2>MuonClip QK diagnostics</h2>
{_table_html(qk_summary)}
<h2>Plots</h2>
{''.join(image_sections)}
<h2>Checkpoint integrity</h2>
<p>{len(checkpoint_index):,} checkpoint files were indexed by byte size and SHA-256. See <a href="checkpoint_sha256.csv">checkpoint_sha256.csv</a>.</p>
<h2>Machine-readable artifacts</h2>
<ul>{artifact_links}</ul>
{warning_html}
</body>
</html>
"""
    destination = output_root / "report.html"
    _atomic_write_text(destination, document)
    return destination


def _input_artifact_manifest(
    results_root: Path,
    *,
    run_dirs: Mapping[tuple[str, int], Path],
    checkpoint_index: pd.DataFrame,
) -> list[dict[str, Any]]:
    candidates: set[Path] = set()
    for _identity, run_dir in run_dirs.items():
        if not run_dir.is_dir():
            continue
        for relative in REQUIRED_RUN_FILES:
            path = run_dir / relative
            if path.is_file():
                candidates.add(path.resolve())
        qk_path = run_dir / "muonclip_qk.csv"
        if qk_path.is_file():
            candidates.add(qk_path.resolve())
        candidates.update(path.resolve() for path in (run_dir / "spectral").glob(
            "status_step_*.json"
        ) if path.is_file())
    if "checkpoint_path" in checkpoint_index.columns:
        candidates.update(
            Path(str(value)).resolve()
            for value in checkpoint_index["checkpoint_path"]
            if Path(str(value)).is_file()
        )

    rows: list[dict[str, Any]] = []
    resolved_root = results_root.resolve()
    for path in sorted(candidates):
        if not _is_within(path, resolved_root):
            raise CampaignValidationError(
                f"report input artifact escapes results root: {path}"
            )
        rows.append({
            "path": str(path.relative_to(resolved_root)),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        })
    return rows


def _artifact_manifest(
    output_root: Path,
    *,
    results_root: Path,
    args: argparse.Namespace,
    campaign_runs: pd.DataFrame,
    checkpoint_index: pd.DataFrame,
    input_artifacts: Sequence[Mapping[str, Any]],
    source_git_commit: str,
    warnings: Sequence[str],
) -> dict[str, Any]:
    manifest_path = output_root / "results_manifest.json"
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        relative = path.relative_to(output_root)
        if (
            not path.is_file()
            or path == manifest_path
            or path.name.endswith(".tmp")
            or (relative.parts and relative.parts[0] == "notebooks")
        ):
            continue
        artifacts.append({
            "path": str(relative),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        })
    run_records = campaign_runs.to_dict("records")
    return {
        "schema_version": 2,
        "campaign": "nanogpt_one_head_2026_08_21_baseline",
        "generated_at_utc": _utc_now(),
        "report_builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source_git_commit": source_git_commit,
        "frozen_config": {
            "path": str(FROZEN_CONFIG.relative_to(REPOSITORY_ROOT)),
            "canonical_sha256": FROZEN_CONFIG_SHA256,
        },
        "results_root": str(results_root.resolve()),
        "output_root": str(output_root.resolve()),
        "exact_campaign": {
            "optimizers": list(OPTIMIZERS),
            "seeds": list(SEEDS),
            "expected_run_count": len(OPTIMIZERS) * len(SEEDS),
            "require_complete": _requires_complete(args),
            "allow_extra_runs": bool(args.allow_extra_runs),
            "allow_mixed_runtime": False,
        },
        "statistical_contract": {
            "replicate": "complete seeded run",
            "paired_seed_differences": True,
            "paired_n": 5,
            "student_t_critical_95_df4": T_975_DF4,
            "alpha_aggregation": (
                "six-matrix median within run, then across-seed Student-t summary"
            ),
            "saturation_uses_test_metrics": False,
            "saturation_delta_nats": SATURATION_DELTA_NATS,
        },
        "run_records": run_records,
        "valid_run_count": int(
            campaign_runs.get("complete", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "checkpoint_file_count": int(len(checkpoint_index)),
        "checkpoint_total_bytes": int(
            pd.to_numeric(checkpoint_index.get("bytes", pd.Series(dtype=float)), errors="coerce")
            .fillna(0)
            .sum()
        ),
        "input_artifacts": [dict(record) for record in input_artifacts],
        "artifacts": artifacts,
        "warnings": list(warnings),
        "command": list(sys.argv),
    }


def build_report(args: argparse.Namespace) -> Path:
    results_root = Path(args.results_root).resolve(strict=False)
    output_root = Path(args.output_root).resolve(strict=False)
    _validate_paths(results_root, output_root)
    if bool(getattr(args, "allow_mixed_runtime", False)):
        raise CampaignValidationError(
            "mixed-runtime reporting is disabled because the registered "
            "paired statistics require one homogeneous hardware/runtime block"
        )
    require_complete = _requires_complete(args)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)
    _initialize_matplotlib(output_root)

    campaign = _load_campaign(
        results_root,
        require_complete=require_complete,
        allow_extra_runs=args.allow_extra_runs,
        allow_mixed_runtime=False,
    )
    campaign_runs = campaign["campaign_runs"]
    metrics = campaign["metrics"]
    epoch_metrics = campaign["epoch_metrics"]
    layers = campaign["spectral_layers"]
    spectral_summary = campaign["spectral_summary"]
    test_results = campaign["test_results"]
    qk = campaign["qk"]

    performance = _summarize_performance(test_results, epoch_metrics)
    paired = _paired_seed_differences(
        test_results,
        epoch_metrics,
        require_complete=require_complete,
    )
    alpha_run_medians, alpha_summary = _alpha_run_medians(layers)
    saturation, saturation_integer, saturation_summary = _validation_saturation(
        epoch_metrics
    )
    qk_summary = _qk_summary(qk)

    print("[report] hashing checkpoints; this can take several minutes", flush=True)
    checkpoint_index = _checkpoint_index(
        results_root,
        epoch_metrics,
        campaign["run_dirs"],
        campaign["completions"],
        require_complete=require_complete,
    )

    table_map = {
        "campaign_runs.csv": campaign_runs,
        "metrics_all.csv": metrics,
        "epoch_metrics_all.csv": epoch_metrics,
        "spectral_layers_all.csv": layers,
        "spectral_summary_all.csv": spectral_summary,
        "test_results_all.csv": test_results,
        "qk_diagnostics_all.csv": qk,
        "qk_summary.csv": qk_summary,
        "performance_summary.csv": performance,
        "paired_seed_differences.csv": paired,
        "alpha_run_medians.csv": alpha_run_medians,
        "alpha_across_seed_summary.csv": alpha_summary,
        "saturation_diagnostics.csv": saturation,
        "saturation_integer_epoch_validation.csv": saturation_integer,
        "saturation_across_seed_summary.csv": saturation_summary,
        "checkpoint_sha256.csv": checkpoint_index,
    }
    for filename, frame in table_map.items():
        _atomic_csv(output_root / filename, frame)

    for optimizer in OPTIMIZERS:
        _optimizer_performance_plot(
            metrics,
            test_results,
            optimizer=optimizer,
            path=output_root / "plots" / f"{optimizer}_performance.png",
        )
        _optimizer_alpha_plot(
            layers,
            optimizer=optimizer,
            path=(
                output_root
                / "plots"
                / f"{optimizer}_alpha_raw_vs_clip_xmax.png"
            ),
        )
        _optimizer_erg_plot(
            layers,
            optimizer=optimizer,
            path=output_root / "plots" / f"{optimizer}_erg_gap_num_traps.png",
        )

    warnings = [*campaign["validation_errors"]]
    if campaign["unexpected_runs"]:
        warnings.append(
            "Unexpected seed directories were ignored: "
            + ", ".join(campaign["unexpected_runs"])
        )
    if not require_complete:
        warnings.append(
            "Incomplete runs were permitted; rows with n<5 are not valid exact "
            "five-seed comparisons."
        )

    _write_summary(
        output_root,
        campaign_runs=campaign_runs,
        performance=performance,
        paired=paired,
        alpha_summary=alpha_summary,
        saturation=saturation,
        qk_summary=qk_summary,
        warnings=warnings,
    )
    report_path = _write_html_report(
        output_root,
        campaign_runs=campaign_runs,
        performance=performance,
        paired=paired,
        alpha_summary=alpha_summary,
        saturation=saturation,
        qk_summary=qk_summary,
        checkpoint_index=checkpoint_index,
        warnings=warnings,
    )
    input_artifacts = _input_artifact_manifest(
        results_root,
        run_dirs=campaign["run_dirs"],
        checkpoint_index=checkpoint_index,
    )
    source_git_commit = str(
        _campaign_invariants(campaign["manifests"][0]).get("git_commit", "")
    )
    manifest = _artifact_manifest(
        output_root,
        results_root=results_root,
        args=args,
        campaign_runs=campaign_runs,
        checkpoint_index=checkpoint_index,
        input_artifacts=input_artifacts,
        source_git_commit=source_git_commit,
        warnings=warnings,
    )
    _atomic_json(output_root / "results_manifest.json", manifest)
    print(f"[report] complete: {report_path}", flush=True)
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and report the exact 2026-08-21 one-head nanoGPT "
            "AdamW/MuonClip five-seed campaign"
        )
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help=(
            "input results directory containing optimizer/seed_<seed> runs; "
            "must be strictly below resolved /tmp or /private/tmp"
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="report destination strictly below resolved /tmp or /private/tmp",
    )
    completion_group = parser.add_mutually_exclusive_group()
    completion_group.add_argument(
        "--require-complete",
        dest="require_complete",
        action="store_true",
        help=(
            "require all exact twenty completed runs (the default; this explicit "
            "form is used by the executed notebook contract)"
        ),
    )
    completion_group.add_argument(
        "--allow-incomplete",
        dest="require_complete",
        action="store_false",
        help=(
            "build a diagnostic report from available runs; default behavior "
            "requires all exact twenty completed runs"
        ),
    )
    parser.set_defaults(require_complete=True)
    parser.add_argument(
        "--allow-extra-runs",
        action="store_true",
        help="ignore additional seed_* directories under the two optimizer roots",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        build_report(args)
    except CampaignValidationError as exc:
        print(f"[report] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

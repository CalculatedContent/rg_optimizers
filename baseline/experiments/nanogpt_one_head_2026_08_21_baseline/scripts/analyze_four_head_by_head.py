#!/usr/bin/env python3
"""Head-resolved WeightWatcher analysis for the four-head nanoGPT campaign.

This is a post-hoc, read-only analysis.  It loads the permanent epoch
checkpoints, splits the packed attention projections into logical head
matrices, and runs WeightWatcher on every head matrix separately.

For Q/K/V, head h is the corresponding output-row slice.  For O, head h is
the corresponding input-column slice because the concatenated head outputs
are the inputs to out_proj.

The source results are never modified.  Per-checkpoint CSVs are cached in the
analysis directory, so rerunning the program resumes safely after interruption.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn


SCHEMA_VERSION = 1
DEFAULT_OPTIMIZERS = ("adamw", "muon_clip")
DEFAULT_SEEDS = (1337, 2027, 4099, 31415, 271828)
ATTENTION_MATRICES = ("W_Q", "W_K", "W_V", "W_O")
EXPECTED_EPOCHS = tuple(round(0.25 * index, 8) for index in range(17))
OPTIMIZER_LABELS = {"adamw": "AdamW", "muon_clip": "MuonClip"}
HEAD_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
STATE_SUFFIXES = {
    "W_Q": "blocks.0.attn.q_proj.weight",
    "W_K": "blocks.0.attn.k_proj.weight",
    "W_V": "blocks.0.attn.v_proj.weight",
    "W_O": "blocks.0.attn.out_proj.weight",
}
PLOT_METRICS = ("alpha_raw", "alpha_clip_xmax")
SUMMARY_METRICS = (
    "alpha_raw",
    "alpha_clip_xmax",
    "alpha_delta",
    "ERG_gap",
    "num_traps",
    "rand_distance",
    "D",
    "stable_rank",
    "mp_softrank",
    "entropy",
    "Lambda",
    "rank_loss",
)


def parse_csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return result


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Match the campaign's exact model-state integrity hash."""

    if not state:
        raise ValueError("model state is empty")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"model state entry is not a tensor: {name}")
        tensor = value.detach().to("cpu").contiguous()
        metadata = json.dumps(
            {
                "name": str(name),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def find_state_tensor(
    state: dict[str, torch.Tensor], suffix: str
) -> tuple[str, torch.Tensor]:
    matches = [(name, value) for name, value in state.items() if name.endswith(suffix)]
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise RuntimeError(
            f"expected one model tensor ending in {suffix!r}; found {names}"
        )
    return matches[0]


def checkpoint_inventory(
    results_root: Path,
    optimizers: Sequence[str],
    seeds: Sequence[int],
) -> list[tuple[str, int, Path]]:
    inventory: list[tuple[str, int, Path]] = []
    failures: list[str] = []
    reference_epochs: tuple[float, ...] | None = None

    for optimizer in optimizers:
        for seed in seeds:
            run_dir = results_root / optimizer / f"seed_{seed}"
            completion_path = run_dir / "run_complete.json"
            if not completion_path.is_file():
                failures.append(f"missing completion marker: {completion_path}")
                continue
            try:
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append(f"invalid completion marker {completion_path}: {exc}")
                continue
            if completion.get("completed") is not True:
                failures.append(f"run is not complete: {run_dir}")
                continue

            checkpoints = sorted((run_dir / "epoch_checkpoints").glob("model_epoch_*.pt"))
            if not checkpoints:
                failures.append(f"no permanent epoch checkpoints: {run_dir}")
                continue

            epochs: list[float] = []
            for path in checkpoints:
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    epochs.append(round(float(payload["nominal_epoch"]), 8))
                except Exception as exc:
                    failures.append(f"cannot inspect {path}: {exc}")
            observed = tuple(sorted(epochs))
            if observed != EXPECTED_EPOCHS:
                failures.append(
                    f"{run_dir}: epochs are {observed}; expected {EXPECTED_EPOCHS}"
                )
            if reference_epochs is None:
                reference_epochs = observed
            elif observed != reference_epochs:
                failures.append(f"checkpoint epochs differ across runs: {run_dir}")
            inventory.extend((optimizer, int(seed), path) for path in checkpoints)

    if failures:
        raise RuntimeError("campaign validation failed:\n  - " + "\n  - ".join(failures))
    return inventory


def validate_checkpoint(
    path: Path,
    *,
    expected_optimizer: str,
    expected_seed: int,
    expected_heads: int,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint is not a mapping: {path}")
    if str(payload.get("optimizer_name")) != expected_optimizer:
        raise RuntimeError(f"optimizer identity mismatch in {path}")
    if int(payload.get("seed", -1)) != int(expected_seed):
        raise RuntimeError(f"seed identity mismatch in {path}")

    config = payload.get("config")
    state = payload.get("model")
    if not isinstance(config, dict) or not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint lacks config or model state: {path}")
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise RuntimeError(f"checkpoint lacks model configuration: {path}")
    n_head = int(model_config.get("n_head", -1))
    n_embd = int(model_config.get("n_embd", -1))
    n_layer = int(model_config.get("n_layer", -1))
    if n_layer != 1:
        raise RuntimeError(f"expected one transformer block, found n_layer={n_layer}: {path}")
    if n_head != expected_heads:
        raise RuntimeError(f"expected {expected_heads} heads, found {n_head}: {path}")
    if n_embd <= 0 or n_embd % n_head:
        raise RuntimeError(f"invalid n_embd={n_embd}, n_head={n_head}: {path}")

    observed_hash = model_state_sha256(state)
    expected_hash = str(payload.get("model_state_sha256", ""))
    if expected_hash and observed_hash != expected_hash:
        raise RuntimeError(f"model-state SHA-256 mismatch: {path}")

    for matrix_type, suffix in STATE_SUFFIXES.items():
        _, weight = find_state_tensor(state, suffix)
        if tuple(weight.shape) != (n_embd, n_embd):
            raise RuntimeError(
                f"{matrix_type} has shape {tuple(weight.shape)}, expected "
                f"{(n_embd, n_embd)}: {path}"
            )
        if not bool(torch.isfinite(weight).all()):
            raise RuntimeError(f"{matrix_type} contains non-finite values: {path}")

    payload["_validated_model_hash"] = observed_hash
    payload["_n_head"] = n_head
    payload["_n_embd"] = n_embd
    return payload


class HeadMatrixHolder(nn.Module):
    """CPU-only Linear modules for the individual Q/K/V/O head slices."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        state = payload["model"]
        n_head = int(payload["_n_head"])
        n_embd = int(payload["_n_embd"])
        head_width = n_embd // n_head
        self.matrix_metadata: list[dict[str, Any]] = []

        for matrix_type in ATTENTION_MATRICES:
            state_name, packed = find_state_tensor(state, STATE_SUFFIXES[matrix_type])
            packed = packed.detach().float().cpu().contiguous()
            for zero_head in range(n_head):
                start = zero_head * head_width
                stop = start + head_width
                if matrix_type in ("W_Q", "W_K", "W_V"):
                    # q/k/v output channels are reshaped to [n_head, head_width].
                    head_weight = packed[start:stop, :].clone()
                    slice_axis = "output_rows"
                else:
                    # out_proj consumes the concatenation of all head outputs.
                    head_weight = packed[:, start:stop].clone()
                    slice_axis = "input_columns"

                module_name = f"L00_{matrix_type}_H{zero_head + 1:02d}"
                layer = nn.Linear(
                    int(head_weight.shape[1]),
                    int(head_weight.shape[0]),
                    bias=False,
                )
                layer.weight = nn.Parameter(head_weight, requires_grad=False)
                self.add_module(module_name, layer)
                self.matrix_metadata.append(
                    {
                        "matrix_name": module_name,
                        "matrix_type": matrix_type,
                        "head": zero_head + 1,
                        "block": 0,
                        "slice_axis": slice_axis,
                        "slice_start": start,
                        "slice_stop": stop,
                        "rows": int(head_weight.shape[0]),
                        "columns": int(head_weight.shape[1]),
                        "packed_state_name": state_name,
                    }
                )


def attach_matrix_metadata(
    details: pd.DataFrame,
    metadata: list[dict[str, Any]],
) -> pd.DataFrame:
    result = details.copy().reset_index(drop=True)
    names = [str(item["matrix_name"]) for item in metadata]
    resolved: list[str | None] = [None] * len(result)
    for row_index, row in result.iterrows():
        description = " ".join(str(row.get(column, "")) for column in ("longname", "name"))
        for name in names:
            if name in description:
                resolved[row_index] = name
                break

    if any(value is None for value in resolved) and len(result) == len(metadata):
        order = list(range(len(result)))
        if "layer_id" in result.columns:
            numeric = pd.to_numeric(result["layer_id"], errors="coerce")
            if numeric.notna().all():
                order = list(numeric.sort_values().index)
        for metadata_index, row_index in enumerate(order):
            resolved[row_index] = names[metadata_index]

    if len(result) != len(metadata) or any(value is None for value in resolved):
        raise RuntimeError(
            "WeightWatcher did not return exactly one row for every head matrix; "
            f"returned={len(result)}, expected={len(metadata)}"
        )
    if len(set(str(value) for value in resolved)) != len(metadata):
        raise RuntimeError("WeightWatcher head rows could not be matched uniquely")

    by_name = {str(item["matrix_name"]): item for item in metadata}
    for column in (
        "matrix_name",
        "matrix_type",
        "head",
        "block",
        "slice_axis",
        "slice_start",
        "slice_stop",
        "rows",
        "columns",
        "packed_state_name",
    ):
        if column in result.columns:
            result = result.rename(columns={column: f"weightwatcher_{column}"})
    identity = pd.DataFrame([by_name[str(name)] for name in resolved])
    return pd.concat([identity.reset_index(drop=True), result], axis=1)


def cached_frame_valid(
    path: Path,
    *,
    optimizer: str,
    seed: int,
    step: int,
    model_hash: str,
    expected_rows: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    required = {
        "optimizer",
        "seed",
        "step",
        "model_state_sha256",
        "matrix_name",
        "matrix_type",
        "head",
        "alpha_raw",
        "alpha_clip_xmax",
    }
    if required.difference(frame.columns) or len(frame) != expected_rows:
        return False
    if frame["matrix_name"].nunique() != expected_rows:
        return False
    return bool(
        frame["optimizer"].astype(str).eq(optimizer).all()
        and pd.to_numeric(frame["seed"], errors="coerce").eq(seed).all()
        and pd.to_numeric(frame["step"], errors="coerce").eq(step).all()
        and frame["model_state_sha256"].astype(str).eq(model_hash).all()
    )


def analyze_checkpoint(
    payload: dict[str, Any],
    *,
    checkpoint_path: Path,
    optimizer: str,
    seed: int,
    min_evals: int,
    max_fingers: int,
) -> pd.DataFrame:
    try:
        import weightwatcher as ww
    except ImportError as exc:
        raise RuntimeError(
            "WeightWatcher is unavailable in this Python environment"
        ) from exc

    step = int(payload["step"])
    nominal_epoch = float(payload["nominal_epoch"])
    actual_epoch = float(payload.get("actual_epoch", nominal_epoch))
    diagnostic_seed = int(seed) + 7_000_003 + step
    holder = HeadMatrixHolder(payload)

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(diagnostic_seed)
    np.random.seed(diagnostic_seed % (2**32 - 1))
    torch.manual_seed(diagnostic_seed)
    try:
        watcher = ww.WeightWatcher(model=holder)
        details = watcher.analyze(
            ERG=True,
            randomize=True,
            plot=False,
            min_evals=int(min_evals),
            fix_fingers="clip_xmax",
            max_fingers=int(max_fingers),
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

    if details is None or len(details) == 0:
        raise RuntimeError(f"WeightWatcher returned no rows for {checkpoint_path}")
    frame = attach_matrix_metadata(pd.DataFrame(details), holder.matrix_metadata)
    if "raw_alpha" not in frame.columns or "alpha" not in frame.columns:
        raise RuntimeError(
            "WeightWatcher did not return alpha and raw_alpha under clip_xmax"
        )
    frame["alpha_raw"] = pd.to_numeric(frame["raw_alpha"], errors="coerce")
    frame["alpha_clip_xmax"] = pd.to_numeric(frame["alpha"], errors="coerce")
    frame["alpha_delta"] = frame["alpha_raw"] - frame["alpha_clip_xmax"]
    required_finite = ("alpha_raw", "alpha_clip_xmax", "alpha_delta")
    if not np.isfinite(frame[list(required_finite)].to_numpy(dtype=float)).all():
        raise RuntimeError(f"non-finite head alpha returned for {checkpoint_path}")

    leading = {
        "schema_version": SCHEMA_VERSION,
        "optimizer": optimizer,
        "seed": int(seed),
        "step": step,
        "nominal_epoch": nominal_epoch,
        "actual_epoch": actual_epoch,
        "diagnostic_seed": diagnostic_seed,
        "model_state_sha256": payload["_validated_model_hash"],
        "protocol_fingerprint": str(payload.get("fingerprint", "")),
        "checkpoint_path": str(checkpoint_path),
        "n_head": int(payload["_n_head"]),
        "n_embd": int(payload["_n_embd"]),
        "head_width": int(payload["_n_embd"]) // int(payload["_n_head"]),
        "finger_policy": "fix_fingers=clip_xmax",
        "weightwatcher_analysis_calls": 1,
    }
    for column, value in reversed(tuple(leading.items())):
        if column in frame.columns:
            frame = frame.rename(columns={column: f"weightwatcher_{column}"})
        frame.insert(0, column, value)
    return frame


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def ci95(values: Iterable[float]) -> dict[str, float | int]:
    array = finite(values)
    n = int(array.size)
    mean = float(array.mean()) if n else math.nan
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "sd": math.nan,
            "sem": math.nan,
            "ci95_lower": math.nan,
            "ci95_upper": math.nan,
        }
    sd = float(array.std(ddof=1))
    sem = sd / math.sqrt(n)
    half = float(stats.t.ppf(0.975, n - 1) * sem)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
    }


def summarize_heads(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = (
        "optimizer",
        "matrix_type",
        "head",
        "nominal_epoch",
        "step",
    )
    available_metrics = [metric for metric in SUMMARY_METRICS if metric in frame.columns]
    for keys, group in frame.groupby(list(group_columns), sort=True):
        identity = dict(zip(group_columns, keys, strict=True))
        for metric in available_metrics:
            summary = ci95(pd.to_numeric(group[metric], errors="coerce"))
            rows.append({**identity, "metric": metric, **summary})
    return pd.DataFrame(rows)


def paired_differences(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"adamw", "muon_clip"}.issubset(set(frame["optimizer"].astype(str))):
        return pd.DataFrame(), pd.DataFrame()
    available_metrics = [metric for metric in SUMMARY_METRICS if metric in frame.columns]
    keys = ["seed", "matrix_type", "head", "nominal_epoch", "step"]
    raw_rows: list[pd.DataFrame] = []
    for metric in available_metrics:
        pivot = frame.pivot(index=keys, columns="optimizer", values=metric).reset_index()
        pivot["metric"] = metric
        pivot["muon_clip_minus_adamw"] = (
            pd.to_numeric(pivot["muon_clip"], errors="coerce")
            - pd.to_numeric(pivot["adamw"], errors="coerce")
        )
        raw_rows.append(pivot)
    raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    if not raw.empty:
        group_keys = ["matrix_type", "head", "nominal_epoch", "step", "metric"]
        for keys_value, group in raw.groupby(group_keys, sort=True):
            identity = dict(zip(group_keys, keys_value, strict=True))
            summary_rows.append(
                {
                    **identity,
                    **ci95(group["muon_clip_minus_adamw"]),
                }
            )
    return raw, pd.DataFrame(summary_rows)


def robust_limits(values: Iterable[float], *, include_two: bool) -> tuple[float, float] | None:
    array = finite(values)
    if array.size == 0:
        return None
    lower, upper = np.percentile(array, [2.0, 98.0])
    if include_two:
        lower = min(float(lower), 2.0)
        upper = max(float(upper), 2.0)
    span = max(float(upper - lower), 0.25)
    return float(lower - 0.08 * span), float(upper + 0.08 * span)


def plot_metric(
    frame: pd.DataFrame,
    *,
    optimizer: str,
    metric: str,
    output_path: Path,
    zoom: bool,
) -> None:
    subset = frame[frame["optimizer"].astype(str).eq(optimizer)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, matrix_type in zip(axes.flat, ATTENTION_MATRICES, strict=True):
        matrix = subset[subset["matrix_type"].astype(str).eq(matrix_type)]
        zoom_values: list[float] = []
        for head in sorted(pd.to_numeric(matrix["head"], errors="coerce").dropna().unique()):
            head_int = int(head)
            head_frame = matrix[pd.to_numeric(matrix["head"], errors="coerce").eq(head_int)]
            color = HEAD_COLORS[(head_int - 1) % len(HEAD_COLORS)]
            for _, seed_frame in head_frame.groupby("seed", sort=True):
                seed_frame = seed_frame.sort_values("nominal_epoch")
                axis.plot(
                    seed_frame["nominal_epoch"],
                    pd.to_numeric(seed_frame[metric], errors="coerce"),
                    color=color,
                    alpha=0.12,
                    linewidth=0.8,
                )
            epochs: list[float] = []
            means: list[float] = []
            lowers: list[float] = []
            uppers: list[float] = []
            for epoch, epoch_frame in head_frame.groupby("nominal_epoch", sort=True):
                summary = ci95(pd.to_numeric(epoch_frame[metric], errors="coerce"))
                epochs.append(float(epoch))
                means.append(float(summary["mean"]))
                lowers.append(float(summary["ci95_lower"]))
                uppers.append(float(summary["ci95_upper"]))
            x = np.asarray(epochs, dtype=float)
            y = np.asarray(means, dtype=float)
            lo = np.asarray(lowers, dtype=float)
            hi = np.asarray(uppers, dtype=float)
            axis.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                linewidth=2.0,
                color=color,
                label=f"head {head_int}",
            )
            axis.fill_between(x, lo, hi, color=color, alpha=0.12)
            zoom_values.extend(pd.to_numeric(head_frame[metric], errors="coerce"))

        axis.axhline(2.0, color="black", linestyle="--", linewidth=1.2, label="alpha = 2")
        axis.set_title(matrix_type)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
        if zoom:
            limits = robust_limits(zoom_values, include_two=True)
            if limits is not None:
                axis.set_ylim(*limits)
        handles, labels = axis.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels, strict=True):
            unique.setdefault(label, handle)
        axis.legend(unique.values(), unique.keys(), fontsize=8, loc="best")

    label = OPTIMIZER_LABELS.get(optimizer, optimizer)
    suffix = " (robust zoom)" if zoom else ""
    fig.suptitle(
        f"{label}: head-resolved {metric}{suffix}\n"
        "solid = mean across seeds; shading = 95% t CI; faint = individual seeds",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dataframe_html(frame: pd.DataFrame, *, digits: int = 4) -> str:
    return frame.to_html(index=False, border=0, float_format=lambda value: f"{value:.{digits}f}")


def write_report(
    output_root: Path,
    *,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    optimizers: Sequence[str],
    seeds: Sequence[int],
    plots: Sequence[Path],
) -> Path:
    final_epoch = float(pd.to_numeric(frame["nominal_epoch"], errors="coerce").max())
    final = summary[
        summary["nominal_epoch"].astype(float).eq(final_epoch)
        & summary["metric"].isin(PLOT_METRICS)
    ][
        [
            "optimizer",
            "matrix_type",
            "head",
            "metric",
            "n",
            "mean",
            "sd",
            "ci95_lower",
            "ci95_upper",
        ]
    ].copy()
    sections = "\n".join(
        f'<h3>{html.escape(path.stem.replace("_", " "))}</h3>'
        f'<img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}">'
        for path in plots
    )
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Four-head nanoGPT: head-resolved spectral analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem auto; max-width: 1500px; color: #222; line-height: 1.45; }}
h1, h2, h3 {{ color: #17324d; }}
img {{ width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 1.5rem; }}
table {{ border-collapse: collapse; font-size: 0.9rem; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: right; }}
th {{ background: #f2f5f8; }}
td:first-child, th:first-child {{ text-align: left; }}
code {{ background: #f3f3f3; padding: 0.1rem 0.25rem; }}
.note {{ background: #fff8dc; border-left: 4px solid #d9a400; padding: 0.8rem 1rem; }}
</style>
</head>
<body>
<h1>Four-head nanoGPT: head-resolved spectral analysis</h1>
<p><b>Optimizers:</b> {html.escape(', '.join(optimizers))}<br>
<b>Seeds:</b> {html.escape(', '.join(str(seed) for seed in seeds))}<br>
<b>Epochs:</b> 0.00 through {final_epoch:.2f} in 0.25-epoch increments</p>

<div class="note"><b>This analysis does not average the four heads before fitting.</b>
At every checkpoint and for every seed, WeightWatcher is run on 16 distinct matrices:
Q1–Q4, K1–K4, V1–V4, and O1–O4. Each plotted head curve is then averaged only
across the independent seeds; its band is a 95% Student-t confidence interval across seeds.</div>

<h2>Exact slicing</h2>
<table>
<tr><th>Matrix</th><th>Packed shape</th><th>Per-head shape</th><th>Slice</th></tr>
<tr><td>Q, K, V</td><td>128 × 128</td><td>32 × 128</td><td>Separate output-row blocks</td></tr>
<tr><td>O</td><td>128 × 128</td><td>128 × 32</td><td>Separate input-column blocks</td></tr>
</table>
<p>The MLP matrices are block-level matrices and are not divided by attention head.</p>

<h2>Final-epoch head statistics</h2>
{dataframe_html(final)}

<h2>Head-resolved alpha trajectories</h2>
{sections}

<h2>Files</h2>
<ul>
<li><code>head_spectral_layers.csv</code>: every seed × checkpoint × matrix type × head result, including the complete WeightWatcher output.</li>
<li><code>head_spectral_summary.csv</code>: mean, SD, SEM, and 95% t interval across seeds for each fixed head.</li>
<li><code>paired_optimizer_differences.csv</code>: MuonClip minus AdamW for the same seed, head, matrix, and checkpoint.</li>
<li><code>paired_optimizer_difference_summary.csv</code>: paired differences summarized across seeds.</li>
<li><code>raw/</code>: resumable per-checkpoint WeightWatcher caches.</li>
</ul>
<p class="note">Head matrices have only 32 singular values, so their power-law fits are intrinsically noisier than the packed 128 × 128 fits. Interpret individual-head alpha values together with D, tail size, and confidence intervals.</p>
</body>
</html>
"""
    path = output_root / "report.html"
    atomic_text(path, report)
    return path


def analyze(args: argparse.Namespace) -> Path:
    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        raise RuntimeError(f"results root does not exist: {results_root}")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else results_root.parent / "analysis_four_head_by_head"
    )
    if output_root == results_root or results_root in output_root.parents:
        raise RuntimeError("analysis output must not be inside the source results directory")
    output_root.mkdir(parents=True, exist_ok=True)

    inventory = checkpoint_inventory(results_root, args.optimizers, args.seeds)
    print(
        f"[head-analysis] validated {len(inventory)} checkpoints "
        f"({len(args.optimizers)} optimizers × {len(args.seeds)} seeds × "
        f"{len(EXPECTED_EPOCHS)} epochs)",
        flush=True,
    )

    raw_frames: list[pd.DataFrame] = []
    total = len(inventory)
    for index, (optimizer, seed, checkpoint_path) in enumerate(inventory, start=1):
        payload = validate_checkpoint(
            checkpoint_path,
            expected_optimizer=optimizer,
            expected_seed=seed,
            expected_heads=args.expected_heads,
        )
        step = int(payload["step"])
        epoch = float(payload["nominal_epoch"])
        cache_path = (
            output_root
            / "raw"
            / optimizer
            / f"seed_{seed}"
            / f"head_spectral_step_{step:07d}.csv"
        )
        expected_rows = len(ATTENTION_MATRICES) * int(payload["_n_head"])
        if not args.force and cached_frame_valid(
            cache_path,
            optimizer=optimizer,
            seed=seed,
            step=step,
            model_hash=payload["_validated_model_hash"],
            expected_rows=expected_rows,
        ):
            checkpoint_frame = pd.read_csv(cache_path)
            action = "cached"
        else:
            checkpoint_frame = analyze_checkpoint(
                payload,
                checkpoint_path=checkpoint_path,
                optimizer=optimizer,
                seed=seed,
                min_evals=args.min_evals,
                max_fingers=args.max_fingers,
            )
            atomic_csv(cache_path, checkpoint_frame)
            action = "analyzed"
        raw_frames.append(checkpoint_frame)
        print(
            f"[head-analysis] {index:03d}/{total:03d} {action}: "
            f"{optimizer} seed={seed} epoch={epoch:.2f} "
            f"({expected_rows} head matrices)",
            flush=True,
        )

    layers = pd.concat(raw_frames, ignore_index=True, sort=False)
    expected_total_rows = (
        len(args.optimizers)
        * len(args.seeds)
        * len(EXPECTED_EPOCHS)
        * len(ATTENTION_MATRICES)
        * args.expected_heads
    )
    if len(layers) != expected_total_rows:
        raise RuntimeError(
            f"head-row count is {len(layers)}, expected {expected_total_rows}"
        )
    identity = ["optimizer", "seed", "step", "matrix_type", "head"]
    if layers[identity].duplicated().any():
        raise RuntimeError("duplicate head spectral rows were produced")
    layers = layers.sort_values(
        ["optimizer", "seed", "nominal_epoch", "matrix_type", "head"]
    ).reset_index(drop=True)
    atomic_csv(output_root / "head_spectral_layers.csv", layers)

    summary = summarize_heads(layers)
    atomic_csv(output_root / "head_spectral_summary.csv", summary)
    paired_raw, paired_summary = paired_differences(layers)
    if not paired_raw.empty:
        atomic_csv(output_root / "paired_optimizer_differences.csv", paired_raw)
        atomic_csv(
            output_root / "paired_optimizer_difference_summary.csv",
            paired_summary,
        )

    plots: list[Path] = []
    for optimizer in args.optimizers:
        for metric in PLOT_METRICS:
            if metric not in layers.columns:
                continue
            for zoom in (False, True):
                suffix = "_zoom" if zoom else ""
                path = output_root / f"{optimizer}_{metric}_by_head{suffix}.png"
                plot_metric(
                    layers,
                    optimizer=optimizer,
                    metric=metric,
                    output_path=path,
                    zoom=zoom,
                )
                plots.append(path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "output_root": str(output_root),
        "optimizers": list(args.optimizers),
        "seeds": list(args.seeds),
        "expected_heads": int(args.expected_heads),
        "epochs": list(EXPECTED_EPOCHS),
        "checkpoint_count": len(inventory),
        "head_matrix_rows": len(layers),
        "min_evals": int(args.min_evals),
        "max_fingers": int(args.max_fingers),
        "slicing": {
            "W_Q": "output rows",
            "W_K": "output rows",
            "W_V": "output rows",
            "W_O": "input columns",
        },
    }
    atomic_text(
        output_root / "analysis_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    report = write_report(
        output_root,
        frame=layers,
        summary=summary,
        optimizers=args.optimizers,
        seeds=args.seeds,
        plots=plots,
    )
    print(f"[head-analysis] complete: {report}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze every attention head separately in the completed four-head "
            "nanoGPT campaign."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/tmp/rg-nanogpt-four-head-20260827/results"),
        help="campaign results directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "analysis directory; defaults to analysis_four_head_by_head next "
            "to the results directory"
        ),
    )
    parser.add_argument(
        "--optimizers",
        type=parse_csv_strings,
        default=DEFAULT_OPTIMIZERS,
        help="comma-separated optimizer names",
    )
    parser.add_argument(
        "--seeds",
        type=parse_csv_ints,
        default=DEFAULT_SEEDS,
        help="comma-separated seed integers",
    )
    parser.add_argument("--expected-heads", type=int, default=4)
    parser.add_argument("--min-evals", type=int, default=20)
    parser.add_argument("--max-fingers", type=int, default=10)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute checkpoint caches that already pass validation",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = analyze(args)
    except Exception as exc:
        print(f"[head-analysis] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

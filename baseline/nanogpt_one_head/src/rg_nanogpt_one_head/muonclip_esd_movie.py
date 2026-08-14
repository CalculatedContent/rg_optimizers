from __future__ import annotations

"""Render a smooth log-log ESD movie for one saved MuonClip matrix.

For every selected checkpoint this command constructs a one-matrix model, runs
WeightWatcher, reads the actual ESD with ``get_ESD()``, and renders the empirical
spectral density itself on fixed log-log axes.  It never guesses which PNG from
WeightWatcher is the ESD and never makes a movie by cross-fading arbitrary plot
files.
"""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import weightwatcher as ww

from .muonclip_capture import load_weightwatcher_checkpoint

MAX_CHECKPOINTS = 500


class OneMatrixModel(nn.Module):
    """PyTorch wrapper containing exactly one matrix for WeightWatcher."""

    def __init__(self, matrix_name: str, weight: torch.Tensor) -> None:
        super().__init__()
        weight = weight.detach().float().cpu()
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.add_module(matrix_name, layer)


def _natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _load_index(walk_dir: Path, cadence: str) -> pd.DataFrame:
    snapshot_index = walk_dir / "snapshot_index.csv"
    if snapshot_index.is_file():
        frame = pd.read_csv(snapshot_index)
        if cadence == "optimizer":
            frame = frame[
                frame["snapshot_kind"].isin(["initial", "optimizer_step"])
            ]
        elif cadence != "microbatch":
            raise ValueError("cadence must be optimizer or microbatch")
        return frame.sort_values("timeline_index").reset_index(drop=True)

    rows = []
    for path in sorted(
        (walk_dir / "weightwatcher_checkpoints").glob("ww_step_*.pt"),
        key=_natural_key,
    ):
        match = re.search(r"ww_step_(\d+)\.pt$", path.name)
        if match:
            step = int(match.group(1))
            rows.append(
                {
                    "timeline_index": step,
                    "snapshot_kind": "initial" if step == 0 else "optimizer_step",
                    "effective_batch": step,
                    "microbatch_index": 0,
                    "weightwatcher_checkpoint": str(path),
                }
            )
    return pd.DataFrame(rows)


def _select_rows(
    frame: pd.DataFrame,
    *,
    first_effective_batch: int,
    last_effective_batch: int | None,
    max_checkpoints: int,
) -> pd.DataFrame:
    if frame.empty:
        raise FileNotFoundError("no capture checkpoints were found")

    effective = pd.to_numeric(frame["effective_batch"], errors="coerce")
    initial = frame["snapshot_kind"].eq("initial")
    selected = frame[initial | effective.ge(first_effective_batch)].copy()
    if last_effective_batch is not None:
        selected = selected[
            selected["snapshot_kind"].eq("initial")
            | pd.to_numeric(
                selected["effective_batch"], errors="coerce"
            ).le(last_effective_batch)
        ]
    selected = selected.sort_values("timeline_index").reset_index(drop=True)

    if len(selected) < 2:
        raise ValueError(
            f"movie selection contains only {len(selected)} checkpoint; "
            "at least two are required"
        )
    if len(selected) > max_checkpoints:
        raise ValueError(
            f"requested {len(selected)} checkpoints, exceeding "
            f"--max-checkpoints={max_checkpoints}"
        )
    if len(selected) > MAX_CHECKPOINTS:
        raise ValueError(f"hard movie cap is {MAX_CHECKPOINTS} checkpoints")

    missing = [
        Path(str(path))
        for path in selected["weightwatcher_checkpoint"]
        if not Path(str(path)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing checkpoint files:\n" + "\n".join(map(str, missing))
        )
    return selected


def _single_matrix(checkpoint: Path, matrix_name: str) -> OneMatrixModel:
    holder, _ = load_weightwatcher_checkpoint(checkpoint, source="weights")
    layers = dict(holder.named_children())
    if matrix_name not in layers:
        raise KeyError(
            f"{matrix_name!r} not found in {checkpoint.name}; "
            f"available={list(layers)}"
        )
    return OneMatrixModel(matrix_name, layers[matrix_name].weight)


def _finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _analyze_checkpoint(
    row: pd.Series,
    *,
    matrix_name: str,
    output_root: Path,
    min_evals: int,
) -> dict[str, Any]:
    checkpoint = Path(str(row["weightwatcher_checkpoint"]))
    timeline = int(row["timeline_index"])
    model = _single_matrix(checkpoint, matrix_name)
    watcher = ww.WeightWatcher(model=model)

    # Keep the requested WeightWatcher call for reproducibility, but do not use
    # an arbitrarily named saved image as a movie frame.  The movie is built
    # from watcher.get_ESD() below.
    native_dir = output_root / "weightwatcher_native" / f"snapshot_{timeline:07d}"
    if native_dir.exists():
        shutil.rmtree(native_dir)
    native_dir.mkdir(parents=True, exist_ok=True)

    old_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        savedir = str(native_dir)
        details_raw = watcher.analyze(
            plot=True,
            savefig=savedir,
            min_evals=min_evals,
            randomize=False,
            ERG=False,
        )
    finally:
        plt.show = old_show
        plt.close("all")

    details = pd.DataFrame(details_raw)
    if details.empty:
        raise RuntimeError(f"WeightWatcher returned no details for {checkpoint}")

    layer_id = (
        int(details.iloc[0]["layer_id"])
        if "layer_id" in details.columns
        else int(details.index[0])
    )
    eigenvalues = np.asarray(watcher.get_ESD(layer=layer_id), dtype=float).reshape(-1)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > 0)]
    eigenvalues.sort()
    if eigenvalues.size < min_evals:
        raise RuntimeError(
            f"{checkpoint.name} has only {eigenvalues.size} positive eigenvalues"
        )

    detail = details.iloc[0]
    alpha = _finite_number(detail.get("alpha"))
    xmin = _finite_number(detail.get("xmin"))
    xmax = _finite_number(detail.get("xmax"))
    if not math.isfinite(xmin) or xmin <= 0:
        xmin = float(eigenvalues.min())
    if not math.isfinite(xmax) or xmax <= xmin:
        xmax = float(eigenvalues.max())

    details.insert(0, "checkpoint", checkpoint.name)
    details.insert(1, "timeline_index", timeline)
    details.insert(2, "matrix_name", matrix_name)
    details.to_csv(native_dir / "weightwatcher_details.csv", index=False)

    print(
        "[one-head-esd-movie] "
        f"snapshot={timeline} kind={row['snapshot_kind']} "
        f"batch={row.get('effective_batch')} matrix={matrix_name} "
        f"evals={eigenvalues.size} alpha={alpha:.4f}",
        flush=True,
    )
    return {
        "timeline_index": timeline,
        "snapshot_kind": str(row["snapshot_kind"]),
        "effective_batch": int(row.get("effective_batch", timeline)),
        "microbatch_index": int(row.get("microbatch_index", 0)),
        "checkpoint": checkpoint.name,
        "eigenvalues": eigenvalues,
        "alpha": alpha,
        "xmin": xmin,
        "xmax": xmax,
        "details": details,
    }


def _density_on_common_log_grid(
    eigenvalues: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    density, _ = np.histogram(eigenvalues, bins=edges, density=True)
    density = np.asarray(density, dtype=float)
    # A tiny fixed convolution reduces histogram flicker but preserves the
    # spectral movement.  It is applied identically to every checkpoint.
    if density.size >= 5:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel /= kernel.sum()
        density = np.convolve(density, kernel, mode="same")
    return density


def _geom_interp(left: float, right: float, fraction: float) -> float:
    if left > 0 and right > 0:
        return float(
            np.exp(
                (1.0 - fraction) * np.log(left)
                + fraction * np.log(right)
            )
        )
    return float((1.0 - fraction) * left + fraction * right)


def _make_loglog_movie(
    snapshots: list[dict[str, Any]],
    *,
    matrix_name: str,
    output_root: Path,
    fps: int,
    frames_per_transition: int,
    bins: int,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required; install on macOS with: brew install ffmpeg"
        )

    all_evals = np.concatenate([snapshot["eigenvalues"] for snapshot in snapshots])
    x_min = float(all_evals.min())
    x_max = float(all_evals.max())
    edges = np.geomspace(x_min, x_max, bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    densities = np.vstack(
        [
            _density_on_common_log_grid(snapshot["eigenvalues"], edges)
            for snapshot in snapshots
        ]
    )
    positive = densities[densities > 0]
    if positive.size == 0:
        raise RuntimeError("all empirical ESD density bins are zero")
    floor = float(positive.min()) * 0.15
    densities = np.maximum(densities, floor)

    frame_map: list[tuple[int, float]] = []
    for index in range(len(snapshots) - 1):
        for subframe in range(frames_per_transition):
            raw = subframe / float(frames_per_transition)
            eased = 0.5 - 0.5 * math.cos(math.pi * raw)
            frame_map.append((index, eased))
    frame_map.append((len(snapshots) - 2, 1.0))

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    esd_line, = ax.plot(
        [], [], marker="o", markersize=3.4, linewidth=2.0, label="Empirical ESD"
    )
    fit_line, = ax.plot([], [], linestyle="--", linewidth=2.0, label="WW power-law fit")
    xmin_line = ax.axvline(x_min, linestyle=":", linewidth=1.3)
    title = ax.set_title("")
    annotation = ax.text(
        0.02,
        0.03,
        "",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min * 0.88, x_max * 1.12)
    ax.set_ylim(floor * 0.55, float(densities.max()) * 1.8)
    ax.set_xlabel(r"Eigenvalue $\lambda$  (log scale)")
    ax.set_ylabel(r"Spectral density $\rho(\lambda)$  (log scale)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right")

    def update(frame_number: int):
        index, fraction = frame_map[frame_number]
        left = snapshots[index]
        right = snapshots[index + 1]

        density = np.exp(
            (1.0 - fraction) * np.log(densities[index])
            + fraction * np.log(densities[index + 1])
        )
        esd_line.set_data(centers, density)

        left_alpha = left["alpha"]
        right_alpha = right["alpha"]
        alpha = (
            (1.0 - fraction) * left_alpha + fraction * right_alpha
            if math.isfinite(left_alpha) and math.isfinite(right_alpha)
            else left_alpha if math.isfinite(left_alpha) else right_alpha
        )
        xmin = _geom_interp(left["xmin"], right["xmin"], fraction)
        xmax = _geom_interp(left["xmax"], right["xmax"], fraction)
        xmin = float(np.clip(xmin, centers.min(), centers.max()))
        xmax = float(np.clip(xmax, xmin, centers.max()))
        xmin_line.set_xdata([xmin, xmin])

        if math.isfinite(alpha) and xmax > xmin:
            fit_x = np.geomspace(xmin, xmax, 180)
            anchor_log_y = np.interp(
                np.log(xmin), np.log(centers), np.log(density)
            )
            anchor_y = float(np.exp(anchor_log_y))
            fit_y = anchor_y * np.power(fit_x / xmin, -alpha)
            fit_line.set_data(fit_x, fit_y)
        else:
            fit_line.set_data([], [])

        displayed_step = (
            (1.0 - fraction) * left["effective_batch"]
            + fraction * right["effective_batch"]
        )
        exact = fraction < 1e-12 or abs(fraction - 1.0) < 1e-12
        state = "actual saved checkpoint" if exact else "smooth interpolation"
        title.set_text(
            f"MuonClip — {matrix_name} log-log ESD\n"
            f"effective optimizer batch {displayed_step:.2f} ({state})"
        )
        annotation.set_text(
            f"matrix: {matrix_name}\n"
            f"checkpoints: {len(snapshots)}\n"
            f"alpha ≈ {alpha:.4f}" if math.isfinite(alpha) else
            f"matrix: {matrix_name}\ncheckpoints: {len(snapshots)}\nalpha: n/a"
        )
        return esd_line, fit_line, xmin_line, title, annotation

    video_dir = output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_dir / f"{matrix_name}_loglog_esd.mp4"
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_map),
        interval=1000 / fps,
        blit=False,
    )
    movie.save(output_path, writer=writer, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a smooth, fixed-axis, log-log ESD movie for exactly one "
            "MuonClip matrix from saved WeightWatcher checkpoints."
        )
    )
    parser.add_argument(
        "--walk-dir",
        default=os.environ.get("WALK_DIR"),
        help="capture directory; defaults to $WALK_DIR",
    )
    parser.add_argument("--matrix", default="L00_W_Q")
    parser.add_argument(
        "--cadence",
        choices=("optimizer", "microbatch"),
        default="optimizer",
    )
    parser.add_argument("--first-effective-batch", type=int, default=1)
    parser.add_argument("--last-effective-batch", type=int, default=10)
    parser.add_argument("--max-checkpoints", type=int, default=500)
    parser.add_argument("--min-evals", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames-per-transition", type=int, default=24)
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if not args.walk_dir:
        raise SystemExit("Set WALK_DIR or pass --walk-dir")
    if not 1 <= args.max_checkpoints <= MAX_CHECKPOINTS:
        raise SystemExit("--max-checkpoints must be between 1 and 500")
    if args.frames_per_transition < 1:
        raise SystemExit("--frames-per-transition must be positive")
    if args.bins < 8:
        raise SystemExit("--bins must be at least 8")

    walk_dir = Path(args.walk_dir).expanduser().resolve()
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else walk_dir
        / "diagnostics"
        / f"loglog_esd_movie_{args.matrix}_{args.cadence}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    index = _select_rows(
        _load_index(walk_dir, args.cadence),
        first_effective_batch=args.first_effective_batch,
        last_effective_batch=args.last_effective_batch,
        max_checkpoints=args.max_checkpoints,
    )
    index.to_csv(output_root / "movie_checkpoint_index.csv", index=False)

    print(
        f"[one-head-esd-movie] selected {len(index)} actual checkpoints",
        flush=True,
    )
    snapshots = [
        _analyze_checkpoint(
            row,
            matrix_name=args.matrix,
            output_root=output_root,
            min_evals=args.min_evals,
        )
        for _, row in index.iterrows()
    ]

    details = pd.concat(
        [snapshot["details"] for snapshot in snapshots],
        ignore_index=True,
    )
    details.to_csv(output_root / "weightwatcher_details_all_checkpoints.csv", index=False)

    output_path = _make_loglog_movie(
        snapshots,
        matrix_name=args.matrix,
        output_root=output_root,
        fps=args.fps,
        frames_per_transition=args.frames_per_transition,
        bins=args.bins,
    )

    print()
    print(f"[one-head-esd-movie] actual checkpoints: {len(snapshots)}")
    print(f"[one-head-esd-movie] movie: {output_path}")


if __name__ == "__main__":
    main()

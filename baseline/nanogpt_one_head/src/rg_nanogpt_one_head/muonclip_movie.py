from __future__ import annotations

"""Create native WeightWatcher ESD frames and a smooth MP4 spectral movie."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import weightwatcher as ww

from .muonclip_capture import load_weightwatcher_checkpoint

_MAX_MOVIE_CHECKPOINTS = 500


class OneMatrixModel(nn.Module):
    """A one-layer model so WeightWatcher analyzes one named matrix only."""

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


def _one_matrix_model(
    checkpoint: Path,
    *,
    matrix_name: str,
    source: str,
) -> tuple[OneMatrixModel, dict[str, Any]]:
    holder, payload = load_weightwatcher_checkpoint(
        checkpoint,
        source=source,
    )
    layers = dict(holder.named_children())
    if matrix_name not in layers:
        raise KeyError(
            f"{matrix_name!r} not found in {checkpoint.name}; "
            f"available={list(layers)}"
        )
    return OneMatrixModel(matrix_name, layers[matrix_name].weight), payload


def _legacy_optimizer_index(walk_dir: Path) -> pd.DataFrame:
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
                    "step": step,
                    "optimizer_step": step,
                    "effective_batch": step,
                    "microbatch_index": 0,
                    "global_microbatch": 0,
                    "weightwatcher_checkpoint": str(path),
                }
            )
    return pd.DataFrame(rows)


def load_snapshot_index(
    walk_dir: Path,
    *,
    cadence: str,
    source: str,
    first_effective_batch: int,
    last_effective_batch: int | None,
    max_checkpoints: int,
) -> pd.DataFrame:
    index_path = walk_dir / "snapshot_index.csv"
    if not index_path.is_file():
        index_path = walk_dir / "checkpoint_index.csv"
    if index_path.is_file():
        frame = pd.read_csv(index_path)
    else:
        frame = _legacy_optimizer_index(walk_dir)

    if frame.empty:
        raise FileNotFoundError(
            f"No captured checkpoints found under {walk_dir}"
        )

    for column in (
        "timeline_index",
        "step",
        "optimizer_step",
        "effective_batch",
        "microbatch_index",
        "global_microbatch",
        "epoch",
        "tokens_seen",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "snapshot_kind" not in frame.columns:
        step_series = (
            frame["step"]
            if "step" in frame.columns
            else pd.Series(0, index=frame.index)
        )
        frame["snapshot_kind"] = np.where(
            step_series.fillna(0).astype(int).eq(0),
            "initial",
            "optimizer_step",
        )
    if "timeline_index" not in frame.columns:
        frame["timeline_index"] = np.arange(len(frame))
    if "effective_batch" not in frame.columns:
        frame["effective_batch"] = frame.get("step", 0)

    frame = frame.dropna(subset=["weightwatcher_checkpoint"]).copy()
    frame["weightwatcher_checkpoint"] = frame[
        "weightwatcher_checkpoint"
    ].astype(str)
    frame = frame[
        frame["weightwatcher_checkpoint"].str.len() > 0
    ]

    if cadence == "optimizer":
        frame = frame[
            frame["snapshot_kind"].isin(["initial", "optimizer_step"])
        ]
    elif cadence != "microbatch":
        raise ValueError("cadence must be 'optimizer' or 'microbatch'")

    if source != "weights":
        # Only after-backward microbatch files contain accumulated gradients.
        frame = frame[frame["snapshot_kind"].eq("microbatch")]

    effective = pd.to_numeric(frame["effective_batch"], errors="coerce")
    initial = frame["snapshot_kind"].eq("initial")
    frame = frame[
        initial | effective.ge(int(first_effective_batch))
    ]
    if last_effective_batch is not None:
        frame = frame[
            initial | effective.le(int(last_effective_batch))
        ]

    frame = (
        frame.sort_values("timeline_index")
        .drop_duplicates("timeline_index", keep="last")
        .reset_index(drop=True)
    )

    paths = [Path(value) for value in frame["weightwatcher_checkpoint"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Captured checkpoint index contains missing files:\n"
            + "\n".join(map(str, missing))
        )

    if len(frame) > int(max_checkpoints):
        raise ValueError(
            f"movie requested {len(frame)} checkpoints, exceeding "
            f"--max-checkpoints={max_checkpoints}"
        )
    if len(frame) > _MAX_MOVIE_CHECKPOINTS:
        raise ValueError(
            f"movie hard cap is {_MAX_MOVIE_CHECKPOINTS} checkpoints"
        )
    return frame


def _choose_first_esd(files: list[Path]) -> Path:
    if not files:
        raise RuntimeError("WeightWatcher did not save a plot")
    preferred = [
        path
        for path in files
        if any(token in path.name.lower() for token in ("esd", "power", "pl"))
    ]
    return preferred[0] if preferred else files[0]


def analyze_snapshot(
    row: pd.Series,
    *,
    matrix_name: str,
    source: str,
    output_root: Path,
    min_evals: int,
) -> dict[str, Any]:
    checkpoint = Path(str(row["weightwatcher_checkpoint"]))
    model, payload = _one_matrix_model(
        checkpoint,
        matrix_name=matrix_name,
        source=source,
    )
    timeline = int(row["timeline_index"])
    native_dir = output_root / "native_weightwatcher" / f"snapshot_{timeline:07d}"
    if native_dir.exists():
        shutil.rmtree(native_dir)
    native_dir.mkdir(parents=True, exist_ok=True)

    watcher = ww.WeightWatcher(model=model)
    savedir = str(native_dir)
    details_raw = watcher.analyze(
        plot=True,
        savefig=savedir,
        min_evals=min_evals,
        randomize=False,
        ERG=False,
    )
    details = pd.DataFrame(details_raw)
    if details.empty:
        raise RuntimeError(
            f"WeightWatcher returned no details for {checkpoint}"
        )

    layer_id = (
        int(details.iloc[0]["layer_id"])
        if "layer_id" in details.columns
        else int(details.index[0])
    )
    eigenvalues = np.asarray(
        watcher.get_ESD(layer=layer_id),
        dtype=float,
    ).reshape(-1)
    eigenvalues = eigenvalues[
        np.isfinite(eigenvalues) & (eigenvalues > 0)
    ]
    eigenvalues.sort()
    if eigenvalues.size < min_evals:
        raise RuntimeError(
            f"Only {eigenvalues.size} positive eigenvalues for {checkpoint}"
        )

    plot_files = sorted(
        [
            path
            for path in native_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=_natural_key,
    )
    first_esd = _choose_first_esd(plot_files)
    first_dir = output_root / "native_first_esd"
    first_dir.mkdir(parents=True, exist_ok=True)
    first_target = first_dir / (
        f"first_esd_snapshot_{timeline:07d}{first_esd.suffix.lower()}"
    )
    shutil.copy2(first_esd, first_target)
    # Keep only the selected first ESD image from the native run.
    for path in plot_files:
        if path.resolve() != first_esd.resolve():
            path.unlink(missing_ok=True)

    detail_row = details.iloc[0]
    alpha = float(pd.to_numeric(detail_row.get("alpha"), errors="coerce"))
    xmin = float(pd.to_numeric(detail_row.get("xmin"), errors="coerce"))
    xmax = float(pd.to_numeric(detail_row.get("xmax"), errors="coerce"))
    if not math.isfinite(xmin) or xmin <= 0:
        xmin = float(eigenvalues.min())
    if not math.isfinite(xmax) or xmax <= xmin:
        xmax = float(eigenvalues.max())

    metadata = {
        key: row.get(key)
        for key in (
            "timeline_index",
            "snapshot_kind",
            "step",
            "optimizer_step",
            "effective_batch",
            "microbatch_index",
            "global_microbatch",
            "tokens_seen",
            "epoch",
        )
    }
    details.insert(0, "checkpoint", checkpoint.name)
    details.insert(1, "matrix_name", matrix_name)
    details.insert(2, "source", source)
    for key, value in reversed(list(metadata.items())):
        if key not in details.columns:
            details.insert(3, key, value)
    details.to_csv(native_dir / "weightwatcher_details.csv", index=False)

    esd_dir = output_root / "esd_data"
    esd_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timeline_index": timeline,
            "eigenvalue": eigenvalues,
        }
    ).to_csv(esd_dir / f"esd_snapshot_{timeline:07d}.csv", index=False)

    print(
        "[one-head-movie] "
        f"snapshot={timeline} kind={row['snapshot_kind']} "
        f"effective_batch={row.get('effective_batch')} "
        f"microbatch={row.get('microbatch_index')} "
        f"alpha={alpha:.4f}",
        flush=True,
    )
    return {
        "metadata": metadata,
        "checkpoint": str(checkpoint),
        "eigenvalues": eigenvalues,
        "alpha": alpha,
        "xmin": xmin,
        "xmax": xmax,
        "details": details,
        "native_first_esd": str(first_target),
    }


def _smooth_density(eigenvalues: np.ndarray, edges: np.ndarray) -> np.ndarray:
    density, _ = np.histogram(eigenvalues, bins=edges, density=True)
    density = np.asarray(density, dtype=float)
    if density.size >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        density = np.convolve(density, kernel, mode="same")
    return density


def _positive_geometric(left: float, right: float, fraction: float) -> float:
    if left > 0 and right > 0:
        return float(
            np.exp(
                (1.0 - fraction) * np.log(left)
                + fraction * np.log(right)
            )
        )
    return float((1.0 - fraction) * left + fraction * right)


def make_movie(
    snapshots: list[dict[str, Any]],
    *,
    matrix_name: str,
    source: str,
    cadence: str,
    output_root: Path,
    fps: int,
    frames_per_transition: int,
    bins: int,
) -> Path:
    if len(snapshots) < 2:
        raise ValueError("at least two snapshots are required for a movie")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for MP4 output; on macOS run: brew install ffmpeg"
        )

    all_values = np.concatenate(
        [snapshot["eigenvalues"] for snapshot in snapshots]
    )
    x_min = float(all_values.min())
    x_max = float(all_values.max())
    edges = np.geomspace(x_min, x_max, int(bins) + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    densities = np.vstack(
        [
            _smooth_density(snapshot["eigenvalues"], edges)
            for snapshot in snapshots
        ]
    )
    positive = densities[densities > 0]
    if positive.size == 0:
        raise RuntimeError("all ESD histogram bins are zero")
    floor = float(positive.min()) * 0.2
    densities = np.maximum(densities, floor)

    frame_map: list[tuple[int, float]] = []
    for index in range(len(snapshots) - 1):
        for subframe in range(int(frames_per_transition)):
            fraction = subframe / float(frames_per_transition)
            # Cosine ease-in/ease-out removes click-click motion.
            eased = 0.5 - 0.5 * math.cos(math.pi * fraction)
            frame_map.append((index, eased))
    frame_map.append((len(snapshots) - 2, 1.0))

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    empirical_line, = ax.plot(
        [],
        [],
        marker="o",
        markersize=3,
        linewidth=2.2,
        label="ESD",
    )
    fit_line, = ax.plot(
        [],
        [],
        linestyle="--",
        linewidth=2.0,
        label="WeightWatcher PL fit",
    )
    xmin_line = ax.axvline(x_min, linestyle=":", linewidth=1.5)
    progress_line, = ax.plot([], [], linewidth=5, solid_capstyle="round")
    title = ax.set_title("")
    annotation = ax.text(
        0.02,
        0.03,
        "",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min * 0.9, x_max * 1.1)
    ax.set_ylim(floor * 0.55, float(densities.max()) * 2.0)
    ax.set_xlabel(r"Eigenvalue $\lambda$")
    ax.set_ylabel(r"ESD density $\rho(\lambda)$")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right")

    total_progress = max(len(snapshots) - 1, 1)

    def update(frame_number: int):
        index, fraction = frame_map[frame_number]
        left = snapshots[index]
        right = snapshots[index + 1]
        density = np.exp(
            (1.0 - fraction) * np.log(densities[index])
            + fraction * np.log(densities[index + 1])
        )
        empirical_line.set_data(centers, density)

        alpha = (
            (1.0 - fraction) * left["alpha"]
            + fraction * right["alpha"]
        )
        xmin = _positive_geometric(left["xmin"], right["xmin"], fraction)
        xmax = _positive_geometric(left["xmax"], right["xmax"], fraction)
        xmin = float(np.clip(xmin, centers.min(), centers.max()))
        xmax = float(np.clip(xmax, xmin, centers.max()))
        xmin_line.set_xdata([xmin, xmin])

        if math.isfinite(alpha) and xmax > xmin:
            fit_x = np.geomspace(xmin, xmax, 180)
            anchor_log_y = np.interp(
                np.log(xmin),
                np.log(centers),
                np.log(density),
            )
            anchor_y = float(np.exp(anchor_log_y))
            fit_y = anchor_y * np.power(fit_x / xmin, -alpha)
            fit_line.set_data(fit_x, fit_y)
        else:
            fit_line.set_data([], [])

        left_meta = left["metadata"]
        right_meta = right["metadata"]
        timeline = (
            (1.0 - fraction) * float(left_meta["timeline_index"])
            + fraction * float(right_meta["timeline_index"])
        )
        exact = fraction < 1e-12 or abs(fraction - 1.0) < 1e-12
        state_text = "actual checkpoint" if exact else "visual interpolation"
        title.set_text(
            f"MuonClip {matrix_name} spectral flow\n"
            f"{cadence} timeline {timeline:.2f} — {state_text}"
        )

        active_meta = left_meta if fraction < 0.5 else right_meta
        annotation.set_text(
            f"source: {source}\n"
            f"kind: {active_meta.get('snapshot_kind')}\n"
            f"effective batch: {active_meta.get('effective_batch')}\n"
            f"microbatch: {active_meta.get('microbatch_index')}\n"
            f"alpha ≈ {alpha:.4f}"
        )

        progress = (index + fraction) / total_progress
        progress_line.set_data(
            [x_min, _positive_geometric(x_min, x_max, progress)],
            [floor * 0.72, floor * 0.72],
        )
        return (
            empirical_line,
            fit_line,
            xmin_line,
            progress_line,
            title,
            annotation,
        )

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_map),
        interval=1000 / int(fps),
        blit=False,
    )
    video_dir = output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_dir / (
        f"{matrix_name}_{source}_{cadence}_"
        f"{len(snapshots):03d}_checkpoints.mp4"
    )
    writer = animation.FFMpegWriter(
        fps=int(fps),
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    movie.save(output_path, writer=writer, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run WeightWatcher on one MuonClip matrix checkpoint-by-checkpoint "
            "and make a smooth log-log ESD MP4."
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
        default="microbatch",
    )
    parser.add_argument(
        "--source",
        choices=("weights", "accumulated_gradients"),
        default="weights",
    )
    parser.add_argument("--first-effective-batch", type=int, default=1)
    parser.add_argument("--last-effective-batch", type=int)
    parser.add_argument("--max-checkpoints", type=int, default=500)
    parser.add_argument("--min-evals", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames-per-transition", type=int, default=8)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if not args.walk_dir:
        raise SystemExit("Set WALK_DIR or pass --walk-dir")
    if not 1 <= args.max_checkpoints <= _MAX_MOVIE_CHECKPOINTS:
        raise SystemExit("--max-checkpoints must be between 1 and 500")
    if args.first_effective_batch < 1:
        raise SystemExit("--first-effective-batch must be at least 1")
    if (
        args.last_effective_batch is not None
        and args.last_effective_batch < args.first_effective_batch
    ):
        raise SystemExit(
            "--last-effective-batch must be >= --first-effective-batch"
        )

    walk_dir = Path(args.walk_dir).expanduser().resolve()
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else walk_dir
        / "diagnostics"
        / f"esd_movie_{args.matrix}_{args.source}_{args.cadence}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    index = load_snapshot_index(
        walk_dir,
        cadence=args.cadence,
        source=args.source,
        first_effective_batch=args.first_effective_batch,
        last_effective_batch=args.last_effective_batch,
        max_checkpoints=args.max_checkpoints,
    )
    snapshots = [
        analyze_snapshot(
            row,
            matrix_name=args.matrix,
            source=args.source,
            output_root=output_root,
            min_evals=args.min_evals,
        )
        for _, row in index.iterrows()
    ]
    details = pd.concat(
        [snapshot["details"] for snapshot in snapshots],
        ignore_index=True,
    )
    details.to_csv(
        output_root / "weightwatcher_details_all_snapshots.csv",
        index=False,
    )
    video = make_movie(
        snapshots,
        matrix_name=args.matrix,
        source=args.source,
        cadence=args.cadence,
        output_root=output_root,
        fps=args.fps,
        frames_per_transition=args.frames_per_transition,
        bins=args.bins,
    )
    movie_manifest = {
        "walk_dir": str(walk_dir),
        "output_root": str(output_root),
        "matrix": args.matrix,
        "source": args.source,
        "cadence": args.cadence,
        "checkpoint_count": len(snapshots),
        "fps": args.fps,
        "frames_per_transition": args.frames_per_transition,
        "video": str(video),
    }
    (output_root / "movie_manifest.json").write_text(
        json.dumps(movie_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[one-head-movie] complete: {video}", flush=True)


if __name__ == "__main__":
    main()

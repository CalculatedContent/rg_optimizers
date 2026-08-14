from __future__ import annotations

"""Render a movie from the native WeightWatcher ESD plot for one matrix.

This command is deliberately headless on macOS: WeightWatcher writes plot files
only; no GUI windows are opened. For each selected checkpoint we analyze exactly
one named matrix, retain the first native WeightWatcher ESD/log-log figure, and
cross-fade those real figures into an H.264 MP4.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import weightwatcher as ww

from .muonclip_capture import load_weightwatcher_checkpoint

MAX_CHECKPOINTS = 500


class OneMatrixModel(nn.Module):
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

    # Compatibility with older captures that only have ww_step_N files.
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
    selected = frame[
        initial | effective.ge(first_effective_batch)
    ].copy()
    if last_effective_batch is not None:
        selected = selected[
            initial | pd.to_numeric(
                selected["effective_batch"], errors="coerce"
            ).le(last_effective_batch)
        ]
    selected = selected.sort_values("timeline_index").reset_index(drop=True)

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


def _choose_first_esd(files: list[Path]) -> Path:
    if not files:
        raise RuntimeError("WeightWatcher saved no plot image")
    # Prefer names that explicitly indicate ESD / power-law. Otherwise the first
    # native WeightWatcher image is used, which is the standard ESD figure for
    # this single-layer analysis.
    preferred = [
        path
        for path in files
        if any(token in path.name.lower() for token in ("esd", "power", "pl"))
    ]
    return preferred[0] if preferred else files[0]


def _run_weightwatcher_frame(
    row: pd.Series,
    *,
    matrix_name: str,
    output_root: Path,
    min_evals: int,
) -> Path:
    checkpoint = Path(str(row["weightwatcher_checkpoint"]))
    timeline = int(row["timeline_index"])
    native_dir = output_root / "native" / f"snapshot_{timeline:07d}"
    if native_dir.exists():
        shutil.rmtree(native_dir)
    native_dir.mkdir(parents=True, exist_ok=True)

    model = _single_matrix(checkpoint, matrix_name)
    watcher = ww.WeightWatcher(model=model)

    # Prevent WeightWatcher/Matplotlib from opening GUI windows even if a user's
    # matplotlibrc requests an interactive backend.
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        savedir = str(native_dir)
        details = watcher.analyze(
            plot=True,
            savefig=savedir,
            min_evals=min_evals,
            randomize=False,
            ERG=False,
        )
    finally:
        plt.show = old_show
        plt.close("all")

    pd.DataFrame(details).to_csv(
        native_dir / "weightwatcher_details.csv",
        index=False,
    )

    images = sorted(
        [
            path
            for path in native_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=_natural_key,
    )
    esd = _choose_first_esd(images)

    frame_dir = output_root / "frames_native_esd"
    frame_dir.mkdir(parents=True, exist_ok=True)
    target = frame_dir / f"frame_{timeline:07d}.png"

    image = mpimg.imread(esd)
    plt.imsave(target, image)

    # Remove every other native plot so this command leaves one ESD image per
    # checkpoint rather than a directory full of unrelated diagnostics.
    for path in images:
        path.unlink(missing_ok=True)

    print(
        "[one-head-esd-movie] "
        f"snapshot={timeline} kind={row['snapshot_kind']} "
        f"batch={row.get('effective_batch')} "
        f"matrix={matrix_name} frame={target.name}",
        flush=True,
    )
    return target


def _normalize_rgba(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 3:
        alpha = np.ones((*image.shape[:2], 1), dtype=np.float32)
        image = np.concatenate([image, alpha], axis=2)
    return np.clip(image, 0.0, 1.0)


def _make_transition_frames(
    native_frames: list[Path],
    *,
    output_root: Path,
    frames_per_transition: int,
) -> Path:
    if len(native_frames) < 2:
        raise ValueError("at least two native ESD frames are required")

    movie_frames = output_root / "movie_frames"
    if movie_frames.exists():
        shutil.rmtree(movie_frames)
    movie_frames.mkdir(parents=True)

    images = [_normalize_rgba(mpimg.imread(path)) for path in native_frames]
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise RuntimeError(
            "WeightWatcher ESD figures changed image dimensions across "
            "checkpoints; cannot cross-fade them safely"
        )

    frame_number = 0
    for index in range(len(images) - 1):
        left = images[index]
        right = images[index + 1]
        for subframe in range(frames_per_transition):
            fraction = subframe / float(frames_per_transition)
            # Cosine easing gives continuous motion rather than click-click cuts.
            eased = 0.5 - 0.5 * np.cos(np.pi * fraction)
            blended = (1.0 - eased) * left + eased * right
            target = movie_frames / f"frame_{frame_number:06d}.png"
            plt.imsave(target, blended)
            frame_number += 1

    plt.imsave(movie_frames / f"frame_{frame_number:06d}.png", images[-1])
    return movie_frames


def _encode_mp4(
    frame_dir: Path,
    *,
    output_path: Path,
    fps: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required; install on macOS with: brew install ffmpeg"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a smooth MP4 from the native WeightWatcher log-log ESD "
            "plot for exactly one MuonClip matrix."
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
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if not args.walk_dir:
        raise SystemExit("Set WALK_DIR or pass --walk-dir")
    if not 1 <= args.max_checkpoints <= MAX_CHECKPOINTS:
        raise SystemExit("--max-checkpoints must be between 1 and 500")
    if args.frames_per_transition < 1:
        raise SystemExit("--frames-per-transition must be positive")

    walk_dir = Path(args.walk_dir).expanduser().resolve()
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else walk_dir
        / "diagnostics"
        / f"native_esd_movie_{args.matrix}_{args.cadence}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    index = _load_index(walk_dir, args.cadence)
    index = _select_rows(
        index,
        first_effective_batch=args.first_effective_batch,
        last_effective_batch=args.last_effective_batch,
        max_checkpoints=args.max_checkpoints,
    )
    index.to_csv(output_root / "movie_checkpoint_index.csv", index=False)

    native_frames = [
        _run_weightwatcher_frame(
            row,
            matrix_name=args.matrix,
            output_root=output_root,
            min_evals=args.min_evals,
        )
        for _, row in index.iterrows()
    ]

    movie_frames = _make_transition_frames(
        native_frames,
        output_root=output_root,
        frames_per_transition=args.frames_per_transition,
    )
    output_path = (
        output_root
        / "videos"
        / f"{args.matrix}_{args.cadence}_native_weightwatcher_esd.mp4"
    )
    _encode_mp4(movie_frames, output_path=output_path, fps=args.fps)

    print()
    print(f"[one-head-esd-movie] checkpoints: {len(native_frames)}")
    print(f"[one-head-esd-movie] movie: {output_path}")


if __name__ == "__main__":
    main()

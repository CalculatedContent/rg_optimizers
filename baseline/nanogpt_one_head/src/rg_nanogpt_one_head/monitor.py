from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .config import roots

_LAYER_COLUMNS = (
    "matrix_name",
    "alpha",
    "D",
    "rand_distance",
    "ERG_gap",
    "num_traps",
)
_NUMERIC_COLUMNS = (
    "step",
    "epoch",
    "alpha",
    "alpha_raw",
    "alpha_clip_xmax",
    "D",
    "rand_distance",
    "ERG_gap",
    "num_traps",
)


def resolve_run_dir(
    *,
    run_dir: str | Path | None,
    results_root: str | Path | None,
    optimizer: str,
    seed: int,
    device: str,
) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    root = (
        Path(results_root)
        if results_root is not None
        else roots(device)["results"]
    )
    return root / str(optimizer) / f"seed_{int(seed)}"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    # metrics.csv is appended in place. A monitor can occasionally catch the
    # writer between bytes, so retry once rather than failing the live display.
    for attempt in range(2):
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            if attempt:
                raise
            time.sleep(0.1)
    return pd.DataFrame()


def load_monitor_frames(
    run_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir)
    metrics = _read_csv(run_dir / "metrics.csv")
    layers = _read_csv(run_dir / "spectral" / "layers.csv")
    return metrics, layers


def load_series_frames(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join retained CSV history even after older model files are pruned."""
    root = Path(root)
    state = json.loads((root / "series.json").read_text())
    dirs = [Path(state["origin_checkpoint"]).parent]
    dirs.extend(root / record["directory"] / "muon_clip" / f"seed_{state['seed']}"
                for record in state["completed_segments"])
    if state.get("active_segment"):
        dirs.append(root / state["active_segment"] / "muon_clip" / f"seed_{state['seed']}")
    collected = [[], []]
    for run_dir in dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        offset = int((manifest.get("continuation") or {}).get("global_step_offset", 0))
        tokens = int(manifest["tokens_per_step"])
        train_tokens = int(manifest["data_metadata"]["splits"]["train"])
        for index, frame in enumerate(load_monitor_frames(run_dir)):
            if frame.empty:
                continue
            frame = frame.copy()
            frame["step"] = pd.to_numeric(frame["step"], errors="coerce") + offset
            frame["epoch"] = frame["step"] * tokens / train_tokens
            frame["segment_run_dir"] = str(run_dir)
            if run_dir == dirs[0]:
                frame = frame.loc[frame["step"] <= state["origin_global_step"]]
            collected[index].append(frame)
    frames = []
    for index, parts in enumerate(collected):
        if not parts:
            frames.append(pd.DataFrame())
            continue
        frame = pd.concat(parts, ignore_index=True)
        keys = ["step", "matrix_name"] if index else ["step"]
        frames.append(frame.drop_duplicates(keys, keep="last").sort_values(keys))
    return tuple(frames)


def _finite_summary(values: pd.Series) -> str:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return "n=0"
    return (
        f"n={finite.size} mean={finite.mean():.4f} "
        f"median={np.median(finite):.4f} "
        f"min={finite.min():.4f} max={finite.max():.4f}"
    )


def _format_table(frame: pd.DataFrame) -> str:
    return frame.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )


def format_monitor_snapshot(
    run_dir: str | Path,
    metrics: pd.DataFrame,
    layers: pd.DataFrame,
    *,
    recent: int = 8,
) -> str:
    run_dir = Path(run_dir)
    lines = [f"RUN: {run_dir}"]

    if metrics.empty:
        lines.extend(["", "Waiting for metrics.csv..."])
    else:
        metrics = metrics.copy()
        metrics["step"] = pd.to_numeric(
            metrics["step"],
            errors="coerce",
        )
        metrics = metrics.dropna(subset=["step"]).sort_values("step")
        if metrics.empty:
            lines.extend(
                ["", "Waiting for a complete training-metric row..."]
            )
        else:
            row = metrics.iloc[-1]
            lines.extend(
                [
                    "",
                    (
                        "TRAINING: "
                        f"step={int(row['step'])} "
                        f"epoch={float(row['epoch']):.3f} "
                        f"lr={float(row['primary_lr']):.6g} "
                        f"val_loss={float(row['val_loss']):.4f} "
                        f"val_acc={100.0 * float(row['val_accuracy']):.2f}%"
                    ),
                ]
            )

    if not metrics.empty and "test_accuracy" in metrics.columns:
        measured = metrics.loc[pd.to_numeric(metrics["test_accuracy"], errors="coerce").notna()]
        if not measured.empty:
            row = measured.iloc[-1]
            lines.extend(["", f"LATEST FIXED TEST PROBE: step={int(row['step'])} "
                          f"test_loss={float(row['test_loss']):.4f} "
                          f"test_acc={100 * float(row['test_accuracy']):.2f}%",
                          "RECENT TEST PROBE HISTORY",
                          _format_table(measured[["step", "epoch", "test_loss", "test_accuracy", "val_accuracy"]].tail(recent))])

    if layers.empty:
        lines.extend(["", "Waiting for spectral/layers.csv..."])
        return "\n".join(lines)

    layers = layers.copy()
    missing = [
        column
        for column in _LAYER_COLUMNS
        if column not in layers.columns
    ]
    if missing:
        lines.extend(
            [
                "",
                "INCOMPATIBLE SPECTRAL OUTPUT: missing "
                + ", ".join(missing),
                (
                    "rand_distance should be returned directly by WeightWatcher "
                    "when randomize=True."
                ),
            ]
        )
        return "\n".join(lines)

    for column in _NUMERIC_COLUMNS:
        if column in layers.columns:
            layers[column] = pd.to_numeric(
                layers[column],
                errors="coerce",
            )
    layers = layers.dropna(subset=["step"]).sort_values(
        ["step", "matrix_name"]
    )
    if layers.empty:
        lines.extend(["", "Waiting for a complete WeightWatcher row..."])
        return "\n".join(lines)

    latest_step = int(layers["step"].max())
    latest = layers[layers["step"] == latest_step].copy()
    latest_epoch = float(latest["epoch"].iloc[0])
    table_columns = list(_LAYER_COLUMNS)
    for column in ("alpha_raw", "alpha_clip_xmax", "num_fingers"):
        if column in latest.columns:
            table_columns.append(column)
    table = latest[table_columns].sort_values("matrix_name")

    lines.extend(
        [
            "",
            (
                "LATEST WEIGHTWATCHER: "
                f"step={latest_step} epoch={latest_epoch:.3f}"
            ),
            "",
            _format_table(table),
            "",
            "ALPHA PRIMARY: " + _finite_summary(latest["alpha"]),
            "RAND_DISTANCE: "
            + _finite_summary(latest["rand_distance"]),
        ]
    )

    if "alpha_raw" in latest.columns:
        lines.append(
            "ALPHA RAW:     " + _finite_summary(latest["alpha_raw"])
        )
    if "alpha_clip_xmax" in latest.columns:
        lines.append(
            "ALPHA CLIPPED: "
            + _finite_summary(latest["alpha_clip_xmax"])
        )

    aggregations = {
        "alpha_median": ("alpha", "median"),
        "rand_distance_median": ("rand_distance", "median"),
        "D_median": ("D", "median"),
        "ERG_gap_median": ("ERG_gap", "median"),
        "num_traps_mean": ("num_traps", "mean"),
    }
    if "alpha_raw" in layers.columns:
        aggregations["alpha_raw_median"] = ("alpha_raw", "median")
    if "alpha_clip_xmax" in layers.columns:
        aggregations["alpha_clip_xmax_median"] = (
            "alpha_clip_xmax",
            "median",
        )
    recent_frame = (
        layers.groupby(["step", "epoch"], as_index=False)
        .agg(**aggregations)
        .sort_values("step")
        .tail(max(1, int(recent)))
    )
    lines.extend(
        [
            "",
            "RECENT SPECTRAL CHECKPOINT MEDIANS",
            _format_table(recent_frame),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor one-head nanoGPT training, layer alpha, fit D, "
            "WeightWatcher rand_distance, ERG gap, and traps"
        )
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--series-root", help="join all continuation segments on cumulative steps")
    parser.add_argument("--export-metrics", type=Path, help="write the displayed metrics history as CSV")
    parser.add_argument("--results-root")
    parser.add_argument("--optimizer", default="muon")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "tpu", "xla", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--recent", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.recent < 1:
        parser.error("--recent must be positive")

    run_dir = Path(args.series_root) if args.series_root else resolve_run_dir(
        run_dir=args.run_dir,
        results_root=args.results_root,
        optimizer=args.optimizer,
        seed=args.seed,
        device=args.device,
    )

    try:
        while True:
            metrics, layers = (load_series_frames(run_dir) if args.series_root else load_monitor_frames(run_dir))
            if args.export_metrics:
                args.export_metrics.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.export_metrics.with_suffix(".csv.tmp")
                metrics.to_csv(temporary, index=False)
                temporary.replace(args.export_metrics)
            if not args.no_clear and not args.once:
                print("\033[2J\033[H", end="")
            print(
                format_monitor_snapshot(
                    run_dir,
                    metrics,
                    layers,
                    recent=args.recent,
                ),
                flush=True,
            )
            if args.once:
                break
            time.sleep(float(args.interval))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

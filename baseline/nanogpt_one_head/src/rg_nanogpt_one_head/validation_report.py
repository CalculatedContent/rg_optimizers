from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .config import roots

_REQUIRED_COLUMNS = (
    "step",
    "epoch",
    "val_accuracy",
    "val_loss",
    "primary_lr",
)
_TRAIN_COLUMNS = (
    "train_accuracy",
    "train_loss",
)
_NUMERIC_COLUMNS = (
    *_REQUIRED_COLUMNS,
    *_TRAIN_COLUMNS,
)


def resolve_metrics_path(
    *,
    metrics_csv: str | Path | None,
    run_dir: str | Path | None,
    results_root: str | Path | None,
    optimizer: str,
    seed: int,
    device: str,
) -> Path:
    """Resolve a metrics.csv path for any one-head baseline run."""

    if metrics_csv is not None and run_dir is not None:
        raise ValueError("choose either metrics_csv or run_dir, not both")
    if metrics_csv is not None:
        return Path(metrics_csv)
    if run_dir is not None:
        return Path(run_dir) / "metrics.csv"

    root = (
        Path(results_root)
        if results_root is not None
        else roots(device)["results"]
    )
    return root / str(optimizer) / f"seed_{int(seed)}" / "metrics.csv"


def read_metrics(path: str | Path) -> pd.DataFrame:
    """Read a metrics file safely while an active run may be appending to it."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"metrics file does not exist: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"metrics file is empty: {path}")

    for attempt in range(2):
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            if attempt:
                raise
            time.sleep(0.1)
    raise RuntimeError(f"could not read metrics file: {path}")


def normalize_metrics(
    frame: pd.DataFrame,
    *,
    include_train: bool = False,
) -> pd.DataFrame:
    """Validate and normalize the standard one-head metrics contract."""

    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if include_train:
        missing.extend(
            column for column in _TRAIN_COLUMNS if column not in frame.columns
        )
    if missing:
        raise ValueError(
            "metrics.csv is missing required columns: "
            + ", ".join(dict.fromkeys(missing))
        )

    result = frame.copy()
    for column in _NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    result = (
        result.dropna(subset=list(_REQUIRED_COLUMNS))
        .sort_values("step")
        .drop_duplicates("step", keep="last")
        .reset_index(drop=True)
    )
    if result.empty:
        raise ValueError("metrics.csv contains no complete validation rows")

    accuracy = result["val_accuracy"].to_numpy(dtype=float)
    finite_accuracy = accuracy[np.isfinite(accuracy)]
    if finite_accuracy.size and (
        float(finite_accuracy.min()) < -1e-9
        or float(finite_accuracy.max()) > 1.0 + 1e-9
    ):
        raise ValueError(
            "val_accuracy must use the baseline fractional convention in [0, 1]"
        )

    if include_train:
        train_accuracy = result["train_accuracy"].to_numpy(dtype=float)
        finite_train = train_accuracy[np.isfinite(train_accuracy)]
        if finite_train.size and (
            float(finite_train.min()) < -1e-9
            or float(finite_train.max()) > 1.0 + 1e-9
        ):
            raise ValueError(
                "train_accuracy must use the baseline fractional convention in [0, 1]"
            )

    return result


def _epoch_targets(
    *,
    maximum_epoch: float,
    interval: float,
    start_epoch: float,
    end_epoch: float | None,
) -> list[float]:
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be positive and finite")
    if not math.isfinite(start_epoch) or start_epoch < 0:
        raise ValueError("start_epoch must be nonnegative and finite")
    if maximum_epoch < start_epoch - 1e-12:
        raise ValueError(
            f"run has reached epoch {maximum_epoch:.6f}, before start_epoch={start_epoch}"
        )

    if end_epoch is None:
        final_target = math.floor((maximum_epoch + 1e-10) / interval) * interval
    else:
        if not math.isfinite(end_epoch) or end_epoch < start_epoch:
            raise ValueError("end_epoch must be finite and no smaller than start_epoch")
        if end_epoch > maximum_epoch + 1e-9:
            raise ValueError(
                f"end_epoch={end_epoch} exceeds current maximum epoch {maximum_epoch:.6f}"
            )
        final_target = end_epoch

    count = int(math.floor((final_target - start_epoch) / interval + 1e-10)) + 1
    return [round(start_epoch + index * interval, 12) for index in range(count)]


def validation_by_epoch(
    metrics: pd.DataFrame,
    *,
    interval: float = 1.0,
    start_epoch: float = 0.0,
    end_epoch: float | None = None,
    include_current: bool = False,
    include_train: bool = False,
) -> pd.DataFrame:
    """Select the validation measurement nearest each requested epoch.

    The default interval of one prints epochs 0, 1, 2, ... through the latest
    completed integer epoch. `include_current=True` additionally appends the
    most recent partial-epoch measurement when it is not already on the grid.
    """

    normalized = normalize_metrics(metrics, include_train=include_train)
    maximum_epoch = float(normalized["epoch"].max())
    targets = _epoch_targets(
        maximum_epoch=maximum_epoch,
        interval=float(interval),
        start_epoch=float(start_epoch),
        end_epoch=end_epoch,
    )

    rows: list[dict[str, float | int]] = []
    epochs = normalized["epoch"].to_numpy(dtype=float)
    for target in targets:
        distances = np.abs(epochs - target)
        minimum = float(np.nanmin(distances))
        candidate_indices = np.flatnonzero(
            np.isclose(distances, minimum, rtol=0.0, atol=1e-12)
        )
        # Prefer the later measurement only in the rare exact tie between an
        # evaluation immediately before and immediately after a target epoch.
        index = int(candidate_indices[-1])
        source = normalized.iloc[index]
        row: dict[str, float | int] = {
            "TARGET_EPOCH": float(target),
            "ACTUAL_EPOCH": float(source["epoch"]),
            "EPOCH_ERROR": float(source["epoch"] - target),
            "STEP": int(source["step"]),
            "VAL_ACC_%": 100.0 * float(source["val_accuracy"]),
            "VAL_LOSS": float(source["val_loss"]),
            "LR": float(source["primary_lr"]),
            "IS_CURRENT": 0,
        }
        if include_train:
            row.update(
                {
                    "TRAIN_ACC_%": 100.0 * float(source["train_accuracy"]),
                    "TRAIN_LOSS": float(source["train_loss"]),
                }
            )
        rows.append(row)

    if include_current:
        current = normalized.iloc[-1]
        current_epoch = float(current["epoch"])
        if not any(
            math.isclose(current_epoch, float(row["ACTUAL_EPOCH"]), abs_tol=1e-12)
            and int(current["step"]) == int(row["STEP"])
            for row in rows
        ):
            row = {
                "TARGET_EPOCH": current_epoch,
                "ACTUAL_EPOCH": current_epoch,
                "EPOCH_ERROR": 0.0,
                "STEP": int(current["step"]),
                "VAL_ACC_%": 100.0 * float(current["val_accuracy"]),
                "VAL_LOSS": float(current["val_loss"]),
                "LR": float(current["primary_lr"]),
                "IS_CURRENT": 1,
            }
            if include_train:
                row.update(
                    {
                        "TRAIN_ACC_%": 100.0 * float(current["train_accuracy"]),
                        "TRAIN_LOSS": float(current["train_loss"]),
                    }
                )
            rows.append(row)

    columns = [
        "TARGET_EPOCH",
        "ACTUAL_EPOCH",
        "EPOCH_ERROR",
        "STEP",
    ]
    if include_train:
        columns.extend(["TRAIN_ACC_%", "VAL_ACC_%", "TRAIN_LOSS", "VAL_LOSS"])
    else:
        columns.extend(["VAL_ACC_%", "VAL_LOSS"])
    columns.extend(["LR", "IS_CURRENT"])

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["ACTUAL_EPOCH", "STEP"]
    ).reset_index(drop=True)


def all_validation_evaluations(
    metrics: pd.DataFrame,
    *,
    include_train: bool = False,
) -> pd.DataFrame:
    normalized = normalize_metrics(metrics, include_train=include_train)
    result = pd.DataFrame(
        {
            "ACTUAL_EPOCH": normalized["epoch"].astype(float),
            "STEP": normalized["step"].astype(int),
            "VAL_ACC_%": 100.0 * normalized["val_accuracy"].astype(float),
            "VAL_LOSS": normalized["val_loss"].astype(float),
            "LR": normalized["primary_lr"].astype(float),
        }
    )
    if include_train:
        result.insert(
            2,
            "TRAIN_ACC_%",
            100.0 * normalized["train_accuracy"].astype(float),
        )
        result.insert(
            4,
            "TRAIN_LOSS",
            normalized["train_loss"].astype(float),
        )
    return result


def format_summary(metrics: pd.DataFrame) -> str:
    normalized = normalize_metrics(metrics)
    current = normalized.iloc[-1]
    best_accuracy = normalized.loc[normalized["val_accuracy"].idxmax()]
    best_loss = normalized.loc[normalized["val_loss"].idxmin()]

    return "\n".join(
        [
            "CURRENT",
            (
                f"epoch={float(current['epoch']):.4f} "
                f"step={int(current['step'])} "
                f"val_acc={100.0 * float(current['val_accuracy']):.2f}% "
                f"val_loss={float(current['val_loss']):.4f} "
                f"lr={float(current['primary_lr']):.6g}"
            ),
            "",
            "BEST VALIDATION ACCURACY (ALL EVALUATIONS)",
            (
                f"epoch={float(best_accuracy['epoch']):.4f} "
                f"step={int(best_accuracy['step'])} "
                f"val_acc={100.0 * float(best_accuracy['val_accuracy']):.2f}% "
                f"val_loss={float(best_accuracy['val_loss']):.4f}"
            ),
            "",
            "MINIMUM VALIDATION LOSS (ALL EVALUATIONS)",
            (
                f"epoch={float(best_loss['epoch']):.4f} "
                f"step={int(best_loss['step'])} "
                f"val_acc={100.0 * float(best_loss['val_accuracy']):.2f}% "
                f"val_loss={float(best_loss['val_loss']):.4f}"
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print validation accuracy and loss by epoch for any one-head "
            "nanoGPT baseline run"
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--metrics-csv")
    source.add_argument("--run-dir")
    parser.add_argument("--results-root")
    parser.add_argument("--optimizer", default="muon")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "tpu", "xla", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="epoch spacing; default prints one row per integer epoch",
    )
    parser.add_argument("--start-epoch", type=float, default=0.0)
    parser.add_argument("--end-epoch", type=float)
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="append the latest partial epoch when it is off the requested grid",
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="also print training accuracy and loss at the selected rows",
    )
    parser.add_argument(
        "--all-evaluations",
        action="store_true",
        help="print every validation evaluation rather than sampling by epoch",
    )
    parser.add_argument(
        "--output-csv",
        help="optionally save the displayed table as CSV",
    )
    args = parser.parse_args()

    metrics_path = resolve_metrics_path(
        metrics_csv=args.metrics_csv,
        run_dir=args.run_dir,
        results_root=args.results_root,
        optimizer=args.optimizer,
        seed=args.seed,
        device=args.device,
    )
    metrics = read_metrics(metrics_path)

    if args.all_evaluations:
        report = all_validation_evaluations(
            metrics,
            include_train=args.include_train,
        )
    else:
        report = validation_by_epoch(
            metrics,
            interval=args.interval,
            start_epoch=args.start_epoch,
            end_epoch=args.end_epoch,
            include_current=args.include_current,
            include_train=args.include_train,
        )

    print(f"METRICS: {metrics_path}\n")
    print(
        report.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\n" + format_summary(metrics))

    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output, index=False)
        print(f"\nWROTE: {output}")


if __name__ == "__main__":
    main()

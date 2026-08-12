from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .model import GPT, transformer_matrix_items
from .runtime import (
    capture_accelerator_rng_state,
    model_device,
    restore_accelerator_rng_state,
    synchronize,
)

SPECTRAL_METRICS = (
    "alpha",
    "alpha_weighted",
    "ERG_gap",
    "num_traps",
    "rand_distance",
    "detX_num",
    "num_pl_spikes",
    "num_ERG_spikes",
    "D",
    "stable_rank",
    "mp_softrank",
    "log_norm",
    "log_spectral_norm",
    "entropy",
    "Lambda",
    "rank_loss",
)


class WeightMatrixHolder(nn.Module):
    """CPU-only Linear view of the six one-block transformer matrices."""

    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.matrix_metadata: list[dict[str, object]] = []
        for (
            name,
            matrix_type,
            block,
            weight,
        ) in transformer_matrix_items(model):
            layer = nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=False,
            )
            layer.weight = nn.Parameter(
                weight.detach().float().cpu().clone(),
                requires_grad=False,
            )
            self.add_module(name, layer)
            self.matrix_metadata.append(
                {
                    "matrix_name": name,
                    "matrix_type": matrix_type,
                    "block": int(block),
                }
            )


def _attach_matrix_metadata(
    frame: pd.DataFrame,
    metadata: list[dict[str, object]],
) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    names = [str(item["matrix_name"]) for item in metadata]
    resolved: list[str | None] = [None] * len(result)
    for row_index, row in result.iterrows():
        text = " ".join(
            str(row.get(column, ""))
            for column in ("longname", "name")
        )
        for name in names:
            if name in text:
                resolved[row_index] = name
                break
    if (
        any(value is None for value in resolved)
        and len(result) == len(metadata)
    ):
        order = list(range(len(result)))
        if "layer_id" in result.columns:
            numeric = pd.to_numeric(
                result["layer_id"],
                errors="coerce",
            )
            if numeric.notna().all():
                order = list(numeric.sort_values().index)
        for metadata_index, row_index in enumerate(order):
            resolved[row_index] = names[metadata_index]
    if any(value is None for value in resolved):
        raise RuntimeError(
            "WeightWatcher rows could not be matched to all "
            "transformer matrices"
        )
    by_name = {
        str(item["matrix_name"]): item
        for item in metadata
    }
    result.insert(0, "matrix_name", resolved)
    result.insert(
        1,
        "matrix_type",
        [
            by_name[str(name)]["matrix_type"]
            for name in resolved
        ],
    )
    result.insert(
        2,
        "block",
        [
            by_name[str(name)]["block"]
            for name in resolved
        ],
    )
    return result


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _append_deduplicated(
    path: Path,
    frame: pd.DataFrame,
    keys: list[str],
) -> None:
    if path.is_file():
        existing = pd.read_csv(path)
        combined = pd.concat(
            [existing, frame],
            ignore_index=True,
            sort=False,
        )
    else:
        combined = frame.copy()
    combined = (
        combined.drop_duplicates(keys, keep="last")
        .sort_values(keys)
    )
    _atomic_csv(path, combined)


def summarize_spectral_frame(
    frame: pd.DataFrame,
    *,
    step: int,
    tokens_seen: int,
    epoch: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "epoch": float(epoch),
        "n_matrices": int(len(frame)),
    }
    for metric in SPECTRAL_METRICS:
        values = (
            pd.to_numeric(
                frame[metric],
                errors="coerce",
            )
            if metric in frame.columns
            else pd.Series(dtype=float)
        )
        array = values.to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        finite = array[np.isfinite(array)]
        summary[f"{metric}_n"] = int(finite.size)
        for statistic in (
            "mean",
            "median",
            "std",
            "min",
            "max",
        ):
            summary[f"{metric}_{statistic}"] = float("nan")
        if finite.size:
            summary[f"{metric}_mean"] = float(
                np.mean(finite)
            )
            summary[f"{metric}_median"] = float(
                np.median(finite)
            )
            summary[f"{metric}_std"] = (
                float(np.std(finite, ddof=1))
                if finite.size > 1
                else 0.0
            )
            summary[f"{metric}_min"] = float(
                np.min(finite)
            )
            summary[f"{metric}_max"] = float(
                np.max(finite)
            )
    return summary


def run_weightwatcher(
    model: GPT,
    run_dir: str | Path,
    *,
    step: int,
    tokens_seen: int,
    train_tokens: int,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run WeightWatcher exactly with ERG=True and randomize=True.

    `alpha`, `ERG_gap`, `num_traps`, and `rand_distance` are retained directly
    from WeightWatcher. `rand_distance` is the Jensen-Shannon distance between
    the empirical ESD and the entry-wise randomized ESD. No fallback alpha,
    proxy trap count, synthesized ERG gap, or replacement random-distance
    statistic is permitted. Every CPU and accelerator RNG stream is restored
    after the randomized diagnostic so measurement cannot change the
    subsequent training path.
    """

    try:
        import weightwatcher as ww
    except ImportError as exc:
        raise RuntimeError(
            "WeightWatcher is required; install the experiment dependencies "
            "with `python -m pip install -e .`"
        ) from exc

    run_dir = Path(run_dir)
    spectral_root = run_dir / "spectral"
    raw_root = spectral_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = (
        raw_root
        / f"weightwatcher_step_{int(step):07d}.csv"
    )
    if raw_path.is_file():
        frame = pd.read_csv(raw_path)
        return summarize_spectral_frame(
            frame,
            step=step,
            tokens_seen=tokens_seen,
            epoch=tokens_seen / max(1, int(train_tokens)),
        )

    device = model_device(model)
    synchronize(device)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    accelerator_state = capture_accelerator_rng_state(device)
    diagnostic_seed = int(seed) + 1_000_003 + int(step)
    random.seed(diagnostic_seed)
    np.random.seed(diagnostic_seed % (2**32 - 1))
    torch.manual_seed(diagnostic_seed)

    try:
        # WeightWatcher remains deliberately CPU/NumPy based. For TPU/XLA,
        # WeightMatrixHolder materializes only the six hidden matrices on the
        # host and never exposes the live accelerator model to NumPy.
        holder = WeightMatrixHolder(model)
        watcher = ww.WeightWatcher(model=holder)
        details = watcher.analyze(
            ERG=True,
            randomize=True,
            plot=False,
            min_evals=int(config.get("min_evals", 20)),
        )
        if details is None or len(details) == 0:
            raise RuntimeError(
                "WeightWatcher returned no transformer-matrix rows"
            )
        frame = _attach_matrix_metadata(
            pd.DataFrame(details),
            holder.matrix_metadata,
        )
        required_columns = (
            "alpha",
            "ERG_gap",
            "num_traps",
            "rand_distance",
        )
        missing = [
            column
            for column in required_columns
            if column not in frame.columns
        ]
        if missing:
            raise RuntimeError(
                "WeightWatcher did not return required "
                "ERG/randomization columns: "
                + ", ".join(missing)
            )
        if frame[list(required_columns)].isna().any().any():
            raise RuntimeError(
                "WeightWatcher required alpha/ERG_gap/num_traps/"
                "rand_distance values contain NaN"
            )
        epoch = tokens_seen / max(1, int(train_tokens))
        frame.insert(0, "step", int(step))
        frame.insert(1, "tokens_seen", int(tokens_seen))
        frame.insert(2, "epoch", float(epoch))
        frame.insert(
            3,
            "diagnostic_seed",
            int(diagnostic_seed),
        )
        _atomic_csv(raw_path, frame)
        _append_deduplicated(
            spectral_root / "layers.csv",
            frame,
            keys=["step", "matrix_name"],
        )
        summary = summarize_spectral_frame(
            frame,
            step=step,
            tokens_seen=tokens_seen,
            epoch=epoch,
        )
        _append_deduplicated(
            spectral_root / "summary.csv",
            pd.DataFrame([summary]),
            keys=["step"],
        )
        status = {
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "epoch": float(epoch),
            "completed": True,
            "raw_path": str(raw_path),
            "alpha_valid_matrices": int(
                summary["alpha_n"]
            ),
            "ERG_gap_valid_matrices": int(
                summary["ERG_gap_n"]
            ),
            "num_traps_valid_matrices": int(
                summary["num_traps_n"]
            ),
            "rand_distance_valid_matrices": int(
                summary["rand_distance_n"]
            ),
        }
        (
            spectral_root
            / f"status_step_{int(step):07d}.json"
        ).write_text(
            json.dumps(
                status,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return summary
    except Exception as exc:
        status = {
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "completed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (
            spectral_root
            / f"status_step_{int(step):07d}.json"
        ).write_text(
            json.dumps(
                status,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if bool(config.get("strict", True)):
            raise
        print(
            f"[one-head-ww] WARNING step={step}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return status
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        restore_accelerator_rng_state(
            accelerator_state,
            device,
        )

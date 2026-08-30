from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .model import GPT, transformer_matrix_items
from .checkpoints import model_state_sha256
from .runtime import (
    capture_accelerator_rng_state,
    model_device,
    restore_accelerator_rng_state,
    synchronize,
)

SPECTRAL_METRICS = (
    "alpha",
    "alpha_raw",
    "alpha_clip_xmax",
    "alpha_delta",
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
    """CPU-only Linear views of every declared transformer matrix."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def _validate_weightwatcher_frame(
    frame: pd.DataFrame,
    *,
    finger_policy: str | bool,
    expected_matrix_names: Sequence[str] | None = None,
) -> None:
    """Reject incomplete results and stale pre-finger-policy caches."""

    required = {
        "matrix_name",
        "alpha",
        "alpha_raw",
        "ERG_gap",
        "num_traps",
        "rand_distance",
        "finger_policy",
        "primary_alpha_variant",
        "weightwatcher_analysis_calls",
        "run_seed",
        "diagnostic_seed",
        "protocol_fingerprint",
        "model_state_sha256",
    }
    if finger_policy == "clip_xmax":
        required.update(
            {
                "raw_alpha",
                "alpha_clip_xmax",
                "alpha_delta",
                "num_fingers",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(
            "WeightWatcher result is missing the required one-pass schema: "
            + ", ".join(missing)
            + ". Remove the stale run directory before restarting."
        )
    expected_names = (
        tuple(str(value) for value in expected_matrix_names)
        if expected_matrix_names is not None
        else ()
    )
    expected_count = len(expected_names) if expected_names else 6
    observed_names = tuple(frame["matrix_name"].astype(str))
    inventory_matches = (
        set(observed_names) == set(expected_names)
        if expected_names
        else len(set(observed_names)) == expected_count
    )
    if (
        len(frame) != expected_count
        or frame["matrix_name"].nunique() != expected_count
        or not inventory_matches
    ):
        raise RuntimeError(
            "WeightWatcher must return exactly the declared transformer "
            f"matrix inventory ({expected_count} matrices)"
        )

    numeric = [
        "alpha",
        "alpha_raw",
        "ERG_gap",
        "num_traps",
        "rand_distance",
        "weightwatcher_analysis_calls",
    ]
    expected_policy = "none"
    expected_variant = "raw"
    if finger_policy == "clip_xmax":
        numeric.extend(
            ["raw_alpha", "alpha_clip_xmax", "alpha_delta", "num_fingers"]
        )
        expected_policy = "fix_fingers=clip_xmax"
        expected_variant = "clip_xmax"
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError(
            "WeightWatcher required one-pass spectral values contain NaN or "
            "infinity"
        )
    calls = pd.to_numeric(
        frame["weightwatcher_analysis_calls"], errors="coerce"
    )
    if not calls.eq(1).all():
        raise RuntimeError(
            "WeightWatcher analysis must be called exactly once per checkpoint"
        )
    if not frame["finger_policy"].astype(str).eq(expected_policy).all():
        raise RuntimeError("WeightWatcher finger-policy metadata is inconsistent")
    if not frame["primary_alpha_variant"].astype(str).eq(
        expected_variant
    ).all():
        raise RuntimeError("WeightWatcher primary-alpha metadata is inconsistent")
    if finger_policy == "clip_xmax":
        alpha = values["alpha"].to_numpy(dtype=float)
        raw = values["raw_alpha"].to_numpy(dtype=float)
        canonical_raw = values["alpha_raw"].to_numpy(dtype=float)
        clipped = values["alpha_clip_xmax"].to_numpy(dtype=float)
        delta = values["alpha_delta"].to_numpy(dtype=float)
        if not np.allclose(alpha, clipped, rtol=0.0, atol=0.0):
            raise RuntimeError("WeightWatcher clipped-alpha aliases disagree")
        if not np.allclose(raw, canonical_raw, rtol=0.0, atol=0.0):
            raise RuntimeError("WeightWatcher raw-alpha aliases disagree")
        if not np.allclose(
            delta,
            canonical_raw - clipped,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError("WeightWatcher alpha-delta values are inconsistent")
        if (values["num_fingers"] < 0).any():
            raise RuntimeError("WeightWatcher num_fingers must be nonnegative")


def _record_successful_frame(
    frame: pd.DataFrame,
    *,
    spectral_root: Path,
    raw_path: Path,
    step: int,
    tokens_seen: int,
    train_tokens: int,
    run_seed: int,
    diagnostic_seed: int,
    protocol_fingerprint: str,
    model_hash: str,
) -> dict[str, Any]:
    epoch = tokens_seen / max(1, int(train_tokens))
    for column, expected in (("step", step), ("tokens_seen", tokens_seen)):
        if column not in frame.columns:
            raise RuntimeError(
                f"WeightWatcher cached result has no {column} column"
            )
        observed = pd.to_numeric(frame[column], errors="coerce")
        if observed.isna().any() or not observed.eq(int(expected)).all():
            raise RuntimeError(
                f"WeightWatcher cached {column} does not match this checkpoint"
            )
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
        "raw_csv_sha256": _sha256(raw_path),
        "run_seed": int(run_seed),
        "diagnostic_seed": int(diagnostic_seed),
        "protocol_fingerprint": str(protocol_fingerprint),
        "model_state_sha256": str(model_hash),
        "weightwatcher_analysis_calls": 1,
        "finger_policy": str(frame["finger_policy"].iloc[0]),
        "alpha_valid_matrices": int(summary["alpha_n"]),
        "alpha_raw_valid_matrices": int(summary["alpha_raw_n"]),
        "alpha_clip_xmax_valid_matrices": int(
            summary["alpha_clip_xmax_n"]
        ),
        "ERG_gap_valid_matrices": int(summary["ERG_gap_n"]),
        "num_traps_valid_matrices": int(summary["num_traps_n"]),
        "rand_distance_valid_matrices": int(summary["rand_distance_n"]),
    }
    (spectral_root / f"status_step_{int(step):07d}.json").write_text(
        json.dumps(status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


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
    for identity in (
        "run_seed",
        "diagnostic_seed",
        "protocol_fingerprint",
        "model_state_sha256",
    ):
        if identity in frame.columns:
            values = frame[identity].drop_duplicates()
            if len(values) != 1:
                raise RuntimeError(
                    f"WeightWatcher frame has multiple {identity} values"
                )
            summary[identity] = values.iloc[0]
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
    fingerprint: str,
) -> dict[str, Any]:
    """Run one WeightWatcher analysis with ERG and randomization enabled.

    With ``fix_fingers='clip_xmax'``, WeightWatcher returns the corrected
    exponent in ``alpha`` and the uncorrected exponent from the same call in
    ``raw_alpha``.  The baseline stores those values canonically as
    ``alpha_clip_xmax`` and ``alpha_raw``.  It does not run WeightWatcher a
    second time.  `ERG_gap`, `num_traps`, and `rand_distance` are retained
    directly from that same call. `rand_distance` is the Jensen-Shannon
    distance between the empirical ESD and the entry-wise randomized ESD. No
    fallback alpha, proxy trap count, synthesized ERG gap, or replacement
    random-distance statistic is permitted. Every CPU and accelerator RNG
    stream is restored after the randomized diagnostic so measurement cannot
    change the subsequent training path.
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
    if not str(fingerprint).strip():
        raise ValueError("WeightWatcher requires a non-empty protocol fingerprint")
    device = model_device(model)
    synchronize(device)
    current_model_hash = model_state_sha256(model.state_dict())
    expected_matrix_names = tuple(
        name for name, _, _, _ in transformer_matrix_items(model)
    )
    diagnostic_seed = int(seed) + 1_000_003 + int(step)
    if raw_path.is_file():
        frame = pd.read_csv(raw_path)
        finger_policy = config.get("fix_fingers", False)
        _validate_weightwatcher_frame(
            frame,
            finger_policy=finger_policy,
            expected_matrix_names=expected_matrix_names,
        )
        expected_identities = {
            "run_seed": int(seed),
            "diagnostic_seed": int(diagnostic_seed),
            "protocol_fingerprint": str(fingerprint),
            "model_state_sha256": str(current_model_hash),
        }
        for column, expected in expected_identities.items():
            if column not in frame.columns or not frame[column].astype(str).eq(
                str(expected)
            ).all():
                raise RuntimeError(
                    f"cached WeightWatcher {column} does not match the "
                    "current checkpoint"
                )
        status_path = spectral_root / f"status_step_{int(step):07d}.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "cached WeightWatcher raw CSV has no valid integrity status"
            ) from exc
        if (
            not isinstance(status, dict)
            or status.get("completed") is not True
            or str(status.get("raw_csv_sha256", "")) != _sha256(raw_path)
            or any(
                str(status.get(column, "")) != str(expected)
                for column, expected in expected_identities.items()
            )
        ):
            raise RuntimeError(
                "cached WeightWatcher raw CSV/status integrity binding failed"
            )
        return _record_successful_frame(
            frame,
            spectral_root=spectral_root,
            raw_path=raw_path,
            step=step,
            tokens_seen=tokens_seen,
            train_tokens=train_tokens,
            run_seed=seed,
            diagnostic_seed=diagnostic_seed,
            protocol_fingerprint=fingerprint,
            model_hash=current_model_hash,
        )

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    accelerator_state = capture_accelerator_rng_state(device)
    random.seed(diagnostic_seed)
    np.random.seed(diagnostic_seed % (2**32 - 1))
    torch.manual_seed(diagnostic_seed)

    try:
        # WeightWatcher remains deliberately CPU/NumPy based. For TPU/XLA,
        # WeightMatrixHolder materializes only the declared hidden matrices on
        # the host and never exposes the live accelerator model to NumPy.
        holder = WeightMatrixHolder(model)
        watcher = ww.WeightWatcher(model=holder)
        analysis_kwargs: dict[str, Any] = {
            "ERG": True,
            "randomize": True,
            "plot": False,
            "min_evals": int(config.get("min_evals", 20)),
        }
        finger_policy = config.get("fix_fingers", False)
        if finger_policy:
            analysis_kwargs["fix_fingers"] = finger_policy
            analysis_kwargs["max_fingers"] = int(
                config.get("max_fingers", 10)
            )
        details = watcher.analyze(
            **analysis_kwargs,
        )
        if details is None or len(details) == 0:
            raise RuntimeError(
                "WeightWatcher returned no transformer-matrix rows"
            )
        frame = _attach_matrix_metadata(
            pd.DataFrame(details),
            holder.matrix_metadata,
        )
        if finger_policy == "clip_xmax":
            if "raw_alpha" not in frame.columns:
                raise RuntimeError(
                    "WeightWatcher fix_fingers='clip_xmax' did not return "
                    "the required raw_alpha column"
                )
            if "num_fingers" not in frame.columns:
                raise RuntimeError(
                    "WeightWatcher fix_fingers='clip_xmax' did not return "
                    "the required num_fingers column"
                )
            frame["alpha_raw"] = pd.to_numeric(
                frame["raw_alpha"], errors="coerce"
            )
            frame["alpha_clip_xmax"] = pd.to_numeric(
                frame["alpha"], errors="coerce"
            )
            frame["alpha_delta"] = (
                frame["alpha_raw"] - frame["alpha_clip_xmax"]
            )
            frame["finger_policy"] = "fix_fingers=clip_xmax"
            frame["primary_alpha_variant"] = "clip_xmax"
        else:
            frame["alpha_raw"] = pd.to_numeric(
                frame["alpha"], errors="coerce"
            )
            frame["alpha_clip_xmax"] = np.nan
            frame["alpha_delta"] = np.nan
            frame["finger_policy"] = "none"
            frame["primary_alpha_variant"] = "raw"
        frame["weightwatcher_analysis_calls"] = 1
        epoch = tokens_seen / max(1, int(train_tokens))
        frame.insert(0, "step", int(step))
        frame.insert(1, "tokens_seen", int(tokens_seen))
        frame.insert(2, "epoch", float(epoch))
        frame.insert(
            3,
            "diagnostic_seed",
            int(diagnostic_seed),
        )
        frame.insert(4, "run_seed", int(seed))
        frame.insert(5, "protocol_fingerprint", str(fingerprint))
        frame.insert(6, "model_state_sha256", str(current_model_hash))
        _validate_weightwatcher_frame(
            frame,
            finger_policy=finger_policy,
            expected_matrix_names=expected_matrix_names,
        )
        _atomic_csv(raw_path, frame)
        return _record_successful_frame(
            frame,
            spectral_root=spectral_root,
            raw_path=raw_path,
            step=step,
            tokens_seen=tokens_seen,
            train_tokens=train_tokens,
            run_seed=seed,
            diagnostic_seed=diagnostic_seed,
            protocol_fingerprint=fingerprint,
            model_hash=current_model_hash,
        )
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

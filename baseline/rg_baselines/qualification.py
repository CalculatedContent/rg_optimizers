"""Validation-only qualification for the committed baseline recipes.

A literature-derived hyperparameter point is a strong candidate, not proof of a
global optimum. This module defines bounded candidate neighborhoods and the
selection/locking machinery needed to promote one candidate to the frozen
reference without consulting protected test metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import BaselineConfig
from .statistics import student_t_critical_95
from .vit_final import ViTBaselineConfig


@dataclass(frozen=True)
class QualificationCandidate:
    baseline: str
    optimizer: str
    label: str
    configuration: Mapping[str, Any]
    candidate_id: str
    is_committed_center: bool = False


def _candidate(
    baseline: str,
    optimizer: str,
    label: str,
    configuration: Mapping[str, Any],
    *,
    is_committed_center: bool = False,
) -> QualificationCandidate:
    payload = {
        "baseline": baseline,
        "optimizer": optimizer,
        "configuration": dict(configuration),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    candidate_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return QualificationCandidate(
        baseline=baseline,
        optimizer=optimizer,
        label=label,
        configuration=dict(configuration),
        candidate_id=candidate_id,
        is_committed_center=bool(is_committed_center),
    )


def mnist_candidates(optimizer: str) -> tuple[QualificationCandidate, ...]:
    """Return a bounded, source-centered MNIST candidate neighborhood."""

    optimizer = str(optimizer)
    base = BaselineConfig(optimizer=optimizer)
    if optimizer == "sgd_momentum":
        configs = (
            ("committed", base, True),
            ("lr_0p03", replace(base, sgd_learning_rate=0.03, sgd_min_learning_rate=3e-4), False),
            ("lr_0p08", replace(base, sgd_learning_rate=0.08, sgd_min_learning_rate=8e-4), False),
            ("no_decay", replace(base, sgd_weight_decay=0.0), False),
            ("wd_5e-4", replace(base, sgd_weight_decay=5e-4), False),
            ("warmup_1", replace(base, sgd_warmup_epochs=1), False),
        )
    elif optimizer == "adamw":
        configs = (
            ("committed", base, True),
            ("lr_5e-4", replace(base, adamw_learning_rate=5e-4, adamw_min_learning_rate=5e-6), False),
            ("lr_2e-3", replace(base, adamw_learning_rate=2e-3, adamw_min_learning_rate=2e-5), False),
            ("wd_1e-3", replace(base, adamw_weight_decay=1e-3), False),
            ("no_decay", replace(base, adamw_weight_decay=0.0), False),
            ("no_warmup", replace(base, adamw_warmup_epochs=0), False),
        )
    elif optimizer == "sgd_momentum_muon":
        configs = (
            ("committed", base, True),
            ("matrix_lr_0p01", replace(base, muon_learning_rate=0.01, muon_min_learning_rate=0.001), False),
            ("matrix_lr_0p04", replace(base, muon_learning_rate=0.04, muon_min_learning_rate=0.004), False),
            ("no_matrix_decay", replace(base, muon_weight_decay=0.0), False),
            ("aux_lr_1e-4", replace(base, muon_aux_learning_rate=1e-4, muon_aux_min_learning_rate=1e-5), False),
            ("aux_lr_6e-4", replace(base, muon_aux_learning_rate=6e-4, muon_aux_min_learning_rate=6e-5), False),
        )
    else:
        raise ValueError(f"unknown MNIST optimizer: {optimizer!r}")

    result = []
    for label, config, center in configs:
        config.validate()
        result.append(
            _candidate(
                "mnist_mlp3",
                optimizer,
                label,
                asdict(config),
                is_committed_center=center,
            )
        )
    return tuple(result)


def vit_candidates(optimizer: str) -> tuple[QualificationCandidate, ...]:
    """Return a bounded candidate neighborhood for the selected small ViT."""

    optimizer = str(optimizer)
    base = ViTBaselineConfig()
    if optimizer == "sgd_momentum":
        configs = (
            ("committed", base, True),
            ("lr_0p05", replace(base, sgd_lr=0.05, sgd_min_lr=5e-4), False),
            ("lr_0p20", replace(base, sgd_lr=0.20, sgd_min_lr=2e-3), False),
            ("wd_1e-3", replace(base, sgd_weight_decay=1e-3), False),
            ("warmup_start_1e-4", replace(base, sgd_warmup_start_lr=1e-4), False),
        )
    elif optimizer == "adamw":
        configs = (
            ("committed", base, True),
            ("lr_2p5e-4", replace(base, adamw_lr=2.5e-4), False),
            ("lr_5e-4", replace(base, adamw_lr=5e-4), False),
            ("lr_1e-3", replace(base, adamw_lr=1e-3), False),
            ("wd_0p03", replace(base, adamw_weight_decay=0.03), False),
            ("wd_0p10", replace(base, adamw_weight_decay=0.10), False),
        )
    elif optimizer == "muon":
        configs = (
            ("committed", base, True),
            ("matrix_lr_0p01", replace(base, muon_lr=0.01, muon_min_lr=0.001), False),
            ("matrix_lr_0p04", replace(base, muon_lr=0.04, muon_min_lr=0.004), False),
            ("no_matrix_decay", replace(base, muon_weight_decay=0.0), False),
            ("aux_lr_1e-4", replace(base, muon_aux_lr=1e-4, muon_aux_min_lr=1e-5), False),
            ("aux_lr_6e-4", replace(base, muon_aux_lr=6e-4, muon_aux_min_lr=6e-5), False),
        )
    else:
        raise ValueError(f"unknown ViT optimizer: {optimizer!r}")

    result = []
    for label, config, center in configs:
        config.validate()
        result.append(
            _candidate(
                "cifar10_small_vit",
                optimizer,
                label,
                asdict(config),
                is_committed_center=center,
            )
        )
    return tuple(result)


def one_head_profile_candidates(optimizer: str) -> tuple[QualificationCandidate, ...]:
    """Return optimizer-profile overrides around the one-head reference point."""

    optimizer = str(optimizer)
    if optimizer == "sgd_momentum":
        overrides = (
            ("committed", {}, True),
            ("lr_0p03", {"learning_rate": 0.03, "min_learning_rate": 0.003}, False),
            ("lr_0p08", {"learning_rate": 0.08, "min_learning_rate": 0.008}, False),
            ("no_decay", {"weight_decay": 0.0}, False),
            ("warmup_0p05", {"warmup_fraction": 0.05}, False),
        )
    elif optimizer == "adamw":
        overrides = (
            ("committed", {}, True),
            ("lr_3e-4", {"learning_rate": 3e-4, "min_learning_rate": 3e-5}, False),
            ("lr_1e-3", {"learning_rate": 1e-3, "min_learning_rate": 1e-4}, False),
            ("wd_0p05", {"weight_decay": 0.05}, False),
            ("wd_0p20", {"weight_decay": 0.20}, False),
        )
    elif optimizer == "muon":
        overrides = (
            ("committed", {}, True),
            ("matrix_lr_0p01", {"matrix_learning_rate": 0.01, "matrix_min_learning_rate": 0.001}, False),
            ("matrix_lr_0p04", {"matrix_learning_rate": 0.04, "matrix_min_learning_rate": 0.004}, False),
            ("no_matrix_decay", {"matrix_weight_decay": 0.0}, False),
            ("aux_lr_1e-4", {"aux_learning_rate": 1e-4, "aux_min_learning_rate": 1e-5}, False),
            ("aux_lr_6e-4", {"aux_learning_rate": 6e-4, "aux_min_learning_rate": 6e-5}, False),
        )
    else:
        raise ValueError(f"unknown one-head optimizer: {optimizer!r}")

    return tuple(
        _candidate(
            "fineweb_one_head_nanogpt",
            optimizer,
            label,
            override,
            is_committed_center=center,
        )
        for label, override, center in overrides
    )


def _mean_ci95(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci95_half_width": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    mean = float(array.mean())
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": np.nan,
            "ci95_half_width": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    std = float(array.std(ddof=1))
    half = student_t_critical_95(n) * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_half_width": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def rank_validation_candidates(
    history: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] | None = None,
    candidate_column: str = "candidate_id",
    seed_column: str = "seed",
    epoch_column: str = "epoch",
    validation_loss_column: str = "validation_loss",
    validation_accuracy_column: str = "validation_accuracy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank candidates using validation loss only.

    One checkpoint is selected per candidate/seed by minimum validation loss,
    with earlier epoch as the deterministic tie-break. Test columns may exist in
    the input but are never read by this function.
    """

    required = {
        candidate_column,
        seed_column,
        epoch_column,
        validation_loss_column,
    }
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"qualification history is missing: {sorted(missing)}")

    selected_rows = []
    for (candidate_id, seed), run in history.groupby(
        [candidate_column, seed_column], sort=True
    ):
        candidates = run.dropna(subset=[validation_loss_column]).sort_values(
            [validation_loss_column, epoch_column], ascending=[True, True]
        )
        if candidates.empty:
            raise RuntimeError(
                f"candidate={candidate_id}, seed={seed} has no finite validation loss"
            )
        selected_rows.append(candidates.iloc[0].copy())
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    expected = (
        tuple(sorted(int(seed) for seed in expected_seeds))
        if expected_seeds is not None
        else None
    )
    rows = []
    for candidate_id, group in selected.groupby(candidate_column, sort=True):
        observed = tuple(sorted(group[seed_column].astype(int).unique()))
        if expected is not None and observed != expected:
            raise RuntimeError(
                f"candidate {candidate_id} has seeds {observed}, expected {expected}"
            )
        loss = _mean_ci95(group[validation_loss_column])
        accuracy = (
            _mean_ci95(group[validation_accuracy_column])
            if validation_accuracy_column in group
            else _mean_ci95([])
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "n": loss["n"],
                "mean_best_validation_loss": loss["mean"],
                "validation_loss_std": loss["std"],
                "validation_loss_ci95_half_width": loss["ci95_half_width"],
                "validation_loss_ci95_low": loss["ci95_low"],
                "validation_loss_ci95_high": loss["ci95_high"],
                "mean_validation_accuracy_at_selection": accuracy["mean"],
                "mean_selected_epoch": float(group[epoch_column].mean()),
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values(
        [
            "mean_best_validation_loss",
            "mean_validation_accuracy_at_selection",
            "candidate_id",
        ],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    leaderboard.insert(0, "validation_rank", np.arange(1, len(leaderboard) + 1))
    return leaderboard, selected


def freeze_winner(
    path: str | Path,
    *,
    candidate: QualificationCandidate,
    leaderboard: pd.DataFrame,
    evidence_paths: Sequence[str | Path],
    source_commit: str,
    data_identity: Mapping[str, Any],
) -> Path:
    """Persist the exact validation-selected reference configuration."""

    if leaderboard.empty:
        raise ValueError("leaderboard is empty")
    winner = str(leaderboard.iloc[0]["candidate_id"])
    if winner != candidate.candidate_id:
        raise ValueError(
            f"candidate {candidate.candidate_id} is not leaderboard winner {winner}"
        )
    payload = {
        "schema_version": 1,
        "baseline": candidate.baseline,
        "optimizer": candidate.optimizer,
        "candidate_id": candidate.candidate_id,
        "candidate_label": candidate.label,
        "configuration": dict(candidate.configuration),
        "source_commit": str(source_commit),
        "data_identity": dict(data_identity),
        "selection_metric": "minimum validation loss per run, averaged across complete runs",
        "tie_break": "higher validation accuracy, then deterministic candidate id",
        "protected_test_used_for_selection": False,
        "evidence_paths": [str(Path(item)) for item in evidence_paths],
        "leaderboard": leaderboard.to_dict("records"),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(destination)
    return destination

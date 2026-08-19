"""Scheduling and filesystem contracts for the tangent-RG baseline.

Epoch zero denotes the initialized model.  Epoch ``e`` denotes the model after
``e`` complete passes over the optimization split.  A dense burst anchored at
epoch ``e`` therefore records the first configured updates *after* that
checkpoint.  In particular, epoch ``epochs`` is a valid sparse analysis point
but cannot be a dense-burst anchor because no training updates remain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .config import TangentRGConfig


TAIL_CHECKPOINT_COUNT = 100


@dataclass(frozen=True)
class BurstWindow:
    """A consecutive range of completed optimizer-step numbers."""

    anchor_epoch: int
    anchor_step: int
    first_completed_step: int
    last_completed_step: int

    def contains(self, completed_step: int) -> bool:
        return self.first_completed_step <= int(completed_step) <= self.last_completed_step

    @property
    def length(self) -> int:
        return self.last_completed_step - self.first_completed_step + 1


@dataclass(frozen=True)
class AnalysisPlan:
    """Immutable sparse-checkpoint and dense-update capture plan."""

    steps_per_epoch: int
    total_steps: int
    lr_schedule_steps: int
    analysis_epochs: tuple[int, ...]
    analysis_steps: tuple[int, ...]
    dense_bursts: tuple[BurstWindow, ...]
    capture_completed_steps: tuple[int, ...]

    def burst_for_completed_step(self, completed_step: int) -> Optional[BurstWindow]:
        """Return the unique burst containing ``completed_step``, if any."""

        matches = [burst for burst in self.dense_bursts if burst.contains(completed_step)]
        if len(matches) > 1:
            raise RuntimeError("overlapping dense capture bursts are not supported")
        return matches[0] if matches else None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["epoch_semantics"] = (
            "epoch 0 is initialization; epoch e is after e complete train passes"
        )
        payload["capture_step_semantics"] = (
            "completed_step s captures the update taking global_step from s-1 to s"
        )
        return payload


@dataclass(frozen=True)
class RunLayout:
    """Stable on-disk layout for one optimizer/seed run."""

    root: Path
    metrics: Path
    checkpoints: Path
    captures: Path
    logs: Path
    manifest: Path
    resolved_config: Path
    completion: Path

    def create(self) -> "RunLayout":
        for path in (self.root, self.metrics, self.checkpoints, self.captures, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class TailCheckpointLayout:
    """Namespaced temporary cache for the final trained epoch boundaries."""

    root: Path
    checkpoints: Path
    manifest: Path
    completion: Path
    quarantine: Path

    def create(self) -> "TailCheckpointLayout":
        for path in (self.root, self.checkpoints):
            path.mkdir(parents=True, exist_ok=True)
        return self


def _rounded_log_grid(first: int, last: int, count: int) -> set[int]:
    """Return a dependency-free, inclusive, integer log grid."""

    if first < 1 or last < first or count < 2:
        raise ValueError("log-grid bounds/count are invalid")
    if first == last:
        return {first}
    log_first, log_last = math.log(float(first)), math.log(float(last))
    return {
        int(round(math.exp(log_first + (log_last - log_first) * index / (count - 1))))
        for index in range(count)
    }


def log_spaced_epochs(total_epochs: int, count: int) -> tuple[int, ...]:
    """Generate epoch zero plus approximately ``count`` log-spaced epochs."""

    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    return tuple(sorted({0, *_rounded_log_grid(1, int(total_epochs), int(count))}))


def build_analysis_plan(
    config: TangentRGConfig,
    *,
    steps_per_epoch: int,
) -> AnalysisPlan:
    """Resolve sparse/log checkpoints, LR horizon, and dense capture bursts."""

    config.validate()
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    total_steps = int(config.epochs) * int(steps_per_epoch)
    schedule_steps = max(
        2,
        min(total_steps, int(math.ceil(config.lr_schedule_epochs * steps_per_epoch))),
    )

    epochs = set(log_spaced_epochs(config.epochs, config.log_analysis_points))
    epochs.update(int(value) for value in config.explicit_analysis_epochs)
    epochs.update({0, config.epochs})
    schedule_epoch = config.lr_schedule_epochs
    epochs.update({int(math.floor(schedule_epoch)), int(math.ceil(schedule_epoch))})
    if config.tail_analysis_start_epoch is not None:
        epochs.update(
            range(
                int(config.tail_analysis_start_epoch),
                int(config.epochs) + 1,
                int(config.tail_analysis_every_epochs),
            )
        )
    analysis_epochs = tuple(sorted(epoch for epoch in epochs if 0 <= epoch <= config.epochs))
    analysis_steps = tuple(epoch * int(steps_per_epoch) for epoch in analysis_epochs)

    windows: list[BurstWindow] = []
    occupied: set[int] = set()
    for anchor_epoch in sorted(set(config.dense_burst_anchor_epochs)):
        anchor_step = int(anchor_epoch) * int(steps_per_epoch)
        if anchor_epoch < 0 or anchor_step >= total_steps:
            continue
        first = anchor_step + 1
        last = min(total_steps, anchor_step + int(config.dense_burst_length_steps))
        step_set = set(range(first, last + 1))
        if occupied.intersection(step_set):
            raise ValueError(
                f"dense capture burst at epoch {anchor_epoch} overlaps an earlier burst"
            )
        occupied.update(step_set)
        windows.append(
            BurstWindow(
                anchor_epoch=int(anchor_epoch),
                anchor_step=anchor_step,
                first_completed_step=first,
                last_completed_step=last,
            )
        )

    return AnalysisPlan(
        steps_per_epoch=int(steps_per_epoch),
        total_steps=total_steps,
        lr_schedule_steps=schedule_steps,
        analysis_epochs=analysis_epochs,
        analysis_steps=analysis_steps,
        dense_bursts=tuple(windows),
        capture_completed_steps=tuple(sorted(occupied)),
    )


def make_run_layout(
    config: TangentRGConfig,
    output_dir: Optional[str | Path] = None,
) -> RunLayout:
    """Construct, but do not create, the canonical run directory layout."""

    root = (
        Path(output_dir)
        if output_dir is not None
        else Path(config.run_root)
        / config.suite_name
        / config.optimizer
        / f"seed_{config.seed}"
    )
    return RunLayout(
        root=root,
        metrics=root / "metrics",
        checkpoints=root / "checkpoints",
        captures=root / "captures",
        logs=root / "logs",
        manifest=root / "manifest.json",
        resolved_config=root / "resolved_config.json",
        completion=root / "run_complete.json",
    )


def tail_checkpoint_epochs(total_epochs: int) -> tuple[int, ...]:
    """Return exactly the final 100 trained epochs, or all when shorter.

    Initialization (epoch zero) is deliberately excluded: these snapshots are
    the final ``min(100, total_epochs)`` *trained* epoch boundaries and are
    independent of the sparse WeightWatcher analysis schedule.
    """

    horizon = int(total_epochs)
    if horizon < 1:
        raise ValueError("total_epochs must be positive")
    first = max(1, horizon - TAIL_CHECKPOINT_COUNT + 1)
    return tuple(range(first, horizon + 1))


def make_tail_checkpoint_layout(config: TangentRGConfig) -> TailCheckpointLayout:
    """Construct, but do not create, the safe optimizer/seed cache layout."""

    config.validate()
    root = (
        Path(config.tail_checkpoint_cache_root)
        / config.suite_name
        / config.optimizer
        / f"seed_{config.seed}"
    )
    return TailCheckpointLayout(
        root=root,
        checkpoints=root / "checkpoints",
        manifest=root / "manifest.json",
        completion=root / "cache_complete.json",
        quarantine=root / "resume_quarantine",
    )


def validate_disjoint_checkpoint_layouts(
    run_layout: RunLayout,
    tail_layout: TailCheckpointLayout,
) -> None:
    """Reject overlapping persistent and temporary seed directories."""

    run_root = run_layout.root.expanduser().resolve()
    tail_root = tail_layout.root.expanduser().resolve()
    if (
        run_root == tail_root
        or run_root in tail_root.parents
        or tail_root in run_root.parents
    ):
        raise ValueError(
            "canonical run and tail-checkpoint cache directories must be disjoint"
        )


def indices_sha256(indices: Iterable[int]) -> str:
    """Hash ordered split indices without requiring NumPy."""

    digest = hashlib.sha256()
    for value in indices:
        digest.update(int(value).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def protocol_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-serializable resolved protocol description."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically replace a small JSON contract file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "AnalysisPlan",
    "BurstWindow",
    "RunLayout",
    "TAIL_CHECKPOINT_COUNT",
    "TailCheckpointLayout",
    "atomic_json",
    "build_analysis_plan",
    "indices_sha256",
    "log_spaced_epochs",
    "make_run_layout",
    "make_tail_checkpoint_layout",
    "protocol_fingerprint",
    "tail_checkpoint_epochs",
    "validate_disjoint_checkpoint_layouts",
]

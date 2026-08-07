"""Rotating, without-replacement training-probe subsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.utils.data._utils.collate import default_collate


@dataclass(frozen=True)
class ProbeDraw:
    indices: tuple[int, ...]
    cycle_start: int
    cycle_end: int


class RotatingSubsetSampler:
    """Cycle through random permutations before reusing an example.

    The sampler is independent of the minibatch shuffling used by the base
    optimizer.  A new contiguous slice of a seeded random permutation is used
    for every TraceWall correction.  Once the permutation is exhausted, a new
    one is generated and the cycle counter advances.
    """

    def __init__(self, dataset_size: int, *, seed: int) -> None:
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive")
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)
        self._cycle = 0
        self._position = 0
        self._permutation = torch.randperm(
            self.dataset_size,
            generator=self._generator,
        )

    @property
    def cycle(self) -> int:
        return int(self._cycle)

    @property
    def position(self) -> int:
        return int(self._position)

    def _reshuffle(self) -> None:
        self._cycle += 1
        self._position = 0
        self._permutation = torch.randperm(
            self.dataset_size,
            generator=self._generator,
        )

    def take(self, count: int) -> ProbeDraw:
        if count < 1:
            raise ValueError("count must be positive")
        if count > self.dataset_size:
            raise ValueError(
                "one probe subset cannot exceed the dataset size; increase the "
                "number of corrections instead"
            )
        start_cycle = self._cycle
        pieces: list[torch.Tensor] = []
        remaining = int(count)
        selected: set[int] = set()
        while remaining > 0:
            available = self.dataset_size - self._position
            take_now = min(available, remaining)
            piece = self._permutation[self._position : self._position + take_now]
            pieces.append(piece)
            selected.update(int(index) for index in piece.tolist())
            self._position += take_now
            remaining -= take_now
            if self._position == self.dataset_size and remaining > 0:
                self._reshuffle()
                # A draw may cross a cycle boundary.  Keep the draw a true
                # subset by moving indices already selected from the previous
                # cycle to the end of the new permutation.  No index is lost;
                # the deferred values remain available later in the new cycle.
                if selected:
                    selected_tensor = torch.tensor(
                        sorted(selected), dtype=torch.long
                    )
                    duplicate_mask = torch.isin(
                        self._permutation, selected_tensor
                    )
                    self._permutation = torch.cat(
                        [
                            self._permutation[~duplicate_mask],
                            self._permutation[duplicate_mask],
                        ]
                    )
        indices = torch.cat(pieces).tolist()
        if self._position == self.dataset_size:
            # The draw owns the final examples from this cycle; the next draw
            # begins from a newly shuffled cycle.
            end_cycle = self._cycle
            self._reshuffle()
        else:
            end_cycle = self._cycle
        return ProbeDraw(
            indices=tuple(int(index) for index in indices),
            cycle_start=int(start_cycle),
            cycle_end=int(end_cycle),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "cycle": self._cycle,
            "position": self._position,
            "permutation": self._permutation.clone(),
            "generator_state": self._generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["dataset_size"]) != self.dataset_size:
            raise ValueError("dataset_size changed while restoring probe sampler")
        self.seed = int(state["seed"])
        self._cycle = int(state["cycle"])
        self._position = int(state["position"])
        self._permutation = torch.as_tensor(
            state["permutation"], dtype=torch.long
        ).clone()
        self._generator.set_state(torch.as_tensor(state["generator_state"]).clone())


def materialize_probe_batches(
    dataset: Any,
    draw: ProbeDraw,
    *,
    batch_size: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    examples = [dataset[index] for index in draw.indices]
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(examples), batch_size):
        inputs, targets = default_collate(examples[start : start + batch_size])
        batches.append(
            (
                inputs.to(device, non_blocking=False),
                targets.to(device, non_blocking=False),
            )
        )
    return batches

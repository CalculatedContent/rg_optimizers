import math
import numpy as np
from .schedule import TrainingSchedule
CANDIDATES_BY_STAGE = {0: (10240,), 1: (10240,), 2: (14336, 14336, 24576)}
ALL_CANDIDATE_COUNTS = tuple(sorted({p for counts in CANDIDATES_BY_STAGE.values() for p in counts}))
NEGATIVE_STRIDE = 20011

def _stage_index(schedule: TrainingSchedule, step: int) -> int:
    """Index of `step`'s stage; the final step (total_steps, validation only) counts as the last stage."""
    return next((i for i, (_, end) in enumerate(schedule.boundaries) if step < end), len(schedule.boundaries) - 1)

def candidate_count_at(schedule: TrainingSchedule, step: int) -> int:
    """Candidate count P for `step`; 0 means the full softmax."""
    stage_index = _stage_index(schedule, step)
    counts = CANDIDATES_BY_STAGE.get(stage_index)
    if counts is None:
        return 0
    start, end = schedule.boundaries[stage_index]
    return counts[(step - start) * len(counts) // (end - start)]

class CandidateBuilder:
    """Host half of the candidate build: numpy only, no device work, so it can run on a worker thread.

    Scratch arrays are allocated once and reused, so at most one build may run at a time.
    """

    def __init__(self, vocab_size: int, max_candidates: int, max_rows: int):
        assert math.gcd(NEGATIVE_STRIDE, vocab_size) == 1
        self.vocab_size = V = vocab_size
        self.mark = np.zeros(V + 1, dtype=bool)
        self.mark_u8 = self.mark.view(np.uint8)
        self.pos = np.empty(V + 1, dtype=np.int64)
        self.pos[V] = -1
        self.arange = np.arange(max_candidates, dtype=np.int64)
        self.prefix_targets, self.target_pos, self.prefix_pos = np.empty((3, max_rows), dtype=np.int64)
        sweep = NEGATIVE_STRIDE * np.arange(V, dtype=np.int64) % V
        self.sweep = np.concatenate((sweep, sweep))
        self.sweep_offset = 0

    def reset(self, rank: int, world_size: int):
        self.sweep_offset = self.vocab_size * rank // world_size

    def _negatives(self, need: int) -> np.ndarray:
        """`need` distinct unmarked classes: the next window of the stride sweep."""
        V, mark = (self.vocab_size, self.mark)
        draw = min(V, int(need * 1.7) + 256)
        idx = self.sweep[self.sweep_offset:self.sweep_offset + draw]
        self.sweep_offset = (self.sweep_offset + draw) % V
        idx = idx[~mark[idx]]
        if idx.size >= need:
            return idx[:need]
        free = np.flatnonzero(~mark[:V])
        assert free.size >= need, f'only {free.size} free classes for {need}'
        return free[::max(1, free.size // need)][:need]

    def build(self, P: int, targets: np.ndarray, prefix_table: np.ndarray | None):
        """(candidates, target_pos, prefix_pos) for one microbatch; views into scratch.

        prefix_pos is None when there is no prefix table yet (warmup), leaving the device buffer at -1.
        """
        V, mark, pos, T = (self.vocab_size, self.mark, self.pos, targets.shape[0])
        mark.fill(False)
        mark[targets] = True
        num_targets = int(np.count_nonzero(mark[:V]))
        assert num_targets <= P, f'{num_targets} distinct targets > P={P}'
        if num_targets != P:
            self.mark_u8[self._negatives(P - num_targets)] = 1
        candidates = np.flatnonzero(mark[:V])
        pos[:V].fill(-1)
        pos[candidates] = self.arange[:P]
        target_pos = self.target_pos[:T]
        np.take(pos, targets, out=target_pos)
        if prefix_table is None:
            return (candidates, target_pos, None)
        prefix_targets, prefix_pos = (self.prefix_targets[:T], self.prefix_pos[:T])
        np.take(prefix_table, targets, out=prefix_targets)
        np.take(pos, prefix_targets, out=prefix_pos)
        return (candidates, target_pos, prefix_pos)

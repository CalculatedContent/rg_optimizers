"""Deterministic synthetic probes and exact full-record presentation schedules.

Natural-span decontamination and the baseline training adapter are deliberately
not implemented here; see TRAINER_CONTRACT.md before running a study.
"""
from __future__ import annotations
from itertools import product
import numpy as np


def canary_universe(alphabet: list[int], length: int) -> list[list[int]]:
    if length < 1 or len(alphabet) < 2 or len(set(alphabet)) != len(alphabet):
        raise ValueError("Use a unique alphabet with at least two tokens and positive length.")
    if len(alphabet) ** length > 65536:
        raise ValueError("Use at most 65,536 candidates for exhaustive exposure.")
    return [list(x) for x in product(alphabet, repeat=length)]


def random_sequence_probes(seed: int, doses=(0, 1, 4, 16, 64), per_dose=32,
                           vocab_size=50257, context_tokens=128, target_tokens=64) -> list[dict]:
    if per_dose < 1 or vocab_size < 3 or min(context_tokens, target_tokens) < 1:
        raise ValueError("Invalid probe dimensions.")
    if len(set(doses)) != len(doses) or any(d < 0 or int(d) != d for d in doses):
        raise ValueError("Doses must be unique nonnegative integers.")
    rng = np.random.default_rng(seed)
    probes = []
    seen = set()
    for dose in doses:
        for index in range(per_dose):
            while True:
                # Exclude GPT-2 EOT; sequences are token-random, not character-random.
                context = rng.integers(0, vocab_size - 1, context_tokens).tolist()
                target = rng.integers(0, vocab_size - 1, target_tokens).tolist()
                if tuple(target) not in seen:
                    seen.add(tuple(target))
                    break
            probes.append({"id": f"random_d{dose}_{index:03d}", "family": "random_tokens",
                           "dose": int(dose), "context": context, "target": target})
    return probes


def presentation_schedule(probes: list[dict], slots: int, seed: int) -> list[dict]:
    """One slot is one COMPLETE context+target training record, not a bin offset.

    Allocate every declared presentation without replacement. The adapter must
    log realized visits and verify these counts; it must never sample windows
    through the injected records and call intended copies actual exposures.
    """
    if slots < 1 or len({p['id'] for p in probes}) != len(probes):
        raise ValueError("Require positive slots and unique probe IDs.")
    for p in probes:
        if int(p["dose"]) != p["dose"] or p["dose"] < 0:
            raise ValueError("Doses must be nonnegative integers.")
    ids = [p["id"] for p in probes for _ in range(int(p["dose"]))]
    if len(ids) > slots:
        raise ValueError("Not enough training slots for the declared presentation counts.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(slots, size=len(ids), replace=False)
    return sorted(({'slot': int(s), 'probe_id': p} for s, p in zip(chosen, ids)), key=lambda r: r['slot'])

"""Behavioral measurements: no spectral metric is used to define memorization."""
from __future__ import annotations
import math
from typing import Callable, Sequence
import numpy as np
import torch
import torch.nn.functional as F


def recall(generated: Sequence[int], target: Sequence[int]) -> dict:
    g, y = np.asarray(generated), np.asarray(target)
    if g.ndim != 1 or y.ndim != 1 or len(y) == 0 or g.shape != y.shape:
        raise ValueError("Generated and target sequences must be nonempty, 1-D, and equal-length.")
    match = g == y
    failures = np.flatnonzero(~match)
    return {"sequence_exact_match": bool(match.all()),
            "free_running_token_match": float(match.mean()),
            "longest_exact_prefix": int(failures[0]) if len(failures) else len(y),
            "continuation_tokens": len(y)}


def exposure(candidate_nll: Sequence[float], target_index: int) -> dict:
    """Exact finite-universe exposure; smaller NLL ranks first.

    Return tie bounds. Primary exposure is conservative: all-equal scores -> 0.
    These are full-universe ranks ONLY if every possible candidate was scored.
    """
    scores = np.asarray(candidate_nll, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("All candidate NLL scores must be finite and present.")
    if not 0 <= target_index < len(scores):
        raise ValueError("Target index is outside the candidate universe.")
    value = scores[target_index]
    rank_min = 1 + int(np.count_nonzero(scores < value))
    rank_max = int(np.count_nonzero(scores <= value))
    bits = math.log2(len(scores))
    return {"candidate_count": len(scores), "rank_min": rank_min, "rank_max": rank_max,
            "exposure_bits_lower": bits - math.log2(rank_max),
            "exposure_bits_upper": bits - math.log2(rank_min),
            "tie_count": rank_max - rank_min + 1}


def prefix_compression(context: Sequence[int], target: Sequence[int],
                       lengths: Sequence[int], generate: Callable[[list[int], int], list[int]]) -> dict:
    """Sweep suffixes of ONE fixed context; never move the target boundary.

    This is prefix-constrained prompt compression, not optimized adversarial ACR.
    Failure is censored, not proof that no shorter/adversarial prompt exists.
    """
    if len(context) == 0 or len(target) == 0 or len(lengths) == 0:
        raise ValueError("Context, target, and prefix grid must be nonempty.")
    if any(int(n) != n for n in lengths):
        raise ValueError("Prefix lengths must be integers.")
    grid = sorted(set(int(n) for n in lengths))
    if grid[0] < 1 or grid[-1] > len(context):
        raise ValueError("Prefix grid is outside the available context.")
    rows = []
    for p in grid:
        rows.append({"prefix_tokens": p, **recall(generate(list(context[-p:]), len(target)), target)})
    successes = [r["prefix_tokens"] for r in rows if r["sequence_exact_match"]]
    shortest = min(successes) if successes else None
    return {"rows": rows, "shortest_successful_prefix_in_grid": shortest,
            "prefix_compression_ratio": len(target) / shortest if shortest is not None else None,
            "search_censored": shortest is None,
            "prompt_class": "fixed_context_suffixes_only"}


@torch.inference_mode()
def score_continuation(model, prefix: Sequence[int], target: Sequence[int]) -> dict:
    """Teacher-forced suffix NLL/accuracy and separate free-running recall."""
    if len(prefix) == 0 or len(target) == 0:
        raise ValueError("Prefix and continuation must both be nonempty.")
    if len(prefix) + len(target) - 1 > model.cfg.block_size:
        raise ValueError("Teacher-forced input would exceed model context.")
    device = next(model.parameters()).device
    full = torch.tensor([list(prefix) + list(target)], dtype=torch.long, device=device)
    previous = model.training
    model.eval()
    try:
        logits, _ = model(full[:, :-1])
        selected = logits[:, len(prefix) - 1:, :]
        labels = full[:, len(prefix):]
        losses = F.cross_entropy(selected.reshape(-1, selected.size(-1)), labels.reshape(-1), reduction="none")
        nll = float(losses.mean().item())
        if not math.isfinite(nll):
            raise RuntimeError("Nonfinite continuation loss; do not report a memorization score.")
        greedy = model.generate_greedy(full[:, :len(prefix)], len(target))[0, len(prefix):].tolist()
        return {**recall(greedy, target), "suffix_nll": nll,
                "suffix_nll_sum": float(losses.sum().item()),
                "suffix_perplexity": math.exp(nll) if nll < 709 else None,
                "teacher_forced_token_accuracy": float((selected.argmax(-1) == labels).float().mean().item())}
    finally:
        model.train(previous)


@torch.inference_mode()
def rank_canary(model, prefix: Sequence[int], candidates: Sequence[Sequence[int]],
                target_index: int, batch_size: int = 4) -> dict:
    """Exhaustively score a small declared universe, not sampled rank estimation."""
    if len(prefix) == 0 or len(candidates) == 0 or batch_size < 1:
        raise ValueError("Nonempty prefix/universe and positive batch size are required.")
    if not 0 <= target_index < len(candidates):
        raise ValueError("Target index is outside the candidate universe.")
    length = len(candidates[0])
    if not length or any(len(c) != length for c in candidates):
        raise ValueError("The declared universe must use equal nonzero token lengths.")
    if len({tuple(c) for c in candidates}) != len(candidates):
        raise ValueError("Candidate universe contains duplicates.")
    if len(prefix) + length - 1 > model.cfg.block_size:
        raise ValueError("Canary scoring would exceed context.")
    device = next(model.parameters()).device
    previous = model.training
    model.eval()
    scores = []
    try:
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            full = torch.tensor([list(prefix) + list(c) for c in chunk], dtype=torch.long, device=device)
            logits, _ = model(full[:, :-1])
            logits = logits[:, len(prefix) - 1:, :]
            target = full[:, len(prefix):]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none")
            scores.extend(loss.reshape(len(chunk), length).sum(-1).double().cpu().tolist())
    finally:
        model.train(previous)
    return {**exposure(scores, target_index), "candidate_nll_sum": scores,
            "rank_scope": "entire_declared_finite_universe"}

"""Live-model hook reusing the repository's audited spectral implementation."""
from __future__ import annotations
from pathlib import Path


def monitor_training_state(model, run_dir: str | Path, *, step: int, tokens_seen: int,
                           reference_tokens: int, seed: int, fingerprint: str, ww_config: dict):
    if ww_config.get("fix_fingers") != "clip_xmax":
        raise ValueError("This study requires fix_fingers='clip_xmax'.")
    for key in ("enabled", "ERG", "randomize", "strict", "require_raw_alpha"):
        if ww_config.get(key) is not True:
            raise ValueError(f"This study requires weightwatcher.{key}=true.")
    if step < 0 or tokens_seen < 0 or reference_tokens < 1 or not fingerprint:
        raise ValueError("Invalid checkpoint identity.")
    from rg_nanogpt_one_head.spectral import run_weightwatcher
    # Upstream creates CPU clones of Q/K/V/O/MLP-in/MLP-out, calls WW once,
    # preserves CPU/accelerator RNG, binds output to the model-state hash,
    # and stores alpha, raw_alpha, num_fingers, ERG_gap, traps and rand_distance.
    # No mutation of weights, clipping of weights, or feedback to optimization.
    return run_weightwatcher(model, run_dir, step=step, tokens_seen=tokens_seen,
                             train_tokens=reference_tokens, config=dict(ww_config),
                             seed=seed, fingerprint=fingerprint)

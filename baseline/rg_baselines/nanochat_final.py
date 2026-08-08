"""Final public interface for the pinned nanochat reference baselines.

This layer adds one last migration guard around :mod:`nanochat_portable`: a
non-empty seed directory created before ``runtime_policy.json`` existed is never
silently blessed as a current reference run. It must be moved aside or removed
and rerun under the audited profile/device/compile/fallback policy.
"""

from __future__ import annotations

from pathlib import Path

from . import nanochat_portable as portable


def _reject_unversioned_existing_run(output_dir: str | Path, seed: int) -> None:
    seed_dir = Path(output_dir).expanduser().resolve() / f"seed_{int(seed)}"
    if not seed_dir.is_dir():
        return
    policy = seed_dir / "runtime_policy.json"
    if policy.is_file():
        return
    artifacts = [
        path
        for path in seed_dir.iterdir()
        if path.name not in {".DS_Store"} and not path.name.endswith(".tmp")
    ]
    if artifacts:
        raise RuntimeError(
            "existing nanochat seed directory predates the audited runtime "
            f"policy: {seed_dir}. Move it aside or remove it before rerunning."
        )


def run_seed(
    checkout_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    config: portable.reference.NanoChatConfig,
    *,
    seed: int,
    device_type: str = "auto",
    nproc_per_node: int = 1,
    resume: bool = True,
) -> Path:
    _reject_unversioned_existing_run(output_dir, int(seed))
    return portable.run_seed(
        checkout_dir,
        cache_dir,
        output_dir,
        config,
        seed=int(seed),
        device_type=device_type,
        nproc_per_node=int(nproc_per_node),
        resume=bool(resume),
    )


# Re-export the audited public interface used by the notebook.
DEFAULT_NANOCHAT_SEEDS = portable.DEFAULT_NANOCHAT_SEEDS
NANOCHAT_COMMIT = portable.NANOCHAT_COMMIT
NanoChatD12Config = portable.NanoChatD12Config
NanoChatMacConfig = portable.NanoChatMacConfig
analyze_weightwatcher_checkpoints = portable.analyze_weightwatcher_checkpoints
collect_metrics = portable.collect_metrics
detect_device_type = portable.detect_device_type
ensure_checkout = portable.ensure_checkout
ensure_environment = portable.ensure_environment
prepare_data = portable.prepare_data
resolve_profile = portable.resolve_profile
summarize_training_metrics = portable.summarize_training_metrics
summarize_weightwatcher = portable.summarize_weightwatcher

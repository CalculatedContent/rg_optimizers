from __future__ import annotations

from pathlib import Path


def test_muonclip_2p15m_resilient_protocol_budget() -> None:
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension

    install_muonclip_extension()

    from rg_nanogpt_one_head.config import (
        epoch_step_map,
        load_config,
        lr_schedule_steps,
        max_steps,
        optimizer_profile,
        tokens_per_step,
        warmup_steps,
    )

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(
        root / "configs" / "muonclip_2p15m_resilient.yaml"
    )
    profile = optimizer_profile(cfg, "muon_clip")
    points = epoch_step_map(cfg)
    schedule_steps = lr_schedule_steps(cfg, profile)

    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["grad_accum_steps"] == 16
    assert tokens_per_step(cfg) == 8192

    assert cfg["training"]["target_epochs"] == 220.16
    assert max_steps(cfg) == 2_150_000
    assert max_steps(cfg) * tokens_per_step(cfg) == 17_612_800_000

    assert len(points) == 100
    assert min(points) == 0
    assert max(points) == 2_150_000
    assert list(points)[1] == 21_717
    assert list(points)[-2] == 2_128_283

    assert schedule_steps == 9_766
    assert warmup_steps(profile, schedule_steps) == 500
    assert profile["min_learning_rate"] == 2.0e-5

    assert cfg["training"]["eval_interval_steps"] == 2_150_000
    assert cfg["training"]["checkpoint_interval_steps"] == 2_500
    assert profile["qk_diagnostics_interval"] == 2_500

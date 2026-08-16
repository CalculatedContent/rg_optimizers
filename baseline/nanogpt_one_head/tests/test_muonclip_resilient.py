from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

import rg_nanogpt_one_head.muonclip_resilient as resilient


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    config = tmp_path / "config.yaml"
    config.write_text("protocol: test\n", encoding="utf-8")
    values = {
        "config": str(config),
        "seed": 20260813,
        "data_root": str(tmp_path / "data"),
        "results_root": str(tmp_path / "results"),
        "max_no_progress_failures": 2,
        "retry_delay_seconds": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _save_checkpoint(path: Path, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": int(step)}, path)


def test_worker_command_runs_one_direct_muonclip_mps_worker(tmp_path) -> None:
    command = resilient._worker_command(_args(tmp_path))

    assert "rg_nanogpt_one_head.muonclip" in command
    assert "--optimizer" in command
    assert "muon_clip" in command
    assert "--mps-worker" in command
    assert command[command.index("--mps-retries") + 1] == "0"
    assert command[command.index("--device") + 1] == "mps"


def test_resilient_supervisor_resets_failure_budget_after_progress(
    tmp_path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, max_no_progress_failures=2)
    run_dir = (
        Path(args.results_root)
        / "muon_clip"
        / f"seed_{args.seed}"
    )
    latest = run_dir / "checkpoint_latest.pt"
    calls = 0

    def fake_run(command, env, check):
        nonlocal calls
        del command, env, check
        calls += 1
        if calls == 1:
            _save_checkpoint(latest, 500)
            return SimpleNamespace(returncode=1)
        if calls == 2:
            return SimpleNamespace(returncode=1)
        if calls == 3:
            _save_checkpoint(latest, 1000)
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(resilient.subprocess, "run", fake_run)
    monkeypatch.setattr(resilient.time, "sleep", lambda seconds: None)

    return_code = resilient.run_resilient(args)

    assert return_code == 0
    assert calls == 4
    assert resilient._checkpoint_step(latest) == 1000


def test_resilient_supervisor_stops_repeated_same_checkpoint_failure(
    tmp_path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, max_no_progress_failures=2)
    run_dir = (
        Path(args.results_root)
        / "muon_clip"
        / f"seed_{args.seed}"
    )
    latest = run_dir / "checkpoint_latest.pt"
    _save_checkpoint(latest, 500)
    calls = 0

    def fake_run(command, env, check):
        nonlocal calls
        del command, env, check
        calls += 1
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(resilient.subprocess, "run", fake_run)
    monkeypatch.setattr(resilient.time, "sleep", lambda seconds: None)

    return_code = resilient.run_resilient(args)

    assert return_code == 9
    assert calls == 2
    assert resilient._checkpoint_step(latest) == 500


def test_resilient_config_preserves_update_and_analysis_budget() -> None:
    from rg_nanogpt_one_head.muonclip import install_muonclip_extension

    install_muonclip_extension()

    from rg_nanogpt_one_head.config import (
        epoch_step_map,
        load_config,
        max_steps,
        tokens_per_step,
    )

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(
        root / "configs" / "muonclip_24h_125k_resilient.yaml"
    )
    points = epoch_step_map(cfg)

    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["grad_accum_steps"] == 16
    assert tokens_per_step(cfg) == 8192
    assert max_steps(cfg) == 125_000
    assert len(points) == 100
    assert min(points) == 0
    assert max(points) == 125_000
    assert cfg["training"]["checkpoint_interval_steps"] == 250

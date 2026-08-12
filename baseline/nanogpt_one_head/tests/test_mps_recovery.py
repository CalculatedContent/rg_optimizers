from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rg_nanogpt_one_head.checkpoints import save_training_checkpoint
from rg_nanogpt_one_head.optimizers import OptimizerHandle
import rg_nanogpt_one_head.training as training


def _worker_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    config = tmp_path / "config.yaml"
    config.write_text("protocol: test\n", encoding="utf-8")
    values = {
        "config": str(config),
        "optimizer": "muon",
        "data_root": str(tmp_path / "data"),
        "results_root": str(tmp_path / "results"),
        "overwrite": False,
        "no_resume": False,
        "mps_retries": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_checkpoint_refuses_nonfinite_optimizer_state(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    handle = OptimizerHandle(
        role="primary",
        optimizer=optimizer,
        peak_lr=1e-3,
        min_lr=1e-4,
    )

    x = torch.ones(2, 3)
    model(x).sum().backward()
    optimizer.step()
    for state in optimizer.state.values():
        state["exp_avg"].fill_(float("nan"))
        break

    path = tmp_path / "checkpoint_latest.pt"
    with pytest.raises(
        FloatingPointError,
        match="refusing to write or load a contaminated checkpoint",
    ):
        save_training_checkpoint(
            path,
            model=model,
            handles=[handle],
            step=25,
            best_validation_loss=1.0,
            best_validation_step=20,
            elapsed_seconds=2.0,
            fingerprint="unit",
            cfg={"protocol": {"name": "unit"}},
            optimizer_name="adamw",
            seed=7,
            train_generator=torch.Generator().manual_seed(11),
        )

    assert not path.exists()
    assert not path.with_suffix(".pt.tmp").exists()


def test_mps_worker_command_uses_the_correct_extension_module(tmp_path) -> None:
    args = _worker_args(
        tmp_path,
        overwrite=True,
        no_resume=True,
    )

    ordinary = training._mps_worker_command(
        args=args,
        optimizer_name="muon",
        seed=1337,
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
        first_attempt=True,
    )
    retry = training._mps_worker_command(
        args=args,
        optimizer_name="muon",
        seed=1337,
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
        first_attempt=False,
    )
    muonclip = training._mps_worker_command(
        args=args,
        optimizer_name="muon_clip",
        seed=1337,
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
        first_attempt=True,
    )

    assert "rg_nanogpt_one_head.training" in ordinary
    assert "rg_nanogpt_one_head.muonclip" in muonclip
    assert "--mps-worker" in ordinary
    assert "--overwrite" in ordinary
    assert "--no-resume" in ordinary
    assert "--overwrite" not in retry
    assert "--no-resume" not in retry


def test_mps_supervisor_retries_from_latest_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    args = _worker_args(tmp_path, mps_retries=1)
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    calls: list[list[str]] = []

    monkeypatch.setattr(
        training,
        "_resolve_roots",
        lambda **kwargs: (
            data_root,
            results_root,
            torch.device("mps"),
        ),
    )
    monkeypatch.setattr(
        training,
        "prepare_fineweb_edu",
        lambda cfg, path: None,
    )
    monkeypatch.setattr(
        training,
        "_release_accelerator",
        lambda device: None,
    )
    monkeypatch.setattr(training.time, "sleep", lambda seconds: None)

    def fake_run(command, env, check):
        del env, check
        calls.append(list(command))
        if len(calls) == 1:
            run_dir = results_root / "muon" / "seed_1337"
            run_dir.mkdir(parents=True)
            (run_dir / "checkpoint_latest.pt").write_bytes(b"verified")
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training.subprocess, "run", fake_run)

    training._run_isolated_mps_workers(
        args=args,
        cfg={"training": {"seeds": [1337]}},
        seeds=(1337,),
    )

    assert len(calls) == 2
    assert "--mps-worker" in calls[0]
    assert "--mps-worker" in calls[1]

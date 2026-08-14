from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rg_nanogpt_one_head.config import max_steps
from rg_nanogpt_one_head.model import GPT, GPTConfig
from rg_nanogpt_one_head.muonclip import MuonClip
from rg_nanogpt_one_head.muonclip_capture import (
    MuonClipCaptureRecorder,
    _validate_walk_profile,
    expected_weightwatcher_checkpoint_count,
    load_weightwatcher_checkpoint,
)


def test_capture_budget_and_committed_config() -> None:
    profile = {
        "walk_capture_effective_batches": 55,
        "walk_capture_microbatches": True,
        "walk_max_checkpoints": 500,
        "walk_capture_root": "/tmp/test-walk",
    }
    assert expected_weightwatcher_checkpoint_count(
        profile, grad_accum_steps=8
    ) == 496
    _validate_walk_profile(profile, grad_accum_steps=8)

    profile["walk_capture_effective_batches"] = 56
    with pytest.raises(ValueError, match="505 WeightWatcher checkpoints"):
        _validate_walk_profile(profile, grad_accum_steps=8)

    cfg = yaml.safe_load(
        (ROOT / "configs" / "muonclip_microbatch10.yaml").read_text()
    )
    committed = cfg["optimizer_profiles"]["muon_clip"]
    assert cfg["training"]["seeds"] == [2027]
    assert max_steps(cfg) == 10
    assert expected_weightwatcher_checkpoint_count(
        committed,
        grad_accum_steps=cfg["training"]["grad_accum_steps"],
    ) == 91


def test_movie_uses_requested_weightwatcher_call() -> None:
    source = (
        ROOT
        / "src"
        / "rg_nanogpt_one_head"
        / "muonclip_movie.py"
    ).read_text()
    assert "savedir = str(native_dir)" in source
    assert "savefig=savedir" in source
    assert "randomize=False" in source
    assert "ERG=False" in source


def test_recorder_writes_explicit_initial_and_microbatch_files(tmp_path) -> None:
    run_dir = tmp_path / "results" / "muon_clip" / "seed_7"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "training": {"batch_size": 2, "grad_accum_steps": 2},
                "model": {"block_size": 8},
                "data_metadata": {"splits": {"train": 2048}},
            }
        ),
        encoding="utf-8",
    )

    model = GPT(
        GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
        )
    )
    hidden = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("blocks.") and parameter.ndim == 2
    ]
    optimizer = MuonClip(
        hidden,
        model=model,
        lr=2e-4,
        momentum=0.95,
        nesterov=False,
        weight_decay=0.1,
        newton_schulz_steps=5,
        eps=1e-7,
        update_rms_scale=0.2,
        qk_clip_threshold=100.0,
        qk_clip_balance=0.5,
    )
    recorder = MuonClipCaptureRecorder(
        model=model,
        optimizer=optimizer,
        profile={
            "walk_capture_effective_batches": 1,
            "walk_capture_microbatches": True,
            "walk_max_checkpoints": 10,
            "walk_capture_root": str(tmp_path / "captures"),
            "walk_save_full_model": True,
            "walk_save_weightwatcher": True,
            "walk_save_optimizer_tensors": True,
            "walk_save_microbatch_gradients": True,
        },
        run_dir=run_dir,
    )
    recorder.capture_initial_state()
    recorder.begin_effective_batch()

    for index, parameter in enumerate(hidden, start=1):
        parameter.grad = torch.full_like(parameter, float(index))
    recorder.capture_after_backward(scaled_loss=torch.tensor(1.0))

    capture = recorder.capture_dir
    initial = capture / "weightwatcher_checkpoints" / "ww_step_0000000.pt"
    micro = capture / "weightwatcher_checkpoints" / "ww_microbatch_0000001.pt"
    assert initial.is_file()
    assert (capture / "step_traces" / "step_0000000.pt").is_file()
    assert micro.is_file()

    holder, initial_payload = load_weightwatcher_checkpoint(initial)
    assert initial_payload["step"] == 0
    assert len(dict(holder.named_children())) == 6
    _, gradient_payload = load_weightwatcher_checkpoint(
        micro,
        source="accumulated_gradients",
    )
    assert gradient_payload["global_microbatch"] == 1
    index_text = (capture / "snapshot_index.csv").read_text()
    assert "initial" in index_text
    assert "microbatch" in index_text

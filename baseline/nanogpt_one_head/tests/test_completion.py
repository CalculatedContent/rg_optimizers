from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from rg_nanogpt_one_head.completion import (
    CompletedRunValidationError,
    validate_completed_run,
)
from rg_nanogpt_one_head.config import (
    load_config,
    max_steps,
    optimizer_profile,
    protocol_fingerprint,
    warmup_steps,
)
from rg_nanogpt_one_head.engine import run_one
from rg_nanogpt_one_head.run_utils import run_directory, run_is_complete


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER = "adamw"
SEED = 13
MATRICES = (
    "W_Q",
    "W_K",
    "W_V",
    "W_O",
    "W_MLP_IN",
    "W_MLP_OUT",
)


def _config() -> dict:
    cfg = deepcopy(
        load_config(EXPERIMENT_ROOT / "configs" / "reference.yaml")
    )
    cfg["dataset"].update(
        {
            "name": "unit/fineweb",
            "config": "unit",
            "revision": "unit-revision",
            "train_tokens": 2_048,
            "val_tokens": 512,
            "test_tokens": 512,
        }
    )
    cfg["model"].update(
        {
            "vocab_size": 64,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 16,
        }
    )
    cfg["training"].update(
        {
            "seeds": [SEED],
            "batch_size": 2,
            "grad_accum_steps": 1,
            "target_epochs": 0.02,
            "epoch_interval": 1.0,
            "eval_interval_steps": 1,
            "eval_batches": 1,
            "checkpoint_interval_steps": 1,
        }
    )
    cfg["evaluation"].update(
        {
            "bleu_examples": 2,
            "bleu_prompt_tokens": 3,
            "bleu_continuation_tokens": 2,
            "bleu_batch_size": 2,
        }
    )
    return cfg


def _data_metadata(cfg: dict) -> dict:
    return {
        "schema_version": 2,
        "dataset_name": cfg["dataset"]["name"],
        "dataset_config": cfg["dataset"]["config"],
        "dataset_revision": cfg["dataset"]["revision"],
        "tokenizer": "gpt2",
        "dtype": "uint16",
        "document_disjoint_splits": True,
        "splits": {
            "train": int(cfg["dataset"]["train_tokens"]),
            "val": int(cfg["dataset"]["val_tokens"]),
            "test": int(cfg["dataset"]["test_tokens"]),
        },
        "files": {
            split: {
                "path": f"{split}.bin",
                "sha256": f"unit-{split}",
                "bytes": 2 * int(cfg["dataset"][f"{split}_tokens"]),
            }
            for split in ("train", "val", "test")
        },
    }


def _write_completed_run(results_root: Path, cfg: dict) -> tuple[Path, str, int]:
    metadata = _data_metadata(cfg)
    total_steps = max_steps(cfg, metadata["splits"]["train"])
    profile = optimizer_profile(cfg, OPTIMIZER)
    fingerprint = protocol_fingerprint(
        cfg,
        optimizer=OPTIMIZER,
        seed=SEED,
        data_metadata=metadata,
    )
    run_dir = run_directory(results_root, OPTIMIZER, SEED)
    (run_dir / "spectral").mkdir(parents=True, exist_ok=True)
    (run_dir / "epoch_checkpoints").mkdir(parents=True, exist_ok=True)

    best_step = 0
    best_loss = 2.0
    completion = {
        "completed": True,
        "optimizer": OPTIMIZER,
        "seed": SEED,
        "optimizer_steps": total_steps,
        "best_validation_step": best_step,
        "best_validation_loss": best_loss,
        "fingerprint": fingerprint,
    }
    (run_dir / "run_complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    manifest = {
        "optimizer": OPTIMIZER,
        "seed": SEED,
        "max_steps": total_steps,
        "model": cfg["model"],
        "warmup_steps": warmup_steps(profile, total_steps),
        "protocol_fingerprint": fingerprint,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run_dir / "test_results.json").write_text(
        json.dumps(
            {
                "policy": "validation selects; test monitoring-only",
                "final": {"step": total_steps, "loss": 1.0},
                "validation_selected": {"step": best_step, "loss": 2.0},
            }
        ),
        encoding="utf-8",
    )

    checkpoint_common = {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "optimizer_name": OPTIMIZER,
        "seed": SEED,
        "best_validation_step": best_step,
        "best_validation_loss": best_loss,
    }
    for filename, step in (
        ("checkpoint_latest.pt", total_steps),
        ("checkpoint_final.pt", total_steps),
        ("checkpoint_best.pt", best_step),
    ):
        torch.save({**checkpoint_common, "step": step}, run_dir / filename)

    steps = (0, total_steps)
    pd.DataFrame(
        [{"step": step, "train_loss": 2.0 - 0.1 * index}
         for index, step in enumerate(steps)]
    ).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": step,
                "nominal_epoch": float(index),
                "test_monitoring_only": 1,
                "checkpoint_path": str(
                    run_dir
                    / "epoch_checkpoints"
                    / f"model_epoch_{index:06d}_step_{step:07d}.pt"
                ),
            }
            for index, step in enumerate(steps)
        ]
    ).to_csv(run_dir / "epoch_metrics.csv", index=False)

    layer_rows = []
    for step in steps:
        for matrix in MATRICES:
            layer_rows.append(
                {
                    "step": step,
                    "matrix_name": matrix,
                    "alpha": 2.0,
                    "ERG_gap": 0,
                    "num_traps": 0,
                }
            )
    pd.DataFrame(layer_rows).to_csv(
        run_dir / "spectral" / "layers.csv", index=False
    )
    summary_rows = [
        {"step": step, "n_matrices": len(MATRICES)}
        for step in steps
    ]
    pd.DataFrame(summary_rows).to_csv(
        run_dir / "spectral" / "summary.csv",
        index=False,
    )
    # Keep the synthetic fixture's checkpoint inventory
    # identical to the paths in epoch_metrics.csv.
    inventory = pd.read_csv(run_dir / "epoch_metrics.csv")
    for value in inventory["checkpoint_path"]:
        checkpoint_path = Path(str(value))
        checkpoint_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        if not checkpoint_path.is_file():
            checkpoint_path.write_bytes(
                b"synthetic epoch checkpoint"
            )
    return run_dir, fingerprint, total_steps


def test_valid_completed_run_is_accepted_and_reported_complete(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    completion = validate_completed_run(
        run_dir,
        expected_fingerprint=fingerprint,
        expected_optimizer=OPTIMIZER,
        expected_seed=SEED,
        expected_total_steps=total_steps,
    )
    assert completion["completed"] is True
    assert run_is_complete(tmp_path, OPTIMIZER, SEED)


def test_engine_validates_before_skipping_completed_run(tmp_path, monkeypatch):
    cfg = _config()
    metadata = _data_metadata(cfg)
    run_dir, _, _ = _write_completed_run(tmp_path, cfg)
    monkeypatch.setattr(
        "rg_nanogpt_one_head.engine.load_memmaps",
        lambda *args, **kwargs: (
            metadata,
            {"train": object(), "val": object(), "test": object()},
        ),
    )
    monkeypatch.setattr(
        "rg_nanogpt_one_head.engine.choose_device",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("device setup should not run for a verified result")
        ),
    )
    observed = run_one(
        cfg=cfg,
        data_root=tmp_path / "data",
        results_root=tmp_path,
        optimizer_name=OPTIMIZER,
        seed=SEED,
        resume=True,
        progress=False,
    )
    assert observed == run_dir


def test_engine_rejects_completed_run_from_changed_protocol(
    tmp_path, monkeypatch
):
    cfg = _config()
    metadata = _data_metadata(cfg)
    _write_completed_run(tmp_path, cfg)
    changed = deepcopy(cfg)
    changed["training"]["target_epochs"] = 0.03
    monkeypatch.setattr(
        "rg_nanogpt_one_head.engine.load_memmaps",
        lambda *args, **kwargs: (
            metadata,
            {"train": object(), "val": object(), "test": object()},
        ),
    )
    with pytest.raises(
        CompletedRunValidationError, match="fingerprint|optimizer_steps"
    ):
        run_one(
            cfg=changed,
            data_root=tmp_path / "data",
            results_root=tmp_path,
            optimizer_name=OPTIMIZER,
            seed=SEED,
            resume=True,
            progress=False,
        )


def test_missing_terminal_artifact_is_not_complete(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    (run_dir / "spectral" / "summary.csv").unlink()
    assert not run_is_complete(tmp_path, OPTIMIZER, SEED)
    with pytest.raises(CompletedRunValidationError, match="missing"):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )


def test_checkpoint_with_stale_fingerprint_is_rejected(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    checkpoint = torch.load(
        run_dir / "checkpoint_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["fingerprint"] = "stale"
    torch.save(checkpoint, run_dir / "checkpoint_final.pt")
    with pytest.raises(CompletedRunValidationError, match="fingerprint"):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )

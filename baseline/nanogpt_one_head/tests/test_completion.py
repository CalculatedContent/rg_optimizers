from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

import rg_nanogpt_one_head.config as config_module
from rg_nanogpt_one_head.completion import (
    CompletedRunValidationError,
    validate_completed_run,
)
from rg_nanogpt_one_head.checkpoints import optimizer_state_sha256
from rg_nanogpt_one_head.config import (
    load_config,
    max_steps,
    optimizer_profile,
    protocol_fingerprint,
    warmup_steps,
)
from rg_nanogpt_one_head.engine import run_one
from rg_nanogpt_one_head.run_utils import (
    model_state_sha256,
    run_directory,
    run_is_complete,
    validate_existing_manifest_runtime,
)


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
HELD_OUT_CURVE_COLUMNS = (
    "test_loss",
    "test_perplexity",
    "test_bits_per_token",
    "test_accuracy",
    "test_top5_accuracy",
    "test_bleu",
    "test_continuation_token_accuracy",
    "test_continuation_exact_match",
    "test_generalization_gap",
)


def _runtime_identity() -> dict:
    return {
        "platform": "unit-platform",
        "machine": "unit-machine",
        "python_version": "unit-python",
        "accelerator": "cpu",
        "device": "cpu",
        "torch_version": torch.__version__,
        "float32_matmul_precision": "highest",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "hardware_block_id": "unit-cpu-block",
        "hardware_block_id_source": "test",
    }


def _test_metrics(loss: float, *, step: int) -> dict:
    return {
        "step": int(step),
        "loss": float(loss),
        "perplexity": math.exp(float(loss)),
        "bits_per_token": float(loss) / math.log(2.0),
        "accuracy": 0.10,
        "top5_accuracy": 0.25,
        "bleu": 1.5,
        "continuation_token_accuracy": 0.05,
        "continuation_exact_match": 0.0,
    }


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
    test_model_state = {"weight": torch.tensor([1.0])}
    fingerprint = protocol_fingerprint(
        cfg,
        optimizer=OPTIMIZER,
        seed=SEED,
        data_metadata=metadata,
    )
    run_dir = run_directory(results_root, OPTIMIZER, SEED)
    (run_dir / "spectral").mkdir(parents=True, exist_ok=True)
    (run_dir / "spectral" / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "epoch_checkpoints").mkdir(parents=True, exist_ok=True)

    best_step = 0
    best_loss = 2.0
    final_test = _test_metrics(1.0, step=total_steps)
    selected_test = _test_metrics(2.0, step=best_step)
    completion = {
        "completed": True,
        "optimizer": OPTIMIZER,
        "seed": SEED,
        "optimizer_steps": total_steps,
        "best_validation_step": best_step,
        "best_validation_loss": best_loss,
        "final_test_loss": final_test["loss"],
        "final_test_perplexity": final_test["perplexity"],
        "final_test_bits_per_token": final_test["bits_per_token"],
        "final_test_accuracy": final_test["accuracy"],
        "final_test_top5_accuracy": final_test["top5_accuracy"],
        "final_test_bleu": final_test["bleu"],
        "final_test_continuation_token_accuracy": final_test[
            "continuation_token_accuracy"
        ],
        "final_test_continuation_exact_match": final_test[
            "continuation_exact_match"
        ],
        "fingerprint": fingerprint,
    }
    (run_dir / "run_complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    manifest = {
        "optimizer": OPTIMIZER,
        "seed": SEED,
        "max_steps": total_steps,
        "warmup_steps": warmup_steps(profile, total_steps),
        "protocol_fingerprint": fingerprint,
        "initial_model_sha256": model_state_sha256(test_model_state),
        "runtime_environment": _runtime_identity(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run_dir / "test_results.json").write_text(
        json.dumps(
            {
                "policy": "test is held out; validation selects and test never tunes",
                "final": final_test,
                "validation_selected": selected_test,
            }
        ),
        encoding="utf-8",
    )

    optimizer_states = [{"state": {}, "param_groups": []}]
    checkpoint_common = {
        "schema_version": 5,
        "model": test_model_state,
        "optimizers": optimizer_states,
        "model_state_sha256": model_state_sha256(test_model_state),
        "optimizer_state_sha256": optimizer_state_sha256(optimizer_states),
        "fingerprint": fingerprint,
        "optimizer_name": OPTIMIZER,
        "seed": SEED,
        "best_validation_step": best_step,
        "best_validation_loss": best_loss,
    }
    for filename, step in (
        ("checkpoint_initial.pt", 0),
        ("checkpoint_latest.pt", total_steps),
        ("checkpoint_final.pt", total_steps),
        ("checkpoint_best.pt", best_step),
    ):
        torch.save({**checkpoint_common, "step": step}, run_dir / filename)

    steps = (0, total_steps)
    pd.DataFrame(
        [
            {
                "step": step,
                "train_loss": 2.0 - 0.1 * index,
                **{column: float("nan") for column in HELD_OUT_CURVE_COLUMNS},
            }
            for index, step in enumerate(steps)
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": step,
                "epoch": float(index),
                "nominal_epoch": float(index),
                "test_monitoring_only": 1,
                "test_held_out": 1,
                **{column: float("nan") for column in HELD_OUT_CURVE_COLUMNS},
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
    model_hash = model_state_sha256(test_model_state)
    for step in steps:
        diagnostic_seed = SEED + 1_000_003 + step
        step_rows = []
        for matrix in MATRICES:
            row = {
                "step": step,
                "matrix_name": matrix,
                "alpha": 2.0,
                "ERG_gap": 0,
                "num_traps": 0,
                "rand_distance": 0.0,
                "run_seed": SEED,
                "diagnostic_seed": diagnostic_seed,
                "protocol_fingerprint": fingerprint,
                "model_state_sha256": model_hash,
            }
            step_rows.append(row)
            layer_rows.append(row)
        raw_path = (
            run_dir
            / "spectral"
            / "raw"
            / f"weightwatcher_step_{step:07d}.csv"
        )
        pd.DataFrame(step_rows).to_csv(raw_path, index=False)
        raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        (run_dir / "spectral" / f"status_step_{step:07d}.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "step": step,
                    "run_seed": SEED,
                    "diagnostic_seed": diagnostic_seed,
                    "protocol_fingerprint": fingerprint,
                    "model_state_sha256": model_hash,
                    "raw_csv_sha256": raw_hash,
                }
            ),
            encoding="utf-8",
        )
    pd.DataFrame(layer_rows).to_csv(
        run_dir / "spectral" / "layers.csv", index=False
    )
    summary_rows = [
        {
            "step": step,
            "n_matrices": len(MATRICES),
            "run_seed": SEED,
            "diagnostic_seed": SEED + 1_000_003 + step,
            "protocol_fingerprint": fingerprint,
            "model_state_sha256": model_hash,
        }
        for step in steps
    ]
    pd.DataFrame(summary_rows).to_csv(
        run_dir / "spectral" / "summary.csv",
        index=False,
    )
    inventory = pd.read_csv(run_dir / "epoch_metrics.csv")
    for _, row in inventory.iterrows():
        checkpoint_path = Path(str(row["checkpoint_path"]))
        checkpoint_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        torch.save(
            {
                "schema_version": 3,
                "model": {"weight": torch.tensor([1.0])},
                "model_state_sha256": model_state_sha256(
                    {"weight": torch.tensor([1.0])}
                ),
                "step": int(row["step"]),
                "nominal_epoch": float(row["nominal_epoch"]),
                "actual_epoch": float(row["epoch"]),
                "fingerprint": fingerprint,
                "config": cfg,
                "optimizer_name": OPTIMIZER,
                "seed": SEED,
                "purpose": "per_epoch_model_only_analysis_checkpoint",
            },
            checkpoint_path,
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
        lambda *args, **kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "rg_nanogpt_one_head.engine.configure_runtime",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "rg_nanogpt_one_head.engine.runtime_metadata",
        lambda *args, **kwargs: _runtime_identity(),
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
    monkeypatch.setattr(
        "rg_nanogpt_one_head.engine.choose_device",
        lambda *args, **kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "rg_nanogpt_one_head.engine.configure_runtime",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "rg_nanogpt_one_head.engine.runtime_metadata",
        lambda *args, **kwargs: _runtime_identity(),
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


def test_finite_checkpoint_model_bit_change_is_rejected(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    path = run_dir / "checkpoint_final.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["model"]["weight"][0] += 0.125
    torch.save(checkpoint, path)
    with pytest.raises(CompletedRunValidationError, match="model-state SHA-256"):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )


def test_finite_checkpoint_optimizer_change_is_rejected(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    path = run_dir / "checkpoint_final.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["optimizers"][0]["param_groups"].append({"lr": 0.5})
    torch.save(checkpoint, path)
    with pytest.raises(
        CompletedRunValidationError,
        match="optimizer-state SHA-256",
    ):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )


def test_missing_initial_checkpoint_is_not_complete(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    (run_dir / "checkpoint_initial.pt").unlink()
    with pytest.raises(CompletedRunValidationError, match="checkpoint_initial"):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )


def test_permanent_checkpoint_payload_identity_is_validated(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    epoch_metrics = pd.read_csv(run_dir / "epoch_metrics.csv")
    path = Path(str(epoch_metrics.iloc[-1]["checkpoint_path"]))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["step"] = int(payload["step"]) - 1
    torch.save(payload, path)
    with pytest.raises(CompletedRunValidationError, match="step mismatch"):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )


def test_nonfinite_test_metric_is_not_complete(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    path = run_dir / "test_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final"]["accuracy"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CompletedRunValidationError, match="non-finite"):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )


def test_weightwatcher_raw_csv_hash_is_bound_to_status(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    path = (
        run_dir
        / "spectral"
        / "raw"
        / "weightwatcher_step_0000000.csv"
    )
    raw = pd.read_csv(path)
    raw.loc[0, "alpha"] = 9.0
    raw.to_csv(path, index=False)
    with pytest.raises(CompletedRunValidationError, match="integrity status"):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )


def test_completion_test_metric_must_match_test_results(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    path = run_dir / "run_complete.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_test_loss"] += 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CompletedRunValidationError, match="does not match"):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )


def test_training_curve_cannot_contain_held_out_test_outcome(tmp_path):
    cfg = _config()
    run_dir, fingerprint, total_steps = _write_completed_run(tmp_path, cfg)
    path = run_dir / "metrics.csv"
    metrics = pd.read_csv(path)
    metrics.loc[metrics.index[-1], "test_loss"] = 1.25
    metrics.to_csv(path, index=False)
    with pytest.raises(CompletedRunValidationError, match="leaks held-out"):
        validate_completed_run(
            run_dir,
            expected_fingerprint=fingerprint,
            expected_optimizer=OPTIMIZER,
            expected_seed=SEED,
            expected_total_steps=total_steps,
        )


def test_runtime_guard_rejects_hardware_change_before_artifact_mutation(
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "checkpoint_latest.pt"
    marker.write_bytes(b"unchanged")
    (run_dir / "manifest.json").write_text(
        json.dumps({"runtime_environment": _runtime_identity()}),
        encoding="utf-8",
    )
    changed = {**_runtime_identity(), "accelerator": "cuda", "device": "cuda"}
    changed.update(
        {
            "cuda_version": "12.8",
            "cudnn_version": 9000,
            "cuda_device_name": "NVIDIA H100",
            "cuda_device_capability": [9, 0],
            "cuda_device_count": 1,
            "cuda_device_uuid": "GPU-test-h100",
            "cuda_driver_version": "575.57",
            "cuda_device_total_memory_bytes": 85_899_345_920,
            "cuda_multi_processor_count": 132,
            "cuda_nvidia_smi_memory_mib": 81_920,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
    )
    with pytest.raises(RuntimeError, match="cross-runtime"):
        validate_existing_manifest_runtime(run_dir, changed)
    assert marker.read_bytes() == b"unchanged"


@pytest.mark.parametrize("manifest_text", ["{", "[]"])
def test_runtime_guard_rejects_malformed_existing_manifest(
    tmp_path,
    manifest_text,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "checkpoint_latest.pt"
    marker.write_bytes(b"unchanged")
    (run_dir / "manifest.json").write_text(
        manifest_text,
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="manifest"):
        validate_existing_manifest_runtime(run_dir, _runtime_identity())
    assert marker.read_bytes() == b"unchanged"


def test_runtime_guard_rejects_substantive_artifacts_without_manifest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "checkpoint_latest.pt"
    marker.write_bytes(b"unchanged")
    with pytest.raises(RuntimeError, match="without manifest"):
        validate_existing_manifest_runtime(run_dir, _runtime_identity())
    assert marker.read_bytes() == b"unchanged"


def test_protocol_fingerprint_changes_with_scientific_dependencies(monkeypatch):
    cfg = _config()
    metadata = _data_metadata(cfg)
    monkeypatch.setattr(
        config_module,
        "scientific_dependency_versions",
        lambda: {"torch": "2.7.0", "weightwatcher": "0.7.7"},
    )
    first = protocol_fingerprint(
        cfg,
        optimizer=OPTIMIZER,
        seed=SEED,
        data_metadata=metadata,
    )
    monkeypatch.setattr(
        config_module,
        "scientific_dependency_versions",
        lambda: {"torch": "2.8.0", "weightwatcher": "0.7.7"},
    )
    second = protocol_fingerprint(
        cfg,
        optimizer=OPTIMIZER,
        seed=SEED,
        data_metadata=metadata,
    )
    assert first != second

"""Runtime hardening around the CIFAR-10 small-ViT reference runner.

This wrapper preserves the audited training recipe while adding accelerator RNG
state to checkpoints, isolating randomized WeightWatcher diagnostics from the
training trajectory, skipping completed compatible runs, repairing the valid
edge case in which the untrained epoch-zero model is validation-best, and
writing explicit final versus validation-selected test summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import vit_cifar10 as core


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and torch.backends.mps.is_available()
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if (
        "mps" in state
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(state["mps"])


def _accelerator_rng_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if torch.cuda.is_available():
        payload["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and torch.backends.mps.is_available()
    ):
        payload["mps_random_state"] = torch.mps.get_rng_state()
    return payload


def _restore_accelerator_rng_payload(payload: dict[str, Any]) -> None:
    if torch.cuda.is_available() and "cuda_random_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_random_state_all"])
    if (
        "mps_random_state" in payload
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(payload["mps_random_state"])


def _atomic_resave(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".rng.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _add_accelerator_rng_to_checkpoint(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.update(_accelerator_rng_payload())
    payload["schema_version"] = 2
    _atomic_resave(payload, path)


def _validation_selected_row(history: pd.DataFrame) -> pd.Series:
    if history.empty:
        raise RuntimeError("cannot select a checkpoint from an empty ViT history")
    candidates = history.dropna(subset=["validation_loss"]).sort_values(
        ["validation_loss", "epoch"], ascending=[True, True]
    )
    if candidates.empty:
        raise RuntimeError("ViT history has no finite validation loss")
    return candidates.iloc[0]


def _write_test_results(run_dir: Path, history: pd.DataFrame) -> None:
    final = history.sort_values("epoch").iloc[-1]
    selected = _validation_selected_row(history)
    payload = {
        "policy": (
            "validation loss selects checkpoint_best.pt; the official "
            "CIFAR-10 test set is monitoring-only"
        ),
        "final": {
            "epoch": int(final["epoch"]),
            "test_loss": float(final["test_loss"]),
            "test_accuracy": float(final["test_accuracy"]),
            "validation_loss": float(final["validation_loss"]),
        },
        "validation_selected": {
            "epoch": int(selected["epoch"]),
            "test_loss": float(selected["test_loss"]),
            "test_accuracy": float(selected["test_accuracy"]),
            "validation_loss": float(selected["validation_loss"]),
        },
    }
    (run_dir / "test_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    completion_path = run_dir / "run_complete.json"
    completion = (
        json.loads(completion_path.read_text(encoding="utf-8"))
        if completion_path.is_file()
        else {}
    )
    completion.update(
        {
            "completed": True,
            "best_validation_epoch": int(selected["epoch"]),
            "best_validation_loss": float(selected["validation_loss"]),
            "final_test_loss": float(final["test_loss"]),
            "final_test_accuracy": float(final["test_accuracy"]),
        }
    )
    temporary = completion_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(completion_path)


def _ensure_best_checkpoint(
    run_dir: Path,
    history: pd.DataFrame,
    *,
    optimizer_name: str,
    seed: int,
    config: core.ViTBaselineConfig,
) -> Path:
    """Ensure `checkpoint_best.pt` represents the validation-selected epoch.

    The core runner saves every post-training validation improvement. The only
    unsaved candidate is epoch zero, because no optimizer update has occurred.
    If epoch zero is genuinely best, reconstructing it is exact: the model is
    initialized on CPU immediately after `set_seed(seed)`, and the data-loader
    generator has not yet been consumed.
    """

    best_path = run_dir / "checkpoint_best.pt"
    selected = _validation_selected_row(history)
    selected_epoch = int(selected["epoch"])
    if best_path.is_file():
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != selected_epoch:
            raise RuntimeError(
                "checkpoint_best.pt does not match the validation-selected epoch"
            )
        return best_path
    if selected_epoch != 0:
        raise RuntimeError(
            "checkpoint_best.pt is missing for a post-training validation optimum"
        )

    state = _capture_rng_state()
    try:
        core.set_seed(int(seed))
        model = core.SmallViT(config)
        optimizer = core.build_optimizer(model, optimizer_name, config)
        core.set_learning_rates(optimizer, optimizer_name, config, 0)
        train_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        core._save_checkpoint(
            best_path,
            epoch=0,
            model=model,
            optimizer=optimizer,
            train_generator=train_generator,
            config=config,
            optimizer_name=optimizer_name,
            seed=int(seed),
            best_validation_loss=float(selected["validation_loss"]),
            fingerprint=core._fingerprint(optimizer_name, int(seed), config),
        )
        _add_accelerator_rng_to_checkpoint(best_path)
    finally:
        _restore_rng_state(state)
    return best_path


def run_vit_baseline(
    optimizer_name: str,
    seed: int,
    *,
    data_dir: Path,
    output_dir: Path,
    config: core.ViTBaselineConfig = core.ViTBaselineConfig(),
    device: torch.device | None = None,
    progress: bool = True,
    resume: bool = True,
):
    """Run the audited ViT recipe with deterministic diagnostics and restart."""

    run_dir = Path(output_dir) / optimizer_name / f"seed_{int(seed)}"
    completion_path = run_dir / "run_complete.json"
    history_path = run_dir / "history.csv"
    spectral_path = run_dir / "weightwatcher_by_epoch_layer.csv"
    if completion_path.is_file() and history_path.is_file() and spectral_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = core._fingerprint(optimizer_name, int(seed), config)
        if completion.get("fingerprint") != expected:
            raise RuntimeError(
                "completed ViT run belongs to a different protocol; choose a "
                "new output directory or remove the incompatible run"
            )
        if progress:
            print(f"[vit-baseline] loading completed run: {run_dir}")
        history = pd.read_csv(history_path)
        spectral = pd.read_csv(spectral_path)
        _ensure_best_checkpoint(
            run_dir,
            history,
            optimizer_name=optimizer_name,
            seed=int(seed),
            config=config,
        )
        if not (run_dir / "checkpoint_latest.pt").is_file():
            raise RuntimeError("completed ViT run is missing checkpoint_latest.pt")
        _write_test_results(run_dir, history)
        return history, spectral

    original_snapshot = core._ww_snapshot
    original_save = core._save_checkpoint
    original_load = core._load_checkpoint

    def isolated_snapshot(model, epoch, snapshot_config):
        state = _capture_rng_state()
        try:
            core.set_seed(int(seed) + 200_000 + int(epoch))
            return original_snapshot(model, epoch, snapshot_config)
        finally:
            _restore_rng_state(state)

    def checkpoint_with_accelerator_rng(path, **kwargs):
        original_save(path, **kwargs)
        _add_accelerator_rng_to_checkpoint(path)

    def load_with_accelerator_rng(path, **kwargs):
        result = original_load(path, **kwargs)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _restore_accelerator_rng_payload(payload)
        return result

    core._ww_snapshot = isolated_snapshot
    core._save_checkpoint = checkpoint_with_accelerator_rng
    core._load_checkpoint = load_with_accelerator_rng
    try:
        history, spectral = core.run_vit_baseline(
            optimizer_name,
            int(seed),
            data_dir=Path(data_dir),
            output_dir=Path(output_dir),
            config=config,
            device=device,
            progress=progress,
            resume=resume,
        )
    finally:
        core._ww_snapshot = original_snapshot
        core._save_checkpoint = original_save
        core._load_checkpoint = original_load

    _ensure_best_checkpoint(
        run_dir,
        history,
        optimizer_name=optimizer_name,
        seed=int(seed),
        config=config,
    )
    _write_test_results(run_dir, history)
    if not (run_dir / "checkpoint_latest.pt").is_file():
        raise RuntimeError("ViT run completed without checkpoint_latest.pt")
    return history, spectral


ViTBaselineConfig = core.ViTBaselineConfig
DEFAULT_VIT_SEEDS = core.DEFAULT_VIT_SEEDS
choose_device = core.choose_device

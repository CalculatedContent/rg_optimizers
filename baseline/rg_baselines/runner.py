"""Run one restartable MLP3/MNIST optimizer baseline."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import BaselineConfig
from .diagnostics import measure_weightwatcher_checkpoint
from .engine import (
    choose_device,
    evaluate,
    parameter_l2_norm,
    performance_row,
    set_seed,
    train_one_epoch,
)
from .model import MLP3
from .optimizers import (
    build_optimizer,
    optimizer_group_rows,
    set_scheduled_learning_rates,
)
from .results import BaselineResult, validate_result
from .trap_metrics import attach_correlation_traps


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


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


def _indices_sha256(indices: list[int]) -> str:
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _protocol_fingerprint(
    config: BaselineConfig,
    *,
    train_indices: list[int],
    validation_indices: list[int],
) -> str:
    payload = {
        "config": asdict(config),
        "train_indices_sha256": _indices_sha256(train_indices),
        "validation_indices_sha256": _indices_sha256(validation_indices),
        "model": "MLP3:784-512-512-10:relu",
        "dataset": "torchvision.MNIST",
        "normalization": [0.1307, 0.3081],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _make_datasets_and_loaders(
    config: BaselineConfig,
    *,
    data_dir: str | Path,
    device: torch.device,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    full_train = datasets.MNIST(
        str(data_dir), train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        str(data_dir), train=False, download=True, transform=transform
    )

    split_generator = torch.Generator(device="cpu").manual_seed(
        int(config.split_seed)
    )
    permutation = torch.randperm(
        len(full_train), generator=split_generator
    ).tolist()
    validation_indices = permutation[: int(config.validation_size)]
    train_indices = permutation[int(config.validation_size) :]
    train = Subset(full_train, train_indices)
    validation = Subset(full_train, validation_indices)

    train_generator = torch.Generator(device="cpu").manual_seed(
        int(config.seed) + 101
    )
    workers = 0 if device.type == "mps" else int(config.num_workers)
    common = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    train_eval_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=False,
        **common,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=config.batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test,
        batch_size=config.batch_size,
        shuffle=False,
        **common,
    )
    return (
        train_loader,
        train_eval_loader,
        validation_loader,
        test_loader,
        train_generator,
        train_indices,
        validation_indices,
    )


def _checkpoint_payload(
    *,
    config: BaselineConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    best_validation_epoch: int,
    train_generator: torch.Generator,
    fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "config": asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_epoch": int(best_validation_epoch),
        "train_generator_state": train_generator.get_state(),
        "rng_state": _capture_rng_state(),
        "protocol_fingerprint": str(fingerprint),
    }


def _load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    expected_fingerprint: str,
) -> tuple[int, int, float, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("protocol_fingerprint")) != str(expected_fingerprint):
        raise RuntimeError(
            "MNIST checkpoint protocol fingerprint does not match this run"
        )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    train_generator.set_state(payload["train_generator_state"])
    _restore_rng_state(payload["rng_state"])
    return (
        int(payload["epoch"]),
        int(payload["global_step"]),
        float(payload["best_validation_loss"]),
        int(payload["best_validation_epoch"]),
    )


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _write_progress(
    run_dir: Path,
    *,
    performance: pd.DataFrame,
    spectral: pd.DataFrame,
    details: pd.DataFrame,
    groups: pd.DataFrame,
    esds: dict[str, np.ndarray],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(run_dir / "performance_by_epoch.csv", index=False)
    spectral.to_csv(
        run_dir / "spectral_metrics_by_epoch_and_layer.csv", index=False
    )
    details.to_csv(run_dir / "weightwatcher_details_by_epoch.csv", index=False)
    groups.to_csv(run_dir / "optimizer_groups_by_epoch.csv", index=False)
    np.savez_compressed(run_dir / "esd_history.npz", **esds)


def _load_completed_result(
    config: BaselineConfig,
    run_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    fingerprint: str,
) -> BaselineResult:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("protocol_fingerprint")) != str(fingerprint):
        raise RuntimeError(
            "completed MNIST run was produced by a different protocol"
        )
    final_payload = torch.load(
        run_dir / "final_state.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(final_payload["model"])
    optimizer.load_state_dict(final_payload["optimizer"])
    performance = pd.read_csv(run_dir / "performance_by_epoch.csv")
    spectral = pd.read_csv(
        run_dir / "spectral_metrics_by_epoch_and_layer.csv"
    )
    details = pd.read_csv(run_dir / "weightwatcher_details_by_epoch.csv")
    groups = pd.read_csv(run_dir / "optimizer_groups_by_epoch.csv")
    combined_path = run_dir / "combined_metrics_by_epoch_and_layer.csv"
    combined = (
        pd.read_csv(combined_path)
        if combined_path.is_file()
        else spectral.merge(
            performance,
            on=["run", "epoch", "global_step"],
            how="left",
            validate="many_to_one",
        )
    )
    with np.load(run_dir / "esd_history.npz") as archive:
        esds = {name: archive[name] for name in archive.files}
    result = BaselineResult(
        config,
        performance,
        spectral,
        details,
        groups,
        combined,
        esds,
        model,
        optimizer,
    )
    if config.strict_metrics:
        validate_result(result)
    return result


def run_baseline(
    config: BaselineConfig,
    *,
    data_dir: str | Path = "./data",
    device: Optional[torch.device] = None,
    output_dir: Optional[str | Path] = None,
    progress: bool = True,
    resume: bool = True,
    overwrite: bool = False,
) -> BaselineResult:
    """Run, resume, or load one deterministic MNIST optimizer control."""

    config.validate()
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    set_seed(config.seed)
    device = device or choose_device()

    (
        train_loader,
        train_eval_loader,
        validation_loader,
        test_loader,
        train_generator,
        train_indices,
        validation_indices,
    ) = _make_datasets_and_loaders(config, data_dir=data_dir, device=device)

    model = MLP3().to(device)
    optimizer = build_optimizer(model, config)
    steps_per_epoch = len(train_loader)
    total_steps = int(config.epochs) * steps_per_epoch
    fingerprint = _protocol_fingerprint(
        config,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )

    run_dir = Path(output_dir) if output_dir is not None else None
    if run_dir is not None and run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        completion_path = run_dir / "run_complete.json"
        if completion_path.is_file() and not overwrite:
            if progress:
                print(f"[mnist-baseline] loading completed run: {run_dir}")
            return _load_completed_result(
                config,
                run_dir,
                model=model,
                optimizer=optimizer,
                fingerprint=fingerprint,
            )
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "protocol_fingerprint": fingerprint,
                    "config": asdict(config),
                    "dataset": "torchvision.MNIST",
                    "normalization": {"mean": 0.1307, "std": 0.3081},
                    "train_examples": len(train_indices),
                    "validation_examples": len(validation_indices),
                    "test_examples": 10_000,
                    "train_indices_sha256": _indices_sha256(train_indices),
                    "validation_indices_sha256": _indices_sha256(
                        validation_indices
                    ),
                    "test_monitoring_only": True,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    "device": str(device),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    performance = (
        _read_frame(run_dir / "performance_by_epoch.csv")
        if run_dir is not None
        else pd.DataFrame()
    )
    spectral = (
        _read_frame(run_dir / "spectral_metrics_by_epoch_and_layer.csv")
        if run_dir is not None
        else pd.DataFrame()
    )
    details = (
        _read_frame(run_dir / "weightwatcher_details_by_epoch.csv")
        if run_dir is not None
        else pd.DataFrame()
    )
    groups = (
        _read_frame(run_dir / "optimizer_groups_by_epoch.csv")
        if run_dir is not None
        else pd.DataFrame()
    )
    esds: dict[str, np.ndarray] = {}
    if run_dir is not None and (run_dir / "esd_history.npz").is_file():
        with np.load(run_dir / "esd_history.npz") as archive:
            esds = {name: archive[name] for name in archive.files}

    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    best_validation_epoch = 0
    latest_checkpoint = run_dir / "checkpoint_latest.pt" if run_dir else None
    best_checkpoint = run_dir / "checkpoint_best.pt" if run_dir else None

    if latest_checkpoint is not None and latest_checkpoint.is_file() and resume:
        (
            start_epoch,
            global_step,
            best_validation_loss,
            best_validation_epoch,
        ) = _load_training_checkpoint(
            latest_checkpoint,
            model=model,
            optimizer=optimizer,
            train_generator=train_generator,
            expected_fingerprint=fingerprint,
        )
        model.to(device)
        for frame_name in ("performance", "spectral", "details", "groups"):
            frame = locals()[frame_name]
            if not frame.empty and "epoch" in frame:
                locals()[frame_name] = frame[
                    frame["epoch"].astype(int) <= start_epoch
                ].copy()
        performance = performance[
            performance["epoch"].astype(int) <= start_epoch
        ].copy() if not performance.empty else performance
        spectral = spectral[
            spectral["epoch"].astype(int) <= start_epoch
        ].copy() if not spectral.empty else spectral
        details = details[
            details["epoch"].astype(int) <= start_epoch
        ].copy() if not details.empty else details
        groups = groups[
            groups["epoch"].astype(int) <= start_epoch
        ].copy() if not groups.empty else groups
        if progress:
            print(
                f"[mnist-baseline] resume {config.optimizer_label} "
                f"seed={config.seed} epoch={start_epoch} step={global_step}"
            )
    elif run_dir is not None and any(run_dir.iterdir()) and resume:
        allowed = {"manifest.json"}
        nontrivial = [path for path in run_dir.iterdir() if path.name not in allowed]
        if nontrivial and not (run_dir / "performance_by_epoch.csv").is_file():
            raise FileNotFoundError(
                f"cannot resume incomplete run without {latest_checkpoint}"
            )

    def persist() -> None:
        if run_dir is None:
            return
        _write_progress(
            run_dir,
            performance=performance,
            spectral=spectral,
            details=details,
            groups=groups,
            esds=esds,
        )

    def save_checkpoint(path: Path, epoch: int) -> None:
        _atomic_torch_save(
            _checkpoint_payload(
                config=config,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                best_validation_epoch=best_validation_epoch,
                train_generator=train_generator,
                fingerprint=fingerprint,
            ),
            path,
        )

    def measure(
        epoch: int,
        online: Optional[dict],
        train_time: float,
        current_learning_rates: dict[str, float],
    ) -> tuple[float, float]:
        nonlocal performance, spectral, details, groups, esds
        started = time.perf_counter()
        train_result = evaluate(
            model,
            train_eval_loader,
            device=device,
            max_batches=config.train_eval_max_batches,
        )
        validation_result = evaluate(
            model, validation_loader, device=device
        )
        test_result = evaluate(model, test_loader, device=device)
        evaluation_time = time.perf_counter() - started

        # WeightWatcher randomization is diagnostic-only. Preserve all training
        # RNG streams so measurement cannot alter the next epoch's trajectory.
        rng_state = _capture_rng_state()
        started = time.perf_counter()
        try:
            set_seed(int(config.seed) + 100_000 + int(epoch))
            checkpoint = measure_weightwatcher_checkpoint(
                model,
                run_label=config.optimizer_label,
                epoch=epoch,
                global_step=global_step,
                min_evals=config.ww_min_evals,
                max_evals=config.ww_max_evals,
                svd_method=config.ww_svd_method,
                randomize=config.ww_randomize,
            )
            checkpoint = attach_correlation_traps(checkpoint)
        finally:
            _restore_rng_state(rng_state)
        weightwatcher_time = time.perf_counter() - started

        row = pd.DataFrame(
            [
                performance_row(
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    train_eval=train_result,
                    validation_eval=validation_result,
                    test_eval=test_result,
                    online=online,
                    learning_rates=current_learning_rates,
                    parameter_norm=parameter_l2_norm(model),
                    train_time=train_time,
                    evaluation_time=evaluation_time,
                    ww_time=weightwatcher_time,
                    device=device,
                )
            ]
        )
        performance = pd.concat([performance, row], ignore_index=True)
        performance = performance.drop_duplicates("epoch", keep="last")
        spectral = pd.concat(
            [spectral, checkpoint.metrics], ignore_index=True, sort=False
        ).drop_duplicates(["epoch", "layer_id"], keep="last")
        details = pd.concat(
            [details, checkpoint.details], ignore_index=True, sort=False
        ).drop_duplicates(["epoch", "layer_id"], keep="last")
        group_rows = pd.DataFrame(
            optimizer_group_rows(
                optimizer,
                epoch=epoch,
                optimizer_label=config.optimizer_label,
            )
        )
        groups = pd.concat([groups, group_rows], ignore_index=True, sort=False)
        groups = groups.drop_duplicates(
            ["epoch", "group_index"], keep="last"
        )
        esds.update(checkpoint.esd_arrays)
        persist()
        return float(validation_result["loss"]), float(test_result["loss"])

    if performance.empty:
        initial_lrs = set_scheduled_learning_rates(
            optimizer,
            config,
            update_index=0,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
        )
        validation_loss, _ = measure(0, None, 0.0, initial_lrs)
        best_validation_loss = validation_loss
        best_validation_epoch = 0
        if best_checkpoint is not None:
            save_checkpoint(best_checkpoint, 0)
            save_checkpoint(latest_checkpoint, 0)
        if progress:
            row = performance.iloc[-1]
            print(
                f"epoch=000 | {config.optimizer_label} | "
                f"lr={row['primary_lr']:.3e} | "
                f"train={row['train_accuracy']:.4f} | "
                f"val={row['validation_accuracy']:.4f} | "
                f"test={row['test_accuracy']:.4f}"
            )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        started = time.perf_counter()
        online = train_one_epoch(
            model,
            optimizer,
            train_loader,
            config=config,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
            global_step=global_step,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
        )
        train_time = time.perf_counter() - started
        global_step = int(online["global_step"])
        learning_rates = {
            "primary": float(online["primary_lr"]),
            "auxiliary": float(online["auxiliary_lr"]),
        }
        validation_loss, _ = measure(
            epoch, online, train_time, learning_rates
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_epoch = epoch
            if best_checkpoint is not None:
                save_checkpoint(best_checkpoint, epoch)

        if run_dir is not None:
            save_checkpoint(latest_checkpoint, epoch)
            if (
                config.save_epoch_checkpoints
                and (
                    epoch % config.checkpoint_every_epochs == 0
                    or epoch == config.epochs
                )
            ):
                save_checkpoint(
                    run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                    epoch,
                )
        if progress:
            row = performance.iloc[-1]
            print(
                f"epoch={epoch:03d} | {config.optimizer_label} | "
                f"lr={row['primary_lr']:.3e} | "
                f"train={row['train_accuracy']:.4f} | "
                f"val={row['validation_accuracy']:.4f} | "
                f"test={row['test_accuracy']:.4f}"
            )

    performance = performance.sort_values("epoch").reset_index(drop=True)
    spectral = spectral.sort_values(["epoch", "layer_id"]).reset_index(drop=True)
    details = details.sort_values(["epoch", "layer_id"]).reset_index(drop=True)
    groups = groups.sort_values(["epoch", "group_index"]).reset_index(drop=True)
    combined = spectral.merge(
        performance,
        on=["run", "epoch", "global_step"],
        how="left",
        validate="many_to_one",
    )
    result = BaselineResult(
        config,
        performance,
        spectral,
        details,
        groups,
        combined,
        esds,
        model,
        optimizer,
    )
    if config.strict_metrics:
        validate_result(result)

    if run_dir is not None:
        result.save(run_dir)
        final_row = performance.iloc[-1]
        best_row = performance[
            performance["epoch"].astype(int).eq(best_validation_epoch)
        ].iloc[-1]
        test_results = {
            "policy": (
                "validation loss selects checkpoint_best.pt; the official "
                "test set is monitoring-only"
            ),
            "final": {
                "epoch": int(final_row["epoch"]),
                "loss": float(final_row["test_loss"]),
                "accuracy": float(final_row["test_accuracy"]),
            },
            "validation_selected": {
                "epoch": int(best_row["epoch"]),
                "loss": float(best_row["test_loss"]),
                "accuracy": float(best_row["test_accuracy"]),
                "validation_loss": float(best_row["validation_loss"]),
            },
        }
        (run_dir / "test_results.json").write_text(
            json.dumps(test_results, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        completion = {
            "completed": True,
            "optimizer": config.optimizer,
            "optimizer_label": config.optimizer_label,
            "seed": int(config.seed),
            "epochs": int(config.epochs),
            "global_step": int(global_step),
            "best_validation_epoch": int(best_validation_epoch),
            "best_validation_loss": float(best_validation_loss),
            "final_test_loss": float(final_row["test_loss"]),
            "final_test_accuracy": float(final_row["test_accuracy"]),
            "protocol_fingerprint": fingerprint,
        }
        temporary = run_dir / "run_complete.json.tmp"
        temporary.write_text(
            json.dumps(completion, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(run_dir / "run_complete.json")
    return result

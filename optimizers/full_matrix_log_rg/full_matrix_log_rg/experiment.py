"""Restartable MNIST/MLP3 SGD baseline and FullMatrixLogRG experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import FullMatrixLogConfig
from .support import SupportCheckpoint, analyze_supports
from .wrapper import FullMatrixLogRG


@dataclass
class MNISTRunResult:
    performance: pd.DataFrame
    spectral: pd.DataFrame
    details: pd.DataFrame
    corrections: pd.DataFrame
    output_dir: Path
    completed: bool


@dataclass
class GridSearchResult:
    results: pd.DataFrame
    selected_config: FullMatrixLogConfig
    output_dir: Path


def _baseline_imports():
    try:
        from rg_baselines import (
            BaselineConfig,
            MLP3,
            build_optimizer,
            measure_weightwatcher_checkpoint,
            set_scheduled_learning_rates,
        )
        from rg_baselines.engine import choose_device, evaluate, set_seed
        from rg_baselines.trap_metrics import attach_correlation_traps
    except ImportError as exc:
        raise ImportError(
            "The MNIST experiment requires the sibling baseline package. Install "
            "from the repository root with `python -m pip install -e './baseline[experiment]'`."
        ) from exc
    return (
        BaselineConfig,
        MLP3,
        build_optimizer,
        measure_weightwatcher_checkpoint,
        set_scheduled_learning_rates,
        choose_device,
        evaluate,
        set_seed,
        attach_correlation_traps,
    )


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


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _indices_sha256(indices: list[int]) -> str:
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _make_loaders(config, *, data_dir: Path, device: torch.device):
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
    split_generator = torch.Generator(device="cpu").manual_seed(int(config.split_seed))
    permutation = torch.randperm(
        len(full_train), generator=split_generator
    ).tolist()
    validation_indices = permutation[: int(config.validation_size)]
    train_indices = permutation[int(config.validation_size) :]
    train = Subset(full_train, train_indices)
    validation = Subset(full_train, validation_indices)

    train_generator = torch.Generator(device="cpu").manual_seed(int(config.seed) + 101)
    workers = 0 if device.type == "mps" else int(config.num_workers)
    common = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train,
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=train_generator,
        **common,
    )
    train_eval_loader = DataLoader(
        train,
        batch_size=int(config.batch_size),
        shuffle=False,
        **common,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=int(config.batch_size),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test,
        batch_size=int(config.batch_size),
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


def _protocol_fingerprint(
    baseline_config,
    rg_config: FullMatrixLogConfig | None,
    *,
    train_indices: list[int],
    validation_indices: list[int],
    evaluate_test: bool,
) -> str:
    payload = {
        "baseline_config": asdict(baseline_config),
        "rg_config": asdict(rg_config) if rg_config is not None else None,
        "train_indices_sha256": _indices_sha256(train_indices),
        "validation_indices_sha256": _indices_sha256(validation_indices),
        "model": "MLP3:784-512-512-10:relu",
        "dataset": "torchvision.MNIST",
        "normalization": [0.1307, 0.3081],
        "evaluate_test": bool(evaluate_test),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_esds(esds: dict[str, np.ndarray], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **esds)
    temporary.replace(path)


def _load_esds(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def _run_label(rg_config: FullMatrixLogConfig | None) -> str:
    if rg_config is None:
        return "SGD + Nesterov momentum"
    return f"SGD + Nesterov + FullMatrixLogRG ({rg_config.mode})"


def run_mnist_sgd(
    baseline_config,
    *,
    rg_config: FullMatrixLogConfig | None,
    data_dir: str | Path,
    output_dir: str | Path,
    device: torch.device | None = None,
    evaluate_test: bool = True,
    progress: bool = True,
    resume: bool = True,
    overwrite: bool = False,
) -> MNISTRunResult:
    """Run or resume one baseline-compatible SGD trajectory."""

    (
        _,
        MLP3,
        build_optimizer,
        measure_weightwatcher_checkpoint,
        set_scheduled_learning_rates,
        choose_device,
        evaluate,
        set_seed,
        attach_correlation_traps,
    ) = _baseline_imports()

    baseline_config.validate()
    if baseline_config.optimizer != "sgd_momentum":
        raise ValueError("run_mnist_sgd requires BaselineConfig(optimizer='sgd_momentum')")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if rg_config is not None:
        rg_config.validate()

    run_dir = Path(output_dir).expanduser().resolve()
    data_path = Path(data_dir).expanduser().resolve()
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    set_seed(int(baseline_config.seed))
    device = device or choose_device()
    (
        train_loader,
        train_eval_loader,
        validation_loader,
        test_loader,
        train_generator,
        train_indices,
        validation_indices,
    ) = _make_loaders(baseline_config, data_dir=data_path, device=device)

    model = MLP3().to(device)
    base_optimizer = build_optimizer(model, baseline_config)
    optimizer = (
        base_optimizer
        if rg_config is None
        else FullMatrixLogRG(base_optimizer, model.named_parameters(), rg_config)
    )
    steps_per_epoch = len(train_loader)
    total_steps = int(baseline_config.epochs) * steps_per_epoch
    fingerprint = _protocol_fingerprint(
        baseline_config,
        rg_config,
        train_indices=train_indices,
        validation_indices=validation_indices,
        evaluate_test=evaluate_test,
    )
    label = _run_label(rg_config)

    manifest = {
        "schema_version": 1,
        "protocol_fingerprint": fingerprint,
        "label": label,
        "baseline_config": asdict(baseline_config),
        "rg_config": asdict(rg_config) if rg_config is not None else None,
        "device": str(device),
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "test_examples": 10_000,
        "test_evaluated": bool(evaluate_test),
        "test_monitoring_only": True,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_indices_sha256": _indices_sha256(train_indices),
        "validation_indices_sha256": _indices_sha256(validation_indices),
    }
    existing_manifest = run_dir / "manifest.json"
    if existing_manifest.is_file():
        current = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if current.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing run has a different protocol: {run_dir}. Use a new directory."
            )
    else:
        _atomic_json(manifest, existing_manifest)

    performance = _read_frame(run_dir / "performance_by_epoch.csv")
    spectral = _read_frame(run_dir / "spectral_metrics_by_epoch_and_layer.csv")
    details = _read_frame(run_dir / "weightwatcher_details_by_epoch.csv")
    corrections = _read_frame(run_dir / "rg_corrections_by_step.csv")
    esds = _load_esds(run_dir / "esd_history.npz")

    latest = run_dir / "checkpoint_latest.pt"
    best = run_dir / "checkpoint_best.pt"
    completion = run_dir / "run_complete.json"
    if completion.is_file() and (run_dir / "final_state.pt").is_file() and not overwrite:
        if progress:
            print(f"[full-matrix-log] loading completed run: {run_dir}")
        return MNISTRunResult(performance, spectral, details, corrections, run_dir, True)

    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    best_validation_epoch = 0
    if latest.is_file() and resume:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("Checkpoint protocol fingerprint mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        model.to(device)
        train_generator.set_state(payload["train_generator_state"])
        _restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best_validation_loss = float(payload["best_validation_loss"])
        best_validation_epoch = int(payload["best_validation_epoch"])
        performance = (
            performance[performance["epoch"].astype(int) <= start_epoch].copy()
            if not performance.empty
            else performance
        )
        spectral = (
            spectral[spectral["epoch"].astype(int) <= start_epoch].copy()
            if not spectral.empty
            else spectral
        )
        details = (
            details[details["epoch"].astype(int) <= start_epoch].copy()
            if not details.empty
            else details
        )
        corrections = (
            corrections[corrections["epoch"].astype(int) <= start_epoch].copy()
            if not corrections.empty
            else corrections
        )
        if progress:
            print(
                f"[full-matrix-log] resume seed={baseline_config.seed} "
                f"epoch={start_epoch} step={global_step}"
            )
    elif resume and not latest.is_file():
        nontrivial = [
            path
            for path in run_dir.iterdir()
            if path.name not in {"manifest.json"}
        ]
        if nontrivial:
            raise FileNotFoundError(f"Cannot resume incomplete run without {latest}")

    def persist() -> None:
        _atomic_csv(performance, run_dir / "performance_by_epoch.csv")
        _atomic_csv(spectral, run_dir / "spectral_metrics_by_epoch_and_layer.csv")
        _atomic_csv(details, run_dir / "weightwatcher_details_by_epoch.csv")
        _atomic_csv(corrections, run_dir / "rg_corrections_by_step.csv")
        _write_esds(esds, run_dir / "esd_history.npz")

    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol_fingerprint": fingerprint,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_validation_loss": float(best_validation_loss),
            "best_validation_epoch": int(best_validation_epoch),
            "train_generator_state": train_generator.get_state(),
            "rng_state": _capture_rng_state(),
        }

    def measure(epoch: int, train_time: float, online: dict[str, float]) -> SupportCheckpoint:
        nonlocal performance, spectral, details, esds
        train_result = evaluate(
            model,
            train_eval_loader,
            device=device,
            max_batches=baseline_config.train_eval_max_batches,
        )
        validation_result = evaluate(model, validation_loader, device=device)
        test_result = (
            evaluate(model, test_loader, device=device)
            if evaluate_test
            else {"loss": np.nan, "accuracy": np.nan, "examples": 0}
        )

        rng_state = _capture_rng_state()
        started = time.perf_counter()
        try:
            set_seed(int(baseline_config.seed) + 100_000 + int(epoch))
            if rg_config is not None:
                checkpoint = analyze_supports(
                    model,
                    run_label=label,
                    epoch=epoch,
                    global_step=global_step,
                    min_evals=baseline_config.ww_min_evals,
                    max_evals=baseline_config.ww_max_evals,
                    svd_method=baseline_config.ww_svd_method,
                    randomize=baseline_config.ww_randomize,
                    parameter_names=rg_config.parameter_names,
                    build_bases=True,
                )
                optimizer.set_supports(checkpoint.supports)
            else:
                measured = measure_weightwatcher_checkpoint(
                    model,
                    run_label=label,
                    epoch=epoch,
                    global_step=global_step,
                    min_evals=baseline_config.ww_min_evals,
                    max_evals=baseline_config.ww_max_evals,
                    svd_method=baseline_config.ww_svd_method,
                    randomize=baseline_config.ww_randomize,
                )
                measured = attach_correlation_traps(measured)
                checkpoint = SupportCheckpoint(
                    measured.details, measured.metrics, {}, measured.esd_arrays
                )
        finally:
            _restore_rng_state(rng_state)
        ww_time = time.perf_counter() - started

        row = pd.DataFrame(
            [
                {
                    "run": label,
                    "seed": int(baseline_config.seed),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "train_loss": float(train_result["loss"]),
                    "train_accuracy": float(train_result["accuracy"]),
                    "validation_loss": float(validation_result["loss"]),
                    "validation_accuracy": float(validation_result["accuracy"]),
                    "test_loss": float(test_result["loss"]),
                    "test_accuracy": float(test_result["accuracy"]),
                    "test_evaluated": int(bool(evaluate_test)),
                    "online_train_loss": float(online.get("loss", np.nan)),
                    "online_train_accuracy": float(online.get("accuracy", np.nan)),
                    "primary_lr": float(optimizer.param_groups[0]["lr"]),
                    "train_time_sec": float(train_time),
                    "weightwatcher_time_sec": float(ww_time),
                    "device": str(device),
                }
            ]
        )
        performance = pd.concat([performance, row], ignore_index=True).drop_duplicates(
            "epoch", keep="last"
        )
        spectral_frame = checkpoint.metrics.copy()
        spectral_frame["seed"] = int(baseline_config.seed)
        spectral = pd.concat([spectral, spectral_frame], ignore_index=True, sort=False)
        if {"epoch", "layer_id"}.issubset(spectral.columns):
            spectral = spectral.drop_duplicates(["epoch", "layer_id"], keep="last")
        details_frame = checkpoint.details.copy()
        details_frame["seed"] = int(baseline_config.seed)
        details = pd.concat([details, details_frame], ignore_index=True, sort=False)
        if {"epoch", "layer_id"}.issubset(details.columns):
            details = details.drop_duplicates(["epoch", "layer_id"], keep="last")
        esds.update(checkpoint.esd_arrays)
        return checkpoint

    if start_epoch == 0 and performance.empty:
        measure(0, 0.0, {})
        persist()
        _atomic_torch_save(checkpoint_payload(0), latest)

    for epoch in range(start_epoch + 1, int(baseline_config.epochs) + 1):
        model.train()
        started = time.perf_counter()
        loss_sum = 0.0
        correct = 0
        seen = 0
        correction_rows: list[dict[str, Any]] = []
        for inputs, targets in train_loader:
            set_scheduled_learning_rates(
                optimizer,
                baseline_config,
                update_index=global_step,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(baseline_config.grad_clip_norm)
            )
            optimizer.step()
            global_step += 1

            batch_size = int(targets.numel())
            loss_sum += float(loss.item()) * batch_size
            correct += int((logits.argmax(1) == targets).sum())
            seen += batch_size
            if rg_config is not None:
                for stat in optimizer.pop_step_stats():
                    correction_rows.append(
                        {
                            "run": label,
                            "seed": int(baseline_config.seed),
                            "epoch": int(epoch),
                            **stat,
                        }
                    )
        train_time = time.perf_counter() - started
        if correction_rows:
            corrections = pd.concat(
                [corrections, pd.DataFrame(correction_rows)],
                ignore_index=True,
                sort=False,
            )
        online = {
            "loss": loss_sum / max(seen, 1),
            "accuracy": correct / max(seen, 1),
        }
        measure(epoch, train_time, online)

        validation_loss = float(
            performance.loc[performance["epoch"].eq(epoch), "validation_loss"].iloc[-1]
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_epoch = int(epoch)
            _atomic_torch_save(checkpoint_payload(epoch), best)
        persist()
        _atomic_torch_save(checkpoint_payload(epoch), latest)
        if baseline_config.save_epoch_checkpoints:
            _atomic_torch_save(
                checkpoint_payload(epoch),
                run_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            )
        if progress:
            row = performance[performance["epoch"].eq(epoch)].iloc[-1]
            print(
                f"[{label}] seed={baseline_config.seed} epoch={epoch:03d} "
                f"val={row.validation_accuracy:.4f} test={row.test_accuracy:.4f}"
            )

    final_payload = checkpoint_payload(int(baseline_config.epochs))
    _atomic_torch_save(final_payload, run_dir / "final_state.pt")
    _atomic_json(
        {
            "completed": True,
            "epoch": int(baseline_config.epochs),
            "global_step": int(global_step),
            "best_validation_loss": float(best_validation_loss),
            "best_validation_epoch": int(best_validation_epoch),
            "protocol_fingerprint": fingerprint,
        },
        completion,
    )
    persist()
    return MNISTRunResult(performance, spectral, details, corrections, run_dir, True)


def run_validation_grid(
    baseline_config,
    candidates: list[FullMatrixLogConfig],
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    device: torch.device | None = None,
    progress: bool = True,
    resume: bool = True,
) -> GridSearchResult:
    """Run a bounded single-seed validation-only grid and select one point."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate.validate()
        cap_slug = (
            "none"
            if candidate.max_correction_ratio is None
            else f"{candidate.max_correction_ratio:g}"
        )
        slug = (
            f"{index:02d}_{candidate.mode}_s{candidate.projection_strength:g}_"
            f"c{cap_slug}_e{candidate.apply_every_steps}"
        )
        result = run_mnist_sgd(
            baseline_config,
            rg_config=candidate,
            data_dir=data_dir,
            output_dir=root / slug,
            device=device,
            evaluate_test=False,
            progress=progress,
            resume=resume,
        )
        ranked_epochs = result.performance.sort_values(
            ["validation_loss", "epoch"], ascending=[True, True]
        )
        best_epoch_row = ranked_epochs.iloc[0]
        best_epoch = int(best_epoch_row.epoch)
        ok = (
            result.spectral[
                result.spectral["status"].eq("ok")
                & result.spectral["epoch"].astype(int).eq(best_epoch)
            ]
            if not result.spectral.empty
            and {"status", "epoch"}.issubset(result.spectral.columns)
            else pd.DataFrame()
        )
        rows.append(
            {
                "candidate_index": index,
                "candidate_slug": slug,
                **asdict(candidate),
                "best_epoch": best_epoch,
                "validation_loss": float(best_epoch_row.validation_loss),
                "validation_accuracy": float(best_epoch_row.validation_accuracy),
                "mean_abs_alpha_minus_2": float(ok["abs_alpha_minus_2"].mean())
                if not ok.empty and "abs_alpha_minus_2" in ok
                else np.nan,
                "mean_correction_ratio": float(
                    result.corrections["correction_ratio"].mean()
                )
                if not result.corrections.empty
                else 0.0,
            }
        )
        _atomic_csv(pd.DataFrame(rows), root / "grid_results.csv")

    table = pd.DataFrame(rows).sort_values(
        [
            "validation_loss",
            "validation_accuracy",
            "mean_abs_alpha_minus_2",
            "mean_correction_ratio",
        ],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    selected_index = int(table.iloc[0]["candidate_index"])
    selected = candidates[selected_index]
    _atomic_csv(table, root / "grid_results_ranked.csv")
    _atomic_json(asdict(selected), root / "selected_config.json")
    return GridSearchResult(table, selected, root)

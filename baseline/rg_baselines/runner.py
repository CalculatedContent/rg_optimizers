"""Run one clean MLP3/MNIST optimizer baseline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
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


def run_baseline(
    config: BaselineConfig,
    *,
    data_dir: str | Path = "./data",
    device: Optional[torch.device] = None,
    output_dir: Optional[str | Path] = None,
    progress: bool = True,
) -> BaselineResult:
    config.validate()
    set_seed(config.seed)
    device = device or choose_device()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
    )
    train_eval = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    model = MLP3().to(device)
    optimizer = build_optimizer(model, config)
    learning_rates = set_scheduled_learning_rates(optimizer, config, epoch_index=0)
    step = 0
    performance: list[dict] = []
    spectral: list[pd.DataFrame] = []
    details: list[pd.DataFrame] = []
    groups: list[dict] = []
    esds = {}

    def measure(
        epoch: int,
        online: Optional[dict],
        train_time: float,
        current_learning_rates: dict[str, float],
    ) -> None:
        nonlocal step
        started = time.perf_counter()
        train_result = evaluate(
            model,
            train_eval,
            device=device,
            max_batches=config.train_eval_max_batches,
        )
        test_result = evaluate(model, test_loader, device=device)
        evaluation_time = time.perf_counter() - started

        started = time.perf_counter()
        checkpoint = measure_weightwatcher_checkpoint(
            model,
            run_label=config.optimizer_label,
            epoch=epoch,
            global_step=step,
            min_evals=config.ww_min_evals,
            max_evals=config.ww_max_evals,
            svd_method=config.ww_svd_method,
            randomize=config.ww_randomize,
        )
        checkpoint = attach_correlation_traps(checkpoint)
        weightwatcher_time = time.perf_counter() - started

        performance.append(
            performance_row(
                config=config,
                epoch=epoch,
                global_step=step,
                train_eval=train_result,
                test_eval=test_result,
                online=online,
                learning_rates=current_learning_rates,
                parameter_norm=parameter_l2_norm(model),
                train_time=train_time,
                evaluation_time=evaluation_time,
                ww_time=weightwatcher_time,
                device=device,
            )
        )
        spectral.append(checkpoint.metrics)
        details.append(checkpoint.details)
        esds.update(checkpoint.esd_arrays)
        groups.extend(
            optimizer_group_rows(
                optimizer,
                epoch=epoch,
                optimizer_label=config.optimizer_label,
            )
        )

    measure(0, None, 0.0, learning_rates)
    if progress:
        row = performance[-1]
        print(
            f"epoch=000 | {config.optimizer_label} | "
            f"lr={row['primary_lr']:.3e} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} | "
            f"test loss={row['test_loss']:.4f} acc={row['test_accuracy']:.4f}"
        )

    checkpoint_dir = (
        Path(output_dir) / "checkpoints"
        if output_dir and config.save_epoch_checkpoints
        else None
    )
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        learning_rates = set_scheduled_learning_rates(
            optimizer, config, epoch_index=epoch - 1
        )
        started = time.perf_counter()
        online = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        train_time = time.perf_counter() - started
        step += len(train_loader)
        measure(epoch, online, train_time, learning_rates)
        if checkpoint_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "data_generator_state": generator.get_state(),
                    "learning_rates": learning_rates,
                    "config": config.__dict__,
                },
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )
        if progress:
            row = performance[-1]
            print(
                f"epoch={epoch:03d} | {config.optimizer_label} | "
                f"lr={row['primary_lr']:.3e} | "
                f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} | "
                f"test loss={row['test_loss']:.4f} acc={row['test_accuracy']:.4f}"
            )

    performance_frame = pd.DataFrame(performance)
    spectral_frame = pd.concat(spectral, ignore_index=True)
    details_frame = pd.concat(details, ignore_index=True)
    groups_frame = pd.DataFrame(groups)
    combined = spectral_frame.merge(
        performance_frame,
        on=["run", "epoch", "global_step"],
        how="left",
        validate="many_to_one",
    )
    result = BaselineResult(
        config,
        performance_frame,
        spectral_frame,
        details_frame,
        groups_frame,
        combined,
        esds,
        model,
        optimizer,
    )
    if config.strict_metrics:
        validate_result(result)
    if output_dir:
        result.save(output_dir)
    return result

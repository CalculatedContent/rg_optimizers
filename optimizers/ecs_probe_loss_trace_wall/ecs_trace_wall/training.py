"""Matched baseline and TraceWall epoch training loops."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .optimizer import ECSProbeLossTraceWall
from .reporting import flatten_correction_record
from .runtime import WarmupCosineSchedule
from .sampler import RotatingSubsetSampler, materialize_probe_batches


def train_baseline_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: WarmupCosineSchedule,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    global_step: int,
    gradient_clip_norm: float,
) -> tuple[int, float, float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    current_lr = float(optimizer.param_groups[0]["lr"])
    for inputs, targets in loader:
        current_lr = schedule.apply(global_step)
        optimizer.zero_grad(set_to_none=True)
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_examples = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_examples
        total_correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total_examples += batch_examples
        global_step += 1
    return (
        global_step,
        float(total_loss / total_examples),
        float(total_correct / total_examples),
        current_lr,
    )


def train_trace_wall_epoch(
    model: nn.Module,
    wrapper: ECSProbeLossTraceWall,
    schedule: WarmupCosineSchedule,
    loader: DataLoader[Any],
    *,
    train_dataset: Dataset[Any],
    probe_sampler: RotatingSubsetSampler,
    device: torch.device,
    seed: int,
    epoch: int,
    gradient_clip_norm: float,
) -> tuple[float, float, float, list[dict[str, Any]], int, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    current_lr = float(wrapper.base_optimizer.param_groups[0]["lr"])
    correction_rows: list[dict[str, Any]] = []
    attempts = 0
    accepts = 0
    probe_example_count = (
        wrapper.config.probe_batch_size
        * wrapper.config.probe_batches_per_correction
    )

    for inputs, targets in loader:
        current_lr = schedule.apply(wrapper.global_step)
        wrapper.zero_grad(set_to_none=True)
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        draw_cycle_start: Optional[int] = None
        draw_cycle_end: Optional[int] = None
        probe_batches = None
        if wrapper.correction_due(wrapper.global_step + 1):
            draw = probe_sampler.take(probe_example_count)
            draw_cycle_start = draw.cycle_start
            draw_cycle_end = draw.cycle_end
            probe_batches = materialize_probe_batches(
                train_dataset,
                draw,
                batch_size=wrapper.config.probe_batch_size,
                device=device,
            )
        record = wrapper.step(
            probe_batches=probe_batches,
            loss_function=F.cross_entropy,
        )
        if record.attempted:
            attempts += 1
            accepts += int(record.applied)
            correction_rows.extend(
                flatten_correction_record(
                    record,
                    seed=seed,
                    epoch=epoch,
                    draw_cycle_start=draw_cycle_start,
                    draw_cycle_end=draw_cycle_end,
                )
            )

        batch_examples = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_examples
        total_correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total_examples += batch_examples

    return (
        float(total_loss / total_examples),
        float(total_correct / total_examples),
        current_lr,
        correction_rows,
        attempts,
        accepts,
    )

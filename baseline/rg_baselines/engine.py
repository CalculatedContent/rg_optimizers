"""Device, evaluation, and one-epoch training utilities."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .optimizers import set_scheduled_learning_rates


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
    ):
        torch.mps.manual_seed(seed)


def parameter_l2_norm(model: torch.nn.Module) -> float:
    squared = sum(
        float(torch.sum(parameter.detach().float() ** 2).cpu())
        for parameter in model.parameters()
    )
    return float(squared**0.5)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for index, (inputs, targets) in enumerate(loader, start=1):
        if max_batches is not None and index > int(max_batches):
            break
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss_sum += float(loss.item()) * targets.numel()
        correct += int((logits.argmax(1) == targets).sum())
        seen += targets.numel()
    model.train(was_training)
    return {
        "loss": loss_sum / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "examples": seen,
    }


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    *,
    config,
    device: torch.device,
    grad_clip_norm: float,
    global_step: int,
    total_steps: int,
    steps_per_epoch: int,
) -> dict[str, float | int]:
    """Train one epoch and update the LR before every optimizer step."""

    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    norms: list[float] = []
    last_lrs = {"primary": float("nan"), "auxiliary": float("nan")}
    completed_step = int(global_step)

    for inputs, targets in loader:
        last_lrs = set_scheduled_learning_rates(
            optimizer,
            config,
            update_index=completed_step,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
        )
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(grad_clip_norm)
        )
        optimizer.step()
        completed_step += 1

        batch_examples = targets.numel()
        loss_sum += float(loss.item()) * batch_examples
        correct += int((logits.argmax(1) == targets).sum())
        seen += batch_examples
        norms.append(float(torch.as_tensor(norm).detach().cpu()))

    array = np.asarray(norms, dtype=float)
    return {
        "online_train_loss": loss_sum / max(seen, 1),
        "online_train_accuracy": correct / max(seen, 1),
        "mean_gradient_norm_before_clip": float(array.mean()),
        "median_gradient_norm_before_clip": float(np.median(array)),
        "max_gradient_norm_before_clip": float(array.max()),
        "batches": len(norms),
        "global_step": completed_step,
        "primary_lr": float(last_lrs.get("primary", np.nan)),
        "auxiliary_lr": float(last_lrs.get("auxiliary", np.nan)),
    }


def performance_row(
    *,
    config,
    epoch: int,
    global_step: int,
    train_eval: dict,
    validation_eval: dict,
    test_eval: dict,
    online: Optional[dict],
    learning_rates: dict[str, float],
    parameter_norm: float,
    train_time: float,
    evaluation_time: float,
    ww_time: float,
    device: torch.device,
) -> dict:
    online_values = online or {}
    train_loss = float(train_eval["loss"])
    validation_loss = float(validation_eval["loss"])
    test_loss = float(test_eval["loss"])
    train_accuracy = float(train_eval["accuracy"])
    validation_accuracy = float(validation_eval["accuracy"])
    test_accuracy = float(test_eval["accuracy"])
    return {
        "run": config.optimizer_label,
        "optimizer": config.optimizer,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "primary_lr": float(learning_rates.get("primary", np.nan)),
        "auxiliary_lr": float(learning_rates.get("auxiliary", np.nan)),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "train_examples_evaluated": int(train_eval["examples"]),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "validation_examples_evaluated": int(validation_eval["examples"]),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_examples_evaluated": int(test_eval["examples"]),
        "validation_loss_gap": validation_loss - train_loss,
        "test_loss_gap": test_loss - train_loss,
        "validation_accuracy_gap": train_accuracy - validation_accuracy,
        "test_accuracy_gap": train_accuracy - test_accuracy,
        "test_monitoring_only": int(bool(config.test_monitoring_only)),
        "online_train_loss": float(
            online_values.get("online_train_loss", np.nan)
        ),
        "online_train_accuracy": float(
            online_values.get("online_train_accuracy", np.nan)
        ),
        "mean_gradient_norm_before_clip": float(
            online_values.get("mean_gradient_norm_before_clip", np.nan)
        ),
        "median_gradient_norm_before_clip": float(
            online_values.get("median_gradient_norm_before_clip", np.nan)
        ),
        "max_gradient_norm_before_clip": float(
            online_values.get("max_gradient_norm_before_clip", np.nan)
        ),
        "batches": int(online_values.get("batches", 0)),
        "parameter_l2_norm": float(parameter_norm),
        "train_time_sec": float(train_time),
        "evaluation_time_sec": float(evaluation_time),
        "weightwatcher_time_sec": float(ww_time),
        "epoch_total_time_sec": float(
            train_time + evaluation_time + ww_time
        ),
        "device": str(device),
        "seed": int(config.seed),
    }

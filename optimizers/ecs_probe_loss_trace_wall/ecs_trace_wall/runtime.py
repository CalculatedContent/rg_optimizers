"""Common model, optimizer, schedule, dataset, and evaluation utilities."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .config import BaseOptimizerConfig


class MLP3(nn.Module):
    """The repository-standard 784 -> 512 -> 512 -> 10 ReLU MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class WarmupCosineSchedule:
    """Per-step linear warmup followed by monotone cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        minimum_ratio: float,
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 < minimum_ratio <= 1.0:
            raise ValueError("minimum_ratio must lie in (0, 1]")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(min(warmup_steps, total_steps))
        self.minimum_ratio = float(minimum_ratio)
        self.peak_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = -1
        self.last_factor = float("nan")

    def factor(self, step: int) -> float:
        index = int(np.clip(step, 0, self.total_steps - 1))
        if self.warmup_steps > 0 and index < self.warmup_steps:
            return float((index + 1) / self.warmup_steps)
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 1:
            return self.minimum_ratio
        progress = (index - self.warmup_steps) / (decay_steps - 1)
        progress = float(np.clip(progress, 0.0, 1.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine)

    def apply(self, step: int) -> float:
        factor = self.factor(step)
        for group, peak in zip(self.optimizer.param_groups, self.peak_lrs):
            group["lr"] = peak * factor
        self.last_step = int(step)
        self.last_factor = factor
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "minimum_ratio": self.minimum_ratio,
            "peak_lrs": list(self.peak_lrs),
            "last_step": self.last_step,
            "last_factor": self.last_factor,
        }


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


def state_dict_checksum(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_base_optimizer(
    model: nn.Module,
    config: BaseOptimizerConfig,
) -> torch.optim.Optimizer:
    config.validate()
    if config.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.peak_learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
        )
    if config.name == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.peak_learning_rate,
            momentum=config.momentum,
            dampening=config.dampening,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unknown optimizer {config.name!r}")


def load_mnist(data_dir: str | Path) -> tuple[Dataset[Any], Dataset[Any]]:
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train = datasets.MNIST(
        str(data_dir), train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        str(data_dir), train=False, download=True, transform=transform
    )
    return train, test


def ordered_epoch_indices(dataset_size: int, *, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 1_000_003 + int(epoch) * 97_409)
    return torch.randperm(dataset_size, generator=generator).tolist()


def loader_for_indices(
    dataset: Dataset[Any],
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader[Any]:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def evaluation_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss_sum = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += float(loss_sum.detach().cpu())
        total_correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total_examples += int(targets.shape[0])
    model.train(previous_mode)
    if total_examples < 1:
        raise RuntimeError("evaluation loader produced no examples")
    mean_loss = total_loss / total_examples
    return {
        "loss": float(mean_loss),
        "accuracy": float(total_correct / total_examples),
        "perplexity": float(math.exp(min(mean_loss, 80.0))),
        "examples": float(total_examples),
    }


def parameter_l2_norm(model: nn.Module) -> float:
    squared = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            squared += float(torch.sum(parameter.detach() ** 2).cpu())
    return float(math.sqrt(max(squared, 0.0)))

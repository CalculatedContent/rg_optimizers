"""Paired MLP3-MNIST experiments for local-delta ECS WW-PGD."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import LocalDeltaECSConfig, MNISTRunConfig
from .optimizer import LocalDeltaECSOptimizer
from .weightwatcher import analyze_weightwatcher_or_fallback


class MLP3(nn.Module):
    """Standard MLP3 for MNIST: 784 -> 512 -> 512 -> 10 by default."""

    def __init__(self, hidden_width: int = 512) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class MNISTExperimentResult:
    performance: pd.DataFrame
    spectral: pd.DataFrame
    corrections: pd.DataFrame
    config: MNISTRunConfig

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.performance.to_csv(out / "performance.csv", index=False)
        self.spectral.to_csv(out / "spectral.csv", index=False)
        self.corrections.to_csv(out / "corrections.csv", index=False)
        (out / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_subset(dataset, limit: Optional[int], seed: int):
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[: int(limit)].tolist()
    return Subset(dataset, indices)


def make_loaders(config: MNISTRunConfig, seed: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)
    train_ds = _maybe_subset(train_ds, config.train_limit, seed)
    test_ds = _maybe_subset(test_ds, config.test_limit, seed + 1)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def make_base_optimizer(model: nn.Module, config: MNISTRunConfig) -> torch.optim.Optimizer:
    if config.optimizer_kind == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.adamw_lr,
            weight_decay=config.adamw_weight_decay,
        )
    if config.optimizer_kind == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.sgd_lr,
            momentum=config.sgd_momentum,
            weight_decay=config.sgd_weight_decay,
            nesterov=False,
        )
    raise ValueError(f"unknown optimizer_kind: {config.optimizer_kind!r}")


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | LocalDeltaECSOptimizer,
    loader: DataLoader,
    device: torch.device,
    *,
    grad_clip_norm: Optional[float],
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_grad_norm = 0.0
    num_steps = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        grad_norm = 0.0
        if grad_clip_norm is not None:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            grad_norm = float(grad_norm_tensor.detach().cpu())
        optimizer.step()
        batch_size = int(yb.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
        total_examples += batch_size
        total_grad_norm += grad_norm
        num_steps += 1
    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
        "grad_norm_mean": total_grad_norm / max(num_steps, 1),
        "steps": float(num_steps),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        batch_size = int(yb.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
        total_examples += batch_size
    loss = total_loss / max(total_examples, 1)
    return {
        "loss": loss,
        "acc": total_correct / max(total_examples, 1),
    }


def run_single_arm(
    *,
    config: MNISTRunConfig,
    seed: int,
    arm: str,
    device: torch.device,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    train_loader, test_loader = make_loaders(config, seed)
    model = MLP3(hidden_width=config.hidden_width).to(device)
    base_optimizer = make_base_optimizer(model, config)

    local_optimizer: torch.optim.Optimizer | LocalDeltaECSOptimizer
    if arm == "local_delta_ecs":
        local_optimizer = LocalDeltaECSOptimizer(
            base_optimizer,
            model.named_parameters(),
            config=LocalDeltaECSConfig(
                correction_fraction=config.correction_fraction,
                apply_every_epochs=config.apply_every_epochs,
                warmup_epochs=config.warmup_epochs,
                min_retained=3,
                normalization_gamma=config.normalization_gamma,
                reference="epoch_start",
            ),
        )
    elif arm == "baseline":
        local_optimizer = base_optimizer
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    run_label = f"{config.optimizer_kind}_{arm}_seed{seed}"
    performance_rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    spectral_frames: list[pd.DataFrame] = []

    train0 = evaluate(model, train_loader, device)
    test0 = evaluate(model, test_loader, device)
    performance_rows.append(
        {
            "run_label": run_label,
            "seed": seed,
            "optimizer_kind": config.optimizer_kind,
            "arm": arm,
            "epoch": 0,
            "train_loss": train0["loss"],
            "train_acc": train0["acc"],
            "test_loss": test0["loss"],
            "test_acc": test0["acc"],
            "grad_norm_mean": np.nan,
            "steps": 0.0,
        }
    )
    spectral_frames.append(
        analyze_weightwatcher_or_fallback(
            model,
            epoch=0,
            run_label=run_label,
            seed=seed,
            optimizer_kind=config.optimizer_kind,
            arm=arm,
            ww_enabled=config.ww_enabled,
            normalization_gamma=config.normalization_gamma,
        )
    )

    for epoch in range(1, config.epochs + 1):
        if isinstance(local_optimizer, LocalDeltaECSOptimizer):
            local_optimizer.begin_epoch()
        train_metrics = train_one_epoch(
            model,
            local_optimizer,
            train_loader,
            device,
            grad_clip_norm=config.grad_clip_norm,
        )
        if isinstance(local_optimizer, LocalDeltaECSOptimizer):
            stats = local_optimizer.apply_epoch_delta_correction(epoch=epoch - 1)
            for row in stats:
                row.update(
                    {
                        "run_label": run_label,
                        "seed": seed,
                        "optimizer_kind": config.optimizer_kind,
                        "arm": arm,
                    }
                )
            correction_rows.extend(stats)
        train_eval = evaluate(model, train_loader, device)
        test_eval = evaluate(model, test_loader, device)
        performance_rows.append(
            {
                "run_label": run_label,
                "seed": seed,
                "optimizer_kind": config.optimizer_kind,
                "arm": arm,
                "epoch": epoch,
                "train_loss": train_eval["loss"],
                "train_acc": train_eval["acc"],
                "test_loss": test_eval["loss"],
                "test_acc": test_eval["acc"],
                "grad_norm_mean": train_metrics["grad_norm_mean"],
                "steps": train_metrics["steps"],
            }
        )
        spectral_frames.append(
            analyze_weightwatcher_or_fallback(
                model,
                epoch=epoch,
                run_label=run_label,
                seed=seed,
                optimizer_kind=config.optimizer_kind,
                arm=arm,
                ww_enabled=config.ww_enabled,
                normalization_gamma=config.normalization_gamma,
            )
        )
        if progress:
            print(
                f"{run_label} epoch {epoch:02d}/{config.epochs}: "
                f"train_acc={train_eval['acc']:.4f} test_acc={test_eval['acc']:.4f} "
                f"test_loss={test_eval['loss']:.4f}"
            )

    return (
        pd.DataFrame(performance_rows),
        pd.concat(spectral_frames, ignore_index=True) if spectral_frames else pd.DataFrame(),
        pd.DataFrame(correction_rows),
    )


def run_mnist_comparison(
    config: MNISTRunConfig,
    *,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> MNISTExperimentResult:
    device = device or default_device()
    perf_frames: list[pd.DataFrame] = []
    spectral_frames: list[pd.DataFrame] = []
    correction_frames: list[pd.DataFrame] = []
    for seed in config.seeds:
        for arm in ("baseline", "local_delta_ecs"):
            perf, spectral, corrections = run_single_arm(
                config=config,
                seed=int(seed),
                arm=arm,
                device=device,
                progress=progress,
            )
            perf_frames.append(perf)
            spectral_frames.append(spectral)
            correction_frames.append(corrections)
    result = MNISTExperimentResult(
        performance=pd.concat(perf_frames, ignore_index=True),
        spectral=pd.concat(spectral_frames, ignore_index=True),
        corrections=pd.concat(correction_frames, ignore_index=True),
        config=config,
    )
    return result


def summarize_final_performance(performance: pd.DataFrame) -> pd.DataFrame:
    final_epoch = performance["epoch"].max()
    final = performance[performance["epoch"] == final_epoch]
    return (
        final.groupby(["optimizer_kind", "arm"])[["train_acc", "test_acc", "train_loss", "test_loss"]]
        .agg(["mean", "std"])
        .reset_index()
    )

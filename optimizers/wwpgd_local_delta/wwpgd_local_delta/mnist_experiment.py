"""Paired MLP3-MNIST experiments for local-delta ECS WW-PGD."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

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
        (out / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2), encoding="utf-8"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_subset(dataset, limit: Optional[int], seed: int):
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[: int(limit)].tolist()
    return Subset(dataset, indices)


def make_loaders(
    config: MNISTRunConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build matched train, nonshuffled train-eval, and test loaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(
        config.data_dir, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        config.data_dir, train=False, download=True, transform=transform
    )
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
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, train_eval_loader, test_loader


def make_base_optimizer(
    model: nn.Module, config: MNISTRunConfig
) -> torch.optim.Optimizer:
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


def _clone_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def initial_state_for_seed(
    seed: int, hidden_width: int
) -> dict[str, torch.Tensor]:
    set_seed(seed)
    return _clone_state_dict(MLP3(hidden_width=hidden_width).state_dict())


def state_checksum(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


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
        grad_norm = float("nan")
        if grad_clip_norm is not None:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm
            )
            grad_norm = float(grad_norm_tensor.detach().cpu())
        optimizer.step()
        batch_size = int(yb.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int(
            (logits.argmax(dim=1) == yb).sum().detach().cpu()
        )
        total_examples += batch_size
        if np.isfinite(grad_norm):
            total_grad_norm += grad_norm
        num_steps += 1
    return {
        "online_loss": total_loss / max(total_examples, 1),
        "online_acc": total_correct / max(total_examples, 1),
        "grad_norm_mean": total_grad_norm / max(num_steps, 1),
        "steps": float(num_steps),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
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
        total_correct += int(
            (logits.argmax(dim=1) == yb).sum().detach().cpu()
        )
        total_examples += batch_size
    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
    }


def _spectral_checkpoint(
    model: nn.Module,
    *,
    config: MNISTRunConfig,
    epoch: int,
    run_label: str,
    seed: int,
    arm: str,
) -> pd.DataFrame:
    return analyze_weightwatcher_or_fallback(
        model,
        epoch=epoch,
        run_label=run_label,
        seed=seed,
        optimizer_kind=config.optimizer_kind,
        arm=arm,
        ww_enabled=config.ww_enabled,
        ww_required=config.ww_required,
        ww_min_evals=config.ww_min_evals,
        ww_svd_method=config.ww_svd_method,
        normalization_gamma=config.normalization_gamma,
    )


def run_single_arm(
    *,
    config: MNISTRunConfig,
    seed: int,
    arm: str,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    train_loader, train_eval_loader, test_loader = make_loaders(config, seed)
    model = MLP3(hidden_width=config.hidden_width)
    model.load_state_dict(initial_state, strict=True)
    model = model.to(device)
    checksum = state_checksum(initial_state)
    base_optimizer = make_base_optimizer(model, config)

    optimizer: torch.optim.Optimizer | LocalDeltaECSOptimizer
    if arm == "local_delta_ecs":
        optimizer = LocalDeltaECSOptimizer(
            base_optimizer,
            model.named_parameters(),
            config=LocalDeltaECSConfig(
                correction_fraction=config.correction_fraction,
                apply_every_epochs=config.apply_every_epochs,
                warmup_epochs=config.warmup_epochs,
                min_retained=3,
                normalization_gamma=config.normalization_gamma,
                reference=config.ecs_reference,
                parameter_name_filter=config.corrected_parameters,
            ),
        )
    elif arm == "baseline":
        optimizer = base_optimizer
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    run_label = f"{config.optimizer_kind}_{arm}_seed{seed}"
    performance_rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    spectral_frames: list[pd.DataFrame] = []

    train0 = evaluate(model, train_eval_loader, device)
    test0 = evaluate(model, test_loader, device)
    performance_rows.append(
        {
            "run_label": run_label,
            "seed": seed,
            "optimizer_kind": config.optimizer_kind,
            "arm": arm,
            "epoch": 0,
            "initial_state_checksum": checksum,
            "online_train_loss": np.nan,
            "online_train_acc": np.nan,
            "pre_correction_train_loss": train0["loss"],
            "pre_correction_train_acc": train0["acc"],
            "pre_correction_test_loss": test0["loss"],
            "pre_correction_test_acc": test0["acc"],
            "train_loss": train0["loss"],
            "train_acc": train0["acc"],
            "test_loss": test0["loss"],
            "test_acc": test0["acc"],
            "correction_train_loss_delta": 0.0,
            "correction_train_acc_delta": 0.0,
            "correction_test_loss_delta": 0.0,
            "correction_test_acc_delta": 0.0,
            "grad_norm_mean": np.nan,
            "steps": 0.0,
        }
    )
    spectral_frames.append(
        _spectral_checkpoint(
            model,
            config=config,
            epoch=0,
            run_label=run_label,
            seed=seed,
            arm=arm,
        )
    )

    for epoch in range(1, config.epochs + 1):
        if isinstance(optimizer, LocalDeltaECSOptimizer):
            optimizer.begin_epoch()
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            grad_clip_norm=config.grad_clip_norm,
        )

        train_pre = evaluate(model, train_eval_loader, device)
        test_pre = evaluate(model, test_loader, device)

        if isinstance(optimizer, LocalDeltaECSOptimizer):
            stats = optimizer.apply_epoch_delta_correction(epoch=epoch - 1)
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
            train_post = evaluate(model, train_eval_loader, device)
            test_post = evaluate(model, test_loader, device)
        else:
            train_post = train_pre
            test_post = test_pre

        performance_rows.append(
            {
                "run_label": run_label,
                "seed": seed,
                "optimizer_kind": config.optimizer_kind,
                "arm": arm,
                "epoch": epoch,
                "initial_state_checksum": checksum,
                "online_train_loss": train_metrics["online_loss"],
                "online_train_acc": train_metrics["online_acc"],
                "pre_correction_train_loss": train_pre["loss"],
                "pre_correction_train_acc": train_pre["acc"],
                "pre_correction_test_loss": test_pre["loss"],
                "pre_correction_test_acc": test_pre["acc"],
                "train_loss": train_post["loss"],
                "train_acc": train_post["acc"],
                "test_loss": test_post["loss"],
                "test_acc": test_post["acc"],
                "correction_train_loss_delta": train_post["loss"]
                - train_pre["loss"],
                "correction_train_acc_delta": train_post["acc"]
                - train_pre["acc"],
                "correction_test_loss_delta": test_post["loss"]
                - test_pre["loss"],
                "correction_test_acc_delta": test_post["acc"]
                - test_pre["acc"],
                "grad_norm_mean": train_metrics["grad_norm_mean"],
                "steps": train_metrics["steps"],
            }
        )
        spectral_frames.append(
            _spectral_checkpoint(
                model,
                config=config,
                epoch=epoch,
                run_label=run_label,
                seed=seed,
                arm=arm,
            )
        )
        if progress:
            print(
                f"{run_label} epoch {epoch:02d}/{config.epochs}: "
                f"train_acc={train_post['acc']:.4f} "
                f"test_acc={test_post['acc']:.4f} "
                f"test_loss={test_post['loss']:.4f}"
            )

    return (
        pd.DataFrame(performance_rows),
        pd.concat(spectral_frames, ignore_index=True)
        if spectral_frames
        else pd.DataFrame(),
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
        initial_state = initial_state_for_seed(int(seed), config.hidden_width)
        for arm in ("baseline", "local_delta_ecs"):
            perf, spectral, corrections = run_single_arm(
                config=config,
                seed=int(seed),
                arm=arm,
                device=device,
                initial_state=initial_state,
                progress=progress,
            )
            perf_frames.append(perf)
            spectral_frames.append(spectral)
            if not corrections.empty:
                correction_frames.append(corrections)
    return MNISTExperimentResult(
        performance=pd.concat(perf_frames, ignore_index=True),
        spectral=pd.concat(spectral_frames, ignore_index=True),
        corrections=(
            pd.concat(correction_frames, ignore_index=True)
            if correction_frames
            else pd.DataFrame()
        ),
        config=config,
    )


def summarize_final_performance(performance: pd.DataFrame) -> pd.DataFrame:
    final_epoch = performance["epoch"].max()
    final = performance[performance["epoch"] == final_epoch]
    return (
        final.groupby(["optimizer_kind", "arm"])[
            ["train_acc", "test_acc", "train_loss", "test_loss"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

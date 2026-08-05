"""Paired AdamW baseline versus trace-log RG experiment on MNIST."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .weightwatcher import analyze_weightwatcher_checkpoint
from .wrapper import TraceLogConfig, TraceLogRGWrapper


@dataclass(frozen=True)
class MNISTExperimentConfig:
    seed: int = 1337
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    rg_mode: str = "one_sided"
    rg_normalization: str = "weightwatcher"
    rg_gamma: float = 0.10
    rg_ridge_relative: float = 1e-6
    rg_min_retained: int = 5
    rg_correction_scale: float = 1.0
    rg_max_correction_ratio: Optional[float] = 0.10
    rg_apply_every_steps: int = 25
    rg_warmup_steps: int = 0
    ww_min_evals: int = 10
    ww_max_evals: Optional[int] = None
    n_log_shells: int = 5
    min_retained_for_beta: int = 20
    min_decades_for_beta: float = 0.50
    train_eval_max_batches: Optional[int] = 50


@dataclass
class MNISTExperimentResult:
    performance: pd.DataFrame
    weightwatcher: pd.DataFrame
    rg_steps: pd.DataFrame
    correction_summary: pd.DataFrame
    baseline_model: nn.Module
    rg_model: nn.Module
    rg_optimizer: TraceLogRGWrapper

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.performance.to_csv(output / "performance_history.csv", index=False)
        self.weightwatcher.to_csv(output / "weightwatcher_rg_history.csv", index=False)
        self.rg_steps.to_csv(output / "rg_step_history.csv", index=False)
        self.correction_summary.to_csv(output / "rg_correction_summary.csv", index=False)


class MLP3(nn.Module):
    """784 -> 512 -> 512 -> 10 ReLU MLP used in the RG draft tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_index, (x, y) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > int(max_batches):
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * y.numel()
        total_correct += int((logits.argmax(1) == y).sum().item())
        total_seen += int(y.numel())
    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
    }


def _measure(
    model: nn.Module,
    *,
    run_label: str,
    epoch: int,
    global_step: int,
    config: MNISTExperimentConfig,
):
    return analyze_weightwatcher_checkpoint(
        model,
        run_label=run_label,
        epoch=epoch,
        global_step=global_step,
        min_evals=config.ww_min_evals,
        max_evals=config.ww_max_evals,
        n_shells=config.n_log_shells,
        min_count_per_shell=1,
        min_retained_for_reliable=config.min_retained_for_beta,
        min_shells_for_reliable=3,
        min_decades_for_reliable=config.min_decades_for_beta,
    )


def _train_pair_one_epoch(
    baseline_model: nn.Module,
    rg_model: nn.Module,
    baseline_optimizer: torch.optim.Optimizer,
    rg_optimizer: TraceLogRGWrapper,
    train_loader: DataLoader,
    *,
    epoch: int,
    device: torch.device,
    grad_clip_norm: float,
) -> pd.DataFrame:
    baseline_model.train()
    rg_model.train()
    rows: list[dict[str, Any]] = []

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        baseline_optimizer.zero_grad(set_to_none=True)
        baseline_loss = F.cross_entropy(baseline_model(x), y)
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), grad_clip_norm)
        baseline_optimizer.step()

        rg_optimizer.zero_grad(set_to_none=True)
        rg_loss = F.cross_entropy(rg_model(x), y)
        rg_loss.backward()
        torch.nn.utils.clip_grad_norm_(rg_model.parameters(), grad_clip_norm)
        rg_optimizer.step()

        for row in rg_optimizer.pop_step_stats():
            record = dict(row)
            record["epoch"] = int(epoch)
            rows.append(record)

    return pd.DataFrame(rows)


def _summarize_corrections(rg_steps: pd.DataFrame) -> pd.DataFrame:
    if rg_steps.empty or "status" not in rg_steps.columns:
        return pd.DataFrame()
    applied = rg_steps[rg_steps["status"] == "ok"]
    if applied.empty:
        return pd.DataFrame()
    return applied.groupby(["epoch", "parameter"], as_index=False).agg(
        corrections=("correction_ratio", "size"),
        mean_correction_ratio=("correction_ratio", "mean"),
        max_correction_ratio=("correction_ratio", "max"),
        mean_base_drift=("base_trace_log_drift", "mean"),
        mean_corrected_drift=("corrected_trace_log_drift", "mean"),
        cap_fraction=("correction_capped", "mean"),
    )


def run_mnist_comparison(
    config: MNISTExperimentConfig = MNISTExperimentConfig(),
    *,
    data_dir: str | Path = "./data",
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> MNISTExperimentResult:
    """Train paired AdamW and AdamW+RG models on identical MNIST batches."""
    set_seed(config.seed)
    device = device or choose_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(
        str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        str(data_dir), train=False, download=True, transform=transform
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    initial_state = copy.deepcopy(MLP3().state_dict())
    baseline_model = MLP3().to(device)
    baseline_model.load_state_dict(initial_state)
    rg_model = MLP3().to(device)
    rg_model.load_state_dict(initial_state)

    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rg_base_optimizer = torch.optim.AdamW(
        rg_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rg_optimizer = TraceLogRGWrapper(
        rg_base_optimizer,
        rg_model.named_parameters(),
        config=TraceLogConfig(
            mode=config.rg_mode,
            gamma=config.rg_gamma,
            normalization=config.rg_normalization,
            ridge_relative=config.rg_ridge_relative,
            min_retained=config.rg_min_retained,
            correction_scale=config.rg_correction_scale,
            max_correction_ratio=config.rg_max_correction_ratio,
            apply_every_steps=config.rg_apply_every_steps,
            warmup_steps=config.rg_warmup_steps,
        ),
    )

    initial_test = evaluate(rg_model, test_loader, device=device)
    initial_rg = _measure(
        rg_model,
        run_label="AdamW + TraceLogRG",
        epoch=0,
        global_step=0,
        config=config,
    )
    rg_optimizer.set_supports(initial_rg.supports)
    initial_baseline = initial_rg.metrics.copy()
    if not initial_baseline.empty:
        initial_baseline["run"] = "AdamW baseline"

    performance_rows = [
        {
            "epoch": 0,
            "run": "AdamW baseline",
            "train_loss": np.nan,
            "train_acc": np.nan,
            "test_loss": initial_test["loss"],
            "test_acc": initial_test["acc"],
        },
        {
            "epoch": 0,
            "run": "AdamW + TraceLogRG",
            "train_loss": np.nan,
            "train_acc": np.nan,
            "test_loss": initial_test["loss"],
            "test_acc": initial_test["acc"],
        },
    ]
    ww_frames = [initial_baseline, initial_rg.metrics.copy()]
    step_frames: list[pd.DataFrame] = []
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        epoch_steps = _train_pair_one_epoch(
            baseline_model,
            rg_model,
            baseline_optimizer,
            rg_optimizer,
            train_loader,
            epoch=epoch,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        if not epoch_steps.empty:
            step_frames.append(epoch_steps)
        global_step += len(train_loader)

        baseline_train = evaluate(
            baseline_model,
            train_eval_loader,
            device=device,
            max_batches=config.train_eval_max_batches,
        )
        baseline_test = evaluate(baseline_model, test_loader, device=device)
        rg_train = evaluate(
            rg_model,
            train_eval_loader,
            device=device,
            max_batches=config.train_eval_max_batches,
        )
        rg_test = evaluate(rg_model, test_loader, device=device)

        performance_rows.extend([
            {
                "epoch": epoch,
                "run": "AdamW baseline",
                "train_loss": baseline_train["loss"],
                "train_acc": baseline_train["acc"],
                "test_loss": baseline_test["loss"],
                "test_acc": baseline_test["acc"],
            },
            {
                "epoch": epoch,
                "run": "AdamW + TraceLogRG",
                "train_loss": rg_train["loss"],
                "train_acc": rg_train["acc"],
                "test_loss": rg_test["loss"],
                "test_acc": rg_test["acc"],
            },
        ])

        baseline_checkpoint = _measure(
            baseline_model,
            run_label="AdamW baseline",
            epoch=epoch,
            global_step=global_step,
            config=config,
        )
        rg_checkpoint = _measure(
            rg_model,
            run_label="AdamW + TraceLogRG",
            epoch=epoch,
            global_step=global_step,
            config=config,
        )
        ww_frames.extend([baseline_checkpoint.metrics, rg_checkpoint.metrics])
        rg_optimizer.set_supports(rg_checkpoint.supports)

        if progress:
            print(
                f"epoch={epoch:03d} | baseline test={baseline_test['acc']:.4f} "
                f"| RG test={rg_test['acc']:.4f} | supports={rg_optimizer.get_supports()}"
            )

    performance = pd.DataFrame(performance_rows)
    weightwatcher = pd.concat(ww_frames, ignore_index=True)
    rg_steps = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    correction_summary = _summarize_corrections(rg_steps)

    return MNISTExperimentResult(
        performance=performance,
        weightwatcher=weightwatcher,
        rg_steps=rg_steps,
        correction_summary=correction_summary,
        baseline_model=baseline_model,
        rg_model=rg_model,
        rg_optimizer=rg_optimizer,
    )

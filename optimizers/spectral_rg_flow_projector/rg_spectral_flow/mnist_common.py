"""Shared MNIST model, configuration, evaluation, and result types."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .wrapper import SpectralRGFlowProjector


@dataclass(frozen=True)
class MNISTExperimentConfig:
    seed: int = 1337
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    collapse_potential: str = "participation_ratio"
    projection_strength: float = 1.0
    min_alignment_cosine: float = 0.0
    max_abs_log_eigenvalue_correction: Optional[float] = 0.20
    max_correction_ratio: Optional[float] = 0.10
    preserve_frobenius_norm: bool = True
    apply_every_steps: int = 25
    warmup_epochs: int = 1
    min_retained: int = 20

    sc_effective_rank_method: str = "participation_ratio"
    sc_normalization_gamma: float = 0.0
    sc_support_policy: str = "midpoint"
    sc_min_ecs_size: int = 2

    ww_min_evals: int = 8
    ww_max_evals: Optional[int] = None
    ww_svd_method: str = "accurate"
    train_eval_max_batches: Optional[int] = 50


@dataclass
class MNISTExperimentResult:
    performance: pd.DataFrame
    weightwatcher: pd.DataFrame
    flow_steps: pd.DataFrame
    correction_summary: pd.DataFrame
    baseline_model: nn.Module
    flow_model: nn.Module
    flow_optimizer: SpectralRGFlowProjector

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.performance.to_csv(output / "performance_history.csv", index=False)
        self.weightwatcher.to_csv(
            output / "weightwatcher_self_consistent_history.csv",
            index=False,
        )
        self.flow_steps.to_csv(output / "spectral_flow_step_history.csv", index=False)
        self.correction_summary.to_csv(
            output / "spectral_flow_correction_summary.csv",
            index=False,
        )
        torch.save(
            {
                "baseline_model": self.baseline_model.state_dict(),
                "flow_model": self.flow_model.state_dict(),
                "flow_optimizer": self.flow_optimizer.state_dict(),
            },
            output / "final_states.pt",
        )


class MLP3(nn.Module):
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



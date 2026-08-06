"""Paired AdamW versus spectral RG-flow projection on MNIST."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .mnist_common import (
    MLP3,
    MNISTExperimentConfig,
    MNISTExperimentResult,
    choose_device,
    evaluate,
    set_seed,
)
from .weightwatcher import SpectralFlowCheckpoint, analyze_weightwatcher_checkpoint
from .wrapper import SpectralRGFlowConfig, SpectralRGFlowProjector

def _measure(
    model: nn.Module,
    *,
    run_label: str,
    epoch: int,
    global_step: int,
    config: MNISTExperimentConfig,
    previous_supports: Optional[dict[str, Any]] = None,
) -> SpectralFlowCheckpoint:
    return analyze_weightwatcher_checkpoint(
        model,
        run_label=run_label,
        epoch=epoch,
        global_step=global_step,
        min_evals=config.ww_min_evals,
        max_evals=config.ww_max_evals,
        svd_method=config.ww_svd_method,
        effective_rank_method=config.sc_effective_rank_method,  # type: ignore[arg-type]
        normalization_gamma=config.sc_normalization_gamma,
        support_policy=config.sc_support_policy,  # type: ignore[arg-type]
        min_ecs_size=config.sc_min_ecs_size,
        min_retained=config.min_retained,
        previous_supports=previous_supports,
    )


def _train_pair_one_epoch(
    baseline_model: nn.Module,
    flow_model: nn.Module,
    baseline_optimizer: torch.optim.Optimizer,
    flow_optimizer: SpectralRGFlowProjector,
    train_loader: DataLoader,
    *,
    epoch: int,
    device: torch.device,
    grad_clip_norm: float,
) -> pd.DataFrame:
    baseline_model.train()
    flow_model.train()
    rows: list[dict[str, Any]] = []

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        baseline_optimizer.zero_grad(set_to_none=True)
        baseline_loss = F.cross_entropy(baseline_model(x), y)
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), grad_clip_norm)
        baseline_optimizer.step()

        flow_optimizer.zero_grad(set_to_none=True)
        flow_loss = F.cross_entropy(flow_model(x), y)
        flow_loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), grad_clip_norm)
        flow_optimizer.step()

        for row in flow_optimizer.pop_step_stats():
            record = dict(row)
            record["epoch"] = int(epoch)
            rows.append(record)

    return pd.DataFrame(rows)


def _summarize(flow_steps: pd.DataFrame) -> pd.DataFrame:
    if flow_steps.empty or "status" not in flow_steps.columns:
        return pd.DataFrame()
    valid = flow_steps[flow_steps["status"].isin(["ok", "skipped"])]
    if valid.empty:
        return pd.DataFrame()
    return valid.groupby(["epoch", "parameter"], as_index=False).agg(
        evaluations=("status", "size"),
        corrections=("status", lambda values: int(np.sum(values == "ok"))),
        correction_fraction=("status", lambda values: float(np.mean(values == "ok"))),
        mean_correction_ratio=("correction_ratio", "mean"),
        max_correction_ratio=("correction_ratio", "max"),
        mean_base_flow_component=("base_flow_component", "mean"),
        mean_corrected_flow_component=("corrected_flow_component", "mean"),
        mean_effective_rank_before=("effective_rank_before", "mean"),
        mean_effective_rank_base=("effective_rank_base", "mean"),
        mean_effective_rank_corrected=("effective_rank_corrected", "mean"),
        mean_rank_alpha_proxy_before=("rank_alpha_proxy_before", "mean"),
        mean_rank_alpha_proxy_base=("rank_alpha_proxy_base", "mean"),
        mean_rank_alpha_proxy_corrected=("rank_alpha_proxy_corrected", "mean"),
        cap_fraction=("correction_capped", "mean"),
    )


def run_mnist_comparison(
    config: MNISTExperimentConfig = MNISTExperimentConfig(),
    *,
    data_dir: str | Path = "./data",
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> MNISTExperimentResult:
    """Train matched AdamW and AdamW+spectral-flow-projector models."""
    set_seed(config.seed)
    device = device or choose_device()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
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
    flow_model = MLP3().to(device)
    flow_model.load_state_dict(initial_state)

    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    flow_base = torch.optim.AdamW(
        flow_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup_steps = int(config.warmup_epochs) * len(train_loader)
    flow_optimizer = SpectralRGFlowProjector(
        flow_base,
        flow_model.named_parameters(),
        config=SpectralRGFlowConfig(
            collapse_potential=config.collapse_potential,  # type: ignore[arg-type]
            projection_strength=config.projection_strength,
            min_alignment_cosine=config.min_alignment_cosine,
            max_abs_log_eigenvalue_correction=(
                config.max_abs_log_eigenvalue_correction
            ),
            max_correction_ratio=config.max_correction_ratio,
            preserve_frobenius_norm=config.preserve_frobenius_norm,
            min_retained=config.min_retained,
            apply_every_steps=config.apply_every_steps,
            warmup_steps=warmup_steps,
        ),
    )

    initial_test = evaluate(flow_model, test_loader, device=device)
    initial_flow = _measure(
        flow_model,
        run_label="AdamW + SpectralRGFlow",
        epoch=0,
        global_step=0,
        config=config,
    )
    flow_optimizer.set_support_states(initial_flow.supports, replace=True)
    initial_baseline = initial_flow.metrics.copy()
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
            "run": "AdamW + SpectralRGFlow",
            "train_loss": np.nan,
            "train_acc": np.nan,
            "test_loss": initial_test["loss"],
            "test_acc": initial_test["acc"],
        },
    ]
    ww_frames = [initial_baseline, initial_flow.metrics.copy()]
    step_frames: list[pd.DataFrame] = []
    baseline_supports = initial_flow.supports
    flow_supports = initial_flow.supports
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        steps = _train_pair_one_epoch(
            baseline_model,
            flow_model,
            baseline_optimizer,
            flow_optimizer,
            train_loader,
            epoch=epoch,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        if not steps.empty:
            step_frames.append(steps)
        global_step += len(train_loader)

        baseline_train = evaluate(
            baseline_model,
            train_eval_loader,
            device=device,
            max_batches=config.train_eval_max_batches,
        )
        baseline_test = evaluate(baseline_model, test_loader, device=device)
        flow_train = evaluate(
            flow_model,
            train_eval_loader,
            device=device,
            max_batches=config.train_eval_max_batches,
        )
        flow_test = evaluate(flow_model, test_loader, device=device)

        performance_rows.extend(
            [
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
                    "run": "AdamW + SpectralRGFlow",
                    "train_loss": flow_train["loss"],
                    "train_acc": flow_train["acc"],
                    "test_loss": flow_test["loss"],
                    "test_acc": flow_test["acc"],
                },
            ]
        )

        baseline_checkpoint = _measure(
            baseline_model,
            run_label="AdamW baseline",
            epoch=epoch,
            global_step=global_step,
            config=config,
            previous_supports=baseline_supports,
        )
        flow_checkpoint = _measure(
            flow_model,
            run_label="AdamW + SpectralRGFlow",
            epoch=epoch,
            global_step=global_step,
            config=config,
            previous_supports=flow_supports,
        )
        baseline_supports = baseline_checkpoint.supports
        flow_supports = flow_checkpoint.supports
        ww_frames.extend([baseline_checkpoint.metrics, flow_checkpoint.metrics])
        flow_optimizer.set_support_states(flow_supports, replace=True)

        if progress:
            compact = {
                name: state.working_rank
                for name, state in flow_optimizer.get_support_states().items()
            }
            print(
                f"epoch={epoch:03d} | AdamW test={baseline_test['acc']:.4f} "
                f"| flow test={flow_test['acc']:.4f} | supports={compact}"
            )

    performance = pd.DataFrame(performance_rows)
    weightwatcher = pd.concat(ww_frames, ignore_index=True)
    flow_steps = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    summary = _summarize(flow_steps)
    return MNISTExperimentResult(
        performance=performance,
        weightwatcher=weightwatcher,
        flow_steps=flow_steps,
        correction_summary=summary,
        baseline_model=baseline_model,
        flow_model=flow_model,
        flow_optimizer=flow_optimizer,
    )

"""Matched optimizer-versus-projector experiments on MNIST.

Each experiment starts two MLP3 models from the same state and trains them on
the same minibatches.  The first model uses the selected base optimizer.  The
second uses the identical base optimizer wrapped by :class:`SpectralRGFlowProjector`.

The supported base optimizers are AdamW, Adam, and ordinary SGD with classical
momentum.  In particular, ``sgd_momentum`` does not use Muon, polar
orthogonalization, or Newton--Schulz iterations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
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
from .mnist_experiment import _measure, _summarize, _train_pair_one_epoch
from .wrapper import SpectralRGFlowConfig, SpectralRGFlowProjector


OptimizerFamily = Literal["adamw", "adam", "sgd_momentum"]


@dataclass(frozen=True)
class OptimizerBenchmarkConfig(MNISTExperimentConfig):
    """Configuration for a matched base-versus-projector experiment."""

    optimizer_family: OptimizerFamily = "adamw"

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_amsgrad: bool = False

    sgd_momentum: float = 0.9
    sgd_dampening: float = 0.0
    sgd_nesterov: bool = False

    def validate(self) -> None:
        if self.optimizer_family not in {"adamw", "adam", "sgd_momentum"}:
            raise ValueError(f"Unknown optimizer_family: {self.optimizer_family!r}")
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive.")
        if not 0.0 <= self.adam_beta1 < 1.0:
            raise ValueError("adam_beta1 must lie in [0, 1).")
        if not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must lie in [0, 1).")
        if self.adam_eps <= 0.0:
            raise ValueError("adam_eps must be positive.")
        if not 0.0 <= self.sgd_momentum < 1.0:
            raise ValueError("sgd_momentum must lie in [0, 1).")
        if self.sgd_dampening < 0.0:
            raise ValueError("sgd_dampening must be non-negative.")
        if self.sgd_nesterov and (
            self.sgd_momentum <= 0.0 or self.sgd_dampening != 0.0
        ):
            raise ValueError(
                "Nesterov SGD requires positive momentum and zero dampening."
            )
        if self.apply_every_steps < 1:
            raise ValueError("apply_every_steps must be positive.")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative.")

    @property
    def optimizer_label(self) -> str:
        return {
            "adamw": "AdamW",
            "adam": "Adam",
            "sgd_momentum": "SGD + momentum",
        }[self.optimizer_family]

    @property
    def flow_label(self) -> str:
        return f"{self.optimizer_label} + SpectralRGFlow"

    @property
    def run_slug(self) -> str:
        return {
            "adamw": "adamw",
            "adam": "adam",
            "sgd_momentum": "sgd_momentum",
        }[self.optimizer_family]


def build_base_optimizer(
    model: torch.nn.Module,
    config: OptimizerBenchmarkConfig,
) -> torch.optim.Optimizer:
    """Build the exact base optimizer used by both arms of the experiment."""
    config.validate()

    if config.optimizer_family == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            amsgrad=config.adam_amsgrad,
        )

    if config.optimizer_family == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            amsgrad=config.adam_amsgrad,
        )

    # This is ordinary torch.optim.SGD with a momentum buffer.  There is no
    # Muon update and no matrix orthogonalization in this branch.
    return torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.sgd_momentum,
        dampening=config.sgd_dampening,
        weight_decay=config.weight_decay,
        nesterov=config.sgd_nesterov,
    )


def _tag_frame(
    frame: pd.DataFrame,
    *,
    config: OptimizerBenchmarkConfig,
    flow_enabled: bool,
) -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty:
        return frame
    frame["optimizer_family"] = config.optimizer_family
    frame["optimizer_label"] = config.optimizer_label
    frame["flow_enabled"] = bool(flow_enabled)
    return frame


def run_optimizer_benchmark(
    config: OptimizerBenchmarkConfig,
    *,
    data_dir: str | Path = "./data",
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> MNISTExperimentResult:
    """Train a matched base optimizer and base-plus-spectral-flow pair.

    Both models receive the same initialization and the same minibatches.  The
    projector is the only difference between the two trajectories.
    """
    config.validate()
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

    baseline_optimizer = build_base_optimizer(baseline_model, config)
    flow_base_optimizer = build_base_optimizer(flow_model, config)
    flow_optimizer = SpectralRGFlowProjector(
        flow_base_optimizer,
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
            warmup_steps=int(config.warmup_epochs) * len(train_loader),
        ),
    )

    initial_test = evaluate(flow_model, test_loader, device=device)
    initial_checkpoint = _measure(
        flow_model,
        run_label=config.flow_label,
        epoch=0,
        global_step=0,
        config=config,
    )
    flow_optimizer.set_support_states(initial_checkpoint.supports, replace=True)

    initial_flow_metrics = _tag_frame(
        initial_checkpoint.metrics,
        config=config,
        flow_enabled=True,
    )
    initial_baseline_metrics = _tag_frame(
        initial_checkpoint.metrics,
        config=config,
        flow_enabled=False,
    )
    if not initial_baseline_metrics.empty:
        initial_baseline_metrics["run"] = config.optimizer_label

    performance_rows: list[dict[str, object]] = [
        {
            "epoch": 0,
            "run": config.optimizer_label,
            "optimizer_family": config.optimizer_family,
            "optimizer_label": config.optimizer_label,
            "flow_enabled": False,
            "train_loss": np.nan,
            "train_acc": np.nan,
            "test_loss": initial_test["loss"],
            "test_acc": initial_test["acc"],
        },
        {
            "epoch": 0,
            "run": config.flow_label,
            "optimizer_family": config.optimizer_family,
            "optimizer_label": config.optimizer_label,
            "flow_enabled": True,
            "train_loss": np.nan,
            "train_acc": np.nan,
            "test_loss": initial_test["loss"],
            "test_acc": initial_test["acc"],
        },
    ]
    ww_frames = [initial_baseline_metrics, initial_flow_metrics]
    step_frames: list[pd.DataFrame] = []
    baseline_supports = initial_checkpoint.supports
    flow_supports = initial_checkpoint.supports
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        step_frame = _train_pair_one_epoch(
            baseline_model,
            flow_model,
            baseline_optimizer,
            flow_optimizer,
            train_loader,
            epoch=epoch,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        if not step_frame.empty:
            step_frames.append(
                _tag_frame(step_frame, config=config, flow_enabled=True)
            )
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
                    "run": config.optimizer_label,
                    "optimizer_family": config.optimizer_family,
                    "optimizer_label": config.optimizer_label,
                    "flow_enabled": False,
                    "train_loss": baseline_train["loss"],
                    "train_acc": baseline_train["acc"],
                    "test_loss": baseline_test["loss"],
                    "test_acc": baseline_test["acc"],
                },
                {
                    "epoch": epoch,
                    "run": config.flow_label,
                    "optimizer_family": config.optimizer_family,
                    "optimizer_label": config.optimizer_label,
                    "flow_enabled": True,
                    "train_loss": flow_train["loss"],
                    "train_acc": flow_train["acc"],
                    "test_loss": flow_test["loss"],
                    "test_acc": flow_test["acc"],
                },
            ]
        )

        baseline_checkpoint = _measure(
            baseline_model,
            run_label=config.optimizer_label,
            epoch=epoch,
            global_step=global_step,
            config=config,
            previous_supports=baseline_supports,
        )
        flow_checkpoint = _measure(
            flow_model,
            run_label=config.flow_label,
            epoch=epoch,
            global_step=global_step,
            config=config,
            previous_supports=flow_supports,
        )
        baseline_supports = baseline_checkpoint.supports
        flow_supports = flow_checkpoint.supports
        ww_frames.extend(
            [
                _tag_frame(
                    baseline_checkpoint.metrics,
                    config=config,
                    flow_enabled=False,
                ),
                _tag_frame(
                    flow_checkpoint.metrics,
                    config=config,
                    flow_enabled=True,
                ),
            ]
        )
        flow_optimizer.set_support_states(flow_supports, replace=True)

        if progress:
            support_ranks = {
                name: state.working_rank
                for name, state in flow_optimizer.get_support_states().items()
            }
            print(
                f"epoch={epoch:03d} | {config.optimizer_label} "
                f"test={baseline_test['acc']:.4f} | flow "
                f"test={flow_test['acc']:.4f} | supports={support_ranks}"
            )

    performance = pd.DataFrame(performance_rows)
    weightwatcher = pd.concat(ww_frames, ignore_index=True)
    flow_steps = (
        pd.concat(step_frames, ignore_index=True)
        if step_frames
        else pd.DataFrame()
    )
    correction_summary = _summarize(flow_steps)
    if not correction_summary.empty:
        correction_summary["optimizer_family"] = config.optimizer_family
        correction_summary["optimizer_label"] = config.optimizer_label

    return MNISTExperimentResult(
        performance=performance,
        weightwatcher=weightwatcher,
        flow_steps=flow_steps,
        correction_summary=correction_summary,
        baseline_model=baseline_model,
        flow_model=flow_model,
        flow_optimizer=flow_optimizer,
    )

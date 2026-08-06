"""Paired MNIST experiment for AdaptiveSpectralGuard."""

from __future__ import annotations

import copy
import random
import time
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

from .config import GuardConfig
from .optimizer import AdaptiveSpectralGuard
from .weightwatcher import analyze_weightwatcher_checkpoint


@dataclass(frozen=True)
class MNISTGuardExperimentConfig:
    seed: int = 1337
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0

    ww_min_evals: int = 10
    ww_max_evals: Optional[int] = None
    n_shells: int = 5
    min_beta_retained: int = 20
    min_beta_decades: float = 0.50

    train_eval_max_batches: Optional[int] = None


@dataclass
class MNISTGuardExperimentResult:
    performance: pd.DataFrame
    weightwatcher: pd.DataFrame
    guard_steps: pd.DataFrame
    controller: pd.DataFrame
    correction_summary: pd.DataFrame
    baseline_model: nn.Module
    guard_model: nn.Module
    guard_optimizer: AdaptiveSpectralGuard
    output_dir: Path

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.performance.to_csv(
            self.output_dir / "performance_history.csv",
            index=False,
        )
        self.weightwatcher.to_csv(
            self.output_dir / "weightwatcher_history.csv",
            index=False,
        )
        self.guard_steps.to_csv(
            self.output_dir / "guard_step_history.csv",
            index=False,
        )
        self.controller.to_csv(
            self.output_dir / "controller_history.csv",
            index=False,
        )
        self.correction_summary.to_csv(
            self.output_dir / "correction_summary.csv",
            index=False,
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
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
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
        if max_batches is not None and batch_index > max_batches:
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
    config: MNISTGuardExperimentConfig,
):
    return analyze_weightwatcher_checkpoint(
        model,
        run_label=run_label,
        epoch=epoch,
        global_step=global_step,
        min_evals=config.ww_min_evals,
        max_evals=config.ww_max_evals,
        n_shells=config.n_shells,
        min_beta_retained=config.min_beta_retained,
        min_beta_decades=config.min_beta_decades,
    )


def _train_pair_one_epoch(
    baseline_model: nn.Module,
    guard_model: nn.Module,
    baseline_optimizer: torch.optim.Optimizer,
    guard_optimizer: AdaptiveSpectralGuard,
    loader: DataLoader,
    *,
    epoch: int,
    device: torch.device,
    grad_clip_norm: float,
) -> pd.DataFrame:
    baseline_model.train()
    guard_model.train()
    rows: list[dict[str, Any]] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        baseline_optimizer.zero_grad(set_to_none=True)
        baseline_loss = F.cross_entropy(baseline_model(x), y)
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            baseline_model.parameters(),
            grad_clip_norm,
        )
        baseline_optimizer.step()

        guard_optimizer.zero_grad(set_to_none=True)
        guard_loss = F.cross_entropy(guard_model(x), y)
        guard_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            guard_model.parameters(),
            grad_clip_norm,
        )
        guard_optimizer.step()

        for row in guard_optimizer.pop_step_stats():
            record = dict(row)
            record["epoch"] = int(epoch)
            rows.append(record)

    return pd.DataFrame(rows)


def _correction_summary(
    steps: pd.DataFrame,
    controller: pd.DataFrame,
    epoch: int,
    expected_batches: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    controller_by_parameter = (
        controller.set_index("parameter").to_dict("index")
        if controller is not None and not controller.empty
        else {}
    )
    parameters = sorted(
        set(controller_by_parameter)
        | (
            set(steps["parameter"].astype(str))
            if steps is not None and not steps.empty
            else set()
        )
    )

    for parameter in parameters:
        state = controller_by_parameter.get(parameter, {})
        policy_cadence = int(state.get("policy_cadence", 1))
        active_during_epoch = bool(
            state.get("regime", "off") != "off"
            and float(state.get("effective_gain", 0.0)) > 0.0
        )
        expected_due = (
            int(np.ceil(expected_batches / max(policy_cadence, 1)))
            if active_during_epoch
            else 0
        )
        group = (
            steps.loc[steps["parameter"].astype(str).eq(parameter)].copy()
            if steps is not None and not steps.empty
            else pd.DataFrame()
        )
        ok = (
            group.loc[group["status"].eq("ok")]
            if not group.empty
            else pd.DataFrame()
        )

        def mean(column: str) -> float:
            if group.empty or column not in group:
                return np.nan
            return float(
                pd.to_numeric(group[column], errors="coerce").mean()
            )

        def max_value(column: str) -> float:
            if group.empty or column not in group:
                return np.nan
            return float(
                pd.to_numeric(group[column], errors="coerce").max()
            )

        rows.append(
            {
                "epoch": int(epoch),
                "parameter": parameter,
                "regime": state.get("regime", "off"),
                "confidence": state.get("confidence", np.nan),
                "task_throttle": state.get("task_throttle", np.nan),
                "effective_gain": state.get("effective_gain", 0.0),
                "policy_cadence": policy_cadence,
                "expected_due_checks": expected_due,
                "recorded_due_checks": int(len(group)),
                "due_coverage": (
                    len(group) / expected_due if expected_due > 0 else np.nan
                ),
                "corrections_applied": int(len(ok)),
                "applied_fraction": (
                    len(ok) / len(group) if len(group) else 0.0
                ),
                "mean_combined_correction_ratio": mean(
                    "combined_correction_ratio"
                ),
                "max_combined_correction_ratio": max_value(
                    "combined_correction_ratio"
                ),
                "mean_volume_correction_ratio": mean(
                    "volume_correction_ratio"
                ),
                "mean_shape_correction_ratio": mean(
                    "shape_correction_ratio"
                ),
                "mean_task_conflict_ratio_pre": mean(
                    "task_conflict_ratio_pre"
                ),
                "mean_task_conflict_ratio_post": mean(
                    "task_conflict_ratio_post"
                ),
                "harmful_attempt_fraction": (
                    float(
                        (
                            pd.to_numeric(
                                group.get(
                                    "task_conflict_ratio_pre",
                                    pd.Series(dtype=float),
                                ),
                                errors="coerce",
                            )
                            > 0
                        ).mean()
                    )
                    if len(group)
                    else np.nan
                ),
                "mean_correction_base_cosine": mean(
                    "correction_base_cosine"
                ),
                "loss_neutral_fraction": (
                    float(
                        group.get(
                            "loss_neutral_applied",
                            pd.Series(dtype=bool),
                        )
                        .fillna(False)
                        .astype(bool)
                        .mean()
                    )
                    if len(group)
                    else np.nan
                ),
                "mean_loss_neutral_removed_fraction": mean(
                    "loss_neutral_removed_fraction"
                ),
                "mean_beta_E_local": mean("beta_E_local"),
                "geometry_failures": int(
                    group["status"].eq("geometry_failed").sum()
                )
                if len(group)
                else 0,
            }
        )
    return pd.DataFrame(rows)


def _controller_frame(
    optimizer: AdaptiveSpectralGuard,
    *,
    checkpoint_epoch: int,
    applies_to_epoch: int,
) -> pd.DataFrame:
    frame = optimizer.controller_frame()
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["checkpoint_epoch"] = int(checkpoint_epoch)
    frame["applies_to_epoch"] = int(applies_to_epoch)
    frame["epoch"] = int(applies_to_epoch)
    frame["policy_cadence"] = frame["parameter"].map(
        lambda name: optimizer.config.policy_for(name).cadence
    )
    frame["policy_volume_max_ratio"] = frame["parameter"].map(
        lambda name: optimizer.config.policy_for(name).volume_max_ratio
    )
    frame["policy_shape_max_ratio"] = frame["parameter"].map(
        lambda name: optimizer.config.policy_for(name).shape_max_ratio
    )
    frame["policy_combined_max_ratio"] = frame["parameter"].map(
        lambda name: optimizer.config.policy_for(name).combined_max_ratio
    )
    return frame


def _online_performance_monitor(
    performance: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for run, group in performance.groupby("run"):
        group = group.sort_values("epoch")
        best_loss = np.inf
        best_loss_epoch = np.nan
        peak_acc = -np.inf
        peak_acc_epoch = np.nan
        for _, row in group.iterrows():
            if row["epoch"] == 0 or pd.isna(row["train_loss"]):
                continue
            if float(row["test_loss"]) < best_loss:
                best_loss = float(row["test_loss"])
                best_loss_epoch = int(row["epoch"])
            if float(row["test_acc"]) > peak_acc:
                peak_acc = float(row["test_acc"])
                peak_acc_epoch = int(row["epoch"])
            rows.append(
                {
                    **row.to_dict(),
                    "best_test_loss_so_far": best_loss,
                    "best_test_loss_epoch": best_loss_epoch,
                    "test_loss_rebound": (
                        float(row["test_loss"]) - best_loss
                    ),
                    "peak_test_acc_so_far": peak_acc,
                    "peak_test_acc_epoch": peak_acc_epoch,
                    "peak_to_current_test_acc_drop": (
                        peak_acc - float(row["test_acc"])
                    ),
                    "loss_generalization_gap": (
                        float(row["test_loss"]) - float(row["train_loss"])
                    ),
                    "accuracy_generalization_gap": (
                        float(row["train_acc"]) - float(row["test_acc"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def _print_epoch_report(
    *,
    epoch: int,
    epochs: int,
    elapsed: float,
    performance_epoch: pd.DataFrame,
    ww_epoch: pd.DataFrame,
    controller_epoch: pd.DataFrame,
    corrections_epoch: pd.DataFrame,
    performance_history: pd.DataFrame,
) -> None:
    print("", flush=True)
    print("=" * 152, flush=True)
    print(
        f"EPOCH {epoch:03d}/{epochs:03d} COMPLETE | "
        f"elapsed={elapsed:,.1f}s",
        flush=True,
    )
    print("-" * 152, flush=True)
    monitor = _online_performance_monitor(performance_history)
    monitor = monitor.loc[monitor["epoch"].eq(epoch)]
    print("PERFORMANCE AND ONLINE OVERFITTING INDICATORS", flush=True)
    print(
        monitor[
            [
                "run",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "test_loss_rebound",
                "peak_to_current_test_acc_drop",
                "loss_generalization_gap",
                "accuracy_generalization_gap",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        flush=True,
    )

    show = ww_epoch.loc[ww_epoch["status"].eq("ok")].copy()
    show["layer"] = show["layer_name"].astype(str).str.split(".").str[-1]
    print("", flush=True)
    print("WEIGHTWATCHER METRICS", flush=True)
    print(
        show[
            [
                "run",
                "layer",
                "alpha",
                "ERG_gap",
                "detX_num",
                "num_pl_spikes",
                "m_midpoint",
                "beta_E_midpoint",
                "scale_balance_reliable",
            ]
        ]
        .sort_values(["run", "layer"])
        .to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        flush=True,
    )

    print("", flush=True)
    print("ADAPTIVE CONTROLLER FOR NEXT EPOCH", flush=True)
    if controller_epoch.empty:
        print("No controlled layers.", flush=True)
    else:
        print(
            controller_epoch[
                [
                    "parameter",
                    "regime",
                    "reason",
                    "alpha",
                    "alpha_trend",
                    "confidence",
                    "task_throttle",
                    "effective_gain",
                    "shape_active",
                    "midpoint_rank",
                    "num_pl_spikes",
                    "policy_cadence",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            ),
            flush=True,
        )

    print("", flush=True)
    print("CORRECTION AND TASK-CONFLICT SUMMARY", flush=True)
    if corrections_epoch.empty:
        print("No correction diagnostics.", flush=True)
    else:
        print(
            corrections_epoch[
                [
                    "parameter",
                    "regime",
                    "expected_due_checks",
                    "recorded_due_checks",
                    "due_coverage",
                    "corrections_applied",
                    "applied_fraction",
                    "mean_combined_correction_ratio",
                    "mean_volume_correction_ratio",
                    "mean_shape_correction_ratio",
                    "mean_task_conflict_ratio_pre",
                    "mean_task_conflict_ratio_post",
                    "harmful_attempt_fraction",
                    "mean_correction_base_cosine",
                    "loss_neutral_fraction",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6g}",
            ),
            flush=True,
        )
    print("=" * 152, flush=True)


def run_mnist_guard_comparison(
    experiment_config: MNISTGuardExperimentConfig,
    guard_config: GuardConfig,
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> MNISTGuardExperimentResult:
    """Train AdamW and AdamW+AdaptiveSpectralGuard on identical batches."""

    guard_config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stop_file = output / "STOP_AFTER_CURRENT_EPOCH"

    set_seed(experiment_config.seed)
    device = device or choose_device()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_data = datasets.MNIST(
        str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(experiment_config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=experiment_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_data,
        batch_size=experiment_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=experiment_config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    initial_state = copy.deepcopy(MLP3().state_dict())
    baseline_model = MLP3().to(device)
    baseline_model.load_state_dict(initial_state)
    guard_model = MLP3().to(device)
    guard_model.load_state_dict(initial_state)

    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=experiment_config.learning_rate,
        weight_decay=experiment_config.weight_decay,
    )
    base_guard_optimizer = torch.optim.AdamW(
        guard_model.parameters(),
        lr=experiment_config.learning_rate,
        weight_decay=experiment_config.weight_decay,
    )
    guard_optimizer = AdaptiveSpectralGuard(
        base_guard_optimizer,
        guard_model.named_parameters(),
        config=guard_config,
    )

    initial_test = evaluate(
        guard_model,
        test_loader,
        device=device,
    )
    initial_guard_ww = _measure(
        guard_model,
        run_label="AdamW + AdaptiveSpectralGuard",
        epoch=0,
        global_step=0,
        config=experiment_config,
    )
    guard_optimizer.update_from_weightwatcher(
        initial_guard_ww.metrics
    )
    initial_baseline_ww = initial_guard_ww.metrics.copy()
    if not initial_baseline_ww.empty:
        initial_baseline_ww["run"] = "AdamW baseline"

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
            "run": "AdamW + AdaptiveSpectralGuard",
            "train_loss": np.nan,
            "train_acc": np.nan,
            "test_loss": initial_test["loss"],
            "test_acc": initial_test["acc"],
        },
    ]
    ww_frames = [
        initial_baseline_ww,
        initial_guard_ww.metrics.copy(),
    ]
    step_frames: list[pd.DataFrame] = []
    controller_frames = [
        _controller_frame(
            guard_optimizer,
            checkpoint_epoch=0,
            applies_to_epoch=1,
        )
    ]
    correction_frames: list[pd.DataFrame] = []
    global_step = 0

    def save_partial() -> None:
        performance = pd.DataFrame(performance_rows)
        weightwatcher = pd.concat(ww_frames, ignore_index=True)
        steps = (
            pd.concat(step_frames, ignore_index=True)
            if step_frames
            else pd.DataFrame()
        )
        controllers = (
            pd.concat(controller_frames, ignore_index=True)
            if controller_frames
            else pd.DataFrame()
        )
        corrections = (
            pd.concat(correction_frames, ignore_index=True)
            if correction_frames
            else pd.DataFrame()
        )
        performance.to_csv(output / "performance_live.csv", index=False)
        weightwatcher.to_csv(
            output / "weightwatcher_live.csv",
            index=False,
        )
        steps.to_csv(output / "guard_steps_live.csv", index=False)
        controllers.to_csv(
            output / "controller_live.csv",
            index=False,
        )
        corrections.to_csv(
            output / "correction_summary_live.csv",
            index=False,
        )

    try:
        for epoch in range(1, experiment_config.epochs + 1):
            started = time.perf_counter()
            controller_used = _controller_frame(
                guard_optimizer,
                checkpoint_epoch=epoch - 1,
                applies_to_epoch=epoch,
            )
            epoch_steps = _train_pair_one_epoch(
                baseline_model,
                guard_model,
                baseline_optimizer,
                guard_optimizer,
                train_loader,
                epoch=epoch,
                device=device,
                grad_clip_norm=experiment_config.grad_clip_norm,
            )
            global_step += len(train_loader)

            baseline_train = evaluate(
                baseline_model,
                train_eval_loader,
                device=device,
                max_batches=experiment_config.train_eval_max_batches,
            )
            baseline_test = evaluate(
                baseline_model,
                test_loader,
                device=device,
            )
            guard_train = evaluate(
                guard_model,
                train_eval_loader,
                device=device,
                max_batches=experiment_config.train_eval_max_batches,
            )
            guard_test = evaluate(
                guard_model,
                test_loader,
                device=device,
            )
            performance_epoch = pd.DataFrame(
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
                        "run": "AdamW + AdaptiveSpectralGuard",
                        "train_loss": guard_train["loss"],
                        "train_acc": guard_train["acc"],
                        "test_loss": guard_test["loss"],
                        "test_acc": guard_test["acc"],
                    },
                ]
            )
            performance_rows.extend(
                performance_epoch.to_dict("records")
            )

            baseline_ww = _measure(
                baseline_model,
                run_label="AdamW baseline",
                epoch=epoch,
                global_step=global_step,
                config=experiment_config,
            )
            guard_ww = _measure(
                guard_model,
                run_label="AdamW + AdaptiveSpectralGuard",
                epoch=epoch,
                global_step=global_step,
                config=experiment_config,
            )
            ww_epoch = pd.concat(
                [baseline_ww.metrics, guard_ww.metrics],
                ignore_index=True,
            )
            ww_frames.append(ww_epoch)

            if not epoch_steps.empty:
                step_frames.append(epoch_steps)
            guard_optimizer.observe_task_feedback(epoch_steps)
            guard_optimizer.update_from_weightwatcher(
                guard_ww.metrics
            )
            controller_next = _controller_frame(
                guard_optimizer,
                checkpoint_epoch=epoch,
                applies_to_epoch=epoch + 1,
            )
            controller_frames.append(controller_next)
            correction_epoch = _correction_summary(
                epoch_steps,
                controller_used,
                epoch,
                len(train_loader),
            )
            correction_frames.append(correction_epoch)

            save_partial()
            if progress:
                _print_epoch_report(
                    epoch=epoch,
                    epochs=experiment_config.epochs,
                    elapsed=time.perf_counter() - started,
                    performance_epoch=performance_epoch,
                    ww_epoch=ww_epoch,
                    controller_epoch=controller_next,
                    corrections_epoch=correction_epoch,
                    performance_history=pd.DataFrame(performance_rows),
                )
                print(
                    f"Clean stop after current epoch: touch {stop_file}",
                    flush=True,
                )
            if stop_file.exists():
                print(
                    f"Stop file detected after epoch {epoch}; stopping.",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        print(
            "KeyboardInterrupt received; preserving completed epochs.",
            flush=True,
        )

    performance = pd.DataFrame(performance_rows)
    weightwatcher = pd.concat(ww_frames, ignore_index=True)
    guard_steps = (
        pd.concat(step_frames, ignore_index=True)
        if step_frames
        else pd.DataFrame()
    )
    controller_history = (
        pd.concat(controller_frames, ignore_index=True)
        if controller_frames
        else pd.DataFrame()
    )
    correction_history = (
        pd.concat(correction_frames, ignore_index=True)
        if correction_frames
        else pd.DataFrame()
    )

    result = MNISTGuardExperimentResult(
        performance=performance,
        weightwatcher=weightwatcher,
        guard_steps=guard_steps,
        controller=controller_history,
        correction_summary=correction_history,
        baseline_model=baseline_model,
        guard_model=guard_model,
        guard_optimizer=guard_optimizer,
        output_dir=output,
    )
    result.save()
    return result

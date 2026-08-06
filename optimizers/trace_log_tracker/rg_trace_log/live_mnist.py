"""Live per-epoch runner for the paired MNIST TraceLogRG experiment."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .live_metrics import (
    LAYERS,
    RUNS,
    append_csv,
    correction_table,
    layer_names,
    monitor_performance,
    validate_weightwatcher,
)
from .mnist_experiment import (
    MLP3,
    MNISTExperimentConfig,
    MNISTExperimentResult,
    _measure,
    _summarize_corrections,
    _train_pair_one_epoch,
    choose_device,
    evaluate,
    set_seed,
)
from .wrapper import TraceLogConfig, TraceLogRGWrapper


@dataclass
class LiveMNISTRun:
    result: MNISTExperimentResult
    output_dir: Path
    completed_epochs: int
    stop_reason: Optional[str]


def run_mnist_comparison_live(
    config: MNISTExperimentConfig,
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    device: Optional[torch.device] = None,
    auto_stop_rg_alpha_below: Optional[float] = None,
    auto_stop_max_correction_ratio: Optional[float] = None,
) -> LiveMNISTRun:
    """Run one epoch at a time, printing and saving complete live diagnostics."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stop_file = output / "STOP_AFTER_CURRENT_EPOCH"
    log_file = output / "epoch_monitor.log"
    perf_csv = output / "performance_live.csv"
    ww_csv = output / "weightwatcher_live.csv"
    steps_csv = output / "rg_steps_live.csv"
    corr_csv = output / "correction_by_layer_epoch_live.csv"
    monitor_csv = output / "performance_monitor_live.csv"

    def emit(value: Any = "") -> None:
        text = str(value)
        print(text, flush=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()

    set_seed(config.seed)
    device = device or choose_device()
    emit(f"Device: {device}")
    emit(f"Live output: {output.resolve()}")
    emit(f"Clean stop file: {stop_file.resolve()}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = datasets.MNIST(
        str(data_dir), train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        str(data_dir), train=False, download=True, transform=transform
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    expected_steps = len(train_loader)

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
        rg_model, run_label=RUNS[1], epoch=0, global_step=0, config=config
    )
    rg_optimizer.set_supports(initial_rg.supports)
    initial_baseline = initial_rg.metrics.copy()
    if not initial_baseline.empty:
        initial_baseline["run"] = RUNS[0]

    performance_rows = [
        dict(
            epoch=0,
            run=RUNS[0],
            train_loss=np.nan,
            train_acc=np.nan,
            test_loss=initial_test["loss"],
            test_acc=initial_test["acc"],
        ),
        dict(
            epoch=0,
            run=RUNS[1],
            train_loss=np.nan,
            train_acc=np.nan,
            test_loss=initial_test["loss"],
            test_acc=initial_test["acc"],
        ),
    ]
    ww_frames = [initial_baseline, initial_rg.metrics.copy()]
    step_frames: list[pd.DataFrame] = []
    global_step = 0
    completed = 0
    stop_reason = None
    started = time.perf_counter()
    monitor_state = {
        run: dict(
            best_loss=np.inf,
            best_loss_epoch=None,
            peak_acc=-np.inf,
            peak_acc_epoch=None,
        )
        for run in RUNS
    }

    initial_perf = pd.DataFrame(performance_rows)
    initial_ww = pd.concat(ww_frames, ignore_index=True)
    validate_weightwatcher(initial_ww)
    append_csv(initial_perf, perf_csv)
    append_csv(initial_ww, ww_csv)

    initial_show = initial_ww.copy()
    initial_show["layer"] = layer_names(initial_show["layer_name"])
    initial_show = initial_show.loc[
        initial_show["status"].eq("ok") & initial_show["layer"].isin(LAYERS)
    ]
    emit("=" * 132)
    emit("INITIALIZATION — WEIGHTWATCHER METRICS")
    emit(
        initial_show[
            [
                "run",
                "layer",
                "alpha",
                "ERG_gap",
                "detX_num",
                "num_pl_spikes",
                "m_midpoint",
                "trace_log_midpoint_per_eval",
            ]
        ]
        .sort_values(["run", "layer"])
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    emit(f"Initial supports: {rg_optimizer.get_supports()}")
    emit("=" * 132)

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
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
            global_step += expected_steps

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

            performance_epoch = pd.DataFrame(
                [
                    dict(
                        epoch=epoch,
                        run=RUNS[0],
                        train_loss=baseline_train["loss"],
                        train_acc=baseline_train["acc"],
                        test_loss=baseline_test["loss"],
                        test_acc=baseline_test["acc"],
                    ),
                    dict(
                        epoch=epoch,
                        run=RUNS[1],
                        train_loss=rg_train["loss"],
                        train_acc=rg_train["acc"],
                        test_loss=rg_test["loss"],
                        test_acc=rg_test["acc"],
                    ),
                ]
            )
            baseline_checkpoint = _measure(
                baseline_model,
                run_label=RUNS[0],
                epoch=epoch,
                global_step=global_step,
                config=config,
            )
            rg_checkpoint = _measure(
                rg_model,
                run_label=RUNS[1],
                epoch=epoch,
                global_step=global_step,
                config=config,
            )
            ww_epoch = pd.concat(
                [baseline_checkpoint.metrics, rg_checkpoint.metrics],
                ignore_index=True,
            )
            validate_weightwatcher(ww_epoch)
            rg_optimizer.set_supports(rg_checkpoint.supports)
            next_supports = rg_optimizer.get_supports()

            performance_rows.extend(performance_epoch.to_dict("records"))
            ww_frames.append(ww_epoch)
            if not epoch_steps.empty:
                step_frames.append(epoch_steps)

            monitor = monitor_performance(performance_epoch, monitor_state)
            corrections = correction_table(epoch_steps, expected_steps, next_supports)
            corrections.insert(0, "epoch", epoch)
            append_csv(performance_epoch, perf_csv)
            append_csv(ww_epoch, ww_csv)
            append_csv(epoch_steps, steps_csv)
            append_csv(monitor, monitor_csv)
            append_csv(corrections, corr_csv)

            ww_show = ww_epoch.copy()
            ww_show["layer"] = layer_names(ww_show["layer_name"])
            ok = ww_show.loc[
                ww_show["status"].eq("ok") & ww_show["layer"].isin(LAYERS)
            ]
            failed = ww_show.loc[~ww_show["status"].eq("ok")]
            warnings = []

            for _, row in ok.iterrows():
                if float(row["alpha"]) < 2:
                    warnings.append(
                        f"{row['run']} {row['layer'].upper()}: "
                        f"alpha={float(row['alpha']):.4f} < 2"
                    )
            for _, row in corrections.iterrows():
                if float(row["coverage"]) < 0.99:
                    warnings.append(
                        f"{row['layer'].upper()}: "
                        f"coverage={float(row['coverage']):.3f} < 0.99"
                    )
                if int(row["geometry_failures"]) > 0:
                    warnings.append(
                        f"{row['layer'].upper()}: "
                        f"{int(row['geometry_failures'])} geometry failures"
                    )
                if float(row["max_correction_ratio"]) > 1:
                    warnings.append(
                        f"{row['layer'].upper()}: max correction="
                        f"{float(row['max_correction_ratio']):.3f} AdamW steps"
                    )
            for _, row in monitor.iterrows():
                if float(row["test_loss_rebound"]) > 0.02:
                    warnings.append(
                        f"{row['run']}: test-loss rebound="
                        f"{float(row['test_loss_rebound']):.4f}"
                    )
                if float(row["peak_to_current_test_acc_drop"]) > 0.002:
                    warnings.append(
                        f"{row['run']}: peak-to-current test-accuracy drop="
                        f"{float(row['peak_to_current_test_acc_drop']):.4f}"
                    )
            for _, row in failed.iterrows():
                warnings.append(
                    f"WeightWatcher failure for {row.get('run', '?')} "
                    f"{row.get('layer_name', '?')}: {row.get('error', '')}"
                )

            elapsed = time.perf_counter() - epoch_started
            total = time.perf_counter() - started
            emit("")
            emit("=" * 132)
            emit(
                f"EPOCH {epoch:03d}/{config.epochs:03d} COMPLETE"
                f" | epoch={elapsed:,.1f}s | total={total / 60:,.1f} min"
            )
            emit("-" * 132)
            emit("PERFORMANCE AND ONLINE OVERFITTING INDICATORS")
            emit(
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
                ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
            )
            emit("")
            emit("WEIGHTWATCHER — DIRECT FROM analyze(..., ERG=True)")
            emit(
                ok[
                    [
                        "run",
                        "layer",
                        "alpha",
                        "ERG_gap",
                        "detX_num",
                        "num_pl_spikes",
                        "m_midpoint",
                        "trace_log_midpoint_per_eval",
                    ]
                ]
                .sort_values(["run", "layer"])
                .to_string(index=False, float_format=lambda x: f"{x:.6f}")
            )
            if not failed.empty:
                emit("")
                emit("WEIGHTWATCHER FAILURES")
                columns = [
                    name
                    for name in ("run", "layer_name", "status", "error")
                    if name in failed
                ]
                emit(failed[columns].to_string(index=False))
            emit("")
            emit("TRACELOGRG CORRECTION BY LAYER FOR THIS EPOCH")
            emit(
                corrections[
                    [
                        "layer",
                        "support_used",
                        "next_epoch_support",
                        "opportunities",
                        "coverage",
                        "fired",
                        "fired_fraction",
                        "negative_drift_fraction",
                        "mean_correction_ratio_all_steps",
                        "mean_correction_ratio_when_fired",
                        "max_correction_ratio",
                        "geometry_failures",
                        "selected_drift_residual",
                    ]
                ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
            )
            emit("")
            if warnings:
                emit(f"WARNINGS ({len(warnings)})")
                for warning in warnings:
                    emit(f"  WARNING: {warning}")
            else:
                emit("WARNINGS: none")
            emit(f"Live files: {output.resolve()}")
            emit(f"Clean stop after this epoch: touch {stop_file.resolve()}")
            emit("=" * 132)
            completed = epoch

            if stop_file.exists():
                stop_reason = f"stop file detected after epoch {epoch}"
            elif auto_stop_rg_alpha_below is not None:
                rg_alpha = pd.to_numeric(
                    ok.loc[ok["run"].eq(RUNS[1]), "alpha"], errors="coerce"
                )
                if rg_alpha.lt(float(auto_stop_rg_alpha_below)).any():
                    stop_reason = (
                        "RG alpha crossed below "
                        f"{auto_stop_rg_alpha_below}"
                    )
            if stop_reason is None and auto_stop_max_correction_ratio is not None:
                if corrections["max_correction_ratio"].gt(
                    float(auto_stop_max_correction_ratio)
                ).any():
                    stop_reason = (
                        "maximum correction ratio crossed "
                        f"{auto_stop_max_correction_ratio}"
                    )
            if stop_reason:
                emit(f"STOPPING CLEANLY: {stop_reason}")
                break

    except KeyboardInterrupt:
        stop_reason = f"KeyboardInterrupt; preserving {completed} completed epochs"
        emit(f"STOPPING: {stop_reason}")

    performance = pd.DataFrame(performance_rows)
    weightwatcher = pd.concat(ww_frames, ignore_index=True)
    rg_steps = (
        pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    )
    result = MNISTExperimentResult(
        performance=performance,
        weightwatcher=weightwatcher,
        rg_steps=rg_steps,
        correction_summary=_summarize_corrections(rg_steps),
        baseline_model=baseline_model,
        rg_model=rg_model,
        rg_optimizer=rg_optimizer,
    )
    result.save(output)
    emit(
        f"FINISHED: {completed} completed epochs"
        + (f" | {stop_reason}" if stop_reason else "")
    )
    emit(f"Final/partial result files: {output.resolve()}")
    return LiveMNISTRun(result, output, completed, stop_reason)

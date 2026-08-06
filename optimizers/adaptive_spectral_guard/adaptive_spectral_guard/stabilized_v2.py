"""Fail-fast runtime and configuration for AdaptiveSpectralGuard Stabilized V2.

The notebook uses this module instead of assembling the V2 experiment through
ad-hoc cells.  It makes a stale V1 import or an accidental V1 configuration a
hard error before MNIST training starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .config import ControllerConfig, GuardConfig, preset_policies
from .controller import AdaptiveSpectralController
from .experiment import (
    MNISTGuardExperimentConfig,
    MNISTGuardExperimentResult,
    run_mnist_guard_comparison,
)


STABILIZED_V2_API = 2


@dataclass(frozen=True)
class StabilizedV2Configuration:
    experiment: MNISTGuardExperimentConfig
    guard: GuardConfig


def build_stabilized_v2_configuration(
    *,
    epochs: int = 30,
    seed: int = 1337,
) -> StabilizedV2Configuration:
    """Return the exact configuration used by the stabilized V2 notebook."""

    experiment = MNISTGuardExperimentConfig(
        seed=int(seed),
        epochs=int(epochs),
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-2,
        grad_clip_norm=1.0,
        ww_min_evals=10,
        ww_max_evals=None,
        n_shells=5,
        min_beta_retained=20,
        min_beta_decades=0.50,
        train_eval_max_batches=None,
    )
    controller = ControllerConfig(
        alpha_on=2.08,
        alpha_strong=1.98,
        alpha_off=2.18,
        alpha_trend_on=-0.04,
        trend_ceiling=2.30,
        off_patience=2,
        min_confidence=0.20,
        support_change_scale=0.20,
        erg_gap_ratio_scale=0.30,
        confidence_ema_decay=0.67,
        separate_channel_confidence=True,
        volume_confidence_floor_below_boundary=0.25,
        volume_confidence_floor_alpha=2.05,
        shape_min_confidence=0.15,
        shape_raw_confidence_floor=0.05,
        beta_on=0.05,
        shape_alpha_on=2.05,
        shape_requires_alpha_boundary=True,
        task_conflict_ema_decay=0.80,
        task_conflict_penalty=2.0,
        minimum_task_throttle=0.10,
    )
    guard = GuardConfig(
        controller=controller,
        policies=preset_policies("stabilized"),
    )
    validate_stabilized_v2_configuration(guard)
    return StabilizedV2Configuration(experiment=experiment, guard=guard)


def validate_stabilized_v2_configuration(config: GuardConfig) -> None:
    """Reject any configuration that could silently reproduce the V1 run."""

    config.validate()
    controller = config.controller
    errors: list[str] = []
    if not controller.separate_channel_confidence:
        errors.append("separate_channel_confidence must be True")
    if controller.confidence_ema_decay <= 0.0:
        errors.append("confidence_ema_decay must be positive")
    if controller.volume_confidence_floor_below_boundary <= 0.0:
        errors.append("volume confidence floor must be positive")
    if not controller.shape_requires_alpha_boundary:
        errors.append("shape_requires_alpha_boundary must be True")
    if controller.shape_raw_confidence_floor <= 0.0:
        errors.append("shape raw-confidence veto must be enabled")

    fc1 = config.policy_for("fc1.weight")
    fc2 = config.policy_for("fc2.weight")
    fc3 = config.policy_for("fc3.weight")
    if not fc1.enabled or not fc2.enabled or fc3.enabled:
        errors.append("expected FC1/FC2 enabled and FC3 disabled")
    if fc1.shape_max_ratio is None or fc1.shape_max_ratio > 0.0200001:
        errors.append("FC1 shape cap must be at most 0.02")
    if fc2.shape_max_ratio is None or fc2.shape_max_ratio > 0.0075001:
        errors.append("FC2 shape cap must be at most 0.0075")
    if fc1.shape_beta_deadband < 0.05 or fc2.shape_beta_deadband < 0.05:
        errors.append("beta-E deadband must be at least 0.05")

    if errors:
        raise RuntimeError(
            "Not a valid AdaptiveSpectralGuard Stabilized V2 configuration: "
            + "; ".join(errors)
        )


def stabilized_v2_policy_table(config: GuardConfig) -> pd.DataFrame:
    """Return a compact layer-policy table for the notebook preflight."""

    rows = []
    for parameter in ("fc1.weight", "fc2.weight", "fc3.weight"):
        policy = config.policy_for(parameter)
        rows.append(
            {
                "parameter": parameter,
                "enabled": policy.enabled,
                "cadence": policy.cadence,
                "weak_gain": policy.weak_gain,
                "strong_gain": policy.strong_gain,
                "volume_max_ratio": policy.volume_max_ratio,
                "shape_scale": policy.shape_scale,
                "shape_max_ratio": policy.shape_max_ratio,
                "shape_beta_deadband": policy.shape_beta_deadband,
                "combined_max_ratio": policy.combined_max_ratio,
            }
        )
    return pd.DataFrame(rows)


def _low_confidence_probe_row() -> pd.DataFrame:
    """Mimic the failed checkpoint: alpha well below two, poor ECS confidence."""

    return pd.DataFrame(
        [
            {
                "parameter_name": "fc1.weight",
                "epoch": 1,
                "status": "ok",
                "alpha": 1.59,
                "alpha_source": "WeightWatcher",
                "ERG_gap": -105,
                "ERG_gap_source": "WeightWatcher",
                "detX_num": 199,
                "num_pl_spikes": 304,
                "m_midpoint": 251,
                "boundary_overlap_ratio": 0.05,
                "beta_E_midpoint": 0.28,
                "scale_balance_reliable": True,
            }
        ]
    )


def run_stabilized_v2_preflight(config: GuardConfig) -> pd.DataFrame:
    """Prove before training that poor confidence cannot recreate V1 behavior."""

    validate_stabilized_v2_configuration(config)
    controller = AdaptiveSpectralController(config)
    controller.update_from_weightwatcher(_low_confidence_probe_row())
    frame = controller.frame()
    assert_stabilized_v2_controller_frame(frame, config)
    state = controller.get_state("fc1.weight")
    if state.regime != "strong":
        raise RuntimeError(
            f"V2 preflight failed: expected FC1 strong, received {state.regime!r}"
        )
    if state.volume_effective_gain <= 0.0:
        raise RuntimeError("V2 preflight failed: FC1 volume gain is zero below alpha=2")
    if state.shape_effective_gain != 0.0:
        raise RuntimeError(
            "V2 preflight failed: low raw confidence did not veto the shape channel"
        )
    return frame


def assert_stabilized_v2_controller_frame(
    frame: pd.DataFrame,
    config: GuardConfig,
) -> None:
    """Detect the exact silent fallback seen in the failed run."""

    if frame is None or frame.empty:
        return
    required = {
        "parameter",
        "regime",
        "reason",
        "alpha",
        "raw_confidence",
        "smoothed_confidence",
        "volume_confidence",
        "shape_confidence",
        "volume_effective_gain",
        "shape_effective_gain",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            "Stabilized V2 controller columns are missing: " + ", ".join(sorted(missing))
        )

    bad_reason = frame["reason"].astype(str).eq("low ECS confidence")
    if bad_reason.any():
        parameters = ", ".join(frame.loc[bad_reason, "parameter"].astype(str))
        raise RuntimeError(
            "V1 all-or-nothing confidence controller detected for " + parameters
        )

    threshold = config.controller.volume_confidence_floor_alpha
    for _, row in frame.iterrows():
        parameter = str(row["parameter"])
        policy = config.policy_for(parameter)
        if not policy.enabled:
            continue
        alpha = float(row["alpha"])
        if np.isfinite(alpha) and alpha <= threshold:
            if str(row["regime"]) == "off":
                raise RuntimeError(
                    f"V2 controller failure: {parameter} is off at alpha={alpha:.6f}"
                )
            if float(row["volume_effective_gain"]) <= 0.0:
                raise RuntimeError(
                    f"V2 controller failure: {parameter} has zero volume gain "
                    f"at alpha={alpha:.6f}"
                )


def install_stabilized_v2_live_report(config: GuardConfig) -> None:
    """Install an idempotent report wrapper with V2 assertions and channel data."""

    validate_stabilized_v2_configuration(config)
    from . import experiment as experiment_module

    current = experiment_module._print_epoch_report
    base = getattr(current, "_stabilized_v2_base", current)

    def report(**kwargs) -> None:
        base(**kwargs)
        controller_epoch = kwargs.get("controller_epoch")
        if controller_epoch is None or controller_epoch.empty:
            return
        assert_stabilized_v2_controller_frame(controller_epoch, config)
        columns = [
            "parameter",
            "regime",
            "reason",
            "alpha",
            "raw_confidence",
            "smoothed_confidence",
            "volume_confidence",
            "shape_confidence",
            "task_throttle",
            "volume_effective_gain",
            "shape_effective_gain",
            "shape_active",
            "policy_cadence",
            "policy_volume_max_ratio",
            "policy_shape_max_ratio",
            "policy_combined_max_ratio",
        ]
        available = [column for column in columns if column in controller_epoch.columns]
        print("", flush=True)
        print("STABILIZED V2 CHANNEL GATES FOR NEXT EPOCH", flush=True)
        print(
            controller_epoch[available].to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            ),
            flush=True,
        )

    report._stabilized_v2_base = base  # type: ignore[attr-defined]
    report._stabilized_v2_api = STABILIZED_V2_API  # type: ignore[attr-defined]
    experiment_module._print_epoch_report = report


def run_stabilized_v2_mnist(
    configuration: StabilizedV2Configuration,
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> MNISTGuardExperimentResult:
    """Run MNIST only after the package and controller pass the V2 preflight."""

    validate_stabilized_v2_configuration(configuration.guard)
    run_stabilized_v2_preflight(configuration.guard)
    install_stabilized_v2_live_report(configuration.guard)
    return run_mnist_guard_comparison(
        configuration.experiment,
        configuration.guard,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        progress=progress,
    )

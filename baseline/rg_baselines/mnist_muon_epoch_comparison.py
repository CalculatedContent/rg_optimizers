"""Compare test accuracy with standard and rectangular RG alphas on MNIST Muon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
import pandas as pd

from .muon_microbatch_capture import (
    load_microbatch_checkpoint,
    load_microbatch_index,
    matrix_esd_eigenvalues,
)
from .rectangular_rg import rectangular_flow_spectra

DEFAULT_LAYERS = ("fc1.weight", "fc2.weight")
DEFAULT_ALPHA_RANGE = (1.01, 20.0)


def fit_powerlaw_ks_mle(
    values: np.ndarray,
    *,
    min_points: int = 8,
    alpha_range: tuple[float, float] = DEFAULT_ALPHA_RANGE,
    boundary_atol: float = 1e-3,
) -> dict[str, Any]:
    """Fit a continuous power law with MLE alpha and KS-selected xmin.

    ``powerlaw.Fit`` searches candidate tail starts and selects ``xmin`` by
    minimizing the Kolmogorov-Smirnov distance. ``test_all_xmin=True`` forces
    the search over every admissible observed tail start rather than a reduced
    candidate grid. The returned ``fit.power_law.alpha`` is the package's MLE.
    """

    import powerlaw

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data) & (data > 0.0)]
    lower, upper = float(alpha_range[0]), float(alpha_range[1])
    if not 1.0 < lower < upper:
        raise ValueError("alpha_range must satisfy 1 < lower < upper")
    if int(min_points) < 2:
        raise ValueError("min_points must be at least two")

    base: dict[str, Any] = {
        "alpha": np.nan,
        "xmin": np.nan,
        "ks_distance": np.nan,
        "n_tail": 0,
        "n_values": int(data.size),
        "tail_fraction": 0.0,
        "alpha_lower_bound": lower,
        "alpha_upper_bound": upper,
        "alpha_at_boundary": False,
        "fit_warning": "",
        "fit_error": "",
        "fit_method": "powerlaw.Fit continuous MLE",
        "xmin_method": "minimum KS distance over all observed candidates",
    }
    if data.size < int(min_points):
        return {**base, "fit_error": "too_few_values"}

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = powerlaw.Fit(
                data,
                discrete=False,
                verbose=False,
                test_all_xmin=True,
                parameter_ranges={"alpha": [lower, upper]},
            )
        alpha = float(fit.power_law.alpha)
        xmin = float(fit.power_law.xmin)
        n_tail = int(np.count_nonzero(data >= xmin))
        at_boundary = bool(
            np.isclose(alpha, lower, atol=float(boundary_atol), rtol=0.0)
            or np.isclose(alpha, upper, atol=float(boundary_atol), rtol=0.0)
        )
        return {
            **base,
            "alpha": alpha,
            "xmin": xmin,
            "ks_distance": float(fit.power_law.D),
            "n_tail": n_tail,
            "tail_fraction": n_tail / max(data.size, 1),
            "alpha_at_boundary": at_boundary,
            "fit_warning": " | ".join(
                sorted({str(item.message) for item in caught})
            ),
        }
    except Exception as exc:
        return {**base, "fit_error": f"{type(exc).__name__}: {exc}"}


def _load_training_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "training_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    metrics = pd.read_csv(path).sort_values("epoch").reset_index(drop=True)
    required = {"epoch", "global_step", "test_accuracy", "test_loss"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"training metrics missing columns: {sorted(missing)}")
    if metrics["test_accuracy"].isna().any():
        raise ValueError(
            "test_accuracy contains missing values; run training with "
            "--test-every-epoch"
        )
    return metrics


def _selected_flow_steps(
    available: Iterable[int],
    *,
    stride: int,
    epoch_end_steps: Iterable[int],
) -> list[int]:
    if int(stride) < 1:
        raise ValueError("stride must be positive")
    steps = sorted({int(step) for step in available if int(step) > 0})
    ends = {int(step) for step in epoch_end_steps}
    selected = [step for step in steps if step % int(stride) == 0 or step in ends]
    return sorted(set(selected))


def _aggregate_epoch_rows(
    raw: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_map = metrics.set_index("epoch").to_dict(orient="index")

    weight = raw[raw["spectrum"].eq("weight_esd")].copy()
    for record in weight.itertuples(index=False):
        meta = metric_map[int(record.epoch)]
        rows.append(
            {
                "epoch": int(record.epoch),
                "global_step": int(record.global_step),
                "test_accuracy": float(meta["test_accuracy"]),
                "test_loss": float(meta["test_loss"]),
                "spectrum": str(record.spectrum),
                "layer": str(record.layer),
                "alpha": float(record.alpha),
                "alpha_mean": float(record.alpha),
                "alpha_std": np.nan,
                "sample_count": 1,
                "xmin": float(record.xmin),
                "ks_distance": float(record.ks_distance),
                "n_tail": float(record.n_tail),
                "n_values": float(record.n_values),
                "tail_fraction": float(record.tail_fraction),
                "boundary_hits": int(bool(record.alpha_at_boundary)),
                "aggregation": "epoch-end checkpoint",
            }
        )

    flow = raw[~raw["spectrum"].eq("weight_esd")].copy()
    flow = flow[np.isfinite(flow["alpha"])].copy()
    grouped = flow.groupby(["epoch", "spectrum", "layer"], sort=True)
    for (epoch, spectrum, layer), group in grouped:
        meta = metric_map[int(epoch)]
        rows.append(
            {
                "epoch": int(epoch),
                "global_step": int(meta["global_step"]),
                "test_accuracy": float(meta["test_accuracy"]),
                "test_loss": float(meta["test_loss"]),
                "spectrum": str(spectrum),
                "layer": str(layer),
                "alpha": float(group["alpha"].median()),
                "alpha_mean": float(group["alpha"].mean()),
                "alpha_std": float(group["alpha"].std()),
                "sample_count": int(len(group)),
                "xmin": float(group["xmin"].median()),
                "ks_distance": float(group["ks_distance"].median()),
                "n_tail": float(group["n_tail"].median()),
                "n_values": float(group["n_values"].median()),
                "tail_fraction": float(group["tail_fraction"].median()),
                "boundary_hits": int(group["alpha_at_boundary"].astype(bool).sum()),
                "aggregation": "median over sampled one-step flows in epoch",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["spectrum", "layer", "epoch"]
    ).reset_index(drop=True)


def _plot_test_accuracy(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(metrics["epoch"], metrics["test_accuracy"], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("MNIST MLP3 Muon: test accuracy through five epochs")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(metrics["epoch"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_alpha_with_accuracy(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    spectra: tuple[str, ...],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected = summary[summary["spectrum"].isin(spectra)].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for (spectrum, layer), group in selected.groupby(
        ["spectrum", "layer"], sort=True
    ):
        group = group[np.isfinite(group["alpha"])].sort_values("epoch")
        if group.empty:
            continue
        ax.plot(
            group["epoch"],
            group["alpha"],
            marker="o",
            linewidth=1.4,
            label=f"{layer}: {spectrum}",
        )
    ax.axhline(2.0, linestyle="--", linewidth=1.3, label="RG hypothesis alpha=2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("power-law exponent alpha")
    ax.set_xticks(metrics["epoch"])
    ax.grid(True, alpha=0.25)

    accuracy_axis = ax.twinx()
    accuracy_axis.plot(
        metrics["epoch"],
        metrics["test_accuracy"],
        marker="s",
        linestyle=":",
        linewidth=1.4,
        label="test accuracy",
    )
    accuracy_axis.set_ylabel("test accuracy")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = accuracy_axis.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_all_alpha(summary: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    _plot_alpha_with_accuracy(
        summary,
        metrics,
        spectra=("weight_esd", "core_log_deviation", "angular_theta_squared"),
        title=(
            "MNIST MLP3 Muon: standard weight and rectangular RG alphas "
            "versus test accuracy"
        ),
        output_path=output_path,
    )


def analyze_five_epoch_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    layers: Sequence[str] = DEFAULT_LAYERS,
    flow_step_stride: int = 10,
    rank_rtol: float = 1e-10,
    log_zero_tol: float = 1e-12,
    angle_zero_tol: float = 1e-12,
    powerlaw_min_points: int = 8,
    powerlaw_alpha_range: tuple[float, float] = DEFAULT_ALPHA_RANGE,
) -> dict[str, Any]:
    root = Path(run_dir)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else root / "five_epoch_comparison"
    )
    destination.mkdir(parents=True, exist_ok=True)

    metrics = _load_training_metrics(root)
    index = load_microbatch_index(root)
    by_step = {
        int(row.global_step): Path(row.checkpoint_path)
        for row in index.itertuples(index=False)
    }
    epoch_end_steps = [int(value) for value in metrics["global_step"]]
    missing_ends = [step for step in epoch_end_steps if step not in by_step]
    if missing_ends:
        raise ValueError(f"epoch-end checkpoints are missing: {missing_ends}")

    selected_layers = tuple(str(layer) for layer in layers)
    fit_rows: list[dict[str, Any]] = []
    spectra: dict[tuple[str, int, str], np.ndarray] = {}

    # Standard analysis: fit the ordinary weight ESD at each epoch endpoint.
    for metric in metrics.itertuples(index=False):
        step = int(metric.global_step)
        payload = load_microbatch_checkpoint(by_step[step])
        for layer in selected_layers:
            values = matrix_esd_eigenvalues(payload["matrices"][layer])
            spectra[("weight_esd", step, layer)] = values
            fit_rows.append(
                {
                    "epoch": int(metric.epoch),
                    "global_step": step,
                    "previous_step": np.nan,
                    "spectrum": "weight_esd",
                    "layer": layer,
                    **fit_powerlaw_ks_mle(
                        values,
                        min_points=powerlaw_min_points,
                        alpha_range=powerlaw_alpha_range,
                    ),
                }
            )

    # Rectangular RG analysis: sample one-step flows throughout every epoch.
    flow_steps = _selected_flow_steps(
        by_step,
        stride=int(flow_step_stride),
        epoch_end_steps=epoch_end_steps,
    )
    flow_steps = [step for step in flow_steps if step - 1 in by_step]
    for step in flow_steps:
        previous = load_microbatch_checkpoint(by_step[step - 1])
        current = load_microbatch_checkpoint(by_step[step])
        epoch = int(current["epoch"])
        for layer in selected_layers:
            result = rectangular_flow_spectra(
                previous["matrices"][layer],
                current["matrices"][layer],
                rank_rtol=float(rank_rtol),
                log_zero_tol=float(log_zero_tol),
                angle_zero_tol=float(angle_zero_tol),
            )
            core = np.asarray(result["core_log_deviation"], dtype=float)
            spectra[("core_log_deviation", step, layer)] = core
            fit_rows.append(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "previous_step": step - 1,
                    "spectrum": "core_log_deviation",
                    "layer": layer,
                    **fit_powerlaw_ks_mle(
                        core,
                        min_points=powerlaw_min_points,
                        alpha_range=powerlaw_alpha_range,
                    ),
                }
            )

            angular = np.asarray(result["angular_eigenvalues"], dtype=float)
            spectra[("angular_theta_squared", step, layer)] = angular
            fit_rows.append(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "previous_step": step - 1,
                    "spectrum": "angular_theta_squared",
                    "layer": layer,
                    **fit_powerlaw_ks_mle(
                        angular,
                        min_points=powerlaw_min_points,
                        alpha_range=powerlaw_alpha_range,
                    ),
                }
            )

    raw = pd.DataFrame(fit_rows).sort_values(
        ["spectrum", "layer", "global_step"]
    ).reset_index(drop=True)
    summary = _aggregate_epoch_rows(raw, metrics)

    raw.to_csv(destination / "powerlaw_fits_by_step.csv", index=False)
    summary.to_csv(destination / "alpha_test_accuracy_by_epoch.csv", index=False)
    metrics.to_csv(destination / "training_metrics_with_test_each_epoch.csv", index=False)
    np.savez_compressed(
        destination / "selected_spectra.npz",
        **{
            f"{kind}__step_{step:07d}__{layer.replace('.', '_')}": values
            for (kind, step, layer), values in spectra.items()
        },
    )

    _plot_test_accuracy(metrics, destination / "test_accuracy_vs_epoch.png")
    _plot_alpha_with_accuracy(
        summary,
        metrics,
        spectra=("weight_esd",),
        title="Standard weight-ESD alpha versus test accuracy",
        output_path=destination / "standard_weight_alpha_vs_epoch.png",
    )
    _plot_alpha_with_accuracy(
        summary,
        metrics,
        spectra=("core_log_deviation",),
        title="Aligned-core RG alpha versus test accuracy",
        output_path=destination / "rg_core_alpha_vs_epoch.png",
    )
    _plot_alpha_with_accuracy(
        summary,
        metrics,
        spectra=("angular_theta_squared",),
        title="Grassmann angular alpha versus test accuracy",
        output_path=destination / "rg_angular_alpha_vs_epoch.png",
    )
    _plot_all_alpha(
        summary,
        metrics,
        destination / "all_alpha_and_test_accuracy_vs_epoch.png",
    )

    manifest = {
        "run_dir": str(root),
        "output_dir": str(destination),
        "layers": list(selected_layers),
        "epochs": [int(value) for value in metrics["epoch"]],
        "epoch_end_steps": epoch_end_steps,
        "flow_step_stride": int(flow_step_stride),
        "flow_pair_lag": 1,
        "flow_steps": flow_steps,
        "powerlaw_fit": {
            "package": "powerlaw",
            "distribution": "continuous",
            "alpha_estimator": "powerlaw package MLE",
            "xmin_selection": (
                "minimum Kolmogorov-Smirnov distance with test_all_xmin=True"
            ),
            "alpha_range": list(powerlaw_alpha_range),
            "minimum_spectrum_points": int(powerlaw_min_points),
        },
        "standard_alpha": "power-law fit to sigma(W)^2 at each epoch endpoint",
        "rg_core_alpha": (
            "epoch median of power-law fits to abs(log(sigma(J_core)^2)) "
            "over sampled one-step flows"
        ),
        "rg_angular_alpha": (
            "epoch median of power-law fits to squared principal angles "
            "over sampled one-step flows"
        ),
        "fit_rows": int(len(raw)),
    }
    (destination / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "raw_fits": raw,
        "epoch_summary": summary,
        "manifest": manifest,
        "output_dir": destination,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-epoch test accuracy with standard weight ESD and "
            "rectangular RG power-law exponents."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--flow-step-stride", type=int, default=10)
    parser.add_argument("--rank-rtol", type=float, default=1e-10)
    parser.add_argument("--log-zero-tol", type=float, default=1e-12)
    parser.add_argument("--angle-zero-tol", type=float, default=1e-12)
    parser.add_argument("--powerlaw-min-points", type=int, default=8)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_RANGE[0])
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_RANGE[1])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = analyze_five_epoch_run(
        args.run_dir,
        output_dir=args.output_dir,
        flow_step_stride=args.flow_step_stride,
        rank_rtol=args.rank_rtol,
        log_zero_tol=args.log_zero_tol,
        angle_zero_tol=args.angle_zero_tol,
        powerlaw_min_points=args.powerlaw_min_points,
        powerlaw_alpha_range=(args.alpha_min, args.alpha_max),
    )
    print(result["epoch_summary"].to_string(index=False))
    print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()

"""Analyze aligned core and Grassmann spectra from MNIST Muon checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd

from .muon_microbatch_capture import (
    load_microbatch_checkpoint,
    load_microbatch_index,
)
from .rectangular_rg import rectangular_flow_spectra

DEFAULT_LAYERS = ("fc1.weight", "fc2.weight")
DEFAULT_ALPHA_RANGE = (1.01, 10.0)


def fit_powerlaw_spectrum(
    values: np.ndarray,
    *,
    min_points: int = 8,
    alpha_range: tuple[float, float] = DEFAULT_ALPHA_RANGE,
    boundary_atol: float = 1e-3,
) -> dict[str, Any]:
    """Fit one positive spectrum with powerlaw and retain quality diagnostics."""

    import powerlaw

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data) & (data > 0.0)]
    lower, upper = (float(alpha_range[0]), float(alpha_range[1]))
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
                parameter_ranges={"alpha": [lower, upper]},
            )
        alpha = float(fit.power_law.alpha)
        xmin = float(fit.power_law.xmin)
        n_tail = int(np.count_nonzero(data >= xmin))
        at_boundary = bool(
            np.isclose(alpha, lower, atol=float(boundary_atol), rtol=0.0)
            or np.isclose(alpha, upper, atol=float(boundary_atol), rtol=0.0)
        )
        warning_text = " | ".join(
            sorted({str(item.message) for item in caught})
        )
        return {
            **base,
            "alpha": alpha,
            "xmin": xmin,
            "ks_distance": float(fit.power_law.D),
            "n_tail": n_tail,
            "tail_fraction": n_tail / max(data.size, 1),
            "alpha_at_boundary": at_boundary,
            "fit_warning": warning_text,
        }
    except Exception as exc:
        return {
            **base,
            "fit_error": f"{type(exc).__name__}: {exc}",
        }


def _selected_steps(
    available_steps: Iterable[int],
    *,
    step_stride: int,
    max_step: int | None,
    include_final: bool,
) -> list[int]:
    steps = sorted({int(step) for step in available_steps if int(step) > 0})
    if max_step is not None:
        steps = [step for step in steps if step <= int(max_step)]
    selected = [step for step in steps if step % int(step_stride) == 0]
    if include_final and steps and steps[-1] not in selected:
        selected.append(steps[-1])
    return sorted(set(selected))


def _late_step_summary(fits: pd.DataFrame, *, tail_points: int = 10) -> pd.DataFrame:
    valid = fits[np.isfinite(fits["alpha"])].copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.sort_values("global_step")
        .groupby(["spectrum", "layer"], as_index=False)
        .tail(int(tail_points))
        .groupby(["spectrum", "layer"], as_index=False)
        .agg(
            alpha_median=("alpha", "median"),
            alpha_mean=("alpha", "mean"),
            alpha_std=("alpha", "std"),
            last_step=("global_step", "max"),
            median_tail_size=("n_tail", "median"),
            median_tail_fraction=("tail_fraction", "median"),
            median_ks_distance=("ks_distance", "median"),
            median_mode_count=("n_values", "median"),
            boundary_hits=("alpha_at_boundary", "sum"),
        )
    )


def _plot_alpha(fits: pd.DataFrame, spectrum: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = fits[fits["spectrum"].eq(spectrum)].copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for layer, group in selected.groupby("layer", sort=True):
        group = group[np.isfinite(group["alpha"])].sort_values("global_step")
        ax.plot(
            group["global_step"],
            group["alpha"],
            marker=".",
            linewidth=1.2,
            label=layer,
        )
        boundary = group[group["alpha_at_boundary"].astype(bool)]
        if not boundary.empty:
            ax.scatter(
                boundary["global_step"],
                boundary["alpha"],
                marker="x",
                s=28,
            )
    ax.axhline(
        2.0,
        linestyle="--",
        linewidth=1.4,
        label="RG hypothesis alpha=2",
    )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("power-law exponent alpha")
    ax.set_title(f"MNIST MLP3 Muon: {spectrum} alpha versus step")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_diagnostic(
    diagnostics: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for layer, group in diagnostics.groupby("layer", sort=True):
        group = group.sort_values("global_step")
        ax.plot(
            group["global_step"],
            group[column],
            marker=".",
            linewidth=1.2,
            label=layer,
        )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_selected_esds(
    spectra: dict[tuple[str, int, str], np.ndarray],
    *,
    spectrum: str,
    layer: str,
    steps: list[int],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for step in steps:
        values = np.asarray(spectra.get((spectrum, step, layer), []), dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size < 2 or values.min() == values.max():
            continue
        edges = np.logspace(np.log10(values.min()), np.log10(values.max()), 30)
        density, edges = np.histogram(values, bins=edges, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        mask = density > 0.0
        ax.loglog(
            centers[mask],
            density[mask],
            marker="o",
            linewidth=1.1,
            label=f"step {step}",
        )
    ax.set_xlabel("mode magnitude")
    ax.set_ylabel("ESD density")
    ax.set_title(f"{spectrum}: {layer}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_rectangular_muon_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    layers: Iterable[str] = DEFAULT_LAYERS,
    step_stride: int = 1,
    max_step: int | None = None,
    include_final: bool = True,
    rank_rtol: float = 1e-10,
    log_zero_tol: float = 1e-12,
    angle_zero_tol: float = 1e-12,
    powerlaw_min_points: int = 8,
    powerlaw_alpha_range: tuple[float, float] = DEFAULT_ALPHA_RANGE,
) -> dict[str, Any]:
    """Analyze FC1/FC2 aligned-core and angular spectra over a checkpoint run."""

    if int(step_stride) < 1:
        raise ValueError("step_stride must be positive")
    root = Path(run_dir)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else root / "rectangular_rg_analysis"
    )
    destination.mkdir(parents=True, exist_ok=True)
    selected_layers = tuple(str(layer) for layer in layers)
    index = load_microbatch_index(root)
    by_step = {
        int(row.global_step): Path(row.checkpoint_path)
        for row in index.itertuples(index=False)
    }
    selected_steps = _selected_steps(
        by_step,
        step_stride=int(step_stride),
        max_step=max_step,
        include_final=bool(include_final),
    )
    selected_steps = [step for step in selected_steps if step - 1 in by_step]
    if not selected_steps:
        raise ValueError("no successive checkpoint pairs matched the selection")

    fit_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    spectra: dict[tuple[str, int, str], np.ndarray] = {}

    for step in selected_steps:
        previous = load_microbatch_checkpoint(by_step[step - 1])
        current = load_microbatch_checkpoint(by_step[step])
        for layer in selected_layers:
            if layer not in previous["matrices"] or layer not in current["matrices"]:
                raise KeyError(f"checkpoint pair does not contain {layer}")
            result = rectangular_flow_spectra(
                previous["matrices"][layer],
                current["matrices"][layer],
                rank_rtol=rank_rtol,
                log_zero_tol=log_zero_tol,
                angle_zero_tol=angle_zero_tol,
            )
            core_values = np.asarray(result["core_log_deviation"], dtype=float)
            spectra[("core_log_deviation", step, layer)] = core_values
            fit_rows.append(
                {
                    "spectrum": "core_log_deviation",
                    "global_step": step,
                    "previous_step": step - 1,
                    "layer": layer,
                    "subspace": result["subspace"],
                    **fit_powerlaw_spectrum(
                        core_values,
                        min_points=powerlaw_min_points,
                        alpha_range=powerlaw_alpha_range,
                    ),
                }
            )

            angular_values = np.asarray(result["angular_eigenvalues"], dtype=float)
            spectra[("angular_theta_squared", step, layer)] = angular_values
            fit_rows.append(
                {
                    "spectrum": "angular_theta_squared",
                    "global_step": step,
                    "previous_step": step - 1,
                    "layer": layer,
                    "subspace": result["subspace"],
                    **fit_powerlaw_spectrum(
                        angular_values,
                        min_points=powerlaw_min_points,
                        alpha_range=powerlaw_alpha_range,
                    ),
                }
            )

            angles = np.asarray(result["principal_angles"], dtype=float)
            positive_angles = np.sqrt(angular_values)
            diagnostic_rows.append(
                {
                    "global_step": step,
                    "previous_step": step - 1,
                    "layer": layer,
                    "shape": "x".join(str(value) for value in result["shape"]),
                    "subspace": result["subspace"],
                    "rank": int(result["rank"]),
                    "ambient_dimension": int(result["ambient_dimension"]),
                    "forced_intersection_dimension": int(
                        result["forced_intersection_dimension"]
                    ),
                    "maximum_angular_modes": int(result["maximum_angular_modes"]),
                    "observed_angular_modes": int(angular_values.size),
                    "maximum_principal_angle": float(
                        angles.max() if angles.size else 0.0
                    ),
                    "median_positive_principal_angle": float(
                        np.median(positive_angles) if positive_angles.size else 0.0
                    ),
                    "previous_condition_number": float(
                        result["previous_condition_number"]
                    ),
                    "current_condition_number": float(
                        result["current_condition_number"]
                    ),
                }
            )

    fits = pd.DataFrame(fit_rows).sort_values(
        ["spectrum", "layer", "global_step"]
    ).reset_index(drop=True)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["layer", "global_step"]
    ).reset_index(drop=True)
    summary = _late_step_summary(fits)

    fits.to_csv(destination / "rectangular_powerlaw_fits.csv", index=False)
    diagnostics.to_csv(destination / "rectangular_flow_diagnostics.csv", index=False)
    summary.to_csv(destination / "late_step_alpha_summary.csv", index=False)
    np.savez_compressed(
        destination / "rectangular_spectra.npz",
        **{
            f"{kind}__step_{step:07d}__{layer.replace('.', '_')}": values
            for (kind, step, layer), values in spectra.items()
        },
    )

    _plot_alpha(
        fits,
        "core_log_deviation",
        destination / "alpha_core_log_deviation_vs_step.png",
    )
    _plot_alpha(
        fits,
        "angular_theta_squared",
        destination / "alpha_angular_theta_squared_vs_step.png",
    )
    _plot_diagnostic(
        diagnostics,
        "maximum_principal_angle",
        "maximum principal angle (radians)",
        "MNIST MLP3 Muon: maximum subspace angle versus step",
        destination / "maximum_principal_angle_vs_step.png",
    )

    display_steps = sorted(
        {
            selected_steps[0],
            selected_steps[len(selected_steps) // 2],
            selected_steps[-1],
        }
    )
    for layer in selected_layers:
        _plot_selected_esds(
            spectra,
            spectrum="core_log_deviation",
            layer=layer,
            steps=display_steps,
            output_path=destination
            / f"esd_core_log_deviation_{layer.replace('.', '_')}.png",
        )
        _plot_selected_esds(
            spectra,
            spectrum="angular_theta_squared",
            layer=layer,
            steps=display_steps,
            output_path=destination
            / f"esd_angular_theta_squared_{layer.replace('.', '_')}.png",
        )

    manifest = {
        "run_dir": str(root),
        "output_dir": str(destination),
        "layers": list(selected_layers),
        "selected_steps": selected_steps,
        "step_stride": int(step_stride),
        "pair_lag": 1,
        "rank_rtol": float(rank_rtol),
        "log_zero_tol": float(log_zero_tol),
        "angle_zero_tol": float(angle_zero_tol),
        "powerlaw_min_points": int(powerlaw_min_points),
        "powerlaw_alpha_range": list(powerlaw_alpha_range),
        "fit_rows": int(len(fits)),
    }
    (destination / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "fits": fits,
        "diagnostics": diagnostics,
        "summary": summary,
        "spectra": spectra,
        "manifest": manifest,
        "output_dir": destination,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze FC1/FC2 rectangular RG spectra from MNIST Muon checkpoints"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--rank-rtol", type=float, default=1e-10)
    parser.add_argument("--log-zero-tol", type=float, default=1e-12)
    parser.add_argument("--angle-zero-tol", type=float, default=1e-12)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_RANGE[0])
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_RANGE[1])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = analyze_rectangular_muon_run(
        args.run_dir,
        output_dir=args.output_dir,
        step_stride=args.step_stride,
        max_step=args.max_step,
        rank_rtol=args.rank_rtol,
        log_zero_tol=args.log_zero_tol,
        angle_zero_tol=args.angle_zero_tol,
        powerlaw_alpha_range=(args.alpha_min, args.alpha_max),
    )
    summary = result["summary"]
    if summary.empty:
        print("No valid power-law fits")
    else:
        print(summary.to_string(index=False))
    print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()

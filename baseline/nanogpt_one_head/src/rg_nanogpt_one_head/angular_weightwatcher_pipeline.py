from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .angular_weightwatcher_core import (
    AnalysisConfig,
    ResolvedRun,
    angular_from_polar,
    angular_spectra,
    assert_gauge_invariance,
    ccdf,
    fit_tail,
    gram_esd,
    load_weight_pairs,
    monte_carlo_ks,
    null_interval,
    polar,
    projective,
    random_polar,
    resolve_run,
    shuffle_entries,
)


def _finish(
    figure,
    output_dir: Path,
    filename: str,
    show: bool,
) -> None:
    figure.tight_layout()
    figure.savefig(
        output_dir / filename,
        dpi=180,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close(figure)


def _angular_plot(
    *,
    name: str,
    kind: str,
    actual: np.ndarray,
    nulls: list[np.ndarray],
    final_step: int,
    output_dir: Path,
    show: bool,
) -> None:
    low, median, high = np.quantile(
        np.stack(nulls),
        [0.025, 0.5, 0.975],
        axis=0,
    )
    quantile = (
        np.arange(actual.size, dtype=float) + 0.5
    ) / actual.size
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(quantile, actual, label="actual initial → final")
    axis.plot(
        quantile,
        median,
        "--",
        label="randomized-initial median",
    )
    axis.fill_between(
        quantile,
        low,
        high,
        alpha=0.2,
        label="randomized-initial 95% interval",
    )
    axis.set(
        xlabel="angular eigenvalue quantile",
        ylabel="sin² θ" if kind == "tilt" else "4 sin²(φ/2)",
        title=f"{name}: {kind} angular ESD, step 0 → {final_step}",
    )
    axis.legend(fontsize=8)
    _finish(
        figure,
        output_dir,
        f"{name}_{kind}_angular_esd.png",
        show,
    )


def _tail_plot(
    *,
    name: str,
    kind: str,
    actual: np.ndarray,
    nulls: list[np.ndarray],
    fit,
    null_alpha_median: float,
    upper: float,
    output_dir: Path,
    show: bool,
) -> None:
    x_actual, y_actual = ccdf(projective(actual, upper))
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    if x_actual.size:
        axis.loglog(x_actual, y_actual, ".", label="actual")
    for sample in nulls[: min(20, len(nulls))]:
        x_null, y_null = ccdf(projective(sample, upper))
        if x_null.size:
            axis.loglog(x_null, y_null, alpha=0.12)
    if fit.success and x_actual.size:
        x_fit = np.geomspace(
            fit.xmin,
            max(x_actual.max(), fit.xmin),
            100,
        )
        y_fit = (
            fit.n_tail / x_actual.size
        ) * (x_fit / fit.xmin) ** (1 - fit.alpha)
        axis.loglog(
            x_fit,
            y_fit,
            "--",
            label=f"actual Pareto α={fit.alpha:.3f}",
        )
    axis.set(
        xlabel="tan² θ" if kind == "tilt" else "tan²(φ/2)",
        ylabel="CCDF",
        title=(
            f"{name}: {kind} projective tail; "
            f"null α median={null_alpha_median:.3f}"
        ),
    )
    axis.legend(fontsize=8)
    _finish(
        figure,
        output_dir,
        f"{name}_{kind}_projective_ccdf.png",
        show,
    )


def _radial_plot(
    *,
    name: str,
    initial: np.ndarray,
    final: np.ndarray,
    config: AnalysisConfig,
    layer_index: int,
    resolved: ResolvedRun,
) -> None:
    initial_esd = gram_esd(initial)
    final_esd = gram_esd(final)
    rng = np.random.default_rng(
        config.null_seed + 70001 * (layer_index + 1)
    )
    initial_null = np.stack(
        [
            gram_esd(shuffle_entries(initial, rng))
            for _ in range(config.entry_nulls)
        ]
    )
    final_null = np.stack(
        [
            gram_esd(shuffle_entries(final, rng))
            for _ in range(config.entry_nulls)
        ]
    )
    rank = np.arange(1, initial_esd.size + 1)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.loglog(rank, initial_esd, label="saved initialization")
    axis.loglog(
        rank,
        final_esd,
        label=f"saved final step {resolved.final_step}",
    )
    axis.loglog(
        rank,
        np.median(initial_null, axis=0),
        "--",
        label="shuffled initial median",
    )
    axis.loglog(
        rank,
        np.median(final_null, axis=0),
        ":",
        label="shuffled final median",
    )
    axis.set(
        xlabel="eigenvalue rank",
        ylabel="Gram eigenvalue / mean",
        title=f"{name}: radial initial vs final",
    )
    axis.legend(fontsize=8)
    _finish(
        figure,
        resolved.output_dir,
        f"{name}_radial_initial_vs_final.png",
        config.show_plots,
    )


def _summary_plots(
    results: pd.DataFrame,
    resolved: ResolvedRun,
    show: bool,
) -> None:
    for kind in ("tilt", "twist"):
        frame = results[
            results.angular_type == kind
        ].reset_index(drop=True)
        x_position = np.arange(len(frame))
        median = frame.null_alpha_median.to_numpy(float)
        low = frame.null_alpha_2p5.to_numpy(float)
        high = frame.null_alpha_97p5.to_numpy(float)
        actual = frame.actual_alpha.to_numpy(float)
        finite_null = (
            np.isfinite(median)
            & np.isfinite(low)
            & np.isfinite(high)
        )
        finite_actual = np.isfinite(actual)

        figure, axis = plt.subplots(figsize=(10.5, 5.4))
        axis.errorbar(
            x_position[finite_null],
            median[finite_null],
            yerr=np.vstack(
                [
                    median[finite_null] - low[finite_null],
                    high[finite_null] - median[finite_null],
                ]
            ),
            fmt="o",
            capsize=4,
            label="randomized-initial median and 95% interval",
        )
        axis.scatter(
            x_position[finite_actual],
            actual[finite_actual],
            marker="x",
            s=70,
            label="actual initial → final",
        )
        axis.axhline(2, linestyle="--", label="α = 2 reference")
        axis.set_xticks(
            x_position,
            frame.matrix_name,
            rotation=35,
            ha="right",
        )
        axis.set_ylabel("fitted Pareto exponent")
        axis.set_title(
            f"{kind.capitalize()} exponent: actual versus randomized initial"
        )
        axis.legend(fontsize=8)
        _finish(
            figure,
            resolved.output_dir,
            f"summary_{kind}_alpha_actual_vs_null.png",
            show,
        )


def run_analysis(
    config: AnalysisConfig | None = None,
) -> tuple[pd.DataFrame, ResolvedRun, dict]:
    config = config or AnalysisConfig.from_env()
    config.validate()
    assert_gauge_invariance()
    resolved = resolve_run(config)
    initial_weights, final_weights, metadata = load_weight_pairs(
        config,
        resolved,
    )

    rows: list[dict] = []
    for layer_index, name in enumerate(sorted(initial_weights)):
        initial = initial_weights[name]
        final = final_weights[name]
        actual = angular_spectra(initial, final)
        final_polar = polar(final)
        rng = np.random.default_rng(
            config.null_seed + 1009 * layer_index
        )
        nulls = {"tilt": [], "twist": []}
        for _ in range(config.angular_nulls):
            sample = angular_from_polar(
                random_polar(initial.shape, rng),
                final_polar,
            )
            for kind in nulls:
                nulls[kind].append(sample[kind])

        for kind, upper in (("tilt", 1.0), ("twist", 4.0)):
            fit = fit_tail(
                projective(actual[kind], upper),
                config.min_tail,
            )
            null_fits = [
                fit_tail(projective(sample, upper), config.min_tail)
                for sample in nulls[kind]
            ]
            null_alphas = [
                item.alpha for item in null_fits if item.success
            ]
            alpha_low, alpha_median, alpha_high = null_interval(
                null_alphas
            )
            distance, probability = monte_carlo_ks(
                actual[kind],
                nulls[kind],
            )
            candidate = bool(
                fit.success
                and fit.ks <= 0.15
                and probability < 0.05
                and (
                    fit.alpha < alpha_low
                    or fit.alpha > alpha_high
                )
            )
            rows.append(
                {
                    "matrix_name": name,
                    "angular_type": kind,
                    "shape": str(initial.shape),
                    "actual_alpha": fit.alpha,
                    "actual_xmin": fit.xmin,
                    "actual_fit_ks": fit.ks,
                    "actual_tail_n": fit.n_tail,
                    "null_alpha_2p5": alpha_low,
                    "null_alpha_median": alpha_median,
                    "null_alpha_97p5": alpha_high,
                    "angular_esd_ks": distance,
                    "angular_esd_mc_p": probability,
                    "candidate_nonrandom_powerlaw": candidate,
                }
            )
            _angular_plot(
                name=name,
                kind=kind,
                actual=actual[kind],
                nulls=nulls[kind],
                final_step=resolved.final_step,
                output_dir=resolved.output_dir,
                show=config.show_plots,
            )
            _tail_plot(
                name=name,
                kind=kind,
                actual=actual[kind],
                nulls=nulls[kind],
                fit=fit,
                null_alpha_median=alpha_median,
                upper=upper,
                output_dir=resolved.output_dir,
                show=config.show_plots,
            )

        _radial_plot(
            name=name,
            initial=initial,
            final=final,
            config=config,
            layer_index=layer_index,
            resolved=resolved,
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        resolved.output_dir / "angular_initial_vs_final_summary.csv",
        index=False,
    )
    _summary_plots(results, resolved, config.show_plots)
    manifest = {
        "config": asdict(config),
        "run_dir": str(resolved.run_dir),
        "run_dir_source": resolved.run_dir_source,
        "initial_checkpoint": str(resolved.initial_path),
        "final_checkpoint": str(resolved.final_path),
        "final_step": resolved.final_step,
        "primary_null": "randomized_initial_to_fixed_final",
        "strict_saved_checkpoint_mode": True,
        **metadata,
    }
    (
        resolved.output_dir / "analysis_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return results, resolved, manifest

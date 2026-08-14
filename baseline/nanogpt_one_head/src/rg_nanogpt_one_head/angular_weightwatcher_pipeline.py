from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw

from .angular_weightwatcher_core import (
    AnalysisConfig,
    ResolvedRun,
    TailFit,
    angular_from_polar,
    angular_spectra,
    assert_gauge_invariance,
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


def _positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array) & (array > 0)]


def _powerlaw_fit(
    values: np.ndarray,
    min_tail: int,
) -> tuple[TailFit, powerlaw.Fit | None]:
    """Fit the largest values with the same powerlaw package used by WeightWatcher.

    powerlaw.Fit performs the Clauset-style MLE/KS search over candidate xmin
    values.  We do not supply xmin: the package chooses the tail start.  The
    min_tail check is applied only after that search so it cannot force the
    optimizer toward a hand-selected part of the spectrum.
    """
    array = _positive(values)
    if array.size < min_tail:
        return TailFit(False), None
    try:
        fit = powerlaw.Fit(
            array,
            discrete=False,
            verbose=False,
        )
        alpha = float(fit.power_law.alpha)
        xmin = float(fit.power_law.xmin)
        ks = float(fit.power_law.D)
        n_tail = int(np.count_nonzero(array >= xmin))
    except Exception:
        return TailFit(False), None
    if (
        not np.isfinite(alpha)
        or not np.isfinite(xmin)
        or not np.isfinite(ks)
        or n_tail < min_tail
    ):
        return TailFit(False), fit
    return TailFit(True, alpha, xmin, ks, n_tail), fit


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


def _powerlaw_native_plots(
    *,
    name: str,
    kind: str,
    projective_values: np.ndarray,
    package_fit: powerlaw.Fit | None,
    fit: TailFit,
    null_alpha_median: float,
    output_dir: Path,
    show: bool,
) -> None:
    """Use powerlaw's own PDF/CDF/CCDF plotting routines.

    These are deliberately the package-native diagnostics rather than a
    separately implemented histogram or fitted line.  The fitted distribution
    uses the same xmin and alpha selected by powerlaw.Fit.
    """
    data = _positive(projective_values)
    if package_fit is None or not data.size:
        return

    xlabel = "tan² θ" if kind == "tilt" else "tan²(φ/2)"
    suffix = (
        f"powerlaw MLE α={fit.alpha:.3f}, xmin={fit.xmin:.4g}, "
        f"n_tail={fit.n_tail}; null α median={null_alpha_median:.3f}"
        if fit.success
        else f"powerlaw MLE tail rejected; null α median={null_alpha_median:.3f}"
    )

    # Native logarithmic PDF/density plot.
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    package_fit.plot_pdf(ax=axis, label="empirical PDF")
    package_fit.power_law.plot_pdf(
        ax=axis,
        linestyle="--",
        label="power-law fit",
    )
    axis.set(
        xlabel=xlabel,
        ylabel="PDF / density",
        title=f"{name}: {kind} projective PDF (powerlaw)\n{suffix}",
    )
    axis.legend(fontsize=8)
    _finish(
        figure,
        output_dir,
        f"{name}_{kind}_powerlaw_pdf_loglog.png",
        show,
    )

    # Native PDF with linear bins, then shown on linear axes to make terminal
    # cutoff/rollover visible instead of hiding it in log-log compression.
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    powerlaw.plot_pdf(
        data,
        ax=axis,
        linear_bins=True,
        label="empirical PDF, linear bins",
    )
    axis.set_xscale("linear")
    axis.set_yscale("linear")
    axis.set(
        xlabel=xlabel,
        ylabel="PDF / density",
        title=f"{name}: {kind} projective PDF, linear scale (powerlaw)",
    )
    axis.legend(fontsize=8)
    _finish(
        figure,
        output_dir,
        f"{name}_{kind}_powerlaw_pdf_linear.png",
        show,
    )

    # Native CDF.
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    package_fit.plot_cdf(ax=axis, label="empirical CDF")
    package_fit.power_law.plot_cdf(
        ax=axis,
        linestyle="--",
        label="power-law fit",
    )
    axis.set(
        xlabel=xlabel,
        ylabel="CDF",
        title=f"{name}: {kind} projective CDF (powerlaw)",
    )
    axis.legend(fontsize=8)
    _finish(
        figure,
        output_dir,
        f"{name}_{kind}_powerlaw_cdf.png",
        show,
    )

    # Native CCDF.  This is the most direct visual check that the largest
    # elements form a long tail instead of rolling over rapidly at the end.
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    package_fit.plot_ccdf(ax=axis, label="empirical CCDF")
    package_fit.power_law.plot_ccdf(
        ax=axis,
        linestyle="--",
        label="power-law fit",
    )
    axis.set(
        xlabel=xlabel,
        ylabel="CCDF",
        title=f"{name}: {kind} projective CCDF (powerlaw)\n{suffix}",
    )
    axis.legend(fontsize=8)
    _finish(
        figure,
        output_dir,
        f"{name}_{kind}_powerlaw_ccdf_loglog.png",
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
        axis.set_ylabel("powerlaw.Fit MLE exponent")
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
            actual_projective = projective(actual[kind], upper)
            fit, package_fit = _powerlaw_fit(
                actual_projective,
                config.min_tail,
            )
            null_fit_pairs = [
                _powerlaw_fit(
                    projective(sample, upper),
                    config.min_tail,
                )
                for sample in nulls[kind]
            ]
            null_fits = [pair[0] for pair in null_fit_pairs]
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
                and np.isfinite(alpha_low)
                and np.isfinite(alpha_high)
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
                    "fit_backend": "powerlaw.Fit",
                    "fit_selection": "package MLE/KS xmin search; largest x >= xmin",
                    "actual_alpha": fit.alpha,
                    "actual_xmin": fit.xmin,
                    "actual_fit_ks": fit.ks,
                    "actual_tail_n": fit.n_tail,
                    "actual_tail_fraction": (
                        fit.n_tail / max(_positive(actual_projective).size, 1)
                        if fit.success
                        else np.nan
                    ),
                    "actual_xmax": (
                        float(np.max(_positive(actual_projective)))
                        if _positive(actual_projective).size
                        else np.nan
                    ),
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
            _powerlaw_native_plots(
                name=name,
                kind=kind,
                projective_values=actual_projective,
                package_fit=package_fit,
                fit=fit,
                null_alpha_median=alpha_median,
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
        "tail_fit_backend": "powerlaw.Fit",
        "tail_fit_selection": (
            "powerlaw package continuous MLE/KS search over xmin; "
            "tail is the largest projective angular values x >= xmin"
        ),
        "native_powerlaw_plots": [
            "PDF log-log",
            "PDF linear bins/linear axes",
            "CDF",
            "CCDF log-log",
        ],
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

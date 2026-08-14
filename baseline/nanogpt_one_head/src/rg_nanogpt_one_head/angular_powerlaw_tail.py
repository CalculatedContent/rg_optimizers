from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw

from .angular_weightwatcher_core import (
    AnalysisConfig,
    ResolvedRun,
    angular_from_polar,
    angular_spectra,
    load_weight_pairs,
    polar,
    projective,
    random_polar,
    resolve_run,
)


@dataclass(frozen=True)
class PowerLawTailFit:
    success: bool
    alpha: float = np.nan
    xmin: float = np.nan
    D: float = np.nan
    n_tail: int = 0
    n_total: int = 0
    xmax_observed: float = np.nan
    tail_decades: float = np.nan


def _positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.sort(values[np.isfinite(values) & (values > 0.0)])


def fit_powerlaw_tail(
    values: np.ndarray,
    *,
    min_tail: int = 20,
) -> tuple[PowerLawTailFit, powerlaw.Fit | None]:
    """Fit the far tail with the same powerlaw MLE/KS procedure as WeightWatcher.

    Scientific contract:

    * every positive value is passed to ``powerlaw.Fit``;
    * ``xmin`` is NOT supplied, so the package searches for the tail start;
    * ``xmax`` is NOT supplied, so the largest observed values are retained;
    * ``min_tail`` is checked only AFTER the package has chosen ``xmin``.

    Thus the fit always means ``x >= xmin`` through the largest observed value.
    """
    data = _positive(values)
    if data.size < min_tail:
        return PowerLawTailFit(False, n_total=int(data.size)), None

    try:
        package_fit = powerlaw.Fit(
            data,
            discrete=False,
            verbose=False,
        )
        alpha = float(package_fit.power_law.alpha)
        xmin = float(package_fit.power_law.xmin)
        D = float(package_fit.power_law.D)
        n_tail = int(np.count_nonzero(data >= xmin))
        xmax_observed = float(data[-1])
        tail_decades = (
            float(np.log10(xmax_observed / xmin))
            if xmin > 0.0 and xmax_observed >= xmin
            else np.nan
        )
    except Exception:
        return PowerLawTailFit(False, n_total=int(data.size)), None

    success = bool(
        np.isfinite(alpha)
        and np.isfinite(xmin)
        and np.isfinite(D)
        and n_tail >= min_tail
    )
    return (
        PowerLawTailFit(
            success=success,
            alpha=alpha,
            xmin=xmin,
            D=D,
            n_tail=n_tail,
            n_total=int(data.size),
            xmax_observed=xmax_observed,
            tail_decades=tail_decades,
        ),
        package_fit,
    )


def _finish(fig, path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _empirical_ccdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    data = _positive(values)
    if data.size == 0:
        return np.full(grid.shape, np.nan, dtype=np.float64)
    return np.asarray([np.mean(data >= x) for x in grid], dtype=np.float64)


def _native_powerlaw_plots(
    *,
    name: str,
    kind: str,
    values: np.ndarray,
    fit: PowerLawTailFit,
    package_fit: powerlaw.Fit | None,
    output_dir: Path,
    show: bool,
) -> None:
    if package_fit is None:
        return

    xlabel = "tan^2(theta)" if kind == "tilt" else "tan^2(phi/2)"
    suffix = (
        f"alpha={fit.alpha:.3f}, xmin={fit.xmin:.4g}, "
        f"n_tail={fit.n_tail}, decades={fit.tail_decades:.2f}"
        if fit.success
        else "powerlaw.Fit did not yield an accepted tail"
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    package_fit.plot_pdf(ax=ax, label="empirical PDF")
    package_fit.power_law.plot_pdf(ax=ax, linestyle="--", label="power-law fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDF / density")
    ax.set_title(f"{name}: {kind} angular PDF (powerlaw)\n{suffix}")
    ax.legend(fontsize=8)
    _finish(fig, output_dir / f"{name}_{kind}_powerlaw_pdf_loglog.png", show)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    powerlaw.plot_pdf(
        _positive(values),
        ax=ax,
        linear_bins=True,
        label="empirical PDF, linear bins",
    )
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDF / density")
    ax.set_title(f"{name}: {kind} angular PDF, linear scale")
    ax.legend(fontsize=8)
    _finish(fig, output_dir / f"{name}_{kind}_powerlaw_pdf_linear.png", show)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    package_fit.plot_cdf(ax=ax, label="empirical CDF")
    package_fit.power_law.plot_cdf(ax=ax, linestyle="--", label="power-law fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(f"{name}: {kind} angular CDF (powerlaw)")
    ax.legend(fontsize=8)
    _finish(fig, output_dir / f"{name}_{kind}_powerlaw_cdf.png", show)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    package_fit.plot_ccdf(ax=ax, label="empirical CCDF")
    package_fit.power_law.plot_ccdf(ax=ax, linestyle="--", label="power-law fit")
    if fit.success:
        ax.axvline(fit.xmin, linestyle=":", label=f"MLE xmin={fit.xmin:.4g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CCDF")
    ax.set_title(f"{name}: {kind} angular CCDF (powerlaw)\n{suffix}")
    ax.legend(fontsize=8)
    _finish(fig, output_dir / f"{name}_{kind}_powerlaw_ccdf_loglog.png", show)


def _tail_zoom_plot(
    *,
    name: str,
    kind: str,
    actual_values: np.ndarray,
    actual_fit: PowerLawTailFit,
    package_fit: powerlaw.Fit | None,
    null_values: list[np.ndarray],
    null_fits: list[PowerLawTailFit],
    output_dir: Path,
    show: bool,
) -> None:
    """Zoom the far tail without changing what is supplied to the MLE fit."""
    actual = _positive(actual_values)
    null_positive = [_positive(sample) for sample in null_values]
    null_positive = [sample for sample in null_positive if sample.size]
    if actual.size == 0 or not null_positive:
        return

    null_xmins = np.asarray(
        [item.xmin for item in null_fits if item.success and np.isfinite(item.xmin)],
        dtype=np.float64,
    )
    null_xmin_median = (
        float(np.median(null_xmins)) if null_xmins.size else float(actual[0])
    )
    actual_xmin = actual_fit.xmin if actual_fit.success else float(actual[0])

    # This lower bound affects only the display zoom.  It is never passed to
    # powerlaw.Fit and therefore cannot change the selected tail.
    display_xmin = max(min(actual_xmin, null_xmin_median), float(actual[0]))
    xmax = max(float(actual[-1]), *(float(sample[-1]) for sample in null_positive))
    if xmax <= display_xmin:
        return
    grid = np.geomspace(display_xmin, xmax, 240)

    null_ccdf = np.stack(
        [_empirical_ccdf(sample, grid) for sample in null_positive],
        axis=0,
    )
    low, median, high = np.nanquantile(null_ccdf, [0.025, 0.5, 0.975], axis=0)
    actual_ccdf = _empirical_ccdf(actual, grid)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.loglog(grid, actual_ccdf, label="trained initial -> final")
    ax.loglog(grid, median, linestyle="--", label="random-angular null median")
    ax.fill_between(
        grid,
        np.maximum(low, 1e-12),
        np.maximum(high, 1e-12),
        alpha=0.2,
        label="random-angular null 95% interval",
    )
    if package_fit is not None and actual_fit.success:
        package_fit.power_law.plot_ccdf(
            ax=ax,
            linestyle="-.",
            label="powerlaw.Fit tail",
        )
        ax.axvline(
            actual_fit.xmin,
            linestyle=":",
            label=f"trained MLE xmin={actual_fit.xmin:.4g}",
        )
    if np.isfinite(null_xmin_median):
        ax.axvline(
            null_xmin_median,
            linestyle="--",
            linewidth=1.0,
            label=f"null median xmin={null_xmin_median:.4g}",
        )
    ax.set_xlim(display_xmin, xmax)
    ax.set_ylim(bottom=max(np.nanmin(low[low > 0]) if np.any(low > 0) else 1e-6, 1e-6))
    ax.set_xlabel("projective angular value x")
    ax.set_ylabel("CCDF")
    ax.set_title(
        f"{name}: {kind} FAR-TAIL ZOOM\n"
        "fit uses all positive values; no manual xmin or xmax"
    )
    ax.legend(fontsize=8)
    _finish(fig, output_dir / f"{name}_{kind}_far_tail_zoom_ccdf.png", show)


def run_powerlaw_tail_analysis(
    config: AnalysisConfig | None = None,
) -> tuple[pd.DataFrame, ResolvedRun]:
    """Run the canonical angular far-tail PL-vs-random-null test for all layers."""
    config = config or AnalysisConfig.from_env()
    config.validate()
    resolved = resolve_run(config)
    initial_weights, final_weights, _ = load_weight_pairs(config, resolved)

    rows: list[dict] = []
    for layer_index, name in enumerate(sorted(initial_weights)):
        initial = initial_weights[name]
        final = final_weights[name]
        actual = angular_spectra(initial, final)
        final_polar = polar(final)
        rng = np.random.default_rng(config.null_seed + 17011 * (layer_index + 1))

        null_spectra = {"tilt": [], "twist": []}
        for _ in range(config.angular_nulls):
            sample = angular_from_polar(
                random_polar(initial.shape, rng),
                final_polar,
            )
            for kind in null_spectra:
                null_spectra[kind].append(sample[kind])

        for kind, upper in (("tilt", 1.0), ("twist", 4.0)):
            actual_projective = projective(actual[kind], upper)
            actual_fit, package_fit = fit_powerlaw_tail(
                actual_projective,
                min_tail=config.min_tail,
            )
            null_projective = [projective(sample, upper) for sample in null_spectra[kind]]
            null_fit_pairs = [
                fit_powerlaw_tail(sample, min_tail=config.min_tail)
                for sample in null_projective
            ]
            null_fits = [pair[0] for pair in null_fit_pairs]
            good_nulls = [item for item in null_fits if item.success]
            null_alphas = np.asarray([item.alpha for item in good_nulls], dtype=np.float64)
            null_xmins = np.asarray([item.xmin for item in good_nulls], dtype=np.float64)
            null_decades = np.asarray([item.tail_decades for item in good_nulls], dtype=np.float64)

            def interval(values: np.ndarray) -> tuple[float, float, float]:
                if values.size == 0:
                    return np.nan, np.nan, np.nan
                return tuple(float(x) for x in np.quantile(values, [0.025, 0.5, 0.975]))

            alpha_low, alpha_med, alpha_high = interval(null_alphas)
            xmin_low, xmin_med, xmin_high = interval(null_xmins)
            decades_low, decades_med, decades_high = interval(null_decades)

            rows.append(
                {
                    "matrix_name": name,
                    "angular_type": kind,
                    "fit_backend": "powerlaw.Fit",
                    "fit_contract": "all positive x; package selects xmin; no xmax; fit largest x >= xmin",
                    "actual_alpha": actual_fit.alpha,
                    "actual_xmin": actual_fit.xmin,
                    "actual_D": actual_fit.D,
                    "actual_tail_n": actual_fit.n_tail,
                    "actual_total_n": actual_fit.n_total,
                    "actual_xmax_observed": actual_fit.xmax_observed,
                    "actual_tail_decades": actual_fit.tail_decades,
                    "null_alpha_2p5": alpha_low,
                    "null_alpha_median": alpha_med,
                    "null_alpha_97p5": alpha_high,
                    "null_xmin_2p5": xmin_low,
                    "null_xmin_median": xmin_med,
                    "null_xmin_97p5": xmin_high,
                    "null_tail_decades_2p5": decades_low,
                    "null_tail_decades_median": decades_med,
                    "null_tail_decades_97p5": decades_high,
                    "actual_alpha_outside_null": bool(
                        actual_fit.success
                        and np.isfinite(alpha_low)
                        and np.isfinite(alpha_high)
                        and (actual_fit.alpha < alpha_low or actual_fit.alpha > alpha_high)
                    ),
                    "actual_tail_longer_than_null_97p5": bool(
                        actual_fit.success
                        and np.isfinite(decades_high)
                        and actual_fit.tail_decades > decades_high
                    ),
                }
            )

            _native_powerlaw_plots(
                name=name,
                kind=kind,
                values=actual_projective,
                fit=actual_fit,
                package_fit=package_fit,
                output_dir=resolved.output_dir,
                show=config.show_plots,
            )
            _tail_zoom_plot(
                name=name,
                kind=kind,
                actual_values=actual_projective,
                actual_fit=actual_fit,
                package_fit=package_fit,
                null_values=null_projective,
                null_fits=null_fits,
                output_dir=resolved.output_dir,
                show=config.show_plots,
            )

    results = pd.DataFrame(rows)
    results.to_csv(
        resolved.output_dir / "angular_powerlaw_far_tail_summary.csv",
        index=False,
    )
    return results, resolved

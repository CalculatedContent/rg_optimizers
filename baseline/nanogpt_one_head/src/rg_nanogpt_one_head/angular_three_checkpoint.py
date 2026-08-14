from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .angular_powerlaw_tail import (
    PowerLawTailFit,
    _native_powerlaw_plots,
    fit_powerlaw_tail,
)
from .angular_weightwatcher_core import (
    AnalysisConfig,
    _build_model,
    _extract_weights,
    _load_payload,
    _model_config,
    angular_from_polar,
    angular_spectra,
    polar,
    random_polar,
    resolve_run,
)


PAIR_ORDER = (
    ("initial", "best"),
    ("initial", "final"),
    ("best", "final"),
)

DEFAULT_ENDPOINT_TOL = 1e-10
MIN_KS_SAMPLES = 3


def _finish(fig, path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.sort(values[np.isfinite(values) & (values > 0.0)])


def _payload_step(path: Path) -> int:
    payload = _load_payload(path)
    return int(payload.get("step", -1))


def _best_checkpoint_path(config: AnalysisConfig, run_dir: Path) -> Path:
    best_override = os.environ.get("BEST_CHECKPOINT_PATH", "").strip()
    path = (
        Path(best_override).expanduser().resolve()
        if best_override
        else run_dir / "checkpoint_best.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Best checkpoint is unavailable. Expected checkpoint_best.pt or "
            "set BEST_CHECKPOINT_PATH explicitly: " + str(path)
        )
    return path


def _manifest_max_steps(run_dir: Path) -> int | None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = [
        manifest.get("max_steps"),
        manifest.get("training", {}).get("max_steps")
        if isinstance(manifest.get("training"), dict)
        else None,
        manifest.get("config", {}).get("training", {}).get("max_steps")
        if isinstance(manifest.get("config"), dict)
        and isinstance(manifest.get("config", {}).get("training"), dict)
        else None,
    ]
    for value in candidates:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def resolve_three_checkpoints(config: AnalysisConfig):
    """Resolve actual saved initial, best, and final checkpoints.

    The routine is deliberately strict:

    * ``initial`` must be a real saved step-zero checkpoint;
    * ``best`` must be ``checkpoint_best.pt`` or ``BEST_CHECKPOINT_PATH``;
    * ``final`` must be the strict final checkpoint resolved by ``resolve_run``;
    * no state is reconstructed from a seed and no checkpoint is silently
      substituted for another;
    * checkpoint steps must satisfy ``0 <= best <= final``.

    ``best`` is allowed to occur at the final step because the best-validation
    model can legitimately be the completed model.  That case is recorded in
    the manifest and the ``best->final`` flow may consequently be trivial.
    """
    resolved = resolve_run(config)
    best_path = _best_checkpoint_path(config, resolved.run_dir)

    paths = {
        "initial": resolved.initial_path,
        "best": best_path,
        "final": resolved.final_path,
    }
    steps = {name: _payload_step(path) for name, path in paths.items()}

    if steps["initial"] != 0:
        raise ValueError(
            f"Initial checkpoint must be step 0, got {steps['initial']}"
        )
    if steps["best"] < 0 or steps["final"] <= 0:
        raise ValueError(f"Invalid checkpoint steps: {steps}")
    if steps["best"] > steps["final"]:
        raise ValueError(
            "Best checkpoint occurs after final checkpoint: " + str(steps)
        )

    expected_final = _manifest_max_steps(resolved.run_dir)
    if expected_final is not None and steps["final"] != expected_final:
        raise ValueError(
            "checkpoint_final.pt does not match manifest max_steps: "
            f"final step={steps['final']}, expected={expected_final}"
        )

    return resolved, paths, steps


def _load_three_weight_sets(config: AnalysisConfig, resolved, paths):
    payloads = {name: _load_payload(path) for name, path in paths.items()}

    seeds = {
        int(payload.get("seed", config.seed))
        for payload in payloads.values()
    }
    if len(seeds) != 1:
        raise ValueError(f"Checkpoint seeds differ: {sorted(seeds)}")

    optimizers = {
        str(payload.get("optimizer_name", config.optimizer)).lower()
        for payload in payloads.values()
    }
    if len(optimizers) != 1 or next(iter(optimizers)) != config.optimizer:
        raise ValueError(
            f"Checkpoint optimizers={sorted(optimizers)!r}, "
            f"target={config.optimizer!r}"
        )

    model_cfg = _model_config(
        payloads["initial"],
        payloads["final"],
        resolved.run_dir,
    )
    weights = {}
    for state, payload in payloads.items():
        model = _build_model(payload, model_cfg)
        weights[state] = _extract_weights(model)

    inventories = [set(item) for item in weights.values()]
    if not all(inventory == inventories[0] for inventory in inventories[1:]):
        raise RuntimeError("Initial, best, and final matrix inventories differ")

    for matrix_name in inventories[0]:
        shapes = {
            tuple(weights[state][matrix_name].shape)
            for state in ("initial", "best", "final")
        }
        if len(shapes) != 1:
            raise RuntimeError(
                f"Matrix shape changed across checkpoints for {matrix_name}: "
                f"{sorted(shapes)}"
            )

    return weights, payloads, model_cfg


def projective_continuous(
    values: np.ndarray,
    upper: float,
    *,
    endpoint_tol: float = DEFAULT_ENDPOINT_TOL,
) -> tuple[np.ndarray, int, int]:
    """Map the continuous angular spectrum to projective coordinates.

    For ``y=lambda/upper`` the projective coordinate is

    ``x = y / (1-y)``.

    The exact upper endpoint is *not* part of the continuous tail.  For twist,
    ``lambda=4`` is the ``-1`` eigenvalue of the relative orthogonal rotation
    and can be a discrete reflection/parity atom.  Converting that atom by
    clipping ``y`` to ``1-1e-12`` would manufacture a fake value around 1e12
    and can dominate ``powerlaw.Fit``.  We therefore remove eigenvalues within
    ``endpoint_tol`` of the upper endpoint from the continuous sample and
    report their count separately.

    Values at the lower endpoint are also excluded because ``powerlaw.Fit``
    operates on strictly positive tail observations.
    """
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return np.asarray([], dtype=np.float64), 0, 0

    tol = max(float(endpoint_tol), 0.0)
    scaled = finite / float(upper)
    endpoint_mask = scaled >= (1.0 - tol)
    zero_mask = scaled <= tol
    interior = scaled[~endpoint_mask & ~zero_mask]
    interior = interior[(interior > 0.0) & (interior < 1.0)]
    projective_values = interior / (1.0 - interior)
    return (
        np.sort(projective_values.astype(np.float64, copy=False)),
        int(np.count_nonzero(endpoint_mask)),
        int(np.count_nonzero(zero_mask)),
    )


def _null_interval(values: list[float]) -> tuple[float, float, float]:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    q = np.quantile(finite, [0.025, 0.5, 0.975])
    return float(q[0]), float(q[1]), float(q[2])


def _null_metric_fields(prefix: str, values: list[float]) -> dict:
    low, median, high = _null_interval(values)
    return {
        f"null_{prefix}_2p5": low,
        f"null_{prefix}_median": median,
        f"null_{prefix}_97p5": high,
    }


def _two_sided_null_p(value: float, null_values: list[float]) -> float:
    finite = np.asarray(
        [item for item in null_values if np.isfinite(item)],
        dtype=float,
    )
    if finite.size == 0 or not np.isfinite(value):
        return np.nan
    center = float(np.median(finite))
    observed_distance = abs(float(value) - center)
    null_distance = np.abs(finite - center)
    return float(
        (1.0 + np.count_nonzero(null_distance >= observed_distance))
        / (finite.size + 1.0)
    )


def _ccdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    data = _positive(values)
    if data.size == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    return np.asarray([np.mean(data >= x) for x in grid], dtype=float)


def _monte_carlo_ks_variable(
    actual: np.ndarray,
    nulls: list[np.ndarray],
) -> tuple[float, float]:
    """Monte-Carlo KS test for samples whose lengths may differ."""
    actual = _positive(actual)
    nulls = [_positive(item) for item in nulls]
    nulls = [item for item in nulls if item.size >= MIN_KS_SAMPLES]
    if actual.size < MIN_KS_SAMPLES or len(nulls) < 3:
        return np.nan, np.nan

    pooled = np.concatenate(nulls)
    observed = float(stats.ks_2samp(actual, pooled).statistic)
    reference = []
    for index, sample in enumerate(nulls):
        others = nulls[:index] + nulls[index + 1 :]
        if not others:
            continue
        reference.append(
            float(stats.ks_2samp(sample, np.concatenate(others)).statistic)
        )
    if not reference:
        return observed, np.nan
    p_value = float(
        (1.0 + np.count_nonzero(np.asarray(reference) >= observed))
        / (len(reference) + 1.0)
    )
    return observed, p_value


def _monte_carlo_tail_ks(
    actual: np.ndarray,
    actual_xmin: float,
    nulls: list[np.ndarray],
) -> tuple[float, float, int]:
    """Compare the far tail above the trained package-selected xmin.

    The threshold is chosen only by the trained ``powerlaw.Fit``.  It is then
    applied identically to the actual and null samples for this diagnostic.
    This does not alter any power-law fit; it only asks whether the empirical
    trained tail distribution differs from the matched random-angular tail at
    the same physical threshold.
    """
    if not np.isfinite(actual_xmin) or actual_xmin <= 0.0:
        return np.nan, np.nan, 0
    actual_tail = _positive(actual)
    actual_tail = actual_tail[actual_tail >= actual_xmin]
    null_tails = []
    for item in nulls:
        tail = _positive(item)
        tail = tail[tail >= actual_xmin]
        if tail.size >= MIN_KS_SAMPLES:
            null_tails.append(tail)
    D, p = _monte_carlo_ks_variable(actual_tail, null_tails)
    return D, p, len(null_tails)


def _far_tail_pair_plot(
    *,
    matrix_name: str,
    kind: str,
    pair_label: str,
    actual_values: np.ndarray,
    actual_fit: PowerLawTailFit,
    package_fit,
    null_values: list[np.ndarray],
    null_fits: list[PowerLawTailFit],
    output_dir: Path,
    show: bool,
) -> None:
    actual = _positive(actual_values)
    null_positive = [_positive(item) for item in null_values]
    null_positive = [item for item in null_positive if item.size]
    if actual.size == 0 or not null_positive:
        return

    null_xmins = np.asarray(
        [fit.xmin for fit in null_fits if fit.success and np.isfinite(fit.xmin)],
        dtype=float,
    )
    null_xmin_median = (
        float(np.median(null_xmins)) if null_xmins.size else float(actual[0])
    )
    actual_xmin = actual_fit.xmin if actual_fit.success else float(actual[0])

    # Display-only zoom.  It is never passed into powerlaw.Fit.
    display_xmin = max(min(actual_xmin, null_xmin_median), float(actual[0]))
    xmax = max(float(actual[-1]), *(float(item[-1]) for item in null_positive))
    if xmax <= display_xmin:
        return
    grid = np.geomspace(display_xmin, xmax, 240)
    null_ccdfs = np.stack([_ccdf(item, grid) for item in null_positive])
    low, median, high = np.nanquantile(
        null_ccdfs,
        [0.025, 0.5, 0.975],
        axis=0,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.loglog(grid, _ccdf(actual, grid), label=f"actual {pair_label}")
    ax.loglog(grid, median, "--", label="matched random-angular median")
    ax.fill_between(
        grid,
        np.maximum(low, 1e-12),
        np.maximum(high, 1e-12),
        alpha=0.20,
        label="matched random-angular 95% interval",
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
            label=f"MLE xmin={actual_fit.xmin:.4g}",
        )
    if np.isfinite(null_xmin_median):
        ax.axvline(
            null_xmin_median,
            linestyle="--",
            linewidth=1.0,
            label=f"null median xmin={null_xmin_median:.4g}",
        )
    ax.set_xlim(display_xmin, xmax)
    ax.set_xlabel("projective angular value x")
    ax.set_ylabel("CCDF")
    ax.set_title(
        f"{matrix_name}: {kind} far-tail zoom, {pair_label}\n"
        "continuous sector only; powerlaw.Fit sees all positive x"
    )
    ax.legend(fontsize=8)
    safe_pair = pair_label.replace("->", "_to_")
    _finish(
        fig,
        output_dir
        / f"{matrix_name}_{kind}_{safe_pair}_far_tail_zoom_ccdf.png",
        show,
    )


def _radial_three_state_plot(matrix_name, weight_sets, output_dir, show):
    spectra = {}
    for state in ("initial", "best", "final"):
        s = np.linalg.svd(weight_sets[state][matrix_name], compute_uv=False)
        values = np.sort(s * s)[::-1]
        values = values / max(float(np.mean(values)), 1e-12)
        spectra[state] = values
    rank = np.arange(1, len(spectra["initial"]) + 1)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for state in ("initial", "best", "final"):
        ax.loglog(rank, spectra[state], label=state)
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("Gram eigenvalue / mean")
    ax.set_title(f"{matrix_name}: radial spectra, initial / best / final")
    ax.legend()
    _finish(
        fig,
        output_dir / f"{matrix_name}_radial_initial_best_final.png",
        show,
    )


def _pairwise_metric_plot(
    matrix_name: str,
    kind: str,
    frame: pd.DataFrame,
    output_dir: Path,
    show: bool,
    *,
    actual_column: str,
    null_prefix: str,
    ylabel: str,
    filename_suffix: str,
    reference: float | None = None,
) -> None:
    labels = [f"{a}->{b}" for a, b in PAIR_ORDER]
    subset = frame[
        (frame.matrix_name == matrix_name)
        & (frame.angular_type == kind)
    ].set_index("pair").reindex(labels)
    x = np.arange(len(labels), dtype=float)
    actual = pd.to_numeric(
        subset[actual_column], errors="coerce"
    ).to_numpy(dtype=float)
    med = pd.to_numeric(
        subset[f"null_{null_prefix}_median"], errors="coerce"
    ).to_numpy(dtype=float)
    low = pd.to_numeric(
        subset[f"null_{null_prefix}_2p5"], errors="coerce"
    ).to_numpy(dtype=float)
    high = pd.to_numeric(
        subset[f"null_{null_prefix}_97p5"], errors="coerce"
    ).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    finite_null = np.isfinite(low) & np.isfinite(med) & np.isfinite(high)
    if finite_null.any():
        ax.errorbar(
            x[finite_null],
            med[finite_null],
            yerr=np.vstack(
                [
                    med[finite_null] - low[finite_null],
                    high[finite_null] - med[finite_null],
                ]
            ),
            fmt="o",
            capsize=4,
            label="random-angular median and 95% interval",
        )
    finite_actual = np.isfinite(actual)
    if finite_actual.any():
        ax.scatter(
            x[finite_actual],
            actual[finite_actual],
            marker="x",
            s=80,
            label="actual",
        )
    if reference is not None:
        ax.axhline(
            reference,
            linestyle="--",
            linewidth=1.0,
            label=f"reference = {reference:g}",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{matrix_name}: {kind} pairwise {filename_suffix.replace('_', ' ')}"
    )
    ax.legend(fontsize=8)
    _finish(
        fig,
        output_dir / f"{matrix_name}_{kind}_{filename_suffix}.png",
        show,
    )


def _pair_is_identical(
    weights: dict[str, dict[str, np.ndarray]],
    matrix_name: str,
    reference_state: str,
    target_state: str,
) -> bool:
    return bool(
        np.array_equal(
            weights[reference_state][matrix_name],
            weights[target_state][matrix_name],
        )
    )


def run_three_checkpoint_analysis(
    config: AnalysisConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Analyze actual initial, best, and final checkpoints versus angular nulls.

    For every layer, pair, and angular sector this routine:

    1. computes the gauge-invariant angular spectrum;
    2. removes exact/near-exact endpoint atoms from the *continuous* projective
       sample and reports those atoms separately;
    3. gives every positive continuous projective value to ``powerlaw.Fit``
       with no supplied ``xmin`` or ``xmax``;
    4. performs the identical fit on matched Haar/Stiefel random-angular nulls;
    5. compares alpha, xmin, KS D, tail population, tail extent, endpoint atoms,
       the full continuous ESD, and the conditional far tail against the null.
    """
    config = config or AnalysisConfig.from_env()
    config.validate()
    resolved, paths, steps = resolve_three_checkpoints(config)
    weights, payloads, model_cfg = _load_three_weight_sets(
        config,
        resolved,
        paths,
    )

    endpoint_tol = float(
        os.environ.get("ANGULAR_ENDPOINT_TOL", str(DEFAULT_ENDPOINT_TOL))
    )
    if endpoint_tol < 0.0 or endpoint_tol >= 0.1:
        raise ValueError(
            "ANGULAR_ENDPOINT_TOL must satisfy 0 <= tol < 0.1"
        )

    output_dir = resolved.output_dir / "initial_best_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer_index, matrix_name in enumerate(sorted(weights["initial"])):
        _radial_three_state_plot(
            matrix_name,
            weights,
            output_dir,
            config.show_plots,
        )

        for pair_index, (reference_state, target_state) in enumerate(PAIR_ORDER):
            reference = weights[reference_state][matrix_name]
            target = weights[target_state][matrix_name]
            actual = angular_spectra(reference, target)
            target_polar = polar(target)
            pair_identical = _pair_is_identical(
                weights,
                matrix_name,
                reference_state,
                target_state,
            )

            rng = np.random.default_rng(
                config.null_seed
                + 100003 * (layer_index + 1)
                + 1009 * pair_index
            )
            null_spectra = {"tilt": [], "twist": []}
            for _ in range(config.angular_nulls):
                random_reference = random_polar(reference.shape, rng)
                sample = angular_from_polar(random_reference, target_polar)
                for kind in null_spectra:
                    null_spectra[kind].append(sample[kind])

            pair_label = f"{reference_state}->{target_state}"
            file_pair = f"{reference_state}_to_{target_state}"

            for kind, upper in (("tilt", 1.0), ("twist", 4.0)):
                actual_values, actual_endpoint_atoms, actual_zero_atoms = (
                    projective_continuous(
                        actual[kind],
                        upper,
                        endpoint_tol=endpoint_tol,
                    )
                )
                actual_fit, package_fit = fit_powerlaw_tail(
                    actual_values,
                    min_tail=config.min_tail,
                )

                null_values = []
                null_endpoint_atoms = []
                null_zero_atoms = []
                for sample in null_spectra[kind]:
                    values, endpoint_atoms, zero_atoms = projective_continuous(
                        sample,
                        upper,
                        endpoint_tol=endpoint_tol,
                    )
                    null_values.append(values)
                    null_endpoint_atoms.append(float(endpoint_atoms))
                    null_zero_atoms.append(float(zero_atoms))

                null_fit_pairs = [
                    fit_powerlaw_tail(
                        sample,
                        min_tail=config.min_tail,
                    )
                    for sample in null_values
                ]
                null_fits = [item[0] for item in null_fit_pairs]
                successful_null_fits = [fit for fit in null_fits if fit.success]

                null_metrics = {
                    "alpha": [fit.alpha for fit in successful_null_fits],
                    "xmin": [fit.xmin for fit in successful_null_fits],
                    "D": [fit.D for fit in successful_null_fits],
                    "tail_n": [float(fit.n_tail) for fit in successful_null_fits],
                    "tail_decades": [
                        fit.tail_decades for fit in successful_null_fits
                    ],
                    "xmax_observed": [
                        fit.xmax_observed for fit in successful_null_fits
                    ],
                    "endpoint_atoms": null_endpoint_atoms,
                    "zero_atoms": null_zero_atoms,
                }

                full_ks_D, full_ks_p = _monte_carlo_ks_variable(
                    actual_values,
                    null_values,
                )
                tail_ks_D, tail_ks_p, tail_null_replicates = (
                    _monte_carlo_tail_ks(
                        actual_values,
                        actual_fit.xmin,
                        null_values,
                    )
                )

                row = {
                    "matrix_name": matrix_name,
                    "angular_type": kind,
                    "pair": pair_label,
                    "reference_state": reference_state,
                    "target_state": target_state,
                    "reference_step": steps[reference_state],
                    "target_step": steps[target_state],
                    "pair_identical_weights": pair_identical,
                    "fit_backend": "powerlaw.Fit",
                    "fit_selection": (
                        "all positive continuous projective x; package MLE/KS "
                        "xmin; no xmax; endpoint atoms excluded and counted"
                    ),
                    "endpoint_tol": endpoint_tol,
                    "actual_fit_success": actual_fit.success,
                    "actual_alpha": actual_fit.alpha,
                    "actual_xmin": actual_fit.xmin,
                    "actual_D": actual_fit.D,
                    "actual_tail_n": actual_fit.n_tail,
                    "actual_tail_decades": actual_fit.tail_decades,
                    "actual_xmax_observed": actual_fit.xmax_observed,
                    "actual_continuous_n": int(actual_values.size),
                    "actual_endpoint_atoms": actual_endpoint_atoms,
                    "actual_zero_atoms": actual_zero_atoms,
                    "full_continuous_ks_D": full_ks_D,
                    "full_continuous_ks_mc_p": full_ks_p,
                    "tail_conditional_ks_D": tail_ks_D,
                    "tail_conditional_ks_mc_p": tail_ks_p,
                    "tail_conditional_null_replicates": tail_null_replicates,
                }
                for metric_name, metric_values in null_metrics.items():
                    row.update(_null_metric_fields(metric_name, metric_values))

                row["alpha_null_two_sided_p"] = _two_sided_null_p(
                    actual_fit.alpha,
                    null_metrics["alpha"],
                )
                row["tail_decades_null_two_sided_p"] = _two_sided_null_p(
                    actual_fit.tail_decades,
                    null_metrics["tail_decades"],
                )
                row["alpha_outside_random_null"] = bool(
                    actual_fit.success
                    and np.isfinite(row["null_alpha_2p5"])
                    and np.isfinite(row["null_alpha_97p5"])
                    and (
                        actual_fit.alpha < row["null_alpha_2p5"]
                        or actual_fit.alpha > row["null_alpha_97p5"]
                    )
                )
                row["tail_longer_than_random_97p5"] = bool(
                    actual_fit.success
                    and np.isfinite(row["null_tail_decades_97p5"])
                    and actual_fit.tail_decades
                    > row["null_tail_decades_97p5"]
                )
                row["candidate_nonrandom_long_tail"] = bool(
                    actual_fit.success
                    and np.isfinite(tail_ks_p)
                    and tail_ks_p < 0.05
                    and row["tail_longer_than_random_97p5"]
                )
                rows.append(row)

                plot_name = f"{matrix_name}__{file_pair}"
                _native_powerlaw_plots(
                    name=plot_name,
                    kind=kind,
                    values=actual_values,
                    fit=actual_fit,
                    package_fit=package_fit,
                    output_dir=output_dir,
                    show=config.show_plots,
                )
                _far_tail_pair_plot(
                    matrix_name=matrix_name,
                    kind=kind,
                    pair_label=pair_label,
                    actual_values=actual_values,
                    actual_fit=actual_fit,
                    package_fit=package_fit,
                    null_values=null_values,
                    null_fits=null_fits,
                    output_dir=output_dir,
                    show=config.show_plots,
                )

    results = pd.DataFrame(rows)
    summary_path = output_dir / "angular_initial_best_final_powerlaw_summary.csv"
    results.to_csv(summary_path, index=False)

    for matrix_name in sorted(weights["initial"]):
        for kind in ("tilt", "twist"):
            _pairwise_metric_plot(
                matrix_name,
                kind,
                results,
                output_dir,
                config.show_plots,
                actual_column="actual_alpha",
                null_prefix="alpha",
                ylabel="powerlaw.Fit alpha",
                filename_suffix="pairwise_alpha_vs_random",
                reference=2.0,
            )
            _pairwise_metric_plot(
                matrix_name,
                kind,
                results,
                output_dir,
                config.show_plots,
                actual_column="actual_tail_decades",
                null_prefix="tail_decades",
                ylabel="tail length, log10(xmax/xmin)",
                filename_suffix="pairwise_tail_decades_vs_random",
            )
            _pairwise_metric_plot(
                matrix_name,
                kind,
                results,
                output_dir,
                config.show_plots,
                actual_column="actual_D",
                null_prefix="D",
                ylabel="powerlaw KS D",
                filename_suffix="pairwise_powerlaw_D_vs_random",
            )
            _pairwise_metric_plot(
                matrix_name,
                kind,
                results,
                output_dir,
                config.show_plots,
                actual_column="actual_xmin",
                null_prefix="xmin",
                ylabel="package-selected xmin",
                filename_suffix="pairwise_xmin_vs_random",
            )

    best_equals_final_step = steps["best"] == steps["final"]
    manifest = {
        "seed": int(payloads["final"].get("seed", config.seed)),
        "optimizer": config.optimizer,
        "checkpoints": {name: str(path) for name, path in paths.items()},
        "steps": steps,
        "best_equals_final_step": best_equals_final_step,
        "model_config": model_cfg,
        "angular_nulls": config.angular_nulls,
        "endpoint_tol": endpoint_tol,
        "endpoint_policy": (
            "eigenvalues within endpoint_tol of the bounded upper angular "
            "endpoint are treated as discrete atoms, counted separately, and "
            "never converted into projective tail values"
        ),
        "fit_backend": "powerlaw.Fit",
        "fit_contract": (
            "all positive continuous projective x; xmin chosen by package "
            "MLE/KS; no xmax; largest continuous observed x retained"
        ),
        "null_contract": (
            "independent Haar/Stiefel reference polar factor against the fixed "
            "target checkpoint; identical endpoint removal and powerlaw fit"
        ),
        "pairs": [f"{a}->{b}" for a, b in PAIR_ORDER],
        "summary_csv": str(summary_path),
        "output_dir": str(output_dir),
    }
    (
        output_dir / "angular_initial_best_final_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return results, manifest

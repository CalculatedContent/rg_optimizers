from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    projective,
    random_polar,
    resolve_run,
)


PAIR_ORDER = (
    ("initial", "best"),
    ("initial", "final"),
    ("best", "final"),
)


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


def resolve_three_checkpoints(config: AnalysisConfig):
    """Resolve actual saved initial, best, and final checkpoints.

    Initial and final use the strict shared resolver. Best defaults to
    checkpoint_best.pt and may be overridden with BEST_CHECKPOINT_PATH.
    No checkpoint is reconstructed from a seed and no endpoint is silently
    substituted for another.
    """
    resolved = resolve_run(config)
    best_override = os.environ.get("BEST_CHECKPOINT_PATH", "").strip()
    best_path = (
        Path(best_override).expanduser().resolve()
        if best_override
        else resolved.run_dir / "checkpoint_best.pt"
    )
    if not best_path.is_file():
        raise FileNotFoundError(
            "Best checkpoint is unavailable. Expected checkpoint_best.pt or "
            "set BEST_CHECKPOINT_PATH explicitly: " + str(best_path)
        )

    paths = {
        "initial": resolved.initial_path,
        "best": best_path,
        "final": resolved.final_path,
    }
    steps = {name: _payload_step(path) for name, path in paths.items()}
    if steps["initial"] != 0:
        raise ValueError(f"Initial checkpoint must be step 0, got {steps['initial']}")
    if steps["best"] < 0 or steps["final"] <= 0:
        raise ValueError(f"Invalid checkpoint steps: {steps}")
    return resolved, paths, steps


def _load_three_weight_sets(config: AnalysisConfig, resolved, paths):
    payloads = {name: _load_payload(path) for name, path in paths.items()}
    seeds = {
        int(payload.get("seed", config.seed))
        for payload in payloads.values()
    }
    if len(seeds) != 1:
        raise ValueError(f"Checkpoint seeds differ: {sorted(seeds)}")

    optimizer = str(
        payloads["final"].get("optimizer_name", config.optimizer)
    ).lower()
    if optimizer != config.optimizer:
        raise ValueError(
            f"Checkpoint optimizer={optimizer!r}, target={config.optimizer!r}"
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
    return weights, payloads, model_cfg


def _null_interval(values: list[float]):
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    q = np.quantile(finite, [0.025, 0.5, 0.975])
    return float(q[0]), float(q[1]), float(q[2])


def _ccdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    data = _positive(values)
    if data.size == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    return np.asarray([np.mean(data >= x) for x in grid], dtype=float)


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
    display_xmin = max(min(actual_xmin, null_xmin_median), float(actual[0]))
    xmax = max(float(actual[-1]), *(float(item[-1]) for item in null_positive))
    if xmax <= display_xmin:
        return
    grid = np.geomspace(display_xmin, xmax, 240)
    null_ccdfs = np.stack([_ccdf(item, grid) for item in null_positive])
    low, median, high = np.nanquantile(null_ccdfs, [0.025, 0.5, 0.975], axis=0)

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
        "powerlaw.Fit sees all positive x; no manual xmin or xmax"
    )
    ax.legend(fontsize=8)
    safe_pair = pair_label.replace("->", "_to_")
    _finish(
        fig,
        output_dir / f"{matrix_name}_{kind}_{safe_pair}_far_tail_zoom_ccdf.png",
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
    _finish(fig, output_dir / f"{matrix_name}_radial_initial_best_final.png", show)


def _pairwise_alpha_plot(matrix_name, kind, frame, output_dir, show):
    labels = [f"{a}->{b}" for a, b in PAIR_ORDER]
    subset = frame[
        (frame.matrix_name == matrix_name) & (frame.angular_type == kind)
    ].set_index("pair").reindex(labels)
    x = np.arange(len(labels), dtype=float)
    actual = subset.actual_alpha.to_numpy(float)
    med = subset.null_alpha_median.to_numpy(float)
    low = subset.null_alpha_2p5.to_numpy(float)
    high = subset.null_alpha_97p5.to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    finite_null = np.isfinite(low) & np.isfinite(med) & np.isfinite(high)
    if finite_null.any():
        ax.errorbar(
            x[finite_null],
            med[finite_null],
            yerr=np.vstack([
                med[finite_null] - low[finite_null],
                high[finite_null] - med[finite_null],
            ]),
            fmt="o",
            capsize=4,
            label="random-angular median and 95% interval",
        )
    finite_actual = np.isfinite(actual)
    if finite_actual.any():
        ax.scatter(x[finite_actual], actual[finite_actual], marker="x", s=80, label="actual")
    ax.axhline(2.0, linestyle="--", linewidth=1.0, label="alpha = 2 reference")
    ax.set_xticks(x, labels)
    ax.set_ylabel("powerlaw.Fit alpha")
    ax.set_title(f"{matrix_name}: {kind} pairwise angular tail exponents")
    ax.legend(fontsize=8)
    _finish(fig, output_dir / f"{matrix_name}_{kind}_pairwise_alpha_vs_random.png", show)


def run_three_checkpoint_analysis(
    config: AnalysisConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Analyze actual initial, best, and final checkpoints against random angular nulls."""
    config = config or AnalysisConfig.from_env()
    config.validate()
    resolved, paths, steps = resolve_three_checkpoints(config)
    weights, payloads, model_cfg = _load_three_weight_sets(config, resolved, paths)

    output_dir = resolved.output_dir / "initial_best_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer_index, matrix_name in enumerate(sorted(weights["initial"])):
        _radial_three_state_plot(matrix_name, weights, output_dir, config.show_plots)

        for pair_index, (reference_state, target_state) in enumerate(PAIR_ORDER):
            reference = weights[reference_state][matrix_name]
            target = weights[target_state][matrix_name]
            actual = angular_spectra(reference, target)
            target_polar = polar(target)
            rng = np.random.default_rng(
                config.null_seed + 100003 * (layer_index + 1) + 1009 * pair_index
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
                actual_values = projective(actual[kind], upper)
                actual_fit, package_fit = fit_powerlaw_tail(
                    actual_values,
                    min_tail=config.min_tail,
                )
                null_values = [projective(sample, upper) for sample in null_spectra[kind]]
                null_fit_pairs = [
                    fit_powerlaw_tail(sample, min_tail=config.min_tail)
                    for sample in null_values
                ]
                null_fits = [item[0] for item in null_fit_pairs]
                null_alphas = [fit.alpha for fit in null_fits if fit.success]
                alpha_low, alpha_median, alpha_high = _null_interval(null_alphas)

                rows.append({
                    "matrix_name": matrix_name,
                    "angular_type": kind,
                    "pair": pair_label,
                    "reference_state": reference_state,
                    "target_state": target_state,
                    "reference_step": steps[reference_state],
                    "target_step": steps[target_state],
                    "fit_backend": "powerlaw.Fit",
                    "fit_selection": "all positive x; package MLE/KS xmin; no xmax",
                    "actual_alpha": actual_fit.alpha,
                    "actual_xmin": actual_fit.xmin,
                    "actual_D": actual_fit.D,
                    "actual_tail_n": actual_fit.n_tail,
                    "actual_tail_decades": actual_fit.tail_decades,
                    "actual_xmax_observed": actual_fit.xmax_observed,
                    "null_alpha_2p5": alpha_low,
                    "null_alpha_median": alpha_median,
                    "null_alpha_97p5": alpha_high,
                    "alpha_outside_random_null": bool(
                        actual_fit.success
                        and np.isfinite(alpha_low)
                        and np.isfinite(alpha_high)
                        and (actual_fit.alpha < alpha_low or actual_fit.alpha > alpha_high)
                    ),
                })

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
    results.to_csv(output_dir / "angular_initial_best_final_powerlaw_summary.csv", index=False)

    for matrix_name in sorted(weights["initial"]):
        for kind in ("tilt", "twist"):
            _pairwise_alpha_plot(matrix_name, kind, results, output_dir, config.show_plots)

    manifest = {
        "seed": int(payloads["final"].get("seed", config.seed)),
        "optimizer": config.optimizer,
        "checkpoints": {name: str(path) for name, path in paths.items()},
        "steps": steps,
        "model_config": model_cfg,
        "angular_nulls": config.angular_nulls,
        "fit_backend": "powerlaw.Fit",
        "fit_contract": "all positive x; xmin chosen by package MLE/KS; no xmax",
        "pairs": [f"{a}->{b}" for a, b in PAIR_ORDER],
    }
    (output_dir / "angular_initial_best_final_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return results, manifest

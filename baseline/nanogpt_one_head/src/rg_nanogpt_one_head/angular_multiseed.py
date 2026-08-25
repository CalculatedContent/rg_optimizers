from __future__ import annotations

from dataclasses import replace
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .angular_three_checkpoint import PAIR_ORDER, run_three_checkpoint_analysis
from .angular_weightwatcher_core import AnalysisConfig


CORE_METRICS = {
    "alpha": ("actual_alpha", "null_alpha_median", "powerlaw.Fit alpha", 2.0),
    "tail_decades": (
        "actual_tail_decades",
        "null_tail_decades_median",
        "tail length, log10(xmax/xmin)",
        None,
    ),
    "D": ("actual_D", "null_D_median", "power-law KS D", None),
    "xmin": ("actual_xmin", "null_xmin_median", "package-selected xmin", None),
}


def parse_seed_spec(value: str | None) -> tuple[int, ...]:
    """Parse a comma/space/semicolon separated seed list without inventing seeds."""
    if value is None or not str(value).strip():
        return ()
    tokens = [token for token in re.split(r"[,;\s]+", str(value).strip()) if token]
    seeds = tuple(int(token) for token in tokens)
    if not seeds:
        return ()
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"ANGULAR_SEEDS contains duplicate seeds: {seeds}")
    return seeds


def _results_root(config: AnalysisConfig) -> Path:
    if config.results_root:
        return Path(config.results_root).expanduser().resolve()
    if config.runroot:
        return Path(config.runroot).expanduser().resolve() / "results"
    if config.run_dir:
        run_dir = Path(config.run_dir).expanduser().resolve()
        if run_dir.name.startswith("seed_") and run_dir.parent.name == config.optimizer:
            return run_dir.parent.parent
    raise ValueError(
        "Multi-seed angular analysis requires RESULTS_ROOT, RUNROOT, or a standard "
        "RUN_DIR ending in <optimizer>/seed_<seed>."
    )


def discover_complete_seeds(
    config: AnalysisConfig,
    requested: tuple[int, ...] = (),
) -> tuple[Path, tuple[int, ...]]:
    """Resolve seeds having actual initial, best, and final checkpoints."""
    results_root = _results_root(config)
    optimizer_root = results_root / config.optimizer
    if not optimizer_root.is_dir():
        raise FileNotFoundError(f"Optimizer results directory not found: {optimizer_root}")

    discovered: dict[int, Path] = {}
    for run_dir in sorted(optimizer_root.glob("seed_*")):
        try:
            seed = int(run_dir.name.split("seed_", 1)[1])
        except (IndexError, ValueError):
            continue
        required = [
            run_dir / "checkpoint_initial.pt",
            run_dir / "checkpoint_best.pt",
            run_dir / "checkpoint_final.pt",
        ]
        if all(path.is_file() for path in required):
            discovered[seed] = run_dir

    if requested:
        missing = [seed for seed in requested if seed not in discovered]
        if missing:
            raise FileNotFoundError(
                "Requested seeds do not have all three saved checkpoints "
                f"(initial/best/final): {missing}"
            )
        seeds = requested
    else:
        seeds = tuple(sorted(discovered))

    if not seeds:
        raise FileNotFoundError(
            f"No complete angular runs found under {optimizer_root}. Each seed needs "
            "checkpoint_initial.pt, checkpoint_best.pt, and checkpoint_final.pt."
        )
    return results_root, seeds


def _cross_seed_stats(values: pd.Series) -> dict[str, float | int]:
    finite = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    n = int(finite.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "sem": np.nan, "ci95": np.nan}
    mean = float(np.mean(finite))
    if n == 1:
        return {"n": 1, "mean": mean, "sd": np.nan, "sem": np.nan, "ci95": np.nan}
    sd = float(np.std(finite, ddof=1))
    sem = sd / np.sqrt(n)
    ci95 = float(stats.t.ppf(0.975, df=n - 1) * sem)
    return {"n": n, "mean": mean, "sd": sd, "sem": sem, "ci95": ci95}


def aggregate_seed_results(seed_results: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["matrix_name", "angular_type", "pair"]
    numeric_columns = [
        "actual_alpha",
        "actual_xmin",
        "actual_D",
        "actual_tail_n",
        "actual_tail_decades",
        "actual_endpoint_atoms",
        "full_continuous_ks_D",
        "full_continuous_ks_mc_p",
        "tail_conditional_ks_D",
        "tail_conditional_ks_mc_p",
        "null_alpha_median",
        "null_xmin_median",
        "null_D_median",
        "null_tail_n_median",
        "null_tail_decades_median",
    ]
    rows: list[dict] = []
    for key, frame in seed_results.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, key))
        row["n_seeds_total"] = int(frame["seed"].nunique())
        for column in numeric_columns:
            if column not in frame:
                continue
            stats_row = _cross_seed_stats(frame[column])
            for suffix, value in stats_row.items():
                row[f"{column}_{suffix}"] = value
        for column in (
            "actual_fit_success",
            "alpha_outside_random_null",
            "tail_longer_than_random_97p5",
            "candidate_nonrandom_long_tail",
        ):
            if column in frame:
                vals = frame[column].astype(float)
                row[f"{column}_fraction"] = float(vals.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _zoom_limits(center: np.ndarray, error: np.ndarray, reference: float | None) -> tuple[float, float] | None:
    center = np.asarray(center, dtype=float)
    error = np.asarray(error, dtype=float)
    finite = np.isfinite(center)
    if not finite.any():
        return None
    lo = center[finite] - np.where(np.isfinite(error[finite]), error[finite], 0.0)
    hi = center[finite] + np.where(np.isfinite(error[finite]), error[finite], 0.0)
    lower = float(np.min(lo))
    upper = float(np.max(hi))
    if reference is not None and np.isfinite(reference):
        lower = min(lower, float(reference))
        upper = max(upper, float(reference))
    span = upper - lower
    if not np.isfinite(span):
        return None
    if span <= 0.0:
        span = max(abs(upper), 1.0) * 0.10
    padding = max(0.08 * span, 1e-6)
    return lower - padding, upper + padding


def _plot_metric(
    seed_results: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    metric_name: str,
    output_dir: Path,
    zoom: bool,
    show: bool,
) -> Path:
    actual_col, null_col, ylabel, reference = CORE_METRICS[metric_name]
    matrices = sorted(seed_results["matrix_name"].unique())
    pairs = [f"{a}->{b}" for a, b in PAIR_ORDER]
    angular_types = ("tilt", "twist")

    fig, axes = plt.subplots(
        len(matrices),
        len(angular_types),
        figsize=(12.5, max(4.2, 3.15 * len(matrices))),
        squeeze=False,
        sharex=True,
    )

    for row_index, matrix_name in enumerate(matrices):
        for col_index, angular_type in enumerate(angular_types):
            ax = axes[row_index, col_index]
            subset = aggregate[
                (aggregate["matrix_name"] == matrix_name)
                & (aggregate["angular_type"] == angular_type)
            ].set_index("pair")
            raw = seed_results[
                (seed_results["matrix_name"] == matrix_name)
                & (seed_results["angular_type"] == angular_type)
            ]
            x = np.arange(len(pairs), dtype=float)

            actual_mean = np.asarray([
                subset.loc[pair, f"{actual_col}_mean"] if pair in subset.index else np.nan
                for pair in pairs
            ], dtype=float)
            actual_ci = np.asarray([
                subset.loc[pair, f"{actual_col}_ci95"] if pair in subset.index else np.nan
                for pair in pairs
            ], dtype=float)
            null_mean = np.asarray([
                subset.loc[pair, f"{null_col}_mean"] if pair in subset.index else np.nan
                for pair in pairs
            ], dtype=float)
            null_ci = np.asarray([
                subset.loc[pair, f"{null_col}_ci95"] if pair in subset.index else np.nan
                for pair in pairs
            ], dtype=float)

            actual_mask = np.isfinite(actual_mean)
            null_mask = np.isfinite(null_mean)
            ax.errorbar(
                x[actual_mask] - 0.08,
                actual_mean[actual_mask],
                yerr=np.where(np.isfinite(actual_ci[actual_mask]), actual_ci[actual_mask], 0.0),
                fmt="o",
                capsize=4,
                label="trained mean ± 95% t CI",
            )
            ax.errorbar(
                x[null_mask] + 0.08,
                null_mean[null_mask],
                yerr=np.where(np.isfinite(null_ci[null_mask]), null_ci[null_mask], 0.0),
                fmt="s",
                capsize=4,
                label="random-null median mean ± 95% t CI",
            )

            # Seed-level points remain visible; the CI never hides replication.
            for pair_index, pair in enumerate(pairs):
                values = pd.to_numeric(
                    raw.loc[raw["pair"] == pair, actual_col], errors="coerce"
                ).to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    offsets = np.linspace(-0.035, 0.035, values.size)
                    ax.scatter(
                        pair_index - 0.08 + offsets,
                        values,
                        marker=".",
                        s=28,
                        alpha=0.65,
                        label="individual seeds" if pair_index == 0 else None,
                    )

            if reference is not None:
                ax.axhline(reference, linestyle="--", linewidth=1.0, label=f"reference={reference:g}")
            if zoom:
                combined_center = np.concatenate([actual_mean, null_mean])
                combined_error = np.concatenate([actual_ci, null_ci])
                limits = _zoom_limits(combined_center, combined_error, reference)
                if limits is not None:
                    ax.set_ylim(*limits)
            ax.set_xticks(x, pairs, rotation=18, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{matrix_name} — {angular_type}")
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=7)

    scale_label = "zoomed y" if zoom else "full y"
    fig.suptitle(
        f"Cross-seed angular {metric_name.replace('_', ' ')}: trained vs random ({scale_label})",
        y=1.002,
    )
    fig.tight_layout()
    suffix = "zoom_y" if zoom else "full_y"
    path = output_dir / f"cross_seed_{metric_name}_trained_vs_random_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def run_multiseed_analysis(
    config: AnalysisConfig | None = None,
    *,
    seed_spec: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the strict three-checkpoint angular analysis for every completed seed.

    Cross-seed error bars are 95% Student-t confidence intervals over independent
    seed-level statistics. No null realization is ever treated as an independent
    seed. Individual seed points are retained on the figures.
    """
    config = config or AnalysisConfig.from_env()
    config.validate()
    requested = parse_seed_spec(seed_spec)
    results_root, seeds = discover_complete_seeds(config, requested)

    if config.initial_checkpoint or config.final_checkpoint:
        raise ValueError(
            "Explicit INITIAL_CHECKPOINT_PATH/FINAL_CHECKPOINT_PATH are single-seed "
            "overrides and are forbidden in multi-seed mode. Use the standard files "
            "inside each seed directory."
        )

    # BEST_CHECKPOINT_PATH is similarly ambiguous across seeds; require standard files.
    import os
    if os.environ.get("BEST_CHECKPOINT_PATH", "").strip():
        raise ValueError("BEST_CHECKPOINT_PATH must be unset in multi-seed mode")

    aggregate_root = (
        Path(config.output_dir).expanduser().resolve()
        if config.output_dir
        else results_root / config.optimizer / "angular_multiseed"
    )
    aggregate_root.mkdir(parents=True, exist_ok=True)

    seed_frames: list[pd.DataFrame] = []
    seed_manifests: dict[str, dict] = {}
    for seed in seeds:
        seed_config = replace(
            config,
            seed=int(seed),
            run_dir=None,
            results_root=str(results_root),
            runroot=None,
            initial_checkpoint=None,
            final_checkpoint=None,
            output_dir=None,
            show_plots=False,
            null_seed=int(config.null_seed + 1000003 * int(seed)),
        )
        frame, manifest = run_three_checkpoint_analysis(seed_config)
        frame = frame.copy()
        frame.insert(0, "seed", int(seed))
        seed_frames.append(frame)
        seed_manifests[str(seed)] = manifest

    seed_results = pd.concat(seed_frames, ignore_index=True)
    aggregate = aggregate_seed_results(seed_results)

    seed_csv = aggregate_root / "angular_all_seeds_results.csv"
    aggregate_csv = aggregate_root / "angular_cross_seed_summary.csv"
    seed_results.to_csv(seed_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)

    plot_paths: list[str] = []
    for metric_name in CORE_METRICS:
        for zoom in (False, True):
            plot_paths.append(
                str(
                    _plot_metric(
                        seed_results,
                        aggregate,
                        metric_name=metric_name,
                        output_dir=aggregate_root,
                        zoom=zoom,
                        show=config.show_plots,
                    )
                )
            )

    manifest = {
        "optimizer": config.optimizer,
        "seeds": [int(seed) for seed in seeds],
        "n_seeds": len(seeds),
        "results_root": str(results_root),
        "output_dir": str(aggregate_root),
        "seed_results_csv": str(seed_csv),
        "aggregate_csv": str(aggregate_csv),
        "error_bar_contract": (
            "mean +/- 95% Student-t confidence interval across independent seeds; "
            "individual seed points plotted; random-null draws never counted as seeds"
        ),
        "zoom_contract": (
            "display-only y-axis zoom spans cross-seed means and 95% CI endpoints "
            "with padding; alpha=2 is retained in the alpha zoom; fit data are unchanged"
        ),
        "plots": plot_paths,
        "per_seed_manifests": seed_manifests,
    }
    manifest_path = aggregate_root / "angular_multiseed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return seed_results, aggregate, manifest

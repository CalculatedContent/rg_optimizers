#!/usr/bin/env python3
"""Observable command-line Jacobian analysis for the short-100 experiment.

This deliberately does not execute a notebook.  It consumes the saved model-only
checkpoints directly, logs every significant operation, writes status after every
method, and persists tables and plots after every checkpoint.  A failure therefore
appears immediately in the terminal, in the log file, and in ``errors.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any

BASELINE_ROOT = Path(__file__).resolve().parents[3]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUITE = "mnist_mlp3_tangent_rg_v1_muonclip_short100_10seed"
DEFAULT_RUN_ROOT = Path("/private/tmp/rg-mnist-mlp3-short100-runs")
DEFAULT_CACHE_ROOT = Path("/private/tmp/rg-mnist-mlp3-short100-checkpoints")
DEFAULT_OUTPUT_ROOT = Path("/private/tmp/rg-mnist-mlp3-short100-jacobians")
DEFAULT_OPTIMIZERS = ("muonclip_rms", "adamw")
DEFAULT_LAYERS = ("fc1.weight", "fc2.weight", "fc3.weight")
BASE_METHODS = (
    "polar_pullback",
    "normalized_gram_pullback",
    "centered_log_gram_pullback",
    "centered_log_singular_radial_pullback",
    "finite_muon_ns5_pullback",
)


class MaxInfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def configure_logging(output_root: Path, verbose: bool) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("short100_jacobians")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.DEBUG if verbose else logging.INFO)
    stdout.addFilter(MaxInfoFilter())
    stdout.setFormatter(formatter)
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(formatter)
    logfile = logging.FileHandler(output_root / "jacobians.log", mode="a")
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(formatter)
    logger.addHandler(stdout)
    logger.addHandler(stderr)
    logger.addHandler(logfile)
    return logger


def parse_csv_values(text: str, cast=str) -> tuple[Any, ...]:
    values = tuple(cast(value.strip()) for value in str(text).split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("comma-separated value list cannot be empty")
    return values


def load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object: {path}")
    return value


def resolve_run_identity(run_root: Path, optimizer: str, seed: int) -> dict[str, Any]:
    seed_dir = run_root / SUITE / optimizer / f"seed_{seed}"
    manifest = load_json(seed_dir / "manifest.json", "run manifest")
    resolved = load_json(seed_dir / "resolved_config.json", "resolved config")
    complete = load_json(seed_dir / "run_complete.json", "completion marker")
    config = dict(resolved.get("config", resolved))
    expected = {
        "suite_name": SUITE,
        "optimizer": optimizer,
        "seed": int(seed),
    }
    observed = {
        "suite_name": manifest.get("suite_name"),
        "optimizer": manifest.get("optimizer"),
        "seed": manifest.get("seed"),
    }
    if {key: str(value) for key, value in observed.items()} != {
        key: str(value) for key, value in expected.items()
    }:
        raise RuntimeError(f"run identity mismatch at {seed_dir}: {observed} != {expected}")
    fingerprints = {
        str(manifest.get("protocol_fingerprint", "")),
        str(resolved.get("protocol_fingerprint", "")),
        str(complete.get("protocol_fingerprint", "")),
    }
    if "" in fingerprints or len(fingerprints) != 1:
        raise RuntimeError(f"run fingerprints are missing or unequal beneath {seed_dir}")
    if not bool(complete.get("completed", False)):
        raise RuntimeError(f"run is not complete: {seed_dir}")
    epochs = int(config["epochs"])
    if epochs != int(complete["epochs"]):
        raise RuntimeError(f"resolved and completed epoch counts disagree: {seed_dir}")
    return {
        "seed_dir": seed_dir,
        "fingerprint": fingerprints.pop(),
        "epochs": epochs,
        "config": config,
    }


def selected_checkpoint_refs(
    cache_root: Path,
    optimizer: str,
    seed: int,
    *,
    epoch_stride: int,
    maximum_checkpoints: int | None,
) -> tuple[Any, ...]:
    from rg_baselines.tangent_rg.checkpoints import list_analysis_checkpoints

    checkpoint_dir = cache_root / SUITE / optimizer / f"seed_{seed}" / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"missing checkpoint directory: {checkpoint_dir}")
    all_refs = list_analysis_checkpoints(checkpoint_dir)
    if not all_refs:
        raise RuntimeError(f"no analysis checkpoints found: {checkpoint_dir}")
    selected = tuple(
        ref for ref in all_refs if int(ref.epoch) > 0 and int(ref.epoch) % epoch_stride == 0
    )
    if maximum_checkpoints is not None:
        selected = selected[: int(maximum_checkpoints)]
    if not selected:
        raise RuntimeError(
            f"epoch stride {epoch_stride} selected no checkpoints from {checkpoint_dir}"
        )
    return selected


def checkpoint_matrix(path: Path, fingerprint: str, layer: str) -> np.ndarray:
    from rg_baselines.tangent_rg.checkpoints import load_analysis_checkpoint

    payload = load_analysis_checkpoint(path, expected_fingerprint=fingerprint)
    state = payload["model"]
    candidates = (layer, f"module.{layer}")
    name = next((candidate for candidate in candidates if candidate in state), None)
    if name is None:
        raise KeyError(f"checkpoint {path} has no layer {layer!r}; keys={sorted(state)}")
    tensor = state[name]
    value = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"checkpoint layer is not a finite matrix: {path}:{name}")
    return matrix


def record_row(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        raw = asdict(record)
    else:
        raw = dict(vars(record))
    return {
        key: value
        for key, value in raw.items()
        if not isinstance(value, np.ndarray)
    }


def compute_one_spectrum(
    method: str,
    weight: np.ndarray,
    *,
    ns_steps: int,
    ns_eps: float,
    singular_values: np.ndarray | None = None,
) -> tuple[np.ndarray, Any]:
    from rg_baselines.tangent_rg import polar, single_checkpoint

    singular = (
        np.asarray(singular_values, dtype=float)
        if singular_values is not None
        else np.linalg.svd(weight, compute_uv=False)
    )
    factories = {
        "polar_pullback": lambda: polar.polar_pullback_spectrum(
            weight, precomputed_singular_values=singular, include_mode_labels=False
        ),
        "normalized_gram_pullback": lambda: (
            single_checkpoint.normalized_gram_analytic_spectrum(
                weight, precomputed_singular_values=singular
            )
        ),
        "centered_log_gram_pullback": lambda: (
            single_checkpoint.centered_log_gram_analytic_spectrum(
                weight, precomputed_singular_values=singular
            )
        ),
        "centered_log_singular_radial_pullback": lambda: (
            single_checkpoint.centered_log_singular_analytic_spectrum(
                weight, precomputed_singular_values=singular
            )
        ),
        "finite_muon_ns5_pullback": lambda: polar.muon_newton_schulz_analytic_spectrum(
            weight,
            steps=ns_steps,
            eps=ns_eps,
            precomputed_singular_values=singular,
        ),
    }
    if method not in factories:
        raise KeyError(f"unknown base Jacobian method: {method}")
    record = factories[method]()
    return np.asarray(record.singular_amplitudes, dtype=float), record


def compute_base_spectra(
    weight: np.ndarray,
    *,
    ns_steps: int,
    ns_eps: float,
) -> dict[str, tuple[np.ndarray, Any]]:
    """Small-matrix/test convenience; production calls methods one at a time."""

    return {
        method: compute_one_spectrum(
                method, weight, ns_steps=ns_steps, ns_eps=ns_eps
        )
        for method in BASE_METHODS
    }


def exact_ecs_ranks(
    seed_dir: Path,
    optimizer: str,
    seed: int,
    epoch: int,
    global_step: int,
    layer: str,
    maximum_rank: int,
) -> list[tuple[str, int, int, dict[str, Any]]]:
    """Read exact same-checkpoint ECS boundaries; return available cover variants."""

    from rg_baselines.tangent_rg.single_checkpoint import select_ecs_cover_ranks

    fit_path = seed_dir / "metrics" / "weightwatcher_fits.csv"
    trace_path = seed_dir / "metrics" / "trace_log.csv"
    if not fit_path.is_file() or not trace_path.is_file():
        return []
    fits = pd.read_csv(fit_path)
    traces = pd.read_csv(trace_path)
    identity = (
        fits["optimizer"].astype(str).eq(optimizer)
        & pd.to_numeric(fits["seed"], errors="coerce").eq(seed)
        & pd.to_numeric(fits["epoch"], errors="coerce").eq(epoch)
        & pd.to_numeric(fits["global_step"], errors="coerce").eq(global_step)
        & fits["layer"].astype(str).eq(layer)
        & fits["fit_variant"].astype(str).eq("clip_xmax")
    )
    rows = fits.loc[identity]
    if len(rows) != 1:
        return []
    fit = rows.iloc[0]
    fit_ok = str(fit.get("fit_ok", "")).strip().lower() in {"1", "true", "yes"}
    if not fit_ok:
        return []
    trace_identity = (
        traces["optimizer"].astype(str).eq(optimizer)
        & pd.to_numeric(traces["seed"], errors="coerce").eq(seed)
        & pd.to_numeric(traces["epoch"], errors="coerce").eq(epoch)
        & pd.to_numeric(traces["global_step"], errors="coerce").eq(global_step)
        & traces["layer"].astype(str).eq(layer)
        & traces["fit_variant"].astype(str).eq("clip_xmax")
    )
    exact = traces.loc[trace_identity]
    primary = exact.loc[
        exact["qualification_role"].astype(str).eq(
            "preregistered_independent_fit_support"
        )
        & ~exact["sensitivity_only"].astype(str).str.lower().isin({"1", "true", "yes"})
    ]
    detx = exact.loc[exact["support_rank_source"].astype(str).eq("weightwatcher_detX")]
    if len(primary) != 1 or len(detx) != 1:
        return []
    primary = primary.iloc[0]
    certified = str(primary.get("certification_eligible", "")).strip().lower() in {
        "1", "true", "yes"
    }
    if not certified:
        return []
    k_pl = int(round(float(primary["support_window_end_descending_exclusive"])))
    k_tl = int(round(float(fit["detX_num"])))
    window_start = int(round(float(primary["support_window_start_descending_zero_based"])))
    effective_tail = int(round(float(primary["support_rank"])))
    detx_trace_rank = int(round(float(detx.iloc[0]["support_rank"])))
    if k_pl != window_start + effective_tail:
        raise RuntimeError(
            "ECS PL boundary is not support-window start plus effective-tail rank"
        )
    if detx_trace_rank != k_tl:
        raise RuntimeError("WeightWatcher detX_num disagrees with exact trace audit")
    selection = select_ecs_cover_ranks(k_pl, k_tl, maximum_rank=maximum_rank)
    metadata = {
        "k_pl": k_pl,
        "k_tl": k_tl,
        "k_boundary_mid": selection.boundary_midpoint_rank,
        "rank_source": "exact same-checkpoint WeightWatcher clip_xmax and detX trace audit",
        "fit_path": str(fit_path),
        "trace_path": str(trace_path),
    }
    variants: list[tuple[str, int, int, dict[str, Any]]] = []
    if k_pl < maximum_rank:
        variants.append((
            "ecs_grassmann_cartan_cover_full_row_shell_pullback",
            k_pl,
            maximum_rank,
            metadata,
        ))
    if selection.available:
        variants.append((
            "ecs_grassmann_cartan_cover_detx_shell_pullback",
            k_pl,
            k_tl,
            metadata,
        ))
    return variants


def fit_spectrum(
    amplitudes: np.ndarray,
    record: Any,
    metadata: dict[str, Any],
    top_k_values: tuple[int, ...],
    minimum_tail: int,
) -> list[dict[str, Any]]:
    from rg_baselines.tangent_rg import powerlaw_fit

    positive = np.sort(amplitudes[np.isfinite(amplitudes) & (amplitudes > 0)])
    feasible = tuple(value for value in top_k_values if value <= positive.size - 2) or (0,)
    amplitude = powerlaw_fit.fit_clipping_sensitivity(
        positive,
        top_k_values=feasible,
        minimum_tail=minimum_tail,
        operator_kind=record.operator_kind,
        map_definition=record.map_definition,
        spectrum_kind="amplitude",
        metadata=metadata,
    )
    rows: list[dict[str, Any]] = []
    for row in amplitude.to_dict(orient="records"):
        rows.append(row)
        rows.append(powerlaw_fit.amplitude_fit_to_energy(row))
    return rows


def ecs_fit_amplitudes(record: Any, *, compress_groups: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Optionally replace uniform ECS coordinate copies by physical groups.

    Each retained-core amplitude ``2/sigma_i`` is repeated ``q-k`` times in
    the ambient Jacobian. Uniform repetition leaves the empirical CDF, MLE
    alpha, package-selected xmin, and KS distance unchanged. Compression avoids
    treating deterministic copies as independent observations, so the reported
    uncertainty is based on physical core groups.
    """

    multiplicity = int(record.deterministic_shell_multiplicity)
    expanded = np.asarray(record.singular_amplitudes, dtype=float)
    if not compress_groups:
        return expanded, {
            "ecs_fit_observation_unit": "expanded_jacobian_mode",
            "ecs_uniform_group_multiplicity": multiplicity,
            "ecs_groups_compressed": False,
        }
    core = 2.0 / np.asarray(record.retained_singular_values, dtype=float)
    return core, {
        "ecs_fit_observation_unit": "physical_retained_core_amplitude_group",
        "ecs_uniform_group_multiplicity": multiplicity,
        "ecs_groups_compressed": True,
        "ecs_expanded_mode_count": int(expanded.size),
        "ecs_physical_group_count": int(core.size),
        "ecs_compression_invariance": (
            "uniform replication removed; empirical CDF, alpha, xmin, and KS D "
            "are invariant; sigma reflects physical groups"
        ),
    }


def safe_slug(text: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


def save_spectrum_plot(
    spectra: dict[str, np.ndarray],
    destination: Path,
    *,
    title: str,
) -> None:
    from rg_baselines.tangent_rg.powerlaw_fit import empirical_ccdf

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for method, raw in spectra.items():
        sample = np.sort(raw[np.isfinite(raw) & (raw > 0)])
        if sample.size < 2:
            continue
        x, ccdf = empirical_ccdf(sample)
        axes[0].step(x, ccdf, where="post", linewidth=1.4, label=method)
        axes[1].step(x**2, ccdf, where="post", linewidth=1.4, label=method)
    for axis, xlabel in zip(axes, ("Jacobian singular amplitude", "Jacobian Gram eigenvalue")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("CCDF")
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_alpha_progress(fit_rows: list[dict[str, Any]], destination: Path, title: str) -> None:
    frame = pd.DataFrame(fit_rows)
    if frame.empty:
        return
    usable = frame[
        frame["spectrum_kind"].astype(str).eq("energy_derived_from_amplitude")
        & pd.to_numeric(frame["clip_top_k"], errors="coerce").eq(0)
    ].copy()
    if usable.empty:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for method, curve in usable.groupby("method"):
        curve = curve.sort_values("epoch")
        axis.plot(curve["epoch"], curve["alpha"], marker="o", ms=3, label=method)
    axis.axhline(2.0, color="black", linestyle=":", linewidth=1.2)
    axis.set(xlabel="epoch", ylabel="Jacobian energy alpha", title=title)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run observable weight-only Jacobian analysis without Jupyter/Papermill."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--optimizers", default=",".join(DEFAULT_OPTIMIZERS))
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    parser.add_argument(
        "--methods",
        default=",".join(BASE_METHODS),
        help="comma-separated base Jacobians; ECS covers are controlled separately",
    )
    parser.add_argument("--epoch-stride", type=int, default=10)
    parser.add_argument("--maximum-checkpoints", type=int)
    parser.add_argument("--top-k", default="0", help="PL clipping sensitivities; default 0 only")
    parser.add_argument("--minimum-tail", type=int, default=8)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--ns-eps", type=float, default=1e-7)
    parser.add_argument("--skip-ecs", action="store_true")
    parser.add_argument(
        "--compress-ecs-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fit one physical ECS amplitude per uniformly repeated shell group",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    logger = configure_logging(output_root, args.verbose)
    status_path = output_root / "status.json"
    fit_path = output_root / "jacobian_powerlaw_fits.csv"
    operator_path = output_root / "jacobian_operators.csv"
    spectrum_data_path = output_root / "jacobian_spectra.csv"
    error_path = output_root / "errors.csv"
    completion_path = output_root / "completed_checkpoints.csv"
    optimizers = parse_csv_values(args.optimizers)
    seeds = parse_csv_values(args.seeds, int)
    layers = parse_csv_values(args.layers)
    methods = parse_csv_values(args.methods)
    unknown_methods = set(methods) - set(BASE_METHODS)
    if unknown_methods:
        raise ValueError(f"unknown --methods values: {sorted(unknown_methods)}")
    top_k_values = parse_csv_values(args.top_k, int)
    if top_k_values[0] != 0 or any(value < 0 for value in top_k_values):
        raise ValueError("--top-k must begin with 0 and contain nonnegative integers")
    if args.epoch_stride < 1:
        raise ValueError("--epoch-stride must be positive")

    fit_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    if args.resume and fit_path.is_file():
        fit_rows = pd.read_csv(fit_path).to_dict(orient="records")
    if args.resume and operator_path.is_file():
        operator_rows = pd.read_csv(operator_path).to_dict(orient="records")
    if args.resume and spectrum_data_path.is_file():
        spectrum_rows = pd.read_csv(spectrum_data_path).to_dict(orient="records")
    if args.resume and error_path.is_file():
        errors = pd.read_csv(error_path).to_dict(orient="records")
    if args.resume and completion_path.is_file():
        completion_rows = pd.read_csv(completion_path).to_dict(orient="records")

    identities: dict[tuple[str, int], dict[str, Any]] = {}
    work: list[tuple[str, int, str, Any]] = []
    logger.info("START suite=%s", SUITE)
    logger.info("run_root=%s", args.run_root.resolve())
    logger.info("cache_root=%s", args.cache_root.resolve())
    logger.info("output_root=%s", output_root)
    logger.info("Preflight is lightweight: selected checkpoints are validated when loaded")
    for optimizer in optimizers:
        for seed in seeds:
            identity = resolve_run_identity(args.run_root.resolve(), optimizer, seed)
            identities[(optimizer, seed)] = identity
            refs = selected_checkpoint_refs(
                args.cache_root.resolve(), optimizer, seed,
                epoch_stride=args.epoch_stride,
                maximum_checkpoints=args.maximum_checkpoints,
            )
            logger.info(
                "DISCOVER optimizer=%s seed=%d selected=%d epochs=%s",
                optimizer, seed, len(refs), [int(ref.epoch) for ref in refs],
            )
            for layer in layers:
                work.extend((optimizer, seed, layer, ref) for ref in refs)
    total = len(work)
    started = time.perf_counter()
    completed = 0

    for optimizer, seed, layer, ref in work:
        unit_key = (optimizer, int(seed), layer, int(ref.epoch), int(ref.global_step))
        completed_before = any(
            (
                str(row.get("optimizer")), int(row.get("seed", -1)),
                str(row.get("layer")), int(row.get("epoch", -1)),
                int(row.get("global_step", -1)),
            ) == unit_key
            and str(row.get("base_methods_requested", "")) == ",".join(methods)
            and str(row.get("ecs_groups_compressed", "")).strip().lower()
            == str(bool(args.compress_ecs_groups)).lower()
            and str(row.get("ecs_skipped", "")).strip().lower()
            == str(bool(args.skip_ecs)).lower()
            for row in completion_rows
        )
        spectrum_data_available = any(
            (
                str(row.get("optimizer")), int(row.get("seed", -1)),
                str(row.get("layer")), int(row.get("epoch", -1)),
                int(row.get("global_step", -1)),
            ) == unit_key
            for row in spectrum_rows
        )
        if args.resume and completed_before and spectrum_data_available:
            completed += 1
            logger.info("SKIP completed %d/%d %s", completed, total, unit_key)
            continue

        # Replace an interrupted partial unit rather than appending duplicates.
        def same_unit(row: dict[str, Any]) -> bool:
            try:
                observed = (
                    str(row.get("optimizer")), int(row.get("seed", -1)),
                    str(row.get("layer")), int(row.get("epoch", -1)),
                    int(row.get("global_step", -1)),
                )
            except (TypeError, ValueError):
                return False
            return observed == unit_key

        fit_rows = [row for row in fit_rows if not same_unit(row)]
        operator_rows = [row for row in operator_rows if not same_unit(row)]
        spectrum_rows = [row for row in spectrum_rows if not same_unit(row)]
        errors = [row for row in errors if not same_unit(row)]
        completion_rows = [row for row in completion_rows if not same_unit(row)]

        unit_started = time.perf_counter()
        logger.info(
            "CHECKPOINT START %d/%d optimizer=%s seed=%d layer=%s epoch=%d step=%d",
            completed + 1, total, optimizer, seed, layer, ref.epoch, ref.global_step,
        )
        spectra: dict[str, np.ndarray] = {}
        try:
            identity = identities[(optimizer, seed)]
            load_started = time.perf_counter()
            weight = checkpoint_matrix(ref.path, identity["fingerprint"], layer)
            logger.info(
                "LOAD done shape=%s seconds=%.2f file=%s",
                tuple(weight.shape), time.perf_counter() - load_started, ref.path,
            )
            singular = np.linalg.svd(weight, compute_uv=False)
            method_factories: dict[str, tuple[np.ndarray, Any]] = {}
            method_metadata: dict[str, dict[str, Any]] = {}
            for method in methods:
                build_started = time.perf_counter()
                logger.info("JACOBIAN START method=%s", method)
                method_factories[method] = compute_one_spectrum(
                    method, weight, ns_steps=args.ns_steps, ns_eps=args.ns_eps,
                    singular_values=singular,
                )
                logger.info(
                    "JACOBIAN DONE method=%s seconds=%.2f n_amplitudes=%d",
                    method, time.perf_counter() - build_started,
                    len(method_factories[method][0]),
                )
            if not args.skip_ecs and layer == "fc1.weight":
                from rg_baselines.tangent_rg import single_checkpoint

                numerical_rank = int(np.count_nonzero(singular > args.ns_eps * singular[0]))
                for method, k, q, rank_metadata in exact_ecs_ranks(
                    identity["seed_dir"], optimizer, seed, int(ref.epoch),
                    int(ref.global_step), layer, numerical_rank,
                ):
                    cover = single_checkpoint.ecs_grassmann_cover_analytic_spectrum(
                        weight, retained_rank=k, outer_rank=q, rcond=args.ns_eps,
                        precomputed_singular_values=singular,
                    )
                    amplitudes, compression_metadata = ecs_fit_amplitudes(
                        cover, compress_groups=args.compress_ecs_groups
                    )
                    method_factories[method] = (amplitudes, cover)
                    method_metadata[method] = {
                        **rank_metadata,
                        **compression_metadata,
                    }
                    logger.info(
                        "ECS ranks method=%s k=%d q=%d fit_amplitudes=%d "
                        "expanded_modes=%d compressed=%s",
                        method, k, q, len(amplitudes), int(cover.derivative_rank),
                        args.compress_ecs_groups,
                    )

            base = {
                "optimizer": optimizer,
                "seed": int(seed),
                "layer": layer,
                "epoch": int(ref.epoch),
                "global_step": int(ref.global_step),
                "protocol_fingerprint": identity["fingerprint"],
                "checkpoint_path": str(ref.path),
            }
            for method, (amplitudes, record) in method_factories.items():
                method_started = time.perf_counter()
                logger.info("METHOD START method=%s n_amplitudes=%d", method, len(amplitudes))
                metadata = {**base, **method_metadata.get(method, {}), "method": method}
                rows = fit_spectrum(
                    np.asarray(amplitudes, dtype=float), record, metadata,
                    top_k_values, args.minimum_tail,
                )
                fit_rows.extend(rows)
                operator_rows.append({
                    **base,
                    **method_metadata.get(method, {}),
                    "method": method,
                    **record_row(record),
                })
                stored_amplitudes = np.sort(np.asarray(amplitudes, dtype=float))[::-1]
                spectra[method] = stored_amplitudes
                observation_unit = method_metadata.get(method, {}).get(
                    "ecs_fit_observation_unit", "jacobian_singular_mode"
                )
                uniform_multiplicity = int(
                    method_metadata.get(method, {}).get(
                        "ecs_uniform_group_multiplicity", 1
                    )
                )
                spectrum_rows.extend(
                    {
                        **base,
                        "method": method,
                        "mode_index_descending": int(index),
                        "singular_amplitude": float(amplitude),
                        "gram_eigenvalue": float(amplitude * amplitude),
                        "observation_unit": observation_unit,
                        "represented_uniform_multiplicity": uniform_multiplicity,
                    }
                    for index, amplitude in enumerate(stored_amplitudes)
                )
                elapsed_method = time.perf_counter() - method_started
                primary = next(
                    row for row in rows
                    if row["spectrum_kind"] == "energy_derived_from_amplitude"
                    and int(row["clip_top_k"]) == 0
                )
                logger.info(
                    "METHOD DONE method=%s seconds=%.2f alpha=%s D=%s fit_ok=%s",
                    method, elapsed_method,
                    f"{float(primary['alpha']):.4f}" if pd.notna(primary["alpha"]) else "nan",
                    f"{float(primary['ks_D']):.4f}" if pd.notna(primary["ks_D"]) else "nan",
                    primary["fit_ok"],
                )
                atomic_json(status_path, {
                    "state": "running_method",
                    "optimizer": optimizer, "seed": seed, "layer": layer,
                    "epoch": int(ref.epoch), "global_step": int(ref.global_step),
                    "method": method, "completed_checkpoint_count": completed,
                    "total_checkpoint_count": total,
                    "elapsed_seconds": time.perf_counter() - started,
                    "updated_at_utc": utc_now(),
                })

            atomic_csv(fit_path, fit_rows)
            atomic_csv(operator_path, operator_rows)
            atomic_csv(spectrum_data_path, spectrum_rows)
            spectrum_path = (
                output_root / "plots" / "spectra" / optimizer / safe_slug(layer)
                / f"epoch_{int(ref.epoch):05d}.png"
            )
            save_spectrum_plot(
                spectra, spectrum_path,
                title=f"{optimizer} seed {seed} {layer} epoch {int(ref.epoch)}",
            )
            block_rows = [
                row for row in fit_rows
                if str(row.get("optimizer")) == optimizer
                and int(row.get("seed", -1)) == seed
                and str(row.get("layer")) == layer
            ]
            alpha_path = output_root / "plots" / "alpha_progress" / f"{optimizer}_{safe_slug(layer)}.png"
            save_alpha_progress(block_rows, alpha_path, f"{optimizer} seed {seed} {layer}")
            completion_rows.append({
                **base,
                "methods": ",".join(method_factories),
                "base_methods_requested": ",".join(methods),
                "ecs_groups_compressed": bool(args.compress_ecs_groups),
                "ecs_skipped": bool(args.skip_ecs),
                "completed_at_utc": utc_now(),
            })
            atomic_csv(completion_path, completion_rows)
            completed += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (total - completed) if completed else None
            logger.info(
                "CHECKPOINT DONE %d/%d seconds=%.2f elapsed=%s ETA=%s plot=%s",
                completed, total, time.perf_counter() - unit_started,
                format_duration(elapsed), format_duration(eta), spectrum_path,
            )
            atomic_json(status_path, {
                "state": "running",
                "completed_checkpoint_count": completed,
                "total_checkpoint_count": total,
                "percent_complete": 100.0 * completed / total,
                "last_optimizer": optimizer, "last_seed": seed, "last_layer": layer,
                "last_epoch": int(ref.epoch), "last_global_step": int(ref.global_step),
                "elapsed_seconds": elapsed, "eta_seconds": eta,
                "eta_human": format_duration(eta), "updated_at_utc": utc_now(),
            })
        except Exception as error:
            error_row = {
                "optimizer": optimizer, "seed": seed, "layer": layer,
                "epoch": int(ref.epoch), "global_step": int(ref.global_step),
                "exception_type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "failed_at_utc": utc_now(),
            }
            errors.append(error_row)
            atomic_csv(error_path, errors)
            if fit_rows:
                atomic_csv(fit_path, fit_rows)
            if operator_rows:
                atomic_csv(operator_path, operator_rows)
            if spectrum_rows:
                atomic_csv(spectrum_data_path, spectrum_rows)
            atomic_json(status_path, {
                "state": "error", **error_row,
                "completed_checkpoint_count": completed,
                "total_checkpoint_count": total,
            })
            logger.exception(
                "CHECKPOINT ERROR optimizer=%s seed=%d layer=%s epoch=%d",
                optimizer, seed, layer, ref.epoch,
            )
            if args.fail_fast:
                raise

    elapsed = time.perf_counter() - started
    state = "complete_with_errors" if errors else "complete"
    summary = {
        "state": state,
        "suite": SUITE,
        "completed_checkpoint_count": completed,
        "total_checkpoint_count": total,
        "error_count": len(errors),
        "elapsed_seconds": elapsed,
        "elapsed_human": format_duration(elapsed),
        "output_root": str(output_root),
        "completed_at_utc": utc_now(),
    }
    atomic_json(status_path, summary)
    atomic_json(output_root / "run_summary.json", summary)
    logger.info(
        "FINISH state=%s completed=%d/%d errors=%d elapsed=%s output=%s",
        state, completed, total, len(errors), format_duration(elapsed), output_root,
    )
    return 1 if errors else 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except Exception as error:
        output_root = args.output_root.expanduser().resolve()
        fatal = {
            "state": "fatal_error",
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "failed_at_utc": utc_now(),
        }
        atomic_json(output_root / "status.json", fatal)
        atomic_json(output_root / "fatal_error.json", fatal)
        logging.getLogger("short100_jacobians").exception("FATAL analysis terminated")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

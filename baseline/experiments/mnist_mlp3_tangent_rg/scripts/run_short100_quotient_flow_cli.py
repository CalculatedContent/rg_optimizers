#!/usr/bin/env python3
"""Command-line weight-quotient and two-checkpoint RG-flow analysis.

This program deliberately separates two questions that are often conflated:

1. ``state quotient``: construct an auditable representative of a declared
   equivalence class of ``W`` and ask whether its Gram ESD is heavy tailed;
2. ``checkpoint flow``: measure finite radial and angular motion from ``W_t``
   to ``W_{t+1}``.  These are secant/transfer observables, not ``D beta(W)``.

The transformed matrices are actually installed into an MLP3 and analyzed by
WeightWatcher both raw and with ``fix_fingers='clip_xmax'``.  Every phase logs
to stdout/stderr, writes resumable CSVs after each unit, and records failures.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = Path(__file__).resolve().parents[3]
for candidate in (SCRIPT_DIR, BASELINE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import pandas as pd

import run_short100_jacobians_cli as jacobian_cli
from rg_baselines.model import MLP3
from rg_baselines.tangent_rg import single_checkpoint, two_checkpoint, weight_quotients
from rg_baselines.tangent_rg.weightwatcher_fit import analyze_weightwatcher_dual


LAYERS = ("fc1.weight", "fc2.weight")
QUOTIENT_PROFILES = (
    ("midpoint_ecs_control", "midpoint", {}),
    ("gram_ridge", "tau_fraction_0p25", {"tau_fraction": 0.25}),
    ("gram_ridge", "tau_fraction_0p50", {"tau_fraction": 0.50}),
    ("gram_ridge", "tau_fraction_0p75", {"tau_fraction": 0.75}),
    (
        "feshbach_downfolding",
        "ridge_ratio_1em2",
        {"regularization_ratio": 1.0e-2},
    ),
)


def atomic_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except pd.errors.EmptyDataError:
        return []


def configure_logging(root: Path, verbose: bool) -> logging.Logger:
    root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("short100_quotient_flow")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for stream in (sys.stdout, sys.stderr):
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO if stream is sys.stdout else logging.WARNING)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    file_handler = logging.FileHandler(root / "quotient_flow.log", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def checkpoint_model(path: Path, fingerprint: str) -> MLP3:
    from rg_baselines.tangent_rg.checkpoints import load_analysis_checkpoint

    payload = load_analysis_checkpoint(path, expected_fingerprint=fingerprint)
    model = MLP3().to("cpu")
    state = payload["model"]
    normalized = {
        (name.removeprefix("module.")): value for name, value in state.items()
    }
    model.load_state_dict(normalized, strict=True)
    model.eval()
    return model


def model_matrix(model: MLP3, layer: str) -> np.ndarray:
    parameter = dict(model.named_parameters())[layer]
    return parameter.detach().cpu().double().numpy()


def replace_matrix(model: MLP3, layer: str, value: np.ndarray) -> None:
    import torch

    parameter = dict(model.named_parameters())[layer]
    candidate = torch.as_tensor(value, dtype=parameter.dtype)
    if tuple(candidate.shape) != tuple(parameter.shape):
        raise ValueError(f"shape mismatch for {layer}: {candidate.shape} != {parameter.shape}")
    with torch.no_grad():
        parameter.copy_(candidate)


def midpoint_metadata(
    identity: dict[str, Any], optimizer: str, seed: int, ref: Any, layer: str,
    numerical_rank: int,
) -> dict[str, Any]:
    records = jacobian_cli.exact_ecs_ranks(
        identity["seed_dir"], optimizer, seed, int(ref.epoch),
        int(ref.global_step), layer, numerical_rank,
    )
    if not records:
        raise RuntimeError(f"no certified same-checkpoint ECS ranks for {layer} epoch={ref.epoch}")
    metadata = dict(records[0][3])
    metadata["ecs_rank"] = weight_quotients.midpoint_ecs_rank(
        metadata["k_pl"], metadata["k_tl"], maximum_rank=numerical_rank
    )
    return metadata


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def record_metadata(record: Any) -> dict[str, Any]:
    raw = asdict(record) if is_dataclass(record) else dict(vars(record))
    return {
        key: value for key, value in raw.items()
        if not isinstance(value, np.ndarray)
    }


def run_state_quotients(
    *, root: Path, run_root: Path, cache_root: Path, optimizers: tuple[str, ...],
    seed: int, epoch_stride: int, logger: logging.Logger,
) -> None:
    fit_path = root / "weight_quotient_weightwatcher_fits.csv"
    spectrum_path = root / "weight_quotient_spectra.csv"
    operator_path = root / "weight_quotient_operators.csv"
    fits = read_rows(fit_path)
    spectra = read_rows(spectrum_path)
    operators = read_rows(operator_path)

    for optimizer in optimizers:
        identity = jacobian_cli.resolve_run_identity(run_root, optimizer, seed)
        refs = jacobian_cli.selected_checkpoint_refs(
            cache_root, optimizer, seed, epoch_stride=epoch_stride,
            maximum_checkpoints=None,
        )
        anchor_model = checkpoint_model(refs[0].path, identity["fingerprint"])
        for ref in refs:
            base_model = checkpoint_model(ref.path, identity["fingerprint"])
            rank_by_layer = {}
            for layer in LAYERS:
                weight = model_matrix(base_model, layer)
                singular = np.linalg.svd(weight, compute_uv=False)
                numerical_rank = int(np.count_nonzero(singular > 1.0e-9 * singular[0]))
                rank_by_layer[layer] = midpoint_metadata(
                    identity, optimizer, seed, ref, layer, numerical_rank
                )
            for method, profile_id, parameters in QUOTIENT_PROFILES:
                key = (optimizer, seed, int(ref.epoch), method, profile_id)
                observed = {
                    (str(row.get("layer")), str(row.get("fit_variant")))
                    for row in fits
                    if (
                        str(row.get("optimizer")), int(row.get("seed", -1)),
                        int(row.get("epoch", -1)), str(row.get("method")),
                        str(row.get("profile_id")),
                    ) == key
                }
                if observed == {(layer, variant) for layer in LAYERS for variant in ("raw", "clip_xmax")}:
                    logger.info("QUOTIENT SKIP optimizer=%s epoch=%d method=%s", optimizer, ref.epoch, method)
                    continue
                started = time.perf_counter()
                logger.info("QUOTIENT START optimizer=%s epoch=%d method=%s profile=%s", optimizer, ref.epoch, method, profile_id)
                model = copy.deepcopy(base_model)
                results = {}
                for layer in LAYERS:
                    result = weight_quotients.apply_weight_quotient(
                        method,
                        model_matrix(base_model, layer),
                        ecs_rank=int(rank_by_layer[layer]["ecs_rank"]),
                        anchor_weight=model_matrix(anchor_model, layer),
                        parameters=parameters,
                    )
                    replace_matrix(model, layer, result.weight)
                    results[layer] = result
                measurement = analyze_weightwatcher_dual(
                    model,
                    min_evals=8,
                    max_fingers=10,
                    svd_method="accurate",
                    randomize=True,
                    analysis_seed=stable_seed(optimizer, seed, ref.epoch, method, profile_id),
                    primary_variant="clip_xmax",
                )
                fits = [
                    row for row in fits
                    if not (
                        str(row.get("optimizer")), int(row.get("seed", -1)),
                        int(row.get("epoch", -1)), str(row.get("method")),
                        str(row.get("profile_id")),
                    ) == key
                ]
                spectra = [
                    row for row in spectra
                    if not (
                        str(row.get("optimizer")), int(row.get("seed", -1)),
                        int(row.get("epoch", -1)), str(row.get("method")),
                        str(row.get("profile_id")),
                    ) == key
                ]
                operators = [
                    row for row in operators
                    if not (
                        str(row.get("optimizer")), int(row.get("seed", -1)),
                        int(row.get("epoch", -1)), str(row.get("method")),
                        str(row.get("profile_id")),
                    ) == key
                ]
                details = measurement.details[
                    measurement.details["layer"].astype(str).isin(LAYERS)
                ].copy()
                details["optimizer"] = optimizer
                details["seed"] = seed
                details["epoch"] = int(ref.epoch)
                details["global_step"] = int(ref.global_step)
                details["method"] = method
                details["profile_id"] = profile_id
                details["analysis_family"] = "weight_state_quotient_representative"
                details["equivalence_relation"] = "O(m)_left x O(n)_right orbit plus declared nuisance map"
                fits.extend(details.to_dict(orient="records"))
                for layer, result in results.items():
                    base = {
                        "optimizer": optimizer, "seed": seed,
                        "epoch": int(ref.epoch), "global_step": int(ref.global_step),
                        "layer": layer, "method": method, "profile_id": profile_id,
                        "ecs_rank": int(result.ecs_rank),
                        "retained_rank": int(result.retained_rank),
                        "operator_kind": result.operator_kind,
                        "map_definition": result.map_definition,
                        "quotient_parameters": json.dumps(result.parameters, default=str, sort_keys=True),
                    }
                    operators.append(base)
                    spectra.extend({
                        **base,
                        "mode_index_descending": index,
                        "singular_value": float(value),
                        "gram_eigenvalue": float(value * value),
                    } for index, value in enumerate(result.singular_values))
                atomic_frame(fit_path, fits)
                atomic_frame(spectrum_path, spectra)
                atomic_frame(operator_path, operators)
                logger.info("QUOTIENT DONE optimizer=%s epoch=%d method=%s seconds=%.2f", optimizer, ref.epoch, method, time.perf_counter() - started)


def ecs_topk_rates(first: np.ndarray, second: np.ndarray, rank: int, delta_s: float) -> np.ndarray:
    k = int(rank)
    right0 = np.linalg.svd(first, full_matrices=False)[2][:k].T
    right1 = np.linalg.svd(second, full_matrices=False)[2][:k].T
    cosines = np.linalg.svd(right0.T @ right1, compute_uv=False)
    angles = np.arccos(np.clip(cosines, 0.0, 1.0))
    return angles / abs(float(delta_s))


def run_checkpoint_flows(
    *, root: Path, run_root: Path, cache_root: Path, optimizers: tuple[str, ...],
    seed: int, epoch_stride: int, logger: logging.Logger,
) -> None:
    fit_path = root / "two_checkpoint_flow_fits.csv"
    spectrum_path = root / "two_checkpoint_flow_spectra.csv"
    operator_path = root / "two_checkpoint_flow_operators.csv"
    transport_path = root / "two_checkpoint_jacobian_transport.csv"
    fits = read_rows(fit_path)
    spectra = read_rows(spectrum_path)
    operators = read_rows(operator_path)
    transports = read_rows(transport_path)
    for optimizer in optimizers:
        identity = jacobian_cli.resolve_run_identity(run_root, optimizer, seed)
        refs = jacobian_cli.selected_checkpoint_refs(
            cache_root, optimizer, seed, epoch_stride=epoch_stride,
            maximum_checkpoints=None,
        )
        for ref0, ref1 in zip(refs[:-1], refs[1:]):
            delta_s = float(ref1.epoch - ref0.epoch)
            for layer in LAYERS:
                unit = (optimizer, seed, int(ref0.epoch), int(ref1.epoch), layer)
                expected = {
                    "two_checkpoint_generalized_gram_radial",
                    "two_checkpoint_aligned_transfer_core",
                    "two_checkpoint_ecs_topk_grassmann",
                    "two_checkpoint_relative_polar_tilt",
                    "two_checkpoint_radial_quotient_observed_secant",
                    "two_checkpoint_radial_jacobian_prediction",
                }
                observed = {
                    str(row.get("method")) for row in operators
                    if (
                        str(row.get("optimizer")), int(row.get("seed", -1)),
                        int(row.get("epoch_start", -1)), int(row.get("epoch_end", -1)),
                        str(row.get("layer")),
                    ) == unit
                }
                if expected.issubset(observed):
                    logger.info("FLOW SKIP optimizer=%s %d->%d layer=%s", optimizer, ref0.epoch, ref1.epoch, layer)
                    continue
                started = time.perf_counter()
                first = jacobian_cli.checkpoint_matrix(ref0.path, identity["fingerprint"], layer)
                second = jacobian_cli.checkpoint_matrix(ref1.path, identity["fingerprint"], layer)
                singular0 = np.linalg.svd(first, compute_uv=False)
                singular1 = np.linalg.svd(second, compute_uv=False)
                rank0 = midpoint_metadata(identity, optimizer, seed, ref0, layer, singular0.size)["ecs_rank"]
                rank1 = midpoint_metadata(identity, optimizer, seed, ref1, layer, singular1.size)["ecs_rank"]
                k = min(int(rank0), int(rank1))
                radial = two_checkpoint.generalized_gram_log_rates(first, second, delta_s)
                transfer = two_checkpoint.aligned_rectangular_transfer(first, second, delta_s)
                angular = two_checkpoint.relative_polar_angular_flow(first, second, delta_s)
                radial_map0 = single_checkpoint.centered_log_singular_map(first)
                radial_map1 = single_checkpoint.centered_log_singular_map(second)
                radial_jvp = single_checkpoint.centered_log_singular_jvp(
                    first, second - first
                )
                observed_radial = (radial_map1.value - radial_map0.value) / delta_s
                predicted_radial = radial_jvp.jvp / delta_s
                observed_norm = float(np.linalg.norm(observed_radial))
                predicted_norm = float(np.linalg.norm(predicted_radial))
                residual_norm = float(np.linalg.norm(predicted_radial - observed_radial))
                denominator = max(
                    observed_norm * predicted_norm, np.finfo(np.float64).tiny
                )
                transport = {
                    "optimizer": optimizer, "seed": seed, "layer": layer,
                    "epoch_start": int(ref0.epoch), "epoch_end": int(ref1.epoch),
                    "global_step_start": int(ref0.global_step),
                    "global_step_end": int(ref1.global_step),
                    "delta_s": delta_s,
                    "map": "centered_log_singular_radial_quotient",
                    "observed_flow_norm": observed_norm,
                    "jacobian_prediction_norm": predicted_norm,
                    "linearization_residual_norm": residual_norm,
                    "relative_linearization_error": residual_norm / max(
                        observed_norm, np.finfo(np.float64).tiny
                    ),
                    "cosine_observed_vs_jacobian": float(
                        np.dot(observed_radial, predicted_radial) / denominator
                    ),
                    "is_actual_map_jvp": True,
                    "is_optimizer_beta_jacobian": False,
                    "map_definition": (
                        "compare [R(W1)-R(W0)]/delta_s with "
                        "D R_W0[W1-W0]/delta_s for centered log singular R"
                    ),
                }
                cases = (
                    (
                        "two_checkpoint_generalized_gram_radial",
                        radial.radial_rate_amplitudes,
                        radial,
                    ),
                    (
                        "two_checkpoint_aligned_transfer_core",
                        transfer.core_rate_amplitudes,
                        transfer,
                    ),
                    (
                        "two_checkpoint_ecs_topk_grassmann",
                        ecs_topk_rates(first, second, k, delta_s),
                        SimpleNamespace(
                            operator_kind="two_checkpoint_topk_right_grassmann_geodesic_rates",
                            map_definition="principal angles between top-k right singular subspaces divided by epoch separation; finite flow, not a Jacobian",
                        ),
                    ),
                    (
                        "two_checkpoint_relative_polar_tilt",
                        angular.tilt_geodesic_rates,
                        angular,
                    ),
                    (
                        "two_checkpoint_radial_quotient_observed_secant",
                        np.abs(observed_radial),
                        SimpleNamespace(
                            operator_kind="observed_centered_log_singular_quotient_secant",
                            map_definition="absolute components of [R(W1)-R(W0)]/delta_s",
                        ),
                    ),
                    (
                        "two_checkpoint_radial_jacobian_prediction",
                        np.abs(predicted_radial),
                        SimpleNamespace(
                            operator_kind="centered_log_singular_jacobian_prediction_along_checkpoint_step",
                            map_definition="absolute components of D R_W0[W1-W0]/delta_s; actual map JVP, not D beta(W)",
                        ),
                    ),
                )
                base = {
                    "optimizer": optimizer, "seed": seed, "layer": layer,
                    "epoch_start": int(ref0.epoch), "epoch_end": int(ref1.epoch),
                    "global_step_start": int(ref0.global_step),
                    "global_step_end": int(ref1.global_step),
                    "epoch": int(ref1.epoch), "delta_s": delta_s,
                    "ecs_comparison_rank": k,
                    "analysis_family": "two_checkpoint_finite_rg_flow_not_jacobian",
                    "is_training_jacobian": False,
                }
                fits = [row for row in fits if not (
                    str(row.get("optimizer")), int(row.get("seed", -1)),
                    int(row.get("epoch_start", -1)), int(row.get("epoch_end", -1)),
                    str(row.get("layer")),
                ) == unit]
                spectra = [row for row in spectra if not (
                    str(row.get("optimizer")), int(row.get("seed", -1)),
                    int(row.get("epoch_start", -1)), int(row.get("epoch_end", -1)),
                    str(row.get("layer")),
                ) == unit]
                operators = [row for row in operators if not (
                    str(row.get("optimizer")), int(row.get("seed", -1)),
                    int(row.get("epoch_start", -1)), int(row.get("epoch_end", -1)),
                    str(row.get("layer")),
                ) == unit]
                transports = [row for row in transports if not (
                    str(row.get("optimizer")), int(row.get("seed", -1)),
                    int(row.get("epoch_start", -1)), int(row.get("epoch_end", -1)),
                    str(row.get("layer")),
                ) == unit]
                transports.append(transport)
                for method, raw, record in cases:
                    amplitudes = np.asarray(raw, dtype=float)
                    amplitudes = amplitudes[np.isfinite(amplitudes) & (amplitudes > 0)]
                    metadata = {**base, "method": method}
                    if amplitudes.size >= 2:
                        fits.extend(jacobian_cli.fit_spectrum(
                            amplitudes, record, metadata, (0,), 8
                        ))
                    spectra.extend({
                        **metadata,
                        "mode_index_descending": index,
                        "flow_rate_amplitude": float(value),
                        "flow_rate_energy": float(value * value),
                    } for index, value in enumerate(np.sort(amplitudes)[::-1]))
                    operators.append({
                        **metadata,
                        "operator_kind": record.operator_kind,
                        "map_definition": record.map_definition,
                        **record_metadata(record),
                    })
                atomic_frame(fit_path, fits)
                atomic_frame(spectrum_path, spectra)
                atomic_frame(operator_path, operators)
                atomic_frame(transport_path, transports)
                logger.info("FLOW DONE optimizer=%s %d->%d layer=%s seconds=%.2f", optimizer, ref0.epoch, ref1.epoch, layer, time.perf_counter() - started)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=jacobian_cli.DEFAULT_RUN_ROOT)
    parser.add_argument("--cache-root", type=Path, default=jacobian_cli.DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("/private/tmp/rg-mnist-mlp3-short100-jacobians-reduced"))
    parser.add_argument("--optimizers", default="muonclip_rms,adamw")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--epoch-stride", type=int, default=10)
    parser.add_argument("--skip-state-quotients", action="store_true")
    parser.add_argument("--skip-checkpoint-flows", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.output_root.expanduser().resolve()
    logger = configure_logging(root, args.verbose)
    optimizers = jacobian_cli.parse_csv_values(args.optimizers)
    started = time.perf_counter()
    try:
        logger.info("START quotient/flow analysis output_root=%s", root)
        if not args.skip_state_quotients:
            run_state_quotients(
                root=root, run_root=args.run_root.resolve(),
                cache_root=args.cache_root.resolve(), optimizers=optimizers,
                seed=args.seed, epoch_stride=args.epoch_stride, logger=logger,
            )
        if not args.skip_checkpoint_flows:
            run_checkpoint_flows(
                root=root, run_root=args.run_root.resolve(),
                cache_root=args.cache_root.resolve(), optimizers=optimizers,
                seed=args.seed, epoch_stride=args.epoch_stride, logger=logger,
            )
        logger.info("COMPLETE seconds=%.2f", time.perf_counter() - started)
        return 0
    except Exception:
        trace = traceback.format_exc()
        logger.error("FAILED\n%s", trace)
        atomic_frame(root / "quotient_flow_errors.csv", [{
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "exception_traceback": trace,
        }])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit and summarize the frozen five-seed modified-AdamW experiment.

This script reads recorded fits; it never trains a model or refits a spectrum.
By default an incomplete five-seed set produces explicit partial outputs and
exit status 2. Use --allow-partial only for progress inspection.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

SEEDS = (1340, 1341, 1342, 1343, 1344)
STEPS = tuple(range(0, 20001, 500))
BUDGETS = (2000, 20000)
MATRIX_COUNT = 18
DELTA = 0.02


def read_json(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite(value, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {label}")
    return value


def describe(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("A summary needs at least one finite observation")
    mean = float(a.mean())
    if len(a) == 1:
        sd = sem = low = high = None
    else:
        sd = float(a.std(ddof=1))
        sem = sd / math.sqrt(len(a))
        half = float(student_t.ppf(0.975, len(a) - 1)) * sem
        low, high = mean - half, mean + half
    return dict(n=int(len(a)), mean=mean, sample_sd=sd, sem=sem,
                ci95_low=low, ci95_high=high,
                min=float(a.min()), max=float(a.max()))


def weighted_alpha(spectrum: dict, seed: int, step: int):
    matrices = spectrum["matrices"]
    names = [row["matrix"] for row in matrices]
    if (spectrum.get("unique_matrices") != MATRIX_COUNT
            or spectrum.get("successful_fits") != MATRIX_COUNT
            or len(matrices) != MATRIX_COUNT or len(set(names)) != MATRIX_COUNT):
        raise ValueError(f"Seed {seed}, step {step}: expected 18 unique successful fits")
    tied = [row for row in matrices if "lm_head.weight" in row.get("aliases", [])]
    if (len(tied) != 1 or tied[0]["matrix"] != "transformer.wte.weight"
            or "lm_head.weight" in names):
        raise ValueError(f"Seed {seed}, step {step}: invalid tied-embedding accounting")
    rows = []
    for matrix in matrices:
        fits = matrix["fits"]
        if len(fits) != 1 or fits[0].get("status") != "success":
            raise ValueError(f"Seed {seed}, step {step}: failed fit for {matrix['matrix']}")
        fit = fits[0]
        alpha = finite(fit["alpha"], "ordinary fitted alpha")
        distance = finite(fit["D"], "KS distance")
        tail = finite(fit["num_pl_spikes"], "tail count")
        if alpha <= 0 or distance < 0 or tail <= 0:
            raise ValueError(f"Seed {seed}, step {step}: invalid fit inputs")
        weight = math.sqrt(tail) / (distance + DELTA)
        rows.append(dict(seed=seed, step=step, matrix=matrix["matrix"],
                         alpha=alpha, ks_distance=distance, n_tail=tail,
                         confidence_tail_weight=weight, fit_status=fit["status"],
                         aliases=json.dumps(matrix.get("aliases", [])),
                         shape=json.dumps(matrix["shape"])))
    denominator = math.fsum(row["confidence_tail_weight"] for row in rows)
    average = math.fsum(row["alpha"] * row["confidence_tail_weight"]
                        for row in rows) / denominator
    return average, rows


def metric_values(value: dict, prefix: str) -> dict:
    loss = finite(value["loss_nats_per_character"], f"{prefix} CE")
    ppl = finite(value["perplexity"], f"{prefix} perplexity")
    accuracy = finite(value["next_character_accuracy"], f"{prefix} accuracy")
    if not math.isclose(ppl, math.exp(loss), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{prefix} perplexity does not equal exp(CE)")
    if not 0 <= accuracy <= 1:
        raise ValueError(f"{prefix} accuracy is outside [0,1]")
    return {f"{prefix}_ce": loss, f"{prefix}_ppl": ppl,
            f"{prefix}_accuracy_percent": 100.0 * accuracy}


def result_path(root: Path, seed: int) -> Path:
    return (root / "runs" / f"seed_{seed}" / "confirmation" / "rate_0.4"
            / "graft" / f"seed_{seed}" / "result.json")


def design_path(root: Path, seed: int, result: Path) -> Path:
    return result.parents[2] / "design.json"


def validate_design(design: dict, expected: dict, seed: int) -> None:
    recipe = design["recipes"]["graft"]
    if (recipe["reference"]["peak_lr"] != 0.4
            or recipe["native"]["peak_lr"] != 0.001
            or recipe["reference_lr_ratio"] != 400.0):
        raise ValueError(f"Seed {seed}: incorrect tuned graft recipe")
    for key in ("schedule", "architecture", "batch_size", "split", "input_sha256"):
        if design[key] != expected[key]:
            raise ValueError(f"Seed {seed}: {key} differs from archived seed 1340")
    if design["sources"] != expected["sources"]:
        raise ValueError(f"Seed {seed}: frozen source digests differ")
    if design["recipes"] != expected["recipes"]:
        raise ValueError(f"Seed {seed}: optimizer recipes differ")
    if design.get("study_seed") != seed or design.get("phase") != "confirmation":
        raise ValueError(f"Seed {seed}: incorrect confirmation identity")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Allow incomplete progress summaries to exit successfully")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    # The four-worker supervisor and five-seed cohort monitor can finish
    # concurrently. Serialize the entire read/validate/write transaction.
    with (output / ".analysis.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return analyze(root, args.allow_partial)


def analyze(root: Path, allow_partial: bool) -> int:
    output = root / "analysis"
    expected_design_path = root / "existing_seed1340" / "design.json"
    expected_design = read_json(expected_design_path)
    historical_path = root / "existing_seed1340" / "result.json"
    historical = read_json(historical_path)
    baseline_path = root / "data" / "original_comparison_audit.json"
    baseline = {(int(row["seed"]), int(row["completed_updates"])): row
                for row in read_json(baseline_path)["original_per_seed"]
                if row["arm"] == "AdamW"}

    available, missing = [], []
    metrics, histories, layers, paired = [], [], [], []
    source_paths = {baseline_path, expected_design_path, historical_path,
                    Path(__file__).resolve()}
    if (root / "plan.json").is_file():
        source_paths.add(root / "plan.json")
    environments = {}
    training_times = []

    for seed in SEEDS:
        path = result_path(root, seed)
        if not path.exists():
            missing.append(seed)
            continue
        record = read_json(path)
        if not record.get("completed"):
            missing.append(seed)
            continue
        if (record.get("seed") != seed or record.get("arm") != "graft"
                or record["identity"]["seed"] != seed
                or record["identity"]["arm"] != "graft"
                or record["endpoint"]["step"] != 20000):
            raise ValueError(f"Seed {seed}: incorrect result identity or endpoint")
        dp = design_path(root, seed, path)
        design = read_json(dp)
        validate_design(design, expected_design, seed)
        if digest(dp) != record["design_sha256"]:
            raise ValueError(f"Seed {seed}: design digest does not match result")
        source_paths.update((path, dp))
        environments[str(seed)] = dict(device=design["device"], packages=design["packages"])
        available.append(seed)
        training_times.append(finite(record["training_seconds"], "training seconds"))
        history = {int(row["step"]): row for row in record["history"]}
        if len(history) != len(record["history"]) or tuple(sorted(history)) != STEPS:
            raise ValueError(f"Seed {seed}: expected all 41 unique 500-step measurements")
        for step in STEPS:
            item = history[step]
            alpha, layer_rows = weighted_alpha(item["alpha"], seed, step)
            layers.extend(layer_rows)
            values = metric_values(item["validation"], "validation")
            histories.append(dict(seed=seed, step=step,
                                  confidence_tail_weighted_alpha=alpha,
                                  valid_unique_matrix_count=MATRIX_COUNT, **values))
            if step not in BUDGETS:
                continue
            row = dict(seed=seed, step=step, **values,
                       confidence_tail_weighted_alpha=alpha,
                       valid_unique_matrix_count=MATRIX_COUNT,
                       test_ce=None, test_ppl=None, test_accuracy_percent=None)
            if step == 20000:
                endpoint = record["endpoint"]
                endpoint_alpha, _ = weighted_alpha(endpoint["alpha"], seed, step)
                if not math.isclose(endpoint_alpha, alpha, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Seed {seed}: endpoint/history spectral mismatch")
                for name, value in metric_values(endpoint["metrics"]["validation"], "validation").items():
                    if not math.isclose(row[name], value, rel_tol=0, abs_tol=1e-12):
                        raise ValueError(f"Seed {seed}: endpoint/history metric mismatch")
                row.update(metric_values(endpoint["metrics"]["test"], "test"))
            metrics.append(row)
            control = baseline[(seed, step)]
            pair = dict(seed=seed, step=step)
            for name in ("validation_ce", "validation_ppl", "validation_accuracy_percent"):
                delta_name = name.replace("_accuracy_percent", "_accuracy_pp") + "_difference"
                pair[delta_name] = row[name] - float(control[name])
            for name in ("test_ce", "test_ppl", "test_accuracy_percent"):
                delta_name = name.replace("_accuracy_percent", "_accuracy_pp") + "_difference"
                pair[delta_name] = (row[name] - float(control["full_split_endpoint"][name])
                                    if step == 20000 else None)
            pair["confidence_tail_weighted_alpha_difference"] = (
                alpha - float(control["confidence_tail_weighted_alpha"])
                if control["confidence_tail_weighted_alpha"] is not None else None)
            paired.append(pair)

    complete = tuple(available) == SEEDS
    if environments:
        first_environment = next(iter(environments.values()))
        if any(env != first_environment for env in environments.values()):
            raise ValueError("Current cohort has inconsistent recorded runtime/package metadata")
    metric_fields = ["seed", "step", "validation_ce", "validation_ppl", "validation_accuracy_percent",
                     "confidence_tail_weighted_alpha", "valid_unique_matrix_count",
                     "test_ce", "test_ppl", "test_accuracy_percent"]
    history_fields = ["seed", "step", "confidence_tail_weighted_alpha", "valid_unique_matrix_count",
                      "validation_ce", "validation_ppl", "validation_accuracy_percent"]
    layer_fields = ["seed", "step", "matrix", "alpha", "ks_distance", "n_tail",
                    "confidence_tail_weight", "fit_status", "aliases", "shape"]
    pair_fields = ["seed", "step", "validation_ce_difference", "validation_ppl_difference",
                   "validation_accuracy_pp_difference", "test_ce_difference", "test_ppl_difference",
                   "test_accuracy_pp_difference", "confidence_tail_weighted_alpha_difference"]
    write_csv(output / "metrics_per_seed.csv", metrics, metric_fields)
    write_csv(output / "model_alpha_history.csv", histories, history_fields)
    write_csv(output / "layer_alpha_history.csv", layers, layer_fields)
    write_csv(output / "paired_metrics_per_seed.csv", paired, pair_fields)
    endpoint_layers = [row for row in layers if row["step"] == 20000]
    write_csv(output / "layer_alpha_endpoints.csv", endpoint_layers, layer_fields)

    summary_metrics, summary_pairs = {}, {}
    for step in BUDGETS:
        subset = [row for row in metrics if row["step"] == step]
        summary_metrics[str(step)] = {
            key: describe(row[key] for row in subset)
            for key in metric_fields if key not in ("seed", "step", "valid_unique_matrix_count")
            and subset and all(row[key] is not None for row in subset)}
        pairs = [row for row in paired if row["step"] == step]
        summary_pairs[str(step)] = {
            key: describe(row[key] for row in pairs)
            for key in pair_fields if key not in ("seed", "step")
            and pairs and all(row[key] is not None for row in pairs)}

    history_summary = []
    for step in STEPS:
        values = [row["confidence_tail_weighted_alpha"] for row in histories if row["step"] == step]
        if values:
            history_summary.append(dict(step=step, **describe(values)))
    stat_fields = ["n", "mean", "sample_sd", "sem", "ci95_low", "ci95_high", "min", "max"]
    write_csv(output / "model_alpha_history_summary.csv", history_summary, ["step"] + stat_fields)

    endpoint_summary = []
    for matrix in dict.fromkeys(row["matrix"] for row in endpoint_layers):
        subset = [row for row in endpoint_layers if row["matrix"] == matrix]
        if len(subset) != len(available):
            raise ValueError(f"Incomplete endpoint matrix {matrix}")
        endpoint_summary.append(dict(matrix=matrix, **describe(row["alpha"] for row in subset)))
    if available and len(endpoint_summary) != MATRIX_COUNT:
        raise ValueError("Expected 18 endpoint layer summaries")
    write_csv(output / "endpoint_layer_summary.csv", endpoint_summary, ["matrix"] + stat_fields)

    # The old seed 1340 is an immutable historical comparator, never the
    # fifth member of the current-runtime cohort and never a sixth replicate.
    historical_history = {}
    historical_metrics = []
    for item in historical["history"]:
        step = int(item["step"])
        alpha, _ = weighted_alpha(item["alpha"], 1340, step)
        values = metric_values(item["validation"], "validation")
        historical_history[step] = dict(**values, confidence_tail_weighted_alpha=alpha)
        if step in BUDGETS:
            row = dict(seed=1340, step=step, **values,
                       confidence_tail_weighted_alpha=alpha,
                       valid_unique_matrix_count=MATRIX_COUNT,
                       test_ce=None, test_ppl=None, test_accuracy_percent=None)
            if step == 20000:
                row.update(metric_values(historical["endpoint"]["metrics"]["test"], "test"))
            historical_metrics.append(row)
    if tuple(sorted(historical_history)) != STEPS:
        raise ValueError("Historical seed 1340 does not contain all 41 measurements")
    write_csv(output / "historical_seed1340_metrics.csv", historical_metrics, metric_fields)
    current_1340 = {row["step"]: row for row in histories if row["seed"] == 1340}
    discrepancies = []
    for step, current in current_1340.items():
        for metric, old_value in historical_history[step].items():
            discrepancies.append(dict(step=step, metric=metric,
                                      current_value=current[metric], historical_value=old_value,
                                      current_minus_historical=current[metric] - old_value))
    current_endpoint = next((row for row in metrics if row["seed"] == 1340
                             and row["step"] == 20000), None)
    if current_endpoint is not None:
        old_endpoint = next(row for row in historical_metrics if row["step"] == 20000)
        for metric in ("test_ce", "test_ppl", "test_accuracy_percent"):
            discrepancies.append(dict(step=20000, metric=metric,
                                      current_value=current_endpoint[metric],
                                      historical_value=old_endpoint[metric],
                                      current_minus_historical=current_endpoint[metric] - old_endpoint[metric]))
    write_csv(output / "historical_seed1340_discrepancies.csv", discrepancies,
              ["step", "metric", "current_value", "historical_value", "current_minus_historical"])

    def local_name(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return str(path)

    package_matches = (all(env["packages"] == expected_design["packages"]
                           for env in environments.values()) if environments else None)
    summary = dict(
        schema=2, status="complete" if complete else "partial", complete_five_seeds=complete,
        generated_utc=datetime.now(timezone.utc).isoformat(), seeds_expected=list(SEEDS),
        seeds_available=available, seeds_missing=missing, n_completed=len(available),
        cohort="Five current-runtime runs under runs/seed_1340 through runs/seed_1344; archived seed 1340 is excluded",
        reference_peak_lr=0.4, native_peak_lr=0.001,
        estimator=dict(value="ordinary fitted alpha", weight="sqrt(num_pl_spikes)/(D+0.02)",
                       unique_matrices=18, tied_embedding_count=1, equal_seed_weights=True),
        uncertainty=dict(sd="Sample SD across independent seed-level measurements (ddof=1)",
                         sem="sample_sd/sqrt(n)",
                         ci95="Student-t interval for the seed mean, df=n-1",
                         one_seed="SD, SEM and confidence limits are null when n=1",
                         spectral_weights="Fit-quality weights are not confidence intervals",
                         paired_differences="Current modified AdamW minus historical native AdamW, matched by seed label and budget; accuracy differences are percentage points. These are descriptive cross-runtime comparisons, not isolated optimizer effects."),
        metrics_by_budget=summary_metrics, paired_native_adamw_differences=summary_pairs,
        endpoint_layer_summary=endpoint_summary,
        recorded_training_seconds=describe(training_times) if training_times else None,
        historical_seed1340=dict(included_in_current_cohort=False,
                                 metrics=historical_metrics,
                                 current_result_available=1340 in available,
                                 descriptive_discrepancies=discrepancies,
                                 interpretation="Single matched-seed runtime replay; no sampling uncertainty, equivalence, or causal inference is assigned"),
        counts=dict(history_model_rows=len(histories), history_layer_rows=len(layers),
                    endpoint_layer_rows=len(endpoint_layers)),
        provenance=dict(sha256={local_name(p): digest(p) for p in sorted(source_paths)},
                        frozen_source_hashes=expected_design["sources"],
                        environments_by_seed=environments,
                        current_runtime_metadata_consistent=True if environments else None,
                        packages_match_archived_seed1340=package_matches,
                        archived_comparators={
                            "native_AdamW": "Historical-runtime records, matched by seed label only; runtime effects may contribute to differences",
                            "SGDM": "Historical-runtime records retained in original_comparison_audit.json; not part of the current modified-AdamW cohort",
                            "graft_seed1340": "Historical-runtime record kept separately in existing_seed1340; excluded from current cohort means and uncertainty"},
                        runtime_caution="Matching package version strings does not establish numerical equivalence of underlying kernels, libraries, or hardware",
                        no_training_or_spectral_refitting=True),
    )
    atomic_json(output / "summary.json", summary)
    print(json.dumps(dict(status=summary["status"], seeds_available=available,
                          seeds_missing=missing, output=str(output)), indent=2))
    return 0 if complete or allow_partial else 2


if __name__ == "__main__":
    raise SystemExit(main())

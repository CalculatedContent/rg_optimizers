"""Audit and summarize the frozen five-seed California Housing cohort.

Run: python -m california_housing.analyze --root RESULTS_DIR
Intervals describe uncertainty across seeds, never dispersion across layers.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

try:
    from .spectral import aggregate_matrices, finite
except ImportError:
    from spectral import aggregate_matrices, finite

SEEDS = tuple(range(1340, 1345))
ARMS = ("sgdm", "adamw", "modified_adamw")
STEPS = tuple(range(0, 20001, 500))
ENDPOINTS = (2000, 20000)
T95 = {1: 12.706204736432095, 2: 4.302652729696142, 3: 3.182446305284263, 4: 2.7764451051977987}
LABELS = {"sgdm": "SGD + momentum", "adamw": "Native AdamW", "modified_adamw": "Modified AdamW"}
METRICS = tuple(f"{split}_{metric}" for split in ("train", "validation", "test")
                for metric in ("rmse", "mae", "mse")) + ("confidence_tail_alpha", "layer_alpha_sd")


def seed_statistics(values):
    """Equal-weight seed mean, sample SD and a two-sided 95% Student-t CI."""
    values = [float(value) for value in values if finite(value)]
    n = len(values)
    if n > 5:
        raise ValueError("This analysis is prespecified for at most five seeds")
    mean = statistics.fmean(values) if values else None
    sd = statistics.stdev(values) if n > 1 else None
    error = T95[n - 1] * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "sample_sd": sd, "ci95_half_width": error,
            "ci95_low": mean - error if error is not None else None,
            "ci95_high": mean + error if error is not None else None,
            "ci_method": "two-sided Student-t over seeds", "degrees_of_freedom": n - 1 if n else None,
            "complete_five_seeds": n == 5}


def _fingerprint(record, keys):
    for container in (record, record.get("execution", {}), record.get("execution_metadata", {}),
                      record.get("provenance", {}), record.get("fingerprints", {})):
        if isinstance(container, dict):
            for key in keys:
                value = container.get(key)
                if value is not None:
                    return value
    return None


def _canonical_config(config):
    excluded = {"seed", "optimizer", "arm", "phase", "output", "output_dir", "root", "run_dir",
                "results_dir", "resume", "checkpoint", "checkpoint_path", "start_time", "created_at"}
    return {key: value for key, value in config.items() if key not in excluded}


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load_contract(root, errors, local_reproduction=False):
    """Read the committed calibration decision, not a newly selected recipe."""
    frozen, plan, hashes = {}, {}, {}
    try:
        path = root / "frozen_selection.json"
        raw = path.read_bytes()
        frozen = json.loads(raw)
        hashes["frozen_selection_sha256"] = hashlib.sha256(raw).hexdigest()
        if frozen.get("calibration_complete") is not True:
            errors.append("frozen selection does not declare completed calibration")
        if local_reproduction and (frozen.get("github_run_id") is not None or frozen.get("launch_commit") is not None):
            errors.append("local reproduction cannot contain a GitHub frozen cohort")
        elif not local_reproduction and (not frozen.get("github_run_id") or not frozen.get("launch_commit")):
            errors.append("frozen selection lacks primary GitHub run ID or launch commit")
        if frozen.get("confirmation_seeds") != list(SEEDS):
            errors.append("frozen confirmation seeds do not match the five prespecified seeds")
        if frozen.get("development_seed") in SEEDS or frozen.get("development_seed") is None:
            errors.append("development seed is missing or overlaps confirmation seeds")
        if set(frozen.get("selected", {})) != set(ARMS):
            errors.append("frozen selection does not contain exactly the three optimizer recipes")
        plan = json.loads((root / "plan.json").read_text())
        hashes["plan_sha256"] = _canonical_sha256(plan)
        if plan.get("confirmation_seeds") != list(SEEDS) or plan.get("development_seed") != frozen.get("development_seed"):
            errors.append("archived plan and frozen selection disagree on development/confirmation seeds")
        source_path = root / "SOURCE_SHA256.json"
        if not source_path.exists():
            source_path = Path(__file__).with_name("SOURCE_SHA256.json")
        hashes["source_manifest_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        hashes["source_files"] = json.loads(source_path.read_text())["files"]
        if (not frozen.get("plan_file_sha256") or frozen["plan_file_sha256"] !=
                hashes["source_files"].get("california_housing/plan.json")):
            errors.append("original plan file hash disagrees with the frozen source manifest")
        original_plan = json.loads(source_path.with_name("plan.json").read_text())
        if _canonical_sha256(original_plan) != hashes["plan_sha256"]:
            errors.append("archived plan content differs from the frozen source plan")
        data_path = root / "calibration" / "data_manifest.json"
        hashes["data_manifest_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
        for field in ("source_manifest_sha256", "data_manifest_sha256"):
            if frozen.get(field) != hashes[field]:
                errors.append(f"archived {field} disagrees with frozen selection")
    except Exception as exc:
        errors.append(f"frozen cohort contract: {type(exc).__name__}: {exc}")
    return frozen, plan, hashes


def _history_row(record, measurement, issues):
    row = {"seed": record["seed"], "optimizer": record["optimizer"], "step": measurement["step"],
           "elapsed_seconds": measurement.get("elapsed_seconds")}
    for split in ("train", "validation", "test"):
        metrics = measurement.get(split) or {}
        for metric in ("rmse", "mae", "mse"):
            value = metrics.get(metric)
            row[f"{split}_{metric}"] = value if finite(value) else None
            if not finite(value) or value < 0:
                issues.append(f"seed {row['seed']} {row['optimizer']} step {row['step']}: invalid {split} {metric}")
        if finite(metrics.get("rmse")) and finite(metrics.get("mse")):
            if not math.isclose(metrics["rmse"] ** 2, metrics["mse"], rel_tol=1e-6, abs_tol=1e-10):
                issues.append(f"seed {row['seed']} {row['optimizer']} step {row['step']}: RMSE/MSE disagree")
    spectrum = measurement.get("spectral") or {}
    matrices = spectrum.get("matrices") or []
    recalculated = aggregate_matrices(matrices)
    row["confidence_tail_alpha"] = recalculated["confidence_tail_alpha"]
    row["layer_alpha_sd"] = recalculated["layer_alpha_sd"]
    row["n_valid_matrices"] = recalculated["n_valid"]
    row["spectral_complete"] = recalculated["complete"]
    saved = spectrum.get("confidence_tail_alpha")
    computed = recalculated["confidence_tail_alpha"]
    if (saved is None) != (computed is None) or (computed is not None and
            (not finite(saved) or not math.isclose(saved, computed, rel_tol=1e-10, abs_tol=1e-10))):
        issues.append(f"seed {row['seed']} {row['optimizer']} step {row['step']}: stored alpha disagrees with raw fits")
    if row["step"] in ENDPOINTS and not recalculated["complete"]:
        issues.append(f"seed {row['seed']} {row['optimizer']} step {row['step']}: incomplete eligible-matrix fits")
    return row


def audit_records(root, local_reproduction=False):
    root = Path(root)
    if local_reproduction:
        primary = Path(__file__).resolve().parent / "results"
        resolved = root.resolve()
        if resolved == primary or primary in resolved.parents or any(
                left == "california_housing" and right == "results"
                for left, right in zip(resolved.parts, resolved.parts[1:])):
            raise ValueError("Local reproduction must use a separate directory outside primary results")
    records, rows, errors, warnings, sources = [], [], [], [], []
    frozen, plan, expected_hashes = _load_contract(root, errors, local_reproduction)
    for seed in SEEDS:
        for arm in ARMS:
            path = root / "runs" / f"seed_{seed}" / arm / "result.json"
            if not path.exists():
                errors.append(f"missing result: {path.relative_to(root)}")
                continue
            try:
                raw = path.read_bytes()
                record = json.loads(raw)
                if record.get("seed") != seed or record.get("optimizer") != arm:
                    raise ValueError("seed/optimizer identity mismatch")
                if record.get("phase") != "confirmation":
                    raise ValueError("result is not from the confirmation phase")
                if record.get("status") != "complete" or record.get("completed_updates") != 20000:
                    raise ValueError("run has not completed exactly 20,000 updates")
                history = record.get("history", [])
                observed_steps = [measurement.get("step") for measurement in history]
                if observed_steps != list(STEPS):
                    raise ValueError("history must contain exactly steps 0..20000 at interval 500, in order")
                for field in ("plan_sha256", "data_manifest_sha256", "frozen_selection_sha256", "source_manifest_sha256", "source_bundle_sha256"):
                    if not record.get(field):
                        raise ValueError(f"missing {field}")
                    if field in expected_hashes and record[field] != expected_hashes[field]:
                        raise ValueError(f"{field} disagrees with archived frozen cohort contract")
                if not isinstance(record.get("config"), dict):
                    raise ValueError("missing config")
                if record["config"] != frozen.get("selected", {}).get(arm):
                    raise ValueError("optimizer config differs from frozen selection")
                for checkpoint in ("checkpoint_step_2000.pt", "checkpoint_step_20000.pt", "resume.pt"):
                    checkpoint_path = path.parent / checkpoint
                    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
                        raise ValueError(f"missing or empty retained checkpoint: {checkpoint}")
                executions = [record.get("execution", {})] + record.get("resume_executions", [])
                for execution in executions:
                    github = execution.get("github", {})
                    if local_reproduction:
                        if github.get("GITHUB_RUN_ID") is not None or github.get("GITHUB_SHA") is not None:
                            raise ValueError("local reproduction cannot mix GitHub execution records")
                    else:
                        if not github.get("GITHUB_RUN_ID") or str(github["GITHUB_RUN_ID"]) != str(frozen.get("github_run_id")):
                            raise ValueError("execution is outside the primary GitHub workflow run")
                        if github.get("GITHUB_SHA") != frozen.get("launch_commit"):
                            raise ValueError("execution source commit differs from frozen launch commit")
                    if (execution.get("precision") != "float32" or execution.get("torch_num_threads") != 1
                            or execution.get("torch_num_interop_threads") != 1):
                        raise ValueError("execution violates CPU float32/one-thread protocol")
                    if execution.get("deterministic_algorithms") is not True or any(
                            execution.get("thread_environment", {}).get(key) != "1"
                            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
                        raise ValueError("execution deterministic/thread environment violates frozen protocol")
                    if execution.get("source_bundle_sha256") != record["source_bundle_sha256"]:
                        raise ValueError("execution source bundle differs from result identity")
                    source = execution.get("source_sha256")
                    if not isinstance(source, dict) or _canonical_sha256(source) != record["source_bundle_sha256"]:
                        raise ValueError("execution source bundle hash does not match source file hashes")
                    for name in ("data.py", "train.py", "optimizer.py", "spectral.py"):
                        if source.get(name) != expected_hashes.get("source_files", {}).get(f"california_housing/{name}") or not source.get(name):
                            raise ValueError(f"execution {name} source hash differs from frozen source manifest")
                    packages = execution.get("packages", {})
                    if (not execution.get("python") or any(not packages.get(key) for key in
                            ("torch", "numpy", "scipy", "scikit-learn", "weightwatcher", "powerlaw"))):
                        raise ValueError("execution numerical package/Python provenance is incomplete")
                    if packages["weightwatcher"] != "0.7.7" or packages["powerlaw"] != "2.0.0":
                        raise ValueError("execution spectral package versions differ from frozen protocol")
                sources.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()})
                records.append(record)
                rows.extend(_history_row(record, item, errors) for item in history)
            except Exception as exc:
                errors.append(f"{path.relative_to(root)}: {type(exc).__name__}: {exc}")
    for field in ("plan_sha256", "data_manifest_sha256", "source_bundle_sha256", "source_manifest_sha256", "frozen_selection_sha256"):
        if len({record[field] for record in records}) > 1:
            errors.append(f"{field} differs between cohort runs")
    numerical_environments = [json.dumps({"python": execution["python"], "packages": execution["packages"]}, sort_keys=True)
                              for record in records
                              for execution in [record["execution"]] + record.get("resume_executions", [])]
    if len(set(numerical_environments)) > 1:
        errors.append("numerical package/Python versions differ between cohort runs")
    # Each arm's entire numerical configuration must remain frozen across seeds.
    for arm in ARMS:
        configs = [json.dumps(_canonical_config(record["config"]), sort_keys=True)
                   for record in records if record["optimizer"] == arm]
        if len(set(configs)) > 1:
            errors.append(f"{arm}: numerical configuration differs across seeds")
    # Common design settings must also match between optimizer arms.
    common_fields = ("architecture", "model", "hidden_dim", "hidden_dims", "hidden_layers", "batch_size",
                     "max_steps", "updates", "total_updates", "eval_interval", "spectral_interval",
                     "grad_clip", "clip_grad_norm", "schedule", "lr_schedule", "warmup_steps",
                     "min_lr_ratio", "dtype", "num_threads")
    for field in common_fields:
        values = [json.dumps(record["config"][field], sort_keys=True)
                  for record in records if field in record["config"]]
        if len(set(values)) > 1:
            errors.append(f"common numerical setting {field} differs between cohort runs")
    for seed in SEEDS:
        paired = [record for record in records if record["seed"] == seed]
        for description, keys in (
            ("initialization", ("initialization_sha256", "initial_state_sha256", "initial_weights_sha256")),
            ("minibatch order", ("minibatch_sha256", "minibatch_rng_sha256", "minibatch_order_sha256", "batch_sequence_sha256")),
        ):
            values = [_fingerprint(record, keys) for record in paired]
            if paired and any(value is None for value in values):
                errors.append(f"seed {seed}: {description} fingerprint unavailable in one or more results")
            supplied = [json.dumps(value, sort_keys=True) for value in values if value is not None]
            if len(set(supplied)) > 1:
                errors.append(f"seed {seed}: paired {description} fingerprints differ")
    return records, rows, errors, warnings, sources


def summarize_rows(rows):
    summaries = []
    for step in STEPS:
        for arm in ARMS:
            subset = [row for row in rows if row["step"] == step and row["optimizer"] == arm]
            for metric in METRICS:
                summaries.append({"optimizer": arm, "step": step, "metric": metric,
                                  **seed_statistics([row.get(metric) for row in subset])})
    paired = []
    index = {(row["seed"], row["optimizer"], row["step"]): row for row in rows}
    for step in ENDPOINTS:
        for metric in METRICS:
            differences = []
            seed_differences = {}
            for seed in SEEDS:
                modified = index.get((seed, "modified_adamw", step), {}).get(metric)
                native = index.get((seed, "adamw", step), {}).get(metric)
                if finite(modified) and finite(native):
                    differences.append(modified - native)
                    seed_differences[str(seed)] = modified - native
            paired.append({"comparison": "modified_adamw_minus_adamw", "step": step, "metric": metric,
                           "seed_differences": seed_differences, **seed_statistics(differences)})
    return summaries, paired


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _display(stat):
    if stat["mean"] is None:
        return "unavailable (n=0)"
    if stat["ci95_half_width"] is None:
        return f"{stat['mean']:.4f} (n={stat['n']}; CI unavailable)"
    return f"{stat['mean']:.4f} ± {stat['ci95_half_width']:.4f} (n={stat['n']})"


def write_outputs(root, allow_incomplete=False, local_reproduction=False):
    records, rows, errors, warnings, sources = audit_records(root, local_reproduction)
    statistics_rows, paired = summarize_rows(rows)
    complete = len(records) == 15 and not errors
    analysis = Path(root) / "analysis"
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    n_complete_seeds = sum(all(any(record["seed"] == seed and record["optimizer"] == arm
                                  for record in records) for arm in ARMS) for seed in SEEDS)
    cohort_type = "local_reproduction" if local_reproduction else "primary_github"
    summary = {"schema_version": 1, "cohort_type": cohort_type, "complete_five_seeds": complete, "n_completed": n_complete_seeds,
               "n_completed_runs": len(records), "n_expected_runs": 15, "seeds": list(SEEDS),
               "n_complete_seed_triplets": n_complete_seeds,
               "optimizers": list(ARMS), "audit_errors": errors, "audit_warnings": warnings,
               "paired_fingerprints_available": not any("fingerprint unavailable" in error for error in errors),
               "method": "Equal-weight seed means; sample SD (ddof=1); 95% Student-t CI over seeds.",
               "spectral_scope": "fc2.weight, fc3.weight, fc4.weight only; 128x128 hidden matrices. Input and readout excluded a priori.",
               "source_files": sources,
               "statistics": statistics_rows, "paired_modified_minus_native": paired}
    cohort = {"schema_version": 1, "cohort_type": cohort_type,
              "primary_source": "Local CPU reproduction; excluded from primary cohort" if local_reproduction else "GitHub Actions only",
              "complete_five_seeds": complete,
              "n_completed": n_complete_seeds, "n_completed_runs": len(records),
              "executions": [{"seed": record["seed"], "optimizer": record["optimizer"],
                              "execution": record["execution"], "resume_executions": record.get("resume_executions", []),
                              **{field: record[field] for field in ("plan_sha256", "data_manifest_sha256",
                                 "source_bundle_sha256", "source_manifest_sha256", "frozen_selection_sha256")}}
                             for record in records], "audit_errors": errors}
    (analysis / "execution_cohort.json").write_text(json.dumps(cohort, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(analysis / "seed_metrics.csv", rows)
    _write_csv(analysis / "summary.csv", statistics_rows)
    _write_csv(analysis / "paired_effects.csv", [{key: value for key, value in row.items() if key != "seed_differences"} for row in paired])
    lookup = {(row["optimizer"], row["step"], row["metric"]): row for row in statistics_rows}
    text = ["# California Housing: three-optimizer confirmation cohort", "",
            "**LOCAL REPRODUCTION — excluded from the primary GitHub cohort.**" if local_reproduction else "Primary source: the frozen GitHub Actions cohort.", "",
            "**COMPLETE: 15 runs, five seeds per optimizer.**" if complete else "**INCOMPLETE OR FAILED AUDIT — not a completed five-seed comparison.**", "",
            "All ± values below are two-sided 95% Student-t confidence half-widths over independent seed results; they are not layer error bars.",
            "Sample SD and interval endpoints are retained separately in summary.csv and summary.json.", "",
            "| Updates | Optimizer | Test RMSE ↓ | Test MAE ↓ | Confidence-tail average α |",
            "|---:|---|---:|---:|---:|"]
    for step in ENDPOINTS:
        for arm in ARMS:
            cells = [_display(lookup[(arm, step, metric)]) for metric in
                     ("test_rmse", "test_mae", "confidence_tail_alpha")]
            text.append(f"| {step:,} | {LABELS[arm]} | " + " | ".join(cells) + " |")
    text += ["", "## Paired modified-minus-native effects", "",
             "Negative RMSE/MAE differences favor modified AdamW; an α difference has no universal better/worse sign.", "",
             "| Updates | Metric | Mean paired difference ± 95% CI |", "|---:|---|---:|"]
    for row in paired:
        if row["metric"] in ("test_rmse", "test_mae", "confidence_tail_alpha"):
            text.append(f"| {row['step']:,} | {row['metric']} | {_display(row)} |")
    text += ["", "## Interpretation and provenance", "",
             "The spectral average uses three prespecified 128×128 hidden matrices, not all five linear weights. Each fit uses ordinary α with weight sqrt(num_pl_spikes)/(D+0.02), without a matrix-size factor. A seed average requires all three fits. Layer α sample SD is recorded separately and never used as the seed uncertainty.",
             "", "Prediction error and spectral behavior answer separate questions. Neither α near 2 nor a lower training error alone establishes better held-out performance.",
             "", "Test RMSE and MAE are primary performance outcomes, in units of $100,000. Validation metrics served calibration and remain secondary results in the machine-readable tables. Test data never select the frozen recipes.",
             "", "The figures show seed means with pointwise 95% Student-t confidence bands; they are not simultaneous trajectory bands."]
    if errors:
        text += ["", "## Audit errors", ""] + [f"- {error}" for error in errors]
    if warnings:
        text += ["", "## Audit warnings", ""] + [f"- {warning}" for warning in warnings]
    (analysis / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    _plot(statistics_rows, figures, complete, local_reproduction)
    if not complete and not allow_incomplete:
        raise ValueError("Cohort incomplete or audit failed; diagnostic outputs saved. Use --allow-incomplete only for clearly labeled progress analysis.")
    return summary


def _plot(rows, destination, complete, local_reproduction=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"sgdm": "#276f4b", "adamw": "#bd6330", "modified_adamw": "#18497a"}
    for metric, filename, ylabel in (("confidence_tail_alpha", "alpha_vs_updates.png", "Confidence-tail average α"),
                                      ("test_rmse", "rmse_vs_updates.png", "Test RMSE ($100,000; lower is better)")):
        fig, axis = plt.subplots(figsize=(6.4, 3.7), constrained_layout=True)
        for arm in ARMS:
            subset = [row for row in rows if row["optimizer"] == arm and row["metric"] == metric and row["n"] > 0]
            if not subset:
                continue
            x = [row["step"] for row in subset]
            # A complete-cohort curve must not silently change its seed count.
            y = [row["mean"] if not complete or row["n"] == 5 else float("nan") for row in subset]
            axis.plot(x, y, label=LABELS[arm], color=colors[arm], linewidth=1.8)
            low = [row["ci95_low"] if row["ci95_low"] is not None and (not complete or row["n"] == 5) else float("nan") for row in subset]
            high = [row["ci95_high"] if row["ci95_high"] is not None and (not complete or row["n"] == 5) else float("nan") for row in subset]
            axis.fill_between(x, low, high, color=colors[arm], alpha=0.14, linewidth=0)
        if metric == "confidence_tail_alpha":
            axis.axhline(2, color="#6a6a6a", linewidth=1, linestyle="--", zorder=0)
        axis.set(xlabel="Optimizer updates", ylabel=ylabel, xlim=(0, 20000))
        title = "Five seeds; pointwise 95% Student-t CI" if complete else "INCOMPLETE cohort; available seeds only"
        if local_reproduction:
            title = "LOCAL reproduction: " + title
        axis.set_title(title, fontsize=9 if local_reproduction else 10)
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False, fontsize=8)
        fig.savefig(destination / filename, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--local-reproduction", action="store_true",
                        help="Explicitly label a separate local-only cohort; forbidden inside primary results")
    args = parser.parse_args()
    summary = write_outputs(args.root, args.allow_incomplete, args.local_reproduction)
    print(json.dumps({key: summary[key] for key in ("cohort_type", "complete_five_seeds", "n_completed_runs", "n_complete_seed_triplets")}, indent=2))


if __name__ == "__main__":
    main()

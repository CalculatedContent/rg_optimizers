"""Audit the published-MLP optimizer experiment and compute seed uncertainty.

Run: python -m california_housing_benchmark.analysis --root RESULTS_DIR
Fixed-update results (2,000/20,000) and the first validation-stopped result
are distinct estimands. No test metric selects a checkpoint or recipe.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t as student_t

from .spectral import aggregate_matrices, finite

SEEDS = tuple(range(2340, 2345))
ARMS = ("sgdm", "adamw", "modified_adamw")
STEPS = tuple(range(0, 20001, 500))
ENDPOINTS = (2000, 20000)
LABELS = {"sgdm": "SGD + momentum", "adamw": "Native AdamW", "modified_adamw": "Modified AdamW"}
SPLITS = ("train", "validation", "test")
PREDICTION_METRICS = ("rmse", "mae", "mse", "r2", "mape_percent")
METRICS = tuple(f"{split}_{metric}" for split in SPLITS for metric in PREDICTION_METRICS) + ("confidence_tail_alpha",)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seed_statistics(values, expected_n=5):
    """Equal-weight seed mean, sample SD, two-sided Student-t 95% interval.

    Missing/nonfinite values remain visibly incomplete through n and
    complete_expected_seeds. A single observation never receives a zero CI.
    """
    valid = [float(value) for value in values if finite(value)]
    n = len(valid)
    mean = statistics.fmean(valid) if n else None
    sd = statistics.stdev(valid) if n > 1 else None
    critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else None
    half = critical * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "sample_sd": sd,
            "standard_error": sd / math.sqrt(n) if sd is not None else None,
            "ci95_half_width": half, "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None,
            "ci_method": "two-sided Student-t over seeds", "confidence_level": 0.95,
            "degrees_of_freedom": n - 1 if n else None, "t_critical": critical,
            "expected_n": expected_n, "complete_expected_seeds": n == expected_n,
            "complete_five_seeds": n == 5 and expected_n == 5}


def independent_difference(current, reference):
    """Welch interval for independent seed means; not an equivalence test."""
    left, right = current.get("mean"), reference.get("mean")
    result = {"mean_difference": left - right if finite(left) and finite(right) else None,
              "ci95_low": None, "ci95_high": None, "ci95_half_width": None,
              "degrees_of_freedom": None,
              "method": "independent-seed Welch Student-t interval; not paired; not an equivalence test"}
    if current.get("n", 0) < 2 or reference.get("n", 0) < 2:
        return result
    a = current["sample_sd"] ** 2 / current["n"]
    b = reference["sample_sd"] ** 2 / reference["n"]
    if a + b == 0:
        # A degenerate empirical estimate does not justify a finite-df t claim.
        return result
    df = (a + b) ** 2 / (a ** 2 / (current["n"] - 1) + b ** 2 / (reference["n"] - 1))
    half = float(student_t.ppf(0.975, df)) * math.sqrt(a + b)
    result.update(degrees_of_freedom=df, ci95_half_width=half,
                  ci95_low=result["mean_difference"] - half,
                  ci95_high=result["mean_difference"] + half)
    return result


def _read_contract(root, errors, local_reproduction):
    frozen, plan, expected = {}, {}, {}
    try:
        frozen_raw = (root / "frozen_selection.json").read_bytes()
        frozen = json.loads(frozen_raw)
        expected["frozen_selection_sha256"] = hashlib.sha256(frozen_raw).hexdigest()
        if frozen.get("calibration_complete") is not True:
            errors.append("frozen selection does not declare completed calibration")
        if frozen.get("confirmation_seeds") != list(SEEDS):
            errors.append("frozen selection does not specify confirmation seeds 2340–2344")
        if set(frozen.get("selected", {})) != set(ARMS):
            errors.append("frozen selection does not contain exactly the three optimizer recipes")
        if local_reproduction:
            if frozen.get("github_run_id") or frozen.get("launch_commit"):
                errors.append("a local-only cohort cannot use a frozen primary GitHub execution identity")
        elif not frozen.get("github_run_id") or not frozen.get("launch_commit"):
            errors.append("primary frozen selection lacks GitHub run ID or launch commit")
        plan = json.loads((root / "plan.json").read_text())
        expected["plan_sha256"] = canonical_sha256(plan)
        if plan.get("confirmation_seeds") != list(SEEDS):
            errors.append("archived plan does not specify confirmation seeds 2340–2344")
        development = plan.get("development_seeds", [])
        if not development or set(development).intersection(SEEDS):
            errors.append("development seeds are absent or overlap confirmation seeds")
        if frozen.get("development_seeds") != development:
            errors.append("plan and frozen selection disagree on development seeds")
        source_path = root / "SOURCE_SHA256.json"
        if not source_path.exists():
            source_path = Path(__file__).with_name("SOURCE_SHA256.json")
        source_raw = source_path.read_bytes()
        expected["source_manifest_sha256"] = hashlib.sha256(source_raw).hexdigest()
        expected["source_files"] = json.loads(source_raw)["files"]
        if frozen.get("plan_file_sha256") != expected["source_files"].get("california_housing_benchmark/plan.json"):
            errors.append("original plan file hash disagrees with frozen source manifest")
        if canonical_sha256(json.loads(Path(__file__).with_name("plan.json").read_text())) != expected["plan_sha256"]:
            errors.append("archived plan differs from the frozen source plan")
        data_raw = (root / "calibration" / "data_manifest.json").read_bytes()
        expected["data_manifest_sha256"] = hashlib.sha256(data_raw).hexdigest()
        for key in ("source_manifest_sha256", "data_manifest_sha256"):
            if frozen.get(key) != expected[key]:
                errors.append(f"archived {key} differs from frozen selection")
    except Exception as exc:
        errors.append(f"frozen cohort contract: {type(exc).__name__}: {exc}")
    return frozen, plan, expected


def _history_row(record, measurement, endpoint, issues, warnings):
    seed, arm = record["seed"], record["optimizer"]
    label = f"seed {seed} {arm} {endpoint} step {measurement.get('step')}"
    row = {"seed": seed, "optimizer": arm, "endpoint": endpoint,
           "step": measurement.get("step"), "epoch": measurement.get("epoch"),
           "elapsed_seconds": measurement.get("elapsed_seconds"),
           "minibatch_sha256": measurement.get("minibatch_sha256")}
    for split in SPLITS:
        metrics = measurement.get(split) or {}
        for metric in PREDICTION_METRICS:
            value = metrics.get(metric)
            row[f"{split}_{metric}"] = value if finite(value) else None
            if not finite(value) or (metric != "r2" and value < 0):
                issues.append(f"{label}: invalid {split} {metric}")
        if finite(metrics.get("rmse")) and finite(metrics.get("mse")):
            if not math.isclose(metrics["rmse"] ** 2, metrics["mse"], rel_tol=1e-6, abs_tol=1e-10):
                issues.append(f"{label}: RMSE squared does not equal MSE")
    spectrum = measurement.get("spectral") or {}
    matrices = spectrum.get("matrices") or []
    recalculated = aggregate_matrices(matrices, spectrum.get("eligible_matrices", []))
    row.update(confidence_tail_alpha=recalculated["confidence_tail_alpha"],
               layer_alpha_sd=None, n_valid_matrices=recalculated["n_valid"],
               spectral_complete=recalculated["complete"], n_eligible_matrices=spectrum.get("n_eligible"))
    computed, saved = recalculated["confidence_tail_alpha"], spectrum.get("confidence_tail_alpha")
    if ((saved is None) != (computed is None) or (computed is not None and
            (not finite(saved) or not math.isclose(saved, computed, rel_tol=1e-10, abs_tol=1e-10)))):
        issues.append(f"{label}: stored average alpha differs from the raw fit")
    if spectrum.get("n_eligible") != 1 or spectrum.get("layer_alpha_sd") is not None:
        issues.append(f"{label}: spectral scope must be one matrix, with no layer SD")
    for matrix in matrices:
        eigenvalues = matrix.get("eigenvalues", [])
        if (len(eigenvalues) != 784 or any(not finite(value) or value < 0 for value in eigenvalues)
                or eigenvalues != sorted(eigenvalues)):
            issues.append(f"{label}: retained diagnostic eigenvalues must be 784 finite sorted nonnegative values")
    if not recalculated["complete"]:
        target = issues if endpoint == "primary_best_validation" or row["step"] in ENDPOINTS else warnings
        target.append(f"{label}: missing or invalid fit of the sole eligible 784x784 hidden matrix")
    return row


def _audit_execution(record, frozen, expected, issues, local_reproduction):
    for execution in [record.get("execution", {})] + record.get("resume_executions", []):
        github = execution.get("github", {})
        if local_reproduction:
            if github.get("GITHUB_RUN_ID") or github.get("GITHUB_SHA"):
                issues.append("local-only reproduction includes GitHub execution records")
        else:
            if not github.get("GITHUB_RUN_ID") or str(github["GITHUB_RUN_ID"]) != str(frozen.get("github_run_id")):
                issues.append("execution does not belong to the primary GitHub workflow run")
            if github.get("GITHUB_SHA") != frozen.get("launch_commit"):
                issues.append("execution differs from the frozen launch commit")
        if (execution.get("precision") != "float32" or execution.get("torch_num_threads") != 1
                or execution.get("torch_num_interop_threads") != 1):
            issues.append("execution violates CPU float32/one-thread protocol")
        if execution.get("deterministic_algorithms") is not True or any(
                execution.get("thread_environment", {}).get(k) != "1"
                for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
            issues.append("execution violates deterministic/thread-environment protocol")
        source = execution.get("source_sha256", {})
        if (not source or canonical_sha256(source) != record.get("source_bundle_sha256")
                or execution.get("source_bundle_sha256") != record.get("source_bundle_sha256")):
            issues.append("execution source hashes disagree with result source bundle")
        for name in ("data.py", "train.py", "optimizer.py", "spectral.py"):
            if not source.get(name) or source[name] != expected.get("source_files", {}).get(f"california_housing_benchmark/{name}"):
                issues.append(f"execution {name} differs from frozen source manifest")
        packages = execution.get("packages", {})
        if not execution.get("python") or any(not packages.get(k) for k in
                ("torch", "numpy", "scipy", "scikit-learn", "weightwatcher", "powerlaw", "tabm")):
            issues.append("execution numerical package/Python provenance is incomplete")
        if packages.get("weightwatcher") != "0.7.7" or packages.get("powerlaw") != "2.0.0":
            issues.append("spectral package versions differ from frozen protocol")


def audit_records(root, local_reproduction=False):
    root = Path(root)
    primary = Path(__file__).resolve().parent / "results"
    if local_reproduction and (root.resolve() == primary or primary in root.resolve().parents):
        raise ValueError("Local reproduction must be separate from the primary results directory")
    records, rows, errors, warnings, sources, raw_fit_rows = [], [], [], [], [], []
    frozen, plan, expected = _read_contract(root, errors, local_reproduction)
    for seed in SEEDS:
        metadata = root / "runs" / f"seed_{seed}" / "github_execution.json"
        try:
            provenance = json.loads(metadata.read_text())
            if provenance.get("seed") != seed:
                raise ValueError("wrong seed in execution record")
            if local_reproduction:
                if provenance.get("github_run_id") or provenance.get("launch_commit"):
                    raise ValueError("local cohort includes GitHub provenance")
            elif (str(provenance.get("github_run_id")) != str(frozen.get("github_run_id"))
                    or provenance.get("launch_commit") != frozen.get("launch_commit")):
                raise ValueError("execution record differs from primary GitHub cohort")
        except Exception as exc:
            errors.append(f"{metadata.relative_to(root)}: {type(exc).__name__}: {exc}")
        for arm in ARMS:
            path = root / "runs" / f"seed_{seed}" / arm / "result.json"
            if not path.exists():
                errors.append(f"missing result: {path.relative_to(root)}")
                continue
            issues = []
            try:
                raw = path.read_bytes()
                record = json.loads(raw)
                if record.get("seed") != seed or record.get("optimizer") != arm or record.get("phase") != "confirmation":
                    raise ValueError("result seed, optimizer, or confirmation-phase identity mismatch")
                records.append(record)
                sources.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()})
                if record.get("status") != "complete" or record.get("completed_updates", 0) < 20000:
                    issues.append("run has not completed both 20,000 updates and its primary stopping rule")
                if record.get("parameter_count") != 623281:
                    issues.append("parameter count differs from the frozen 623,281-parameter MLP")
                history = record.get("history", [])
                observed = [item.get("step") for item in history]
                if (any(not isinstance(step, int) or isinstance(step, bool) for step in observed)
                        or observed != sorted(set(observed)) or not set(STEPS).issubset(observed)):
                    issues.append("history must include unique increasing steps 0..20000 every500; later measurements are permitted")
                for key in ("plan_sha256", "data_manifest_sha256", "frozen_selection_sha256", "source_manifest_sha256", "source_bundle_sha256"):
                    if not record.get(key) or (key in expected and record[key] != expected[key]):
                        issues.append(f"missing or mismatched {key}")
                if record.get("config") != frozen.get("selected", {}).get(arm) or not isinstance(record.get("config"), dict):
                    issues.append("optimizer config differs from frozen selection")
                for checkpoint in ("checkpoint_step_2000.pt", "checkpoint_step_20000.pt", "resume.pt", "best_checkpoint.pt"):
                    cp = path.parent / checkpoint
                    if not cp.is_file() or cp.stat().st_size == 0:
                        issues.append(f"missing or empty retained checkpoint: {checkpoint}")
                if not record.get("best_weights_sha256"):
                    issues.append("missing best-validation weights fingerprint")
                _audit_execution(record, frozen, expected, issues, local_reproduction)
                stopping, best = record.get("paper_stopping", {}), record.get("primary_best_validation")
                if not stopping.get("reached") or stopping.get("censored") or stopping.get("patience") != 16:
                    issues.append("first 16-bad-epoch stopping event is absent, censored, or uses different patience")
                measurements = [(item, "fixed_updates") for item in history]
                if not isinstance(best, dict):
                    issues.append("missing primary best-validation checkpoint measurement")
                else:
                    measurements.append((best, "primary_best_validation"))
                    if best.get("step") != stopping.get("best_step"):
                        issues.append("best-validation measurement step differs from the first stopping event")
                    if best.get("step", math.inf) > stopping.get("step", -1):
                        issues.append("primary checkpoint was selected after the first stopping event")
                    if not best.get("selection"):
                        issues.append("primary checkpoint lacks a validation-only selection declaration")
                for measurement, endpoint in measurements:
                    rows.append(_history_row(record, measurement, endpoint, issues, warnings))
                    for matrix in (measurement.get("spectral") or {}).get("matrices", []):
                        raw_fit_rows.append({"seed": seed, "optimizer": arm, "endpoint": endpoint,
                                             "step": measurement.get("step"), **matrix})
            except Exception as exc:
                issues.append(f"{type(exc).__name__}: {exc}")
            errors.extend(f"{path.relative_to(root)}: {issue}" for issue in issues)
    for field in ("plan_sha256", "data_manifest_sha256", "source_bundle_sha256", "source_manifest_sha256", "frozen_selection_sha256"):
        if len({record.get(field) for record in records}) > 1:
            errors.append(f"{field} differs across confirmation runs")
    environments = [canonical_sha256({"python": execution.get("python"), "packages": execution.get("packages")})
                    for record in records for execution in [record.get("execution", {})] + record.get("resume_executions", [])]
    if len(set(environments)) > 1:
        errors.append("Python/package versions differ across the confirmation cohort")
    for seed in SEEDS:
        paired = [record for record in records if record["seed"] == seed]
        initialization = [record.get("initialization_sha256", record.get("initial_weights_sha256")) for record in paired]
        if paired and (any(value is None for value in initialization) or len(set(initialization)) != 1):
            errors.append(f"seed {seed}: paired initialization fingerprints are missing or differ")
        # Terminal digests may differ because stopping times differ. Match
        # minibatch sequences at the fixed budgets, never at final termination.
        for step in ENDPOINTS:
            digests = [next((m.get("minibatch_sha256") for m in record.get("history", []) if m.get("step") == step), None)
                       for record in paired]
            if paired and (any(value is None for value in digests) or len(set(digests)) != 1):
                errors.append(f"seed {seed} step {step}: paired minibatch fingerprints are missing or differ")
    return records, rows, errors, warnings, sources, raw_fit_rows


def summarize_rows(rows):
    summaries, paired = [], []
    endpoints = [("fixed_updates", step) for step in STEPS] + [("primary_best_validation", None)]
    for endpoint, step in endpoints:
        for arm in ARMS:
            subset = [row for row in rows if row["optimizer"] == arm and row["endpoint"] == endpoint
                      and (step is None or row["step"] == step)]
            if len({row["seed"] for row in subset}) != len(subset):
                raise ValueError(f"duplicate seed measurements for {arm} {endpoint} {step}")
            for metric in METRICS:
                summaries.append({"optimizer": arm, "endpoint": endpoint, "step": step, "metric": metric,
                                  **seed_statistics([row.get(metric) for row in subset])})
        if step not in ENDPOINTS and endpoint != "primary_best_validation":
            continue
        for metric in METRICS:
            seed_differences = {}
            for seed in SEEDS:
                values = {row["optimizer"]: row.get(metric) for row in rows if row["seed"] == seed
                          and row["endpoint"] == endpoint and (step is None or row["step"] == step)}
                if finite(values.get("modified_adamw")) and finite(values.get("adamw")):
                    seed_differences[str(seed)] = values["modified_adamw"] - values["adamw"]
            paired.append({"comparison": "modified_adamw_minus_adamw", "endpoint": endpoint,
                           "step": step, "metric": metric, "seed_differences": seed_differences,
                           **seed_statistics(list(seed_differences.values()))})
    return summaries, paired


def published_comparison(statistics_rows, errors):
    """Recompute published seed statistics; compare only the aligned estimand."""
    path = Path(__file__).with_name("reference_manifest.json")
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        for item in manifest["files"]:
            source = path.parent / item["local_path"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
                errors.append(f"published source hash mismatch: {item['local_path']}")
        references = []
        for item in manifest["references"]:
            report_path = path.parent / "reference" / "upstream" / "exp" / "mlp" / item["optimizer"] / "california" / "evaluation" / "report.json"
            report = json.loads(report_path.read_text())
            experiments = sorted(report["experiments"], key=lambda entry: entry["config"]["seed"])
            published_seeds = [entry["config"]["seed"] for entry in experiments]
            values = [entry["report"]["metrics"]["test"]["rmse"] for entry in experiments]
            if published_seeds != item["seeds"] or values != item["individual_test_rmse"]:
                errors.append(f"published {item['optimizer']} manifest disagrees with author per-seed report")
            stats = seed_statistics(values, expected_n=item["n_seeds"])
            for stored, actual in (("test_rmse_mean", "mean"), ("test_rmse_sample_sd", "sample_sd")):
                if not math.isclose(item[stored], stats[actual], rel_tol=1e-10, abs_tol=1e-12):
                    errors.append(f"published {item['optimizer']} {stored} disagrees with individual seed records")
            references.append({**item, "recalculated_test_rmse": stats})
        native = next(row for row in statistics_rows if row["optimizer"] == "adamw"
                      and row["endpoint"] == "primary_best_validation" and row["metric"] == "test_rmse")
        original = next(item for item in references if item["optimizer"] == "adamw")
        return {"paper": manifest["paper"], "upstream_repository": manifest["upstream_repository"],
                "upstream_commit": manifest["upstream_commit"], "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "unit": manifest["unit"], "references": references,
                "native_replication_minus_published_adamw": independent_difference(native, original["recalculated_test_rmse"]),
                "interpretation": ("Published results use validation stopping and different training seeds/runtime. "
                    "The current native AdamW recipe matches the published architecture and settings; CPU results "
                    "need not be numerically identical to the authors' GPU results. Published SGD and Muon-EMA "
                    "use separately tuned architectures, so they are context, not same-model optimizer controls. "
                    "Fixed-update results must not be compared as if they used the paper's checkpoint selection. "
                    "A confidence interval containing zero does not establish equivalence or reproduction within a prespecified tolerance.")}
    except Exception as exc:
        errors.append(f"published-reference audit: {type(exc).__name__}: {exc}")
        return {}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _display(row):
    if row.get("mean") is None:
        return "unavailable (n=0)"
    if row.get("ci95_half_width") is None:
        return f"{row['mean']:.4f} (n={row['n']}; CI unavailable)"
    return f"{row['mean']:.4f} ± {row['ci95_half_width']:.4f} (n={row['n']})"


def write_outputs(root, allow_incomplete=False, local_reproduction=False):
    root = Path(root)
    records, rows, errors, warnings, sources, fit_rows = audit_records(root, local_reproduction)
    stats, paired = summarize_rows(rows)
    references = published_comparison(stats, errors)
    completed = [record for record in records if record.get("status") == "complete"
                 and record.get("completed_updates", 0) >= 20000
                 and record.get("paper_stopping", {}).get("reached")
                 and not record.get("paper_stopping", {}).get("censored")]
    n_seeds = sum(all(any(record["seed"] == seed and record["optimizer"] == arm for record in completed)
                      for arm in ARMS) for seed in SEEDS)
    complete = len(completed) == 15 and not errors
    cohort_type = "local_reproduction" if local_reproduction else "primary_github"
    analysis = root / "analysis"
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "cohort_type": cohort_type, "complete_five_seeds": complete,
               "n_completed": n_seeds, "n_completed_runs": len(completed), "n_expected_runs": 15,
               "n_complete_seed_triplets": n_seeds, "seeds": list(SEEDS), "optimizers": list(ARMS),
               "audit_errors": errors, "audit_warnings": warnings,
               "method": "Equal-weight seed means; sample SD (ddof=1); two-sided 95% Student-t CI across seeds.",
               "spectral_scope": ("One unique 784x784 hidden matrix passes min(shape)>=64. The confidence/tail "
                   "average equals that single ordinary fitted alpha. Input/readout are excluded a priori; "
                   "layer dispersion is undefined. This is not a collective many-layer marginality test."),
               "performance_units": {"rmse": "100000 USD", "mae": "100000 USD", "mse": "(100000 USD)^2",
                                     "r2": "dimensionless", "mape_percent": "percent; supplementary"},
               "endpoint_policy": ("2,000 and 20,000 updates are fixed budgets. primary_best_validation uses "
                   "the minimum validation RMSE before the first 16-bad-epoch stop. Later training cannot "
                   "replace that primary checkpoint. Test outcomes never select a model."),
               "source_files": sources, "statistics": stats,
               "paired_modified_minus_native": paired, "published_comparison": references}
    cohort = {"schema_version": 1, "cohort_type": cohort_type,
              "primary_source": "Local CPU reproduction; excluded from primary cohort" if local_reproduction else "GitHub Actions only",
              "complete_five_seeds": complete, "n_completed": n_seeds, "n_completed_runs": len(completed),
              "executions": [{"seed": record["seed"], "optimizer": record["optimizer"],
                              "execution": record.get("execution"), "resume_executions": record.get("resume_executions", []),
                              "paper_stopping": record.get("paper_stopping"),
                              **{key: record.get(key) for key in ("plan_sha256", "data_manifest_sha256", "source_bundle_sha256",
                                  "source_manifest_sha256", "frozen_selection_sha256", "initialization_sha256", "best_weights_sha256")}}
                             for record in records], "audit_errors": errors}
    _write_json(analysis / "summary.json", summary)
    _write_json(analysis / "execution_cohort.json", cohort)
    _write_json(analysis / "published_comparison.json", references)
    _write_json(analysis / "raw_matrix_fits.json", fit_rows)
    _write_csv(analysis / "seed_metrics.csv", rows)
    _write_csv(analysis / "summary.csv", stats)
    _write_csv(analysis / "paired_effects.csv", [{key: value for key, value in item.items() if key != "seed_differences"} for item in paired])
    _write_report(analysis, summary)
    _plot(stats, figures, complete, local_reproduction)
    if not complete and not allow_incomplete:
        raise ValueError("Cohort incomplete or failed audit; diagnostic outputs saved. --allow-incomplete labels progress only.")
    return summary


def _write_report(destination, summary):
    lookup = {(row["optimizer"], row["endpoint"], row["step"], row["metric"]): row for row in summary["statistics"]}
    text = ["# California Housing: published-MLP optimizer benchmark", "",
            "**LOCAL REPRODUCTION — excluded from the primary GitHub cohort.**" if summary["cohort_type"] == "local_reproduction" else "Primary source: the frozen GitHub Actions cohort.", "",
            "**COMPLETE: 15 runs, five seeds per optimizer.**" if summary["complete_five_seeds"] else "**INCOMPLETE OR FAILED AUDIT — not a completed five-seed comparison.**", "",
            "All ± entries in our tables are two-sided 95% Student-t confidence half-widths across seeds. "
            "The machine-readable tables also provide sample SD and interval endpoints. These are not layer error bars.", "",
            "## Fixed training budgets", "",
            "| Updates | Optimizer | Test RMSE ↓ | Test MAE ↓ | Confidence/tail average α |",
            "|---:|---|---:|---:|---:|"]
    for step in ENDPOINTS:
        for arm in ARMS:
            cells = [_display(lookup[(arm, "fixed_updates", step, metric)]) for metric in ("test_rmse", "test_mae", "confidence_tail_alpha")]
            text.append(f"| {step:,} | {LABELS[arm]} | " + " | ".join(cells) + " |")
    text += ["", "## Primary validation-stopped comparison", "",
             "Each seed contributes the best validation-RMSE checkpoint before its first 16-bad-epoch stop. "
             "Selection can occur at different update counts; the fixed-budget table answers a separate question.", "",
             "| Optimizer | Test RMSE ↓ | Test MAE ↓ | Confidence/tail average α |",
             "|---|---:|---:|---:|"]
    for arm in ARMS:
        cells = [_display(lookup[(arm, "primary_best_validation", None, metric)]) for metric in ("test_rmse", "test_mae", "confidence_tail_alpha")]
        text.append(f"| {LABELS[arm]} | " + " | ".join(cells) + " |")
    references = summary["published_comparison"]
    if references:
        text += ["", "## Published reference", "", "The following ± values are **published seed sample SD**, not confidence half-widths.", "",
                 "| Published optimizer | Parameters | Test RMSE mean ± sample SD | Seeds |",
                 "|---|---:|---:|---:|"]
        for item in references["references"]:
            stat = item["recalculated_test_rmse"]
            text.append(f"| {item['optimizer']} | {item['n_parameters']:,} | {stat['mean']:.6f} ± {stat['sample_sd']:.6f} | {stat['n']} |")
        effect = references["native_replication_minus_published_adamw"]
        if effect["ci95_half_width"] is not None:
            text += ["", f"Our native AdamW primary mean minus the published AdamW mean is {effect['mean_difference']:+.6f} "
                     f"± {effect['ci95_half_width']:.6f} (95% Welch interval over independent seed means)."]
        text += ["", references["interpretation"], "",
                 f"Source: [{references['paper']['title']}]({references['paper']['arxiv']}), {references['paper']['table']}; "
                 f"[authors' repository]({references['upstream_repository']}) at `{references['upstream_commit']}`."]
    text += ["", "## Paired modified-minus-native effects", "",
             "Negative RMSE/MAE differences favor modified AdamW. An α difference has no universal better/worse sign.", "",
             "| Endpoint | Metric | Mean difference ± 95% CI |", "|---|---|---:|"]
    for row in summary["paired_modified_minus_native"]:
        if row["metric"] in ("test_rmse", "test_mae", "confidence_tail_alpha"):
            endpoint = f"{row['step']:,} updates" if row["step"] is not None else "Primary validation stop"
            text.append(f"| {endpoint} | {row['metric']} | {_display(row)} |")
    text += ["", "## Scope and interpretation", "", summary["spectral_scope"], "",
             "The 8→784→784→1 MLP has 623,281 parameters. Each raw spectral fit uses ordinary α with weight "
             "sqrt(num_pl_spikes)/(D+0.02), with no extra matrix-size factor. WeightWatcher's alpha_weighted is not substituted.", "",
             "RMSE and MAE use $100,000 units: RMSE 0.50 is $50,000, not 50% error. Lower is better. "
             "MSE uses squared units; R² and MAPE are supplementary and are retained in the data tables.", "",
             "The calibration runs use separate development seeds. Native AdamW uses the published recipe. "
             "SGD-momentum and graft calibration use validation data only, and their settings are frozen before confirmation.", "",
             "Prediction error and spectral behavior answer separate questions. A fitted α near 2 does not by itself "
             "establish good held-out prediction. This one-matrix architecture cannot establish collective many-layer marginality.", "",
             "Trajectory bands are pointwise 95% Student-t intervals across seeds, not simultaneous bands. "
             "A gap or missing value means the measurement was unavailable or invalid; it is never replaced by zero."]
    for heading, key in (("Audit errors", "audit_errors"), ("Audit warnings", "audit_warnings")):
        if summary[key]:
            text += ["", f"## {heading}", ""] + [f"- {issue}" for issue in summary[key]]
    (destination / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _plot(rows, destination, complete, local_reproduction=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"sgdm": "#276f4b", "adamw": "#bd6330", "modified_adamw": "#18497a"}
    for metric, name, ylabel in (("confidence_tail_alpha", "alpha_vs_updates", "Hidden-matrix α (model average)"),
                                  ("test_rmse", "rmse_vs_updates", "Test RMSE ($100,000; lower is better)")):
        fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
        for arm in ARMS:
            subset = [row for row in rows if row["optimizer"] == arm and row["metric"] == metric and row["endpoint"] == "fixed_updates"]
            if not any(row["n"] for row in subset):
                continue
            x = [row["step"] for row in subset]
            # Keep all scheduled x values so a missing fit makes a visible gap.
            valid = lambda row: row["n"] == 5 if complete else row["n"] > 0
            y = [row["mean"] if valid(row) else float("nan") for row in subset]
            axis.plot(x, y, label=LABELS[arm], color=colors[arm], linewidth=1.8)
            low = [row["ci95_low"] if valid(row) and row["ci95_low"] is not None else float("nan") for row in subset]
            high = [row["ci95_high"] if valid(row) and row["ci95_high"] is not None else float("nan") for row in subset]
            axis.fill_between(x, low, high, color=colors[arm], alpha=0.14, linewidth=0)
        if metric == "confidence_tail_alpha":
            axis.axhline(2, color="#6a6a6a", linewidth=1, linestyle="--", zorder=0)
        axis.set(xlabel="Optimizer updates", ylabel=ylabel, xlim=(0, 20000))
        title = "Five seeds; pointwise 95% Student-t CI" if complete else "INCOMPLETE / audit pending; available seeds only"
        if local_reproduction:
            title = "LOCAL: " + title
        axis.set_title(title, fontsize=9)
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False, fontsize=8)
        fig.savefig(destination / f"{name}.png", dpi=180)
        fig.savefig(destination / f"{name}.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--local-reproduction", action="store_true")
    args = parser.parse_args()
    summary = write_outputs(args.root, args.allow_incomplete, args.local_reproduction)
    print(json.dumps({key: summary[key] for key in ("cohort_type", "complete_five_seeds", "n_completed_runs", "n_complete_seed_triplets")}, indent=2))


if __name__ == "__main__":
    main()

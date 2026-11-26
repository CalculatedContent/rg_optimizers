"""Audit the SGDM search, exact continuations, and fresh confirmation cohort.

Development plots use validation data only. Every RMSE/alpha point is bound
to one weight hash. Configurations are never treated as independent seeds.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t as student_t

DEV_SEEDS = (3338, 3339)
FRESH_SEEDS = tuple(range(3340, 3345))
PARENT_SEEDS = tuple(range(2340, 2345))
PARENT_RUN = "45753214421"
PARENT_SOURCE = "67a0bf203ab0f6b34565e37b4f510f123f669876"
END = 100000
DEV_END = 50000
METRICS = ("test_rmse", "test_mae", "test_mse", "test_r2", "test_mape_percent",
           "validation_rmse", "train_rmse", "confidence_tail_alpha")
GRID_ENDPOINTS = (("fixed_updates", 2000), ("fixed_updates", 20000),
                  ("fixed_updates", 50000), ("best_validation", None))
CONFIG_KEYS = ("optimizer", "lr", "momentum", "weight_decay", "dampening", "nesterov", "dropout")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def seed_statistics(values, expected_n=5):
    values = [float(value) for value in values if finite(value)]
    n = len(values)
    mean = statistics.fmean(values) if n else None
    sd = statistics.stdev(values) if n > 1 else None
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "sample_sd": sd,
            "ci95_half_width": half, "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None,
            "degrees_of_freedom": n - 1 if n else None,
            "ci_method": "two-sided 95% Student-t across seeds",
            "expected_n": expected_n, "complete_expected_seeds": n == expected_n}


def paired_effect(left, right, seeds):
    """A contrast of matched seeds; never subtract independent cohort CIs."""
    differences = {str(seed): left[seed] - right[seed] for seed in seeds
                   if finite(left.get(seed)) and finite(right.get(seed))}
    return {"seed_differences": differences, **seed_statistics(list(differences.values()), len(seeds))}


def welch_effect(left, right):
    a, b = seed_statistics(left), seed_statistics(right)
    result = {"n_left": a["n"], "n_right": b["n"],
              "mean_difference": a["mean"] - b["mean"] if a["n"] and b["n"] else None,
              "ci95_half_width": None, "ci95_low": None, "ci95_high": None,
              "degrees_of_freedom": None,
              "ci_method": "independent-seed 95% Welch Student-t; not an equivalence test"}
    if min(a["n"], b["n"]) < 2:
        return result
    va, vb = a["sample_sd"] ** 2 / a["n"], b["sample_sd"] ** 2 / b["n"]
    if va + vb == 0:
        return result
    df = (va + vb) ** 2 / (va ** 2 / (a["n"] - 1) + vb ** 2 / (b["n"] - 1))
    half = float(student_t.ppf(0.975, df)) * math.sqrt(va + vb)
    result.update(degrees_of_freedom=df, ci95_half_width=half,
                  ci95_low=result["mean_difference"] - half, ci95_high=result["mean_difference"] + half)
    return result


def spectral_alpha(spectrum):
    """One eligible matrix: the confidence/tail average equals its raw alpha."""
    spectrum = spectrum or {}
    matrices = spectrum.get("matrices") or []
    if len(matrices) != 1 or spectrum.get("n_eligible") != 1:
        return None
    row = matrices[0]
    if row.get("shape") != [784, 784] or spectrum.get("eligible_matrices") != [row.get("matrix")]:
        return None
    fit = row.get("fit") or {}
    alpha, distance, tail, size = (fit.get(k) for k in ("alpha", "D", "num_pl_spikes", "num_evals"))
    if not (fit.get("status") == "success" and finite(alpha) and alpha > 0
            and finite(distance) and 0 <= distance <= 1
            and isinstance(tail, int) and not isinstance(tail, bool)
            and isinstance(size, int) and not isinstance(size, bool) and 0 < tail <= size <= 784):
        return None
    # Formula retained explicitly; with one matrix the weight cancels.
    weight = math.sqrt(tail) / (distance + 0.02)
    if not finite(weight) or weight <= 0:
        return None
    return float(alpha)


def measurement_row(measurement, *, cohort, seed, endpoint, issues, require_test=True,
                    require_binding=True, require_spectrum=True):
    label = f"{cohort} seed {seed} {endpoint} step {measurement.get('step')}"
    spectrum = measurement.get("spectral") or {}
    weight_hash = measurement.get("weights_sha256")
    spectral_hash = measurement.get("spectral_weights_sha256", spectrum.get("weights_sha256"))
    bound = bool(isinstance(weight_hash, str) and len(weight_hash) == 64 and weight_hash == spectral_hash)
    if require_binding and not bound:
        issues.append(f"{label}: RMSE/alpha pair lacks matching checkpoint weight hashes")
    alpha = spectral_alpha(spectrum)
    saved = spectrum.get("confidence_tail_alpha")
    if alpha is not None and (not finite(saved) or not math.isclose(alpha, saved, rel_tol=1e-10, abs_tol=1e-10)):
        issues.append(f"{label}: stored alpha disagrees with the raw fit")
    if require_spectrum and alpha is None:
        issues.append(f"{label}: invalid or missing fit of the sole eligible hidden matrix")
    if spectrum.get("layer_alpha_sd") is not None:
        issues.append(f"{label}: layer dispersion is undefined for one eligible matrix")
    row = {"cohort": cohort, "seed": seed, "endpoint": endpoint, "step": measurement.get("step"),
           "weights_sha256": weight_hash, "spectral_weights_sha256": spectral_hash,
           "same_checkpoint_verified": bound, "confidence_tail_alpha": alpha if bound or not require_binding else None,
           "n_eligible_matrices": spectrum.get("n_eligible"), "layer_alpha_sd": None}
    for split in ("train", "validation", "test"):
        metrics = measurement.get(split) or {}
        if split == "test" and not require_test:
            if metrics:
                issues.append(f"{label}: development data contain test measurements")
            continue
        for metric in ("rmse", "mae", "mse", "r2", "mape_percent"):
            value = metrics.get(metric)
            row[f"{split}_{metric}"] = value if finite(value) else None
            if not finite(value) or (metric != "r2" and value < 0):
                issues.append(f"{label}: missing or invalid {split} {metric}")
        if finite(metrics.get("rmse")) and finite(metrics.get("mse")):
            if not math.isclose(metrics["rmse"] ** 2, metrics["mse"], rel_tol=1e-6, abs_tol=1e-10):
                issues.append(f"{label}: RMSE squared differs from MSE")
    return row


def verify_best_selection(record, start, end, issues):
    """Select from start, completed epochs and budget; never arbitrary pauses."""
    best = record.get("best_validation")
    if not isinstance(best, dict):
        issues.append("missing best_validation measurement")
        return None
    if not isinstance(best.get("step"), int) or not start <= best["step"] <= end:
        issues.append("best-validation checkpoint falls outside its declared selection window")
    candidates = [row for row in record.get("history", []) if row.get("step") in (start, end)]
    candidates += [row for row in record.get("epoch_history", []) if start <= row.get("step", -1) <= end]
    candidates = [row for row in candidates if finite((row.get("validation") or {}).get("rmse"))]
    if not candidates:
        issues.append("no retained validation selection candidates")
    else:
        winner = min(candidates, key=lambda row: (row["validation"]["rmse"], row["step"]))
        if best.get("step") != winner["step"] or not math.isclose(
                (best.get("validation") or {}).get("rmse", math.inf), winner["validation"]["rmse"],
                rel_tol=1e-7, abs_tol=1e-9):
            issues.append("best_validation does not match the earliest minimum retained validation RMSE")
    if best.get("weights_sha256") != record.get("best_weights_sha256"):
        issues.append("best-validation metrics are not bound to the saved best weight hash")
    for window in (record.get("selection_window") or {}, best.get("selection_window") or {}):
        if window.get("start") != start or window.get("end") != end or window.get("observed_through") != end or window.get("complete") is not True:
            issues.append("selection-window metadata is incomplete or differs from the declared budget")
    return best


def _candidates(plan):
    return plan.get("candidates", plan.get("calibration", {}).get("candidates", []))


def load_contract(root, errors):
    values = {}
    for key, name in (("plan", "plan.json"), ("frozen", "frozen_selection.json"),
                      ("source", "SOURCE_SHA256.json"), ("data", "data_manifest.json")):
        try:
            values[key] = read_json(root / name)
            values[f"{key}_file_sha256"] = file_sha256(root / name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            values[key] = {}
    plan, frozen = values["plan"], values["frozen"]
    for field, expected in (("development_seeds", list(DEV_SEEDS)), ("confirmation_seeds", list(FRESH_SEEDS))):
        if plan.get(field) != expected or frozen.get(field) != expected:
            errors.append(f"plan/frozen selection has unexpected {field}")
    for field, expected in (("development_updates", DEV_END), ("total_updates", END)):
        if plan.get(field) != expected:
            errors.append(f"plan has unexpected {field}")
    if plan.get("model", {}).get("n_blocks") != 2 or plan.get("model", {}).get("d_block") != 784:
        errors.append("plan changed the frozen two-hidden-layer architecture")
    if not frozen.get("calibration_complete") or not frozen.get("github_run_id") or not frozen.get("launch_commit"):
        errors.append("selection is not frozen to a completed GitHub development cohort")
    for field, key in (("source_manifest_sha256", "source_file_sha256"),
                       ("data_manifest_sha256", "data_file_sha256"), ("plan_file_sha256", "plan_file_sha256")):
        if not values.get(key) or frozen.get(field) != values[key]:
            errors.append(f"frozen selection differs from archived {field}")
    candidates = _candidates(plan)
    configs = [candidate["config"] for candidate in candidates]
    if len(candidates) != 32 or len({candidate["id"] for candidate in candidates}) != 32:
        errors.append("the prespecified grid must contain 32 unique candidate IDs")
    actual = {(c.get("lr"), c.get("momentum"), c.get("weight_decay"), c.get("dropout")) for c in configs}
    expected = {(lr, momentum, wd, dropout) for lr in (.003, .01, .03, .1)
                for momentum in (.9, .99) for wd in (0., .001) for dropout in (0., .3690073279296028)}
    if actual != expected or any(c.get("optimizer") != "sgdm" or c.get("dampening") != 0 or c.get("nesterov") is not False for c in configs):
        errors.append("grid differs from the frozen LR/momentum/weight-decay/dropout design")
    if frozen.get("selected") not in configs:
        errors.append("selected configuration is not a member of the frozen grid")
    package_root = Path(__file__).resolve().parents[1]
    for name, expected_hash in values["source"].get("files", {}).items():
        try:
            if file_sha256(package_root / name) != expected_hash:
                errors.append(f"source hash mismatch: {name}")
        except Exception as exc:
            errors.append(f"source file {name}: {type(exc).__name__}: {exc}")
    return values


def audit_execution(record, contract, issues):
    frozen = contract["frozen"]
    for key, expected in (("plan_sha256", canonical_sha256(contract["plan"])),
                          ("data_manifest_sha256", contract.get("data_file_sha256")),
                          ("source_manifest_sha256", contract.get("source_file_sha256"))):
        if not expected or record.get(key) != expected:
            issues.append(f"missing or mismatched {key}")
    for execution in [record.get("execution", {})] + record.get("resume_executions", []):
        github = execution.get("github", {})
        if str(github.get("GITHUB_RUN_ID")) != str(frozen.get("github_run_id")) or github.get("GITHUB_SHA") != frozen.get("launch_commit"):
            issues.append("execution is outside the frozen GitHub cohort")
        if execution.get("precision") != "float32" or execution.get("torch_num_threads") != 1 or execution.get("torch_num_interop_threads") != 1:
            issues.append("execution violates one-thread CPU float32 settings")
        if execution.get("deterministic_algorithms") is not True or any(execution.get("thread_environment", {}).get(key) != "1"
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
            issues.append("execution violates deterministic numerical thread settings")
        source = execution.get("source_sha256", {})
        if not source.get("long_train.py") or canonical_sha256(source) != record.get("source_bundle_sha256"):
            issues.append("training source bundle is missing or inconsistent")
        for name, digest in source.items():
            folder = "california_housing_sgdm_search" if name == "long_train.py" else "california_housing_benchmark"
            if contract["source"].get("files", {}).get(f"{folder}/{name}") != digest:
                issues.append(f"execution source differs from frozen manifest: {name}")


def has_test_measurement(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if (key == "test" and child not in (None, {})) or (key.startswith("test_") and finite(child)):
                return True
            if has_test_measurement(child):
                return True
    elif isinstance(value, list):
        return any(has_test_measurement(child) for child in value)
    return False


def audit_development(root, contract, errors, warnings):
    points, status_rows, scores = [], [], []
    for candidate in _candidates(contract["plan"]):
        cid, config = candidate["id"], candidate["config"]
        best_values = []
        for seed in DEV_SEEDS:
            path = root / "development" / cid / f"seed_{seed}" / "result.json"
            issues = []
            try:
                record = read_json(path)
                trial = read_json(path.with_name("trial.json"))
                if trial.get("seed") != seed or trial.get("result_sha256") != file_sha256(path):
                    issues.append("development trial seed/result fingerprint does not match retained record")
                if record.get("seed") != seed or record.get("phase") != "development" or record.get("config") != config:
                    issues.append("development identity differs from frozen candidate/seed")
                audit_execution(record, contract, issues)
                if has_test_measurement(record):
                    issues.append("development record contains forbidden test measurements")
                status = trial.get("status")
                numerical_failure = (status == "numerical_failure" and
                                     str(trial.get("error", record.get("error", ""))).startswith("FloatingPointError:"))
                if numerical_failure:
                    warnings.append(f"development {cid} seed {seed}: numerical failure retained and excluded from selection")
                elif status != "complete" or record.get("status") != "complete" or record.get("completed_updates") != DEV_END:
                    issues.append("development run did not complete the full 50,000-update budget")
                best = None if numerical_failure else verify_best_selection(record, 0, DEV_END, issues)
                if best is not None:
                    best_values.append(best["validation"]["rmse"])
                    audit_development_weights(record, path.parent, trial, issues)
                for endpoint, step in GRID_ENDPOINTS:
                    measurement = best if step is None else next((r for r in record.get("history", []) if r.get("step") == step), None)
                    base = {"candidate_id": cid, "seed": seed, "endpoint": endpoint, "step": step,
                            "status": status, "error": trial.get("error", record.get("error")),
                            "source": str(path.relative_to(root)), "source_sha256": file_sha256(path), **config}
                    if measurement is None:
                        points.append({**base, "validation_rmse": None, "confidence_tail_alpha": None,
                                       "weights_sha256": None, "spectral_weights_sha256": None,
                                       "same_checkpoint_verified": False, "fit_available": False})
                    else:
                        # Fit failures stay visible; they do not retroactively change validation-only tuning.
                        row = measurement_row(measurement, cohort="development", seed=seed, endpoint=endpoint,
                                              issues=issues, require_test=False, require_spectrum=False)
                        points.append({**base, "step": measurement["step"],
                                       **{k: row.get(k) for k in ("validation_rmse", "confidence_tail_alpha", "weights_sha256",
                                                                  "spectral_weights_sha256", "same_checkpoint_verified")},
                                       "fit_available": row["confidence_tail_alpha"] is not None})
                status_rows.append({"candidate_id": cid, "seed": seed, "status": status,
                                    "completed_updates": record.get("completed_updates"), "numerical_failure": numerical_failure,
                                    "source_sha256": file_sha256(path)})
            except Exception as exc:
                issues.append(f"{type(exc).__name__}: {exc}")
            errors.extend(f"{path.relative_to(root)}: {issue}" for issue in issues)
        if len(best_values) == 2:
            scores.append((statistics.fmean(best_values), cid, config))
    if scores:
        winner = min(scores, key=lambda entry: (entry[0], entry[1]))
        if contract["frozen"].get("selected") != winner[2]:
            errors.append("frozen winner differs from the full-grid mean best-validation score and candidate-ID tie break")
    else:
        errors.append("no candidate completed both development seeds")
    return points, status_rows


def _load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def audit_development_weights(record, directory, trial, issues):
    """Completed development retains best/final weights, not duplicate full states."""
    from california_housing_benchmark.train import model_hash
    for name, expected, step in (("best_checkpoint.pt", record.get("best_weights_sha256"), record["best_validation"]["step"]),
                                 ("final_weights.pt", record.get("final_weights_sha256"), DEV_END)):
        path = directory / name
        digest = trial.get("retained_files", {}).get(name)
        if not digest or not path.is_file() or file_sha256(path) != digest:
            issues.append(f"{name}: missing weights or mismatch with retained-file SHA256")
            continue
        checkpoint = _load_checkpoint(path)
        if not expected or model_hash(checkpoint["model"]) != expected or checkpoint.get("weights_sha256") != expected:
            issues.append(f"{name}: retained model does not match the metric/spectrum weight hash")
        observed = checkpoint.get("selected_step") if name == "best_checkpoint.pt" else checkpoint.get("completed_updates")
        if observed != step:
            issues.append(f"{name}: retained weights have the wrong update index")
    final = next((row for row in record.get("history", []) if row.get("step") == DEV_END), {})
    if final.get("weights_sha256") != record.get("final_weights_sha256"):
        issues.append("development final metrics do not match retained final weights")


def verify_continuity(record, source, issues):
    continuity = record.get("continuity") or {}
    if not source or record.get("parent_checkpoint_sha256") != source.get("sha256"):
        issues.append("continuation did not load the immutable retained 20k checkpoint")
    if (continuity.get("verified") is not True or continuity.get("parent") != source.get("fingerprints")
            or continuity.get("start") != source.get("fingerprints")):
        issues.append("model/optimizer/RNG/sampler state is not an exact continuation")


def audit_retained_checkpoints(record, directory, start, issues):
    from california_housing_benchmark.train import model_hash
    from .long_train import checkpoint_fingerprints
    needed = {"resume.pt", "best_checkpoint.pt", "checkpoint_step_100000.pt", "checkpoint_step_50000.pt"}
    if start == 0:
        needed.update(("checkpoint_step_2000.pt", "checkpoint_step_20000.pt"))
    for name in sorted(needed):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing retained checkpoint {name}")
        elif name.startswith("checkpoint_step_"):
            step = int(name.removeprefix("checkpoint_step_").removesuffix(".pt"))
            measurement = next((row for row in record.get("history", []) if row.get("step") == step), {})
            checkpoint = _load_checkpoint(path)
            if checkpoint.get("completed_updates") != step or model_hash(checkpoint["model"]) != measurement.get("weights_sha256"):
                issues.append(f"{name}: fixed-budget metric/spectral pair does not match retained weights")
            for key in ("seed", "phase", "config", "plan_sha256", "data_manifest_sha256", "source_bundle_sha256"):
                if checkpoint.get("identity", {}).get(key) != record.get(key):
                    issues.append(f"{name}: retained checkpoint identity differs for {key}")
    for name, expected, expected_step in (("resume.pt", record.get("final_weights_sha256"), END),
                                         ("checkpoint_step_100000.pt", record.get("final_weights_sha256"), END),
                                         ("best_checkpoint.pt", record.get("best_weights_sha256"), (record.get("best_validation") or {}).get("step"))):
        path = directory / name
        if not path.is_file():
            continue
        checkpoint = _load_checkpoint(path)
        if not expected or model_hash(checkpoint["model"]) != expected:
            issues.append(f"{name}: retained weights differ from result fingerprint")
        step = checkpoint.get("selected_step") if name == "best_checkpoint.pt" else checkpoint.get("completed_updates")
        if step != expected_step:
            issues.append(f"{name}: checkpoint step differs from measurement")
        if name == "resume.pt":
            fingerprint = checkpoint_fingerprints(checkpoint)
            if fingerprint.get("completed_updates") != END:
                issues.append("resume checkpoint did not reach the fixed 100,000-update budget")


def audit_parent(root, errors):
    records, native, checkpoints = {}, {}, {}
    try:
        manifest = read_json(root / "parent_manifest.json")
        if str(manifest.get("github_run_id")) != PARENT_RUN or manifest.get("launch_commit") != PARENT_SOURCE or not manifest.get("archive_commit"):
            errors.append("parent manifest lacks the exact immutable original GitHub cohort")
        from california_housing_benchmark.train import model_hash
        from .long_train import checkpoint_fingerprints
        for seed in PARENT_SEEDS:
            entry = manifest.get("seeds", {}).get(str(seed), {})
            directory = root / "parent" / f"seed_{seed}"
            for filename, key in (("result.json", "result_sha256"), ("checkpoint20k.pt", "checkpoint_sha256"),
                                  ("native_adamw_result.json", "native_adamw_result_sha256")):
                if not entry.get(key) or file_sha256(directory / filename) != entry[key]:
                    errors.append(f"parent seed {seed}: {filename} differs from immutable archive hash")
            parent = read_json(directory / "result.json")
            adamw = read_json(directory / "native_adamw_result.json")
            for record, optimizer in ((parent, "sgdm"), (adamw, "adamw")):
                github = record.get("execution", {}).get("github", {})
                if record.get("seed") != seed or record.get("optimizer") != optimizer or record.get("status") != "complete":
                    errors.append(f"parent seed {seed}: invalid completed {optimizer} identity")
                if str(github.get("GITHUB_RUN_ID")) != PARENT_RUN or github.get("GITHUB_SHA") != PARENT_SOURCE:
                    errors.append(f"parent seed {seed}: {optimizer} is outside the original GitHub cohort")
            checkpoint = _load_checkpoint(directory / "checkpoint20k.pt")
            fingerprint = checkpoint_fingerprints(checkpoint)
            if fingerprint["completed_updates"] != 20000:
                errors.append(f"parent seed {seed}: source checkpoint is not at 20,000 updates")
            checkpoint_row = next(row for row in checkpoint["result"]["history"] if row["step"] == 20000)
            parent_row = next(row for row in parent["history"] if row["step"] == 20000)
            for field in ("train", "validation", "test", "spectral", "minibatch_sha256"):
                if checkpoint_row.get(field) != parent_row.get(field):
                    errors.append(f"parent seed {seed}: archived {field} does not match checkpoint-contained measurement")
            records[seed], native[seed] = parent, adamw
            checkpoints[seed] = {"fingerprints": fingerprint, "sha256": entry["checkpoint_sha256"],
                                 "weights_sha256": model_hash(checkpoint["model"])}
    except Exception as exc:
        errors.append(f"parent archive: {type(exc).__name__}: {exc}")
    return records, native, checkpoints


def audit_final(root, contract, parent_records, parent_checkpoints, errors):
    records, rows, sources = [], [], []
    for cohort, seeds, folder, start in (("continuation", PARENT_SEEDS, "continuations", 20000),
                                        ("confirmation", FRESH_SEEDS, "runs", 0)):
        for seed in seeds:
            directory = root / folder / f"seed_{seed}"
            issues = []
            try:
                record = read_json(directory / "result.json")
                records.append(record)
                sources.append({"cohort": cohort, "seed": seed, "source": str((directory / "result.json").relative_to(root)),
                                "sha256": file_sha256(directory / "result.json")})
                if record.get("seed") != seed or record.get("phase") != cohort or record.get("status") != "complete" or record.get("completed_updates") != END:
                    issues.append("run identity or fixed 100,000-update completion is invalid")
                if record.get("parameter_count") != 623281 or record.get("patience_stopping") is not False or record.get("new_updates") != END-start:
                    issues.append("parameter count, no-early-stop rule, or newly performed update count is inconsistent")
                audit_execution(record, contract, issues)
                config = record.get("config") or {}
                if cohort == "confirmation" and config != contract["frozen"].get("selected"):
                    issues.append("fresh confirmation changed the frozen selected configuration")
                if cohort == "continuation":
                    parent = parent_records.get(seed, {})
                    expected_config = {"optimizer": "sgdm", "lr": .03, "momentum": .9, "weight_decay": 0.,
                                       "dampening": 0., "nesterov": False, "dropout": .3690073279296028}
                    if any(config.get(key) != value for key, value in expected_config.items()):
                        issues.append("continuation altered the original optimizer or dropout")
                    if record.get("data_manifest_sha256") != parent.get("data_manifest_sha256"):
                        issues.append("continuation altered the original data/preprocessing manifest")
                    source = parent_checkpoints.get(seed, {})
                    verify_continuity(record, source, issues)
                history = record.get("history", [])
                steps = [row.get("step") for row in history]
                expected_steps = set(range(start, END + 1, 5000)) | ({2000} if start == 0 else set())
                if steps != sorted(set(steps)) or not expected_steps.issubset(steps):
                    issues.append("missing or duplicated declared history steps")
                for measurement in history:
                    if measurement.get("spectral"):
                        rows.append(measurement_row(measurement, cohort=cohort, seed=seed, endpoint="fixed_updates", issues=issues))
                best = verify_best_selection(record, start, END, issues)
                if best:
                    rows.append(measurement_row(best, cohort=cohort, seed=seed, endpoint="best_validation", issues=issues))
                final = next((row for row in history if row.get("step") == END), {})
                if final.get("weights_sha256") != record.get("final_weights_sha256"):
                    issues.append("final metrics are not bound to retained final weights")
                audit_retained_checkpoints(record, directory, start, issues)
            except Exception as exc:
                issues.append(f"{type(exc).__name__}: {exc}")
            errors.extend(f"{folder}/seed_{seed}: {issue}" for issue in issues)
    return records, rows, sources


def summarize_final(rows, parents, native, errors):
    combined = list(rows)
    for cohort, records in (("original_sgdm", parents), ("original_native_adamw", native)):
        for seed, record in records.items():
            measurement = next(row for row in record["history"] if row["step"] == 20000)
            combined.append(measurement_row(measurement, cohort=cohort, seed=seed, endpoint="fixed_updates",
                                            issues=errors, require_binding=False))
    groups = sorted({(row["cohort"], row["endpoint"], row["step"] if row["endpoint"] == "fixed_updates" else -1) for row in combined})
    stats = []
    for cohort, endpoint, step in groups:
        subset = [row for row in combined if row["cohort"] == cohort and row["endpoint"] == endpoint
                  and (step == -1 or row["step"] == step)]
        if len({row["seed"] for row in subset}) != len(subset):
            raise ValueError("duplicate seeds in a cohort/endpoint statistic")
        for metric in METRICS:
            stats.append({"cohort": cohort, "endpoint": endpoint, "step": None if step == -1 else step,
                          "metric": metric, **seed_statistics([row.get(metric) for row in subset])})
    paired, independent = [], []
    for metric in METRICS:
        original = {row["seed"]: row.get(metric) for row in combined if row["cohort"] == "original_sgdm"}
        final = {row["seed"]: row.get(metric) for row in combined if row["cohort"] == "continuation" and row["endpoint"] == "fixed_updates" and row["step"] == END}
        paired.append({"contrast": "same_seed_sgdm_100k_minus_original_20k", "metric": metric,
                       "interpretation": "Within-seed extension of the same retained optimizer state; training horizon changes.",
                       **paired_effect(final, original, PARENT_SEEDS)})
        tuned = [row.get(metric) for row in combined if row["cohort"] == "confirmation" and row["endpoint"] == "best_validation"]
        reference = [row.get(metric) for row in combined if row["cohort"] == "original_native_adamw"]
        independent.append({"contrast": "fresh_tuned_sgdm_best_through_100k_minus_original_native_adamw_20k",
                            "metric": metric, "interpretation": "Independent seeds; unequal training budgets, validation selection and tuning/possibly dropout. Not an optimizer-only causal contrast.",
                            **welch_effect(tuned, reference)})
    return combined, stats, paired, independent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grid_summary(points):
    rows = []
    keys = sorted({(p["candidate_id"], p["endpoint"], p["step"] if p["endpoint"] == "fixed_updates" else -1) for p in points})
    for cid, endpoint, step in keys:
        group = [p for p in points if p["candidate_id"] == cid and p["endpoint"] == endpoint and (step == -1 or p["step"] == step)]
        if len({p["seed"] for p in group}) != len(group):
            raise ValueError("duplicate development seed within one configuration/endpoint")
        entry = {"candidate_id": cid, "endpoint": endpoint, "step": None if step == -1 else step,
                 "uncertainty": "Exploratory two-development-seed sample SD; configurations are not replicates."}
        for metric in ("validation_rmse", "confidence_tail_alpha"):
            values = [p[metric] for p in group if finite(p.get(metric))]
            entry[f"{metric}_n"] = len(values)
            entry[f"{metric}_mean"] = statistics.fmean(values) if values else None
            entry[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(entry)
    return rows


def _plot_grid(points, statuses, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {.003: "#3b6fb6", .01: "#18988a", .03: "#d8872e", .1: "#9b5298"}
    failures = sum(row.get("numerical_failure", False) for row in statuses)
    for endpoint, step in GRID_ENDPOINTS:
        subset = [p for p in points if p["endpoint"] == endpoint and (step is None or p["step"] == step)]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), squeeze=False)
        for axis, dropout in zip(axes[0], (0., .3690073279296028)):
            panel = [p for p in subset if p.get("dropout") == dropout]
            for offset, cid in enumerate(sorted({p["candidate_id"] for p in panel})):
                group = [p for p in panel if p["candidate_id"] == cid and p.get("same_checkpoint_verified")
                         and finite(p.get("confidence_tail_alpha")) and finite(p.get("validation_rmse"))]
                if not group:
                    continue
                x, y = [p["confidence_tail_alpha"] for p in group], [p["validation_rmse"] for p in group]
                config = group[0]
                color = colors[config["lr"]]
                axis.plot(x, y, color=color, alpha=.35, linewidth=.8, zorder=1)
                axis.scatter(x, y, marker="o" if config["momentum"] == .9 else "^", s=38,
                             edgecolors=color, facecolors=color if config["weight_decay"] == 0 else "none", zorder=2)
                axis.annotate(cid, (statistics.fmean(x), statistics.fmean(y)),
                              xytext=(3, 3 if offset % 2 else -8), textcoords="offset points", fontsize=6.5, color=color)
            missing = sum(not (p.get("same_checkpoint_verified") and finite(p.get("confidence_tail_alpha"))
                               and finite(p.get("validation_rmse"))) for p in panel)
            axis.axvline(2, linestyle="--", color=".5", linewidth=1)
            axis.set(xlabel="Confidence/tail α (sole hidden matrix)", ylabel="Validation RMSE ($100,000; lower is better)",
                     title=f"Dropout {dropout:.3g}; {len(panel) - missing}/{len(panel)} available seed points")
            axis.grid(alpha=.15)
            axis.spines[["top", "right"]].set_visible(False)
        legend = [Line2D([], [], color=color, marker="o", linestyle="none", label=f"LR {lr:g}") for lr, color in colors.items()]
        legend += [Line2D([], [], color=".3", marker="o", linestyle="none", label="momentum .9"),
                   Line2D([], [], color=".3", marker="^", linestyle="none", label="momentum .99"),
                   Line2D([], [], color=".3", marker="o", linestyle="none", label="WD 0 (filled)"),
                   Line2D([], [], color=".3", marker="o", markerfacecolor="none", linestyle="none", label="WD .001 (open)")]
        label = f"{step:,} updates" if step is not None else "Each seed's best validation checkpoint through 50,000 updates"
        fig.suptitle(f"Full 32-configuration SGDM development scan — {label}", fontsize=12)
        fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(.5,.03))
        fig.text(.5, .005, f"Two seeds per setting; lines connect seeds. {failures}/64 runs failed numerically; earlier valid points remain. IDs: grid_points.csv. No test data.", ha="center", fontsize=8)
        fig.subplots_adjust(top=.87, bottom=.22, wspace=.25)
        name = f"development_rmse_vs_alpha_{step if step is not None else 'best_validation'}"
        for ext in ("png", "pdf"):
            fig.savefig(destination / f"{name}.{ext}", dpi=180)
        plt.close(fig)


def _plot_final(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    for seed in FRESH_SEEDS:
        for endpoint, marker in (("best_validation", "o"), ("fixed_updates", "s")):
            group = [r for r in rows if r["cohort"] == "confirmation" and r["seed"] == seed and r["endpoint"] == endpoint
                     and (endpoint == "best_validation" or r["step"] == END)]
            for row in group:
                if row.get("same_checkpoint_verified") and finite(row.get("confidence_tail_alpha")) and finite(row.get("test_rmse")):
                    axis.scatter(row["confidence_tail_alpha"], row["test_rmse"], marker=marker,
                                 color="#18497a" if endpoint == "best_validation" else "#b96d2e", s=44)
                    axis.annotate(str(seed), (row["confidence_tail_alpha"], row["test_rmse"]), xytext=(3,3), textcoords="offset points", fontsize=8)
    for endpoint, color in (("best_validation", "#18497a"), ("fixed_updates", "#b96d2e")):
        eligible = [r for r in rows if r["cohort"] == "confirmation" and r["endpoint"] == endpoint
                    and (endpoint == "best_validation" or r["step"] == END)
                    and r.get("same_checkpoint_verified") and finite(r.get("confidence_tail_alpha")) and finite(r.get("test_rmse"))]
        x = seed_statistics([r["confidence_tail_alpha"] for r in eligible])
        y = seed_statistics([r["test_rmse"] for r in eligible])
        if x["n"] == 5 and y["n"] == 5:
            axis.errorbar(x["mean"], y["mean"], xerr=x["ci95_half_width"], yerr=y["ci95_half_width"],
                          fmt="D", color=color, markerfacecolor="white", markersize=7, capsize=3,
                          label=f"{endpoint}: seed mean ± marginal 95% CIs")
    axis.axvline(2, color=".5", linestyle="--", linewidth=1)
    axis.set(xlabel="Confidence/tail α (sole hidden matrix)", ylabel="Test RMSE ($100,000; lower is better)",
             title="Fresh confirmation seeds: circles = selected best; squares = 100k")
    axis.grid(alpha=.15)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=7, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"fresh_confirmation_rmse_vs_alpha.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for seed in PARENT_SEEDS:
        group = sorted([r for r in rows if r["cohort"] == "continuation" and r["seed"] == seed and r["endpoint"] == "fixed_updates"], key=lambda r:r["step"])
        for axis, metric in zip(axes, ("test_rmse", "confidence_tail_alpha")):
            axis.plot([r["step"] for r in group], [r.get(metric) if finite(r.get(metric)) else float("nan") for r in group], label=str(seed), linewidth=1.3)
    axes[0].set(ylabel="Test RMSE ($100,000; lower is better)")
    axes[1].set(ylabel="Confidence/tail α (sole hidden matrix)")
    axes[1].axhline(2, color=".5", linestyle="--", linewidth=1)
    for axis in axes:
        axis.set(xlabel="Total optimizer updates", xlim=(20000, END))
        axis.grid(alpha=.15)
    axes[0].legend(title="Continuation seed", fontsize=8, frameon=False)
    fig.suptitle("Exact same-state SGDM continuations: individual seed trajectories")
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"sgdm_continuation_trajectories.{ext}", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for seed in PARENT_SEEDS:
        group = sorted([r for r in rows if r["cohort"] == "continuation" and r["seed"] == seed
                        and r["endpoint"] == "fixed_updates" and r.get("same_checkpoint_verified")], key=lambda r:r["step"])
        if not group:
            continue
        x = [r["confidence_tail_alpha"] if finite(r.get("confidence_tail_alpha")) else float("nan") for r in group]
        y = [r["test_rmse"] if finite(r.get("test_rmse")) else float("nan") for r in group]
        line, = axis.plot(x, y, linewidth=1.2, label=str(seed))
        axis.scatter(x[0], y[0], marker="o", color=line.get_color(), s=28)
        axis.scatter(x[-1], y[-1], marker="s", color=line.get_color(), s=32)
    axis.axvline(2, color=".5", linestyle="--", linewidth=1)
    axis.set(xlabel="Confidence/tail α (sole hidden matrix)", ylabel="Test RMSE ($100,000; lower is better)",
             title="Same-seed SGDM trajectories: circle20k → square100k")
    axis.grid(alpha=.15)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(title="Continuation seed", fontsize=8, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"sgdm_continuation_rmse_vs_alpha.{ext}", dpi=180)
    plt.close(fig)


def display(stat):
    if stat.get("mean") is None:
        return "unavailable"
    half = stat.get("ci95_half_width")
    return f"{stat['mean']:.5f} ± {half:.5f} (n={stat['n']})" if half is not None else f"{stat['mean']:.5f} (n={stat['n']}; CI unavailable)"


def write_outputs(root, development_only=False, allow_incomplete=False):
    root = Path(root)
    errors, warnings = [], []
    contract = load_contract(root, errors)
    grid, statuses = audit_development(root, contract, errors, warnings)
    development_complete = len(statuses) == 64 and not errors
    destination = root / ("development_analysis" if development_only else "analysis")
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "grid_points.csv", grid)
    write_csv(destination / "grid_seed_status.csv", statuses)
    write_csv(destination / "grid_summary.csv", grid_summary(grid))
    _plot_grid(grid, statuses, figures)
    rows, stats, paired, independent, sources, records = [], [], [], [], [], []
    if not development_only:
        parents, native, checkpoints = audit_parent(root, errors)
        records, rows, sources = audit_final(root, contract, parents, checkpoints, errors)
        rows, stats, paired, independent = summarize_final(rows, parents, native, errors)
        write_csv(destination / "seed_metrics.csv", rows)
        write_csv(destination / "seed_statistics.csv", stats)
        write_json(destination / "paired_horizon_effects.json", paired)
        write_json(destination / "independent_tuned_vs_native.json", independent)
        _plot_final(rows, figures)
    completed = [r for r in records if r.get("status") == "complete" and r.get("completed_updates") == END]
    counts = {phase: len([r for r in completed if r.get("phase") == phase]) for phase in ("continuation", "confirmation")}
    complete = not development_only and counts == {"continuation": 5, "confirmation": 5} and development_complete and not errors
    summary = {"schema_version": 1, "development_only": development_only,
               "complete_development_scan": development_complete, "development_records": len(statuses),
               "development_numerical_failures": sum(r.get("numerical_failure", False) for r in statuses),
               "development_unavailable_spectral_points": sum(not p.get("fit_available") for p in grid),
               "complete_five_seeds": complete, "n_completed": 5 if complete else 0,
               "n_completed_by_cohort": counts, "source_files": sources,
               "statistics": stats, "paired_horizon_effects": paired, "independent_tuned_vs_native": independent,
               "audit_errors": errors, "audit_warnings": warnings,
               "scope": "One 784x784 eligible matrix; confidence/tail average equals its ordinary fitted alpha; no layer SD.",
               "uncertainty": "Final uncertainty is across five seeds. Grid shows individual development seeds and per-configuration two-seed SD; configurations are never pooled as replicates.",
               "interpretation": "The continuation contrast changes horizon while preserving the exact prior state/configuration. The tuned comparison also changes tuning, checkpoint selection and potentially dropout, and cannot isolate an optimizer-only effect. Alpha=2 is a reference, not an imposed optimum."}
    write_json(destination / ("development_summary.json" if development_only else "summary.json"), summary)
    if not development_only:
        write_json(destination / "execution_cohort.json", {"source_files": sources,
                   "parent_manifest_sha256": file_sha256(root / "parent_manifest.json") if (root / "parent_manifest.json").exists() else None,
                   "complete_five_seeds": complete, "executions": [{"seed":r.get("seed"),"phase":r.get("phase"),
                   "execution":r.get("execution"),"resume_executions":r.get("resume_executions",[]),
                   "parent_checkpoint_sha256":r.get("parent_checkpoint_sha256"),"continuity":r.get("continuity")} for r in records], "audit_errors":errors})
    text = ["# SGDM learning-rate and long-horizon study", "",
            "**Complete development scan.**" if development_complete else "**Incomplete development scan or audit failure.**", "",
            f"The full 32-configuration grid contains two development seeds per setting; {len(statuses)}/64 records were found, including {summary['development_numerical_failures']} numerical failures. Development plots contain validation metrics only.", "",
            summary["uncertainty"], "", summary["scope"], "", summary["interpretation"], "",
            "Every plotted RMSE/α pair must carry identical measurement and spectral weight hashes. Missing fits and numerical failures stay visible in grid_points.csv and grid_seed_status.csv.", "",
            "Lower RMSE is better. RMSE 0.50 means $50,000, not 50% error. No regression accuracy percentage is substituted."]
    if not development_only:
        text += ["", "**Final cohort complete.**" if complete else "**Final cohort incomplete or failed audit.**", "",
                 "All ± entries below are 95% Student-t confidence half-widths over seeds; CSV files also contain sample SD.", "",
                 "| Cohort | Endpoint | Test RMSE ↓ | Confidence/tail α |", "|---|---|---:|---:|"]
        for cohort, endpoint, step in (("original_sgdm","fixed_updates",20000),("continuation","fixed_updates",END),
                                      ("continuation","best_validation",None),("confirmation","fixed_updates",END),
                                      ("confirmation","best_validation",None),("original_native_adamw","fixed_updates",20000)):
            selected = {r["metric"]:r for r in stats if r["cohort"]==cohort and r["endpoint"]==endpoint and r["step"]==step}
            text.append(f"| {cohort} | {step if step is not None else endpoint} | {display(selected.get('test_rmse',{}))} | {display(selected.get('confidence_tail_alpha',{}))} |")
        text += ["", "## Same-seed change: original SGDM20k to continued100k", ""]
        for effect in paired:
            if effect["metric"] in ("test_rmse","confidence_tail_alpha"):
                text.append(f"- {effect['metric']}: {display(effect)}")
        text += ["", "Fresh tuned SGDM uses its best validation checkpoint through100k; original native AdamW uses the fixed20k checkpoint. Independent Welch intervals are saved separately, with the unequal-budget and changed-regularization caveats."]
    for title, values in (("Audit errors",errors),("Audit warnings",warnings)):
        if values:
            text += ["",f"## {title}",""]+[f"- {value}" for value in values]
    (destination / ("DEVELOPMENT_RESULTS.md" if development_only else "RESULTS.md")).write_text("\n".join(text)+"\n")
    if errors or (not development_only and not complete) or not development_complete:
        if not allow_incomplete:
            raise ValueError("Incomplete cohort or failed audit; explicitly labeled diagnostic outputs were saved")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = write_outputs(args.root, args.development_only, args.allow_incomplete)
    print(json.dumps({key:result[key] for key in ("complete_development_scan","complete_five_seeds","n_completed_by_cohort","development_numerical_failures")},indent=2))


if __name__ == "__main__":
    main()

"""Audit and summarize the unchanged five-seed SGDM extension to 200k.

All intervals describe variation between seeds, conditional on the fixed data
split. The single eligible 784x784 matrix supplies ordinary fitted alpha; the
confidence/tail weight is still recomputed, never replaced by alpha_weighted.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t as student_t

SEEDS = tuple(range(2340, 2345))
START, END = 100000, 200000
PARENT_RUN = "45758895218"
PARENT_SOURCE = "f89e5e00a3b9ad09c906b272fbc1d627ccecd714"
REPOSITORY = "CalculatedContent/rg_optimizers"
CONFIG = {"optimizer": "sgdm", "lr": .03, "momentum": .9, "weight_decay": 0,
          "dampening": 0, "nesterov": False, "dropout": .3690073279296028}
METRICS = ("confidence_tail_alpha", "test_rmse", "test_mae", "test_mse",
           "test_r2", "validation_rmse", "train_rmse")
PACKAGE = Path(__file__).resolve().parent


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha(path):
    path = Path(path)
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_statistics(values, expected_n=5):
    if any(not finite(value) for value in values):
        raise ValueError("Seed statistics require finite observations")
    values = list(map(float, values))
    n = len(values)
    mean = statistics.fmean(values) if n else None
    sd = statistics.stdev(values) if n > 1 else None
    half = float(student_t.ppf(.975, n - 1)) * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "sample_sd": sd, "ci95_half_width": half,
            "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None,
            "degrees_of_freedom": n - 1 if n else None,
            "ci_method": "two-sided 95% Student-t across seeds",
            "expected_n": expected_n, "complete_expected_seeds": n == expected_n}


def paired_effect(later, earlier, seeds=SEEDS):
    differences = {str(seed): later[seed] - earlier[seed] for seed in seeds
                   if finite(later.get(seed)) and finite(earlier.get(seed))}
    return {"direction": "later minus earlier", "seed_differences": differences,
            **seed_statistics(list(differences.values()), len(seeds))}


def spectral_alpha(spectrum):
    """Return a recomputed ordinary-alpha weighted mean, with its weight."""
    if not isinstance(spectrum, dict):
        raise ValueError("Missing spectral measurement")
    matrices = spectrum.get("matrices") or []
    if (len(matrices) != 1 or spectrum.get("n_eligible") != 1
            or spectrum.get("n_valid") != 1 or spectrum.get("complete") is not True):
        raise ValueError("Expected one complete eligible matrix fit")
    row = matrices[0]
    if (row.get("shape") != [784, 784]
            or spectrum.get("eligible_matrices") != [row.get("matrix")]):
        raise ValueError("The eligible matrix is not the declared 784x784 hidden matrix")
    fit = row.get("fit") or {}
    alpha, distance, tail, size = (fit.get(key) for key in ("alpha", "D", "num_pl_spikes", "num_evals"))
    if not (fit.get("status") == "success" and finite(alpha) and alpha > 0
            and finite(distance) and 0 <= distance <= 1
            and isinstance(tail, int) and not isinstance(tail, bool)
            and isinstance(size, int) and not isinstance(size, bool) and 0 < tail <= size <= 784):
        raise ValueError("Invalid ordinary PL fit or confidence/tail inputs")
    weight = math.sqrt(tail) / (distance + .02)
    average = math.fsum([weight * alpha]) / math.fsum([weight])
    for label, stored in (("confidence_tail_alpha", spectrum.get("confidence_tail_alpha")),
                          ("matrix alpha", row.get("alpha")), ("matrix weight", row.get("weight"))):
        expected = weight if label == "matrix weight" else average
        if not finite(stored) or not math.isclose(stored, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Stored {label} differs from independent recomputation")
    if spectrum.get("layer_alpha_sd") is not None or spectrum.get("layer_dispersion_defined") is True:
        raise ValueError("Layer dispersion is undefined for a single eligible matrix")
    return float(average), float(weight)


def measurement_row(measurement, seed, segment):
    weight_hash = measurement.get("weights_sha256")
    spectral_hash = measurement.get("spectral_weights_sha256")
    if not (isinstance(weight_hash, str) and len(weight_hash) == 64 and weight_hash == spectral_hash):
        raise ValueError("Predictive metrics and alpha lack matching checkpoint weight hashes")
    alpha, fit_weight = spectral_alpha(measurement.get("spectral"))
    row = {"seed": seed, "step": measurement.get("step"), "segment": segment,
           "weights_sha256": weight_hash, "spectral_weights_sha256": spectral_hash,
           "same_checkpoint_verified": True, "confidence_tail_alpha": alpha,
           "confidence_tail_fit_weight": fit_weight, "eligible_matrices": 1,
           "layer_alpha_sd": None}
    for split in ("train", "validation", "test"):
        metrics = measurement.get(split) or {}
        for metric in ("rmse", "mae", "mse", "r2"):
            value = metrics.get(metric)
            if metric == "r2" and value is None:
                row[f"{split}_{metric}"] = None
                continue
            if not finite(value) or (metric != "r2" and value < 0):
                raise ValueError(f"Invalid or missing {split} {metric}")
            row[f"{split}_{metric}"] = float(value)
        if not math.isclose(metrics["rmse"] ** 2, metrics["mse"], rel_tol=1e-6, abs_tol=1e-10):
            raise ValueError(f"{split} RMSE squared differs from MSE")
    if not isinstance(row["step"], int):
        raise ValueError("Measurement update count must be an integer")
    return row


def _check(actual, expected, label, errors):
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def _load_contract(root, errors):
    contract = {}
    for name in ("plan", "SOURCE_SHA256", "data_manifest", "parent_manifest", "preparation", "cohort_origin"):
        path = root / f"{name}.json"
        try:
            contract[name] = read_json(path)
            contract[f"{name}_sha256"] = file_sha256(path)
        except Exception as exc:
            errors.append(f"{name}.json: {type(exc).__name__}: {exc}")
            contract[name] = {}
    plan, source, origin = contract["plan"], contract["SOURCE_SHA256"], contract["cohort_origin"]
    for key, expected in (("total_updates", END), ("start_step", START),
                          ("continuation_seeds", list(SEEDS)), ("continuation_config", CONFIG),
                          ("batch_size", 256), ("grad_clip", 1), ("eval_every", 5000),
                          ("checkpoint_updates", [100000, 125000, 150000, 175000, 200000])):
        _check(plan.get(key), expected, f"plan {key}", errors)
    _check(plan.get("model"), {"n_blocks": 2, "d_block": 784, "dropout": CONFIG["dropout"]},
           "unchanged model", errors)
    parent = plan.get("continuation", {})
    manifest = contract["parent_manifest"]
    for key, expected in (("repository", REPOSITORY), ("run_id", PARENT_RUN),
                          ("launch_commit", PARENT_SOURCE), ("checkpoint_step", START)):
        _check(str(parent.get(key)), str(expected), f"parent contract {key}", errors)
        manifest_key = "github_run_id" if key == "run_id" else key
        _check(str(manifest.get(manifest_key)), str(expected), f"parent manifest {manifest_key}", errors)
    _check(parent.get("archive_commit"), manifest.get("archive_commit"), "parent archive commit", errors)
    _check(manifest.get("data_manifest_sha256"), contract.get("data_manifest_sha256"),
           "parent data identity", errors)
    _check(origin.get("repository"), REPOSITORY, "new repository", errors)
    if not origin.get("github_run_id") or not origin.get("launch_commit"):
        errors.append("New GitHub cohort is not identified")
    _check(origin.get("source_manifest_sha256"), contract.get("SOURCE_SHA256_sha256"),
           "new source identity", errors)
    _check(origin.get("source_plan_sha256"), plan.get("source_plan_sha256"), "source plan identity", errors)
    preparation = contract["preparation"]
    for key, expected in (("complete", True), ("effective_plan_sha256", contract.get("plan_sha256")),
                          ("source_manifest_sha256", contract.get("SOURCE_SHA256_sha256")),
                          ("parent_manifest_sha256", contract.get("parent_manifest_sha256")),
                          ("data_manifest_sha256", contract.get("data_manifest_sha256")),
                          ("github_run_id", origin.get("github_run_id")),
                          ("launch_commit", origin.get("launch_commit"))):
        _check(preparation.get(key), expected, f"preparation {key}", errors)
    files = source.get("files", {})
    required = {"california_housing_sgdm_200k/analysis.py", "california_housing_sgdm_200k/long_train.py",
                "california_housing_sgdm_200k/plan.json", "california_housing_benchmark/spectral.py"}
    if not required.issubset(files):
        errors.append("Source manifest omits required experiment files")
    for name, expected in files.items():
        relative = Path(name)
        try:
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source path")
            _check(file_sha256(PACKAGE.parent / relative), expected, f"source {name}", errors)
        except Exception as exc:
            errors.append(f"source {name}: {type(exc).__name__}: {exc}")
    return contract


def _audit_execution(record, run_id, launch_commit, errors, label):
    for index, execution in enumerate([record.get("execution", {})] + record.get("resume_executions", [])):
        github = execution.get("github", {})
        for key, expected in (("GITHUB_REPOSITORY", REPOSITORY), ("GITHUB_RUN_ID", run_id),
                              ("GITHUB_SHA", launch_commit)):
            _check(str(github.get(key)), str(expected), f"{label} execution {index} {key}", errors)
        for key, expected in (("precision", "float32"), ("torch_num_threads", 1),
                              ("torch_num_interop_threads", 1), ("deterministic_algorithms", True)):
            _check(execution.get(key), expected, f"{label} execution {index} {key}", errors)
        base_dir = PACKAGE.parent / "california_housing_benchmark"
        engine_dir = PACKAGE.parent / "california_housing_sgdm_search" if label == "parent" else PACKAGE
        expected_sources = {name: file_sha256(base_dir / name)
                            for name in ("data.py", "train.py", "optimizer.py", "spectral.py")}
        expected_sources["long_train.py"] = file_sha256(engine_dir / "long_train.py")
        _check(execution.get("source_sha256"), expected_sources, f"{label} execution sources", errors)
        _check(record.get("source_bundle_sha256"), canonical_sha256(expected_sources),
               f"{label} source bundle", errors)


def _history(record, seed, segment, start, end, errors):
    rows = {}
    for measurement in record.get("history", []):
        step = measurement.get("step")
        try:
            if step in rows or not isinstance(step, int) or not start <= step <= end:
                raise ValueError("Duplicate or out-of-window observation")
            rows[step] = measurement_row(measurement, seed, segment)
        except Exception as exc:
            errors.append(f"{segment} step {step}: {type(exc).__name__}: {exc}")
    expected_steps = set(range(start, end + 1, 5000))
    missing = expected_steps - rows.keys()
    if missing:
        errors.append(f"{segment}: missing scheduled measurements {sorted(missing)}")
    return rows


def _checkpoint(path, seed, step, row, record, errors):
    """Check retained bytes and the model/state identity, without training."""
    from california_housing_benchmark.train import model_hash
    from california_housing_sgdm_search.long_train import checkpoint_fingerprints
    import torch
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        _check(state.get("checkpoint_kind"), "full_resume", f"checkpoint {step} kind", errors)
        _check(state.get("completed_updates"), step, f"checkpoint {step} update", errors)
        _check(state.get("identity", {}).get("seed"), seed, f"checkpoint {step} seed", errors)
        for key in ("config", "plan_sha256", "data_manifest_sha256", "source_bundle_sha256"):
            _check(state.get("identity", {}).get(key), record.get(key), f"checkpoint {step} {key}", errors)
        _check(model_hash(state["model"]), row.get("weights_sha256"), f"checkpoint {step} weights", errors)
        for key in ("optimizer", "rng", "batch_generator", "epoch_permutation", "next_offset", "minibatch_sha256"):
            if key not in state:
                errors.append(f"checkpoint {step}: missing full-resume field {key}")
        return {"step": step, "file": path.name, "sha256": file_sha256(path),
                "git_blob_sha": git_blob_sha(path), "weights_sha256": row.get("weights_sha256"),
                "state_fingerprints": checkpoint_fingerprints(state)}
    except Exception as exc:
        errors.append(f"checkpoint {step}: {type(exc).__name__}: {exc}")
        return {"step": step, "file": path.name, "error": str(exc)}


def _audit_seed(root, seed, contract):
    errors, audit = [], {"seed": seed}
    directory, parent_dir = root / "continuations" / f"seed_{seed}", root / "parent" / f"seed_{seed}"
    parent, record = read_json(parent_dir / "result.json"), read_json(directory / "result.json")
    origin, plan = contract["cohort_origin"], contract["plan"]
    parent_entry = contract["parent_manifest"].get("seeds", {}).get(str(seed), {})
    _check(parent.get("source_manifest_sha256"), contract["parent_manifest"].get("parent_source_manifest_sha256"),
           "parent frozen source manifest", errors)
    _check(parent.get("plan_sha256"), canonical_sha256(read_json(PACKAGE.parent / "california_housing_sgdm_search/plan.json")),
           "parent frozen plan", errors)
    for label, result, start, end in (("parent", parent, 20000, START), ("new", record, START, END)):
        for key, expected in (("status", "complete"), ("seed", seed), ("optimizer", "sgdm"),
                              ("phase", "continuation"), ("config", CONFIG), ("start_step", start),
                              ("total_updates", end), ("completed_updates", end),
                              ("new_updates", end - start), ("primary_estimand", "fixed_updates"),
                              ("data_manifest_sha256", contract.get("data_manifest_sha256"))):
            _check(result.get(key), expected, f"{label} {key}", errors)
        _audit_execution(result, PARENT_RUN if label == "parent" else origin.get("github_run_id"),
                         PARENT_SOURCE if label == "parent" else origin.get("launch_commit"), errors, label)
        execution_path = (parent_dir if label == "parent" else directory) / "github_execution.json"
        execution = read_json(execution_path)
        for key, expected in (("seed", seed), ("phase", "continuation"),
                              ("github_run_id", PARENT_RUN if label == "parent" else origin.get("github_run_id")),
                              ("launch_commit", PARENT_SOURCE if label == "parent" else origin.get("launch_commit"))):
            _check(str(execution.get(key)), str(expected), f"{label} github_execution {key}", errors)
    for key, expected in (("plan_sha256", canonical_sha256(plan)),
                          ("source_manifest_sha256", contract.get("SOURCE_SHA256_sha256")),
                          ("plan_file_sha256", contract.get("plan_sha256")),
                          ("effective_plan_sha256", contract.get("plan_sha256")),
                          ("source_plan_sha256", plan.get("source_plan_sha256"))):
        _check(record.get(key), expected, f"new {key}", errors)
    for name, key in (("result.json", "result_sha256"), ("github_execution.json", "github_execution_sha256"),
                       ("checkpoint100k.pt", "checkpoint_sha256"), ("completion.json", "completion_sha256")):
        _check(file_sha256(parent_dir / name), parent_entry.get(key), f"parent {name} SHA256", errors)
    checkpoint = plan.get("continuation", {}).get("checkpoints", {}).get(str(seed), {})
    _check(checkpoint.get("sha256"), parent_entry.get("checkpoint_sha256"), "pinned parent checkpoint", errors)
    _check(git_blob_sha(parent_dir / "checkpoint100k.pt"), checkpoint.get("git_blob_sha"), "parent Git blob", errors)
    _check(record.get("parent_checkpoint_sha256"), checkpoint.get("sha256"), "resumed checkpoint identity", errors)
    _check(record.get("parent", {}).get("checkpoint_sha256"), checkpoint.get("sha256"), "parent provenance", errors)
    _check(record.get("numerical_runtime_sha256"), parent.get("numerical_runtime_sha256"), "numerical runtime", errors)
    continuity = record.get("continuity") or {}
    if continuity.get("verified") is not True or not continuity.get("parent") or continuity.get("parent") != continuity.get("start"):
        errors.append("Exact full-state continuation was not verified")
    parent_rows = _history(parent, seed, "parent_20k_to_100k", 20000, START, errors)
    new_rows = _history(record, seed, "continuation_100k_to_200k", START, END, errors)
    if START in parent_rows and START in new_rows:
        _check(new_rows[START]["weights_sha256"], parent_rows[START]["weights_sha256"], "100k boundary weights", errors)
        audit["100k_reevaluation_differences"] = {metric: new_rows[START][metric] - parent_rows[START][metric]
                                                  for metric in METRICS if finite(new_rows[START].get(metric))
                                                  and finite(parent_rows[START].get(metric))}
    _check(record.get("final_weights_sha256"), new_rows.get(END, {}).get("weights_sha256"), "200k final weights", errors)
    audit["retained_checkpoints"] = [_checkpoint(directory / f"checkpoint_step_{step}.pt", seed, step,
                                                new_rows.get(step, {}), record, errors)
                                     for step in (100000, 125000, 150000, 175000, 200000)]
    if START in parent_rows:
        audit["parent_checkpoint"] = _checkpoint(parent_dir / "checkpoint100k.pt", seed, START,
                                                   parent_rows[START], parent, errors)
        parent_state = audit["parent_checkpoint"].get("state_fingerprints")
        start_state = audit["retained_checkpoints"][0].get("state_fingerprints")
        if not parent_state or parent_state != start_state or parent_state != continuity.get("start"):
            errors.append("Retained 100k states do not independently reproduce full optimizer/RNG/sampler continuity")
    completion = read_json(directory / "completion.json")
    for key, expected in (("seed", seed), ("status", "complete"), ("completed_updates", END),
                          ("github_run_id", origin.get("github_run_id")),
                          ("launch_commit", origin.get("launch_commit")),
                          ("result_sha256", file_sha256(directory / "result.json"))):
        _check(completion.get(key), expected, f"completion {key}", errors)
    retained = completion.get("retained_files", {})
    required = {"resume.pt", "final_weights.pt", "best_checkpoint.pt",
                *[f"checkpoint_step_{step}.pt" for step in (100000, 125000, 150000, 175000, 200000)]}
    if not required.issubset(retained):
        errors.append("Completion record omits required retained checkpoints")
    for name in sorted(required):
        try:
            _check(file_sha256(directory / name), retained.get(name), f"retained {name} SHA256", errors)
        except OSError as exc:
            errors.append(f"retained {name}: {exc}")
    merged = {**parent_rows, **{step: row for step, row in new_rows.items() if step > START}}
    audit.update(audit_errors=errors, valid=not errors,
                 parent_result_sha256=file_sha256(parent_dir / "result.json"),
                 result_sha256=file_sha256(directory / "result.json"),
                 parent_execution=parent.get("execution"), execution=record.get("execution"),
                 continuation=continuity, completion=completion)
    return merged, audit


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _display(stat):
    if stat.get("ci95_half_width") is None:
        return "unavailable"
    label = f"{stat['mean']:.5f} ± {stat['ci95_half_width']:.5f}"
    return label if stat.get("n") == 5 else label + f" (n={stat.get('n')})"


def _report(summary):
    lines = ["# California Housing: unchanged SGDM to 200,000 updates", "",
             f"Complete five seeds: **{summary['complete_five_seeds']}**; audited completions: **{summary['n_completed']}/5**.", "",
             "Primary endpoint: fixed 200,000 updates. Settings remain LR 0.03, momentum 0.9, zero weight decay, and dropout 0.3690073279296028.", "",
             "All ± values below are 95% Student-t confidence-interval half-widths across seeds. A complete cohort has five seeds; partial estimates explicitly label their smaller n. Sample SDs and full interval limits are in summary.json and the CSV files. These intervals are conditional on the fixed train/validation/test split.", "",
             "| Updates | Confidence/tail alpha | Test RMSE | Test MAE | Test R² |", "|---:|---:|---:|---:|---:|"]
    for step in (20000, START, END):
        stats = summary["endpoints"][str(step)]
        lines.append(f"| {step:,} | {_display(stats['confidence_tail_alpha'])} | {_display(stats['test_rmse'])} | {_display(stats['test_mae'])} | {_display(stats['test_r2'])} |")
    lines += ["", "RMSE and MAE use target units of $100,000; lower is better. R² is a regression score, not classification accuracy. The alpha average uses ordinary PL alpha with weights sqrt(num_pl_spikes)/(D+0.02), without matrix-size weighting. Only the 784×784 hidden matrix is eligible, so the weighted average equals that matrix's fitted alpha; the error bars quantify seed uncertainty, not layer dispersion.", "",
              "## Paired fixed-budget changes", "", "Changes are later minus earlier for the same seeds; negative RMSE changes indicate improvement.", "",
              "| Contrast | Change in test RMSE | Change in alpha |", "|---|---:|---:|"]
    for label, changes in summary["paired_changes"].items():
        lines.append(f"| {label} | {_display(changes['test_rmse'])} | {_display(changes['confidence_tail_alpha'])} |")
    if summary["complete_five_seeds"]:
        alpha = summary["endpoints"][str(END)]["confidence_tail_alpha"]
        side = "contains" if alpha["ci95_low"] <= 2 <= alpha["ci95_high"] else "excludes"
        lines += ["", f"The 200,000-update alpha interval {side} 2. An interval containing 2 would not establish exact convergence; an interval excluding 2 should not be described as exact marginality."]
    lines += ["", "## Provenance and scope", "",
              "The 20k→100k and 100k→200k segments retain the same settings and exact checkpoint state at their boundary. The audit compares model, optimizer, RNG, and sampler continuity, retained checkpoint hashes, and the weights used for each RMSE/alpha pair. A change of GitHub runner CPU remains possible; hardware and 100k re-evaluation differences are recorded in audit.json.", "",
              "This horizon extension does not retune the optimizer or select a result by test score or alpha. Native and modified AdamW currently stop at 20,000 updates; comparing their scores with SGDM at 200,000 uses unequal compute and does not isolate optimizer effects."]
    if summary["audit_errors"]:
        lines += ["", "## Incomplete or invalid evidence", ""] + [f"- {error}" for error in summary["audit_errors"]]
    return "\n".join(lines) + "\n"


def _plot(output, trajectory, seed_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
    for ax, metric, label in zip(axes, ("confidence_tail_alpha", "test_rmse"),
                                  (r"Confidence/tail-weighted $\bar{\alpha}$", "Test RMSE ($100,000 units)")):
        selected = [row for row in trajectory if row["metric"] == metric]
        x = np.array([row["step"] for row in selected]) / 1000
        mean = np.array([row["mean"] for row in selected])
        low = np.array([row["ci95_low"] for row in selected])
        high = np.array([row["ci95_high"] for row in selected])
        for seed in SEEDS:
            rows = sorted((row for row in seed_rows if row["seed"] == seed), key=lambda row: row["step"])
            ax.plot([row["step"] / 1000 for row in rows], [row[metric] for row in rows],
                    color="#247552", alpha=.17, linewidth=.7)
        ax.fill_between(x, low, high, color="#4d9b75", alpha=.22, linewidth=0)
        ax.plot(x, mean, color="#176341", linewidth=2, label="SGDM mean; 95% CI across 5 seeds")
        ax.axvline(100, color=".5", linestyle=":", linewidth=1)
        ax.set(xlabel="Training updates (thousands)", ylabel=label, xlim=(20, 200))
        ax.grid(alpha=.18)
    axes[0].axhline(2, color=".35", linestyle="--", linewidth=1, label=r"$\bar{\alpha}=2$")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[1].text(.97, .95, "Lower RMSE is better", ha="right", va="top", transform=axes[1].transAxes, fontsize=9)
    fig.suptitle("California Housing: fixed SGDM settings, continued to 200,000 updates", fontsize=12)
    fig.savefig(output / "sgdm_200k_trajectories.png", dpi=220)
    fig.savefig(output / "sgdm_200k_trajectories.pdf")
    plt.close(fig)


def analyze(root, make_plots=True):
    root = Path(root)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    errors, audits, valid = [], {}, {}
    contract = _load_contract(root, errors)
    for seed in SEEDS:
        try:
            rows, audit = _audit_seed(root, seed, contract)
            audits[str(seed)] = audit
            errors += [f"seed {seed}: {error}" for error in audit["audit_errors"]]
            if audit["valid"]:
                valid[seed] = rows
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append(f"seed {seed}: {message}")
            audits[str(seed)] = {"seed": seed, "valid": False, "audit_errors": [message]}
    all_rows = [row for seed in SEEDS for row in valid.get(seed, {}).values()]
    endpoints = {}
    for step in (20000, START, END):
        endpoints[str(step)] = {metric: seed_statistics([valid[seed][step][metric] for seed in valid
                                                       if step in valid[seed] and finite(valid[seed][step].get(metric))])
                                for metric in METRICS}
    trajectory = []
    for step in sorted({row["step"] for row in all_rows}):
        for metric in METRICS:
            trajectory.append({"step": step, "metric": metric,
                               **seed_statistics([row[metric] for row in all_rows
                                                  if row["step"] == step and finite(row.get(metric))])})
    changes = {}
    for earlier, later in ((20000, START), (START, END), (20000, END)):
        changes[f"{later:,} minus {earlier:,}"] = {
            metric: paired_effect({seed: rows.get(later, {}).get(metric) for seed, rows in valid.items()},
                                  {seed: rows.get(earlier, {}).get(metric) for seed, rows in valid.items()})
            for metric in METRICS}
    complete = len(valid) == len(SEEDS) and not errors
    parent_completed = 0
    for seed in SEEDS:
        try:
            parent = read_json(root / "parent" / f"seed_{seed}" / "result.json")
            parent_completed += int(parent.get("status") == "complete" and parent.get("completed_updates") == START
                                    and parent.get("seed") == seed)
        except (OSError, ValueError, TypeError):
            pass
    summary = {"schema_version": 1, "complete_five_seeds": complete, "n_completed": len(valid),
               "n_parent_completed": parent_completed,
               "expected_seeds": list(SEEDS), "completed_seeds": sorted(valid), "optimizer": "sgdm",
               "config": CONFIG, "start_step": START, "total_updates": END,
               "primary_estimand": "fixed 200,000 updates", "endpoints": endpoints,
               "paired_changes": changes, "audit_errors": errors,
               "uncertainty": "Equal-weight seed means, sample SD, two-sided 95% Student-t intervals across five seeds; fixed data split",
               "alpha": "Ordinary fitted alpha, sqrt(num_pl_spikes)/(D+0.02), no matrix-size weight, one eligible 784x784 matrix",
               "layer_dispersion": None, "metrics_units": {"rmse": "100000 USD", "mae": "100000 USD"},
               "cohort": contract.get("cohort_origin"), "parent_cohort": {"github_run_id": PARENT_RUN, "launch_commit": PARENT_SOURCE,
                                                                        "archive_commit": contract.get("parent_manifest", {}).get("archive_commit")},
               "input_sha256": {key: value for key, value in contract.items() if key.endswith("_sha256")},
               "plots_generated": bool(complete and make_plots)}
    _write_json(output / "summary.json", summary)
    _write_json(output / "audit.json", {"audit_errors": errors, "seeds": audits})
    _write_csv(output / "trajectory_seed_values.csv", all_rows)
    _write_csv(output / "trajectory_statistics.csv", trajectory)
    _write_csv(output / "endpoint_seed_values.csv", [row for row in all_rows if row["step"] in (20000, START, END)])
    _write_csv(output / "paired_changes.csv", [{"contrast": contrast, "metric": metric,
                                                **{key: value for key, value in stats.items() if key != "seed_differences"}}
                                               for contrast, metrics in changes.items() for metric, stats in metrics.items()])
    (output / "RESULTS.md").write_text(_report(summary))
    if complete and make_plots:
        _plot(output, trajectory, all_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.root)
    print(json.dumps({key: summary[key] for key in ("complete_five_seeds", "n_completed", "n_parent_completed", "audit_errors")}, indent=2))
    raise SystemExit(0 if summary["complete_five_seeds"] else 1)


if __name__ == "__main__":
    main()

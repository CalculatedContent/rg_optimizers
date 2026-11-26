"""Restrict the unchanged development grid to 20k; audit fresh capped confirmation.

Selection uses initialization, completed epochs, and the fixed 20k endpoint.
Later source observations remain provenance and never enter this analysis.
Failed spectral fits remain explicit outcomes, not performance exclusions.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from california_housing_sgdm_search.analysis import (
    canonical_sha256, file_sha256, finite, measurement_row, read_json,
    seed_statistics, spectral_alpha, welch_effect)

CAP = 20000
DEV_SEEDS = [3338, 3339]
FRESH_SEEDS = list(range(3340, 3345))
HISTORY_STEPS = [0, 2000, 5000, 10000, 15000, CAP]
METRICS = ("test_rmse", "test_mae", "test_mse", "test_r2",
           "validation_rmse", "train_rmse", "confidence_tail_alpha")
PACKAGE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_cap(plan):
    require(plan.get("development_updates") == CAP and plan.get("total_updates") == CAP,
            "Both development analysis and fresh training must stop at 20000")
    require(plan.get("development_seeds") == DEV_SEEDS and plan.get("confirmation_seeds") == FRESH_SEEDS,
            "The frozen development/confirmation seed separation changed")
    candidates = plan.get("candidates", [])
    require(len(candidates) == 32 and len({row["id"] for row in candidates}) == 32,
            "The complete frozen grid has exactly 32 candidates")
    require(plan.get("checkpoint_updates") == [2000, CAP], "Checkpoint budgets exceed or differ from the cap")


def verified_input(root, metadata, expected_path, source):
    require(metadata.get("local_path") == expected_path, "Input inventory path differs")
    path = Path(root) / expected_path
    require(not Path(expected_path).is_absolute() and ".." not in Path(expected_path).parts,
            "Unsafe source input path")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == metadata.get("sha256"), f"Input hash differs: {expected_path}")
    require(hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() == metadata.get("git_blob_sha"),
            f"Input Git object differs: {expected_path}")
    require(metadata.get("bytes") == len(payload), f"Input size differs: {expected_path}")
    for key in ("repository", "archive_commit"):
        require(metadata.get(key) == source[key], f"Input archive provenance differs: {key}")
    return json.loads(payload)


def minimum_validation_within_cap(record):
    """The exact declared selection set; ignore non-epoch intermediate samples."""
    history = record.get("history", [])
    start = [row for row in history if row.get("step") == 0]
    end = [row for row in history if row.get("step") == CAP]
    require(len(start) == len(end) == 1, "Selection requires unique initialization and 20000 observations")
    epochs = [row for row in record.get("epoch_history", []) if 0 < row.get("step", -1) <= CAP]
    epoch_size = record.get("epoch_size")
    require(epoch_size == 52, "The frozen model/data recipe has 52 minibatches per epoch")
    require([row.get("step") for row in epochs] == list(range(epoch_size, CAP + 1, epoch_size)),
            "Validation epoch coverage through 20000 is incomplete or duplicated")
    observations = [*start, *epochs, *end]
    for row in observations:
        value = (row.get("validation") or {}).get("rmse")
        require(finite(value) and value >= 0, "Invalid within-cap validation RMSE")
    best = min(observations, key=lambda row: (row["validation"]["rmse"], row["step"]))
    return {"step": best["step"], "validation_rmse": float(best["validation"]["rmse"]),
            "n_selection_observations": len(observations)}


def checked_measurement(observation, cohort, seed, *, test, binding=True):
    spectrum = observation.get("spectral")
    require(isinstance(spectrum, dict) and spectrum.get("n_eligible") == 1
            and len(spectrum.get("matrices", [])) == 1,
            "The declared one-matrix spectral measurement is missing")
    matrix = spectrum["matrices"][0]
    require(matrix.get("shape") == [784, 784], "Spectral matrix differs from the frozen architecture")
    alpha = spectral_alpha(spectrum)
    if alpha is None:
        require(spectrum.get("confidence_tail_alpha") is None and spectrum.get("complete") is False,
                "An invalid spectral fit must remain explicitly undefined")
    else:
        require(spectrum.get("complete") is True, "A valid fit is mislabeled incomplete")
    issues = []
    row = measurement_row(observation, cohort=cohort, seed=seed, endpoint=f"fixed_{observation['step']}",
                          issues=issues, require_test=test, require_binding=binding,
                          require_spectrum=False)
    require(not issues, "; ".join(issues))
    fit = matrix.get("fit") or {}
    row.update(spectral_fit_valid=alpha is not None, fit_status=fit.get("status", matrix.get("status")),
               fit_D=fit.get("D"), fit_tail_eigenvalues=fit.get("num_pl_spikes"), fit_xmin=fit.get("xmin"),
               spectral_diagnostic=matrix.get("diagnostic", spectrum.get("diagnostic")),
               binding_method="explicit matching weight hashes" if binding else
               "legacy metrics and spectra share one frozen observation; intermediate weight hashes were not recorded")
    return row


def compute_development(root, plan, identity):
    """Read original files without relabeling their actual status or duration."""
    root = Path(root)
    validate_cap(plan)
    require(file_sha256(root / "source_inventory.json") == identity["source_inventory_sha256"],
            "Source inventory changed")
    inventory = read_json(root / "source_inventory.json")
    require(inventory.get("parent_source") == plan["parent_source"], "Development archive differs")
    rows = inventory.get("development", [])
    indexed = {(row.get("candidate_id"), row.get("seed")): row for row in rows}
    expected = {(candidate["id"], seed) for candidate in plan["candidates"] for seed in DEV_SEEDS}
    require(len(rows) == len(indexed) == 64 and set(indexed) == expected, "All 64 unique development prefixes are required")
    source = plan["parent_source"]
    old_plan = read_json(PACKAGE.parent / "california_housing_sgdm_search/plan.json")
    numerical_sources = {name: file_sha256(PACKAGE.parent / "california_housing_benchmark" / name)
                         for name in ("data.py", "train.py", "optimizer.py", "spectral.py")}
    numerical_sources["long_train.py"] = file_sha256(PACKAGE.parent / "california_housing_sgdm_search/long_train.py")
    seed_rows, scores = [], []
    for candidate in plan["candidates"]:
        candidate_rows = []
        for seed in DEV_SEEDS:
            entry = indexed[candidate["id"], seed]
            prefix = f"inputs/development/{candidate['id']}/seed_{seed}"
            record = verified_input(root, entry["result"], prefix + "/result.json", source)
            execution = verified_input(root, entry["github_execution"], prefix + "/github_execution.json", source)
            required = {"optimizer": "sgdm", "seed": seed, "phase": "development", "config": candidate["config"],
                        "plan_sha256": canonical_sha256(old_plan),
                        "source_bundle_sha256": canonical_sha256(numerical_sources),
                        "source_manifest_sha256": source["source_manifest_sha256"],
                        "data_manifest_sha256": identity["data_manifest_sha256"]}
            for key, value in required.items():
                require(record.get(key) == value, f"{candidate['id']}/{seed}: original {key} differs")
            require(record.get("completed_updates", 0) >= CAP, "Development source never reached 20000")
            require(entry.get("source_completed_updates") == record["completed_updates"]
                    and entry.get("source_status") == record["status"], "Original duration/status was relabeled")
            require(entry.get("action") == "reuse_20k_history" and entry.get("new_updates") == 0,
                    "This experiment may not replay development updates")
            for current in [record.get("execution", {}), *record.get("resume_executions", [])]:
                github = current.get("github", {})
                for key, value in (("GITHUB_RUN_ID", source["run_id"]), ("GITHUB_SHA", source["launch_commit"]),
                                   ("GITHUB_REPOSITORY", source["repository"])):
                    require(str(github.get(key)) == str(value), "Original development execution provenance differs")
            for key, value in (("seed", seed), ("candidate_id", candidate["id"]), ("phase", "development"),
                               ("github_run_id", source["run_id"]), ("launch_commit", source["launch_commit"])):
                require(execution.get(key) == value, f"Original development execution record differs: {key}")
            for observation in [*record.get("history", []), record.get("best_validation") or {}]:
                require(observation.get("test") in (None, {}), "Development record contains test measurements")
            prefix_rows = [row for row in record["history"] if row.get("step", -1) <= CAP]
            require([row.get("step") for row in prefix_rows] == HISTORY_STEPS, "Development history prefix is incomplete")
            best = minimum_validation_within_cap(record)
            endpoint = checked_measurement(prefix_rows[-1], candidate["id"], seed, test=False)
            endpoint.update(candidate_id=candidate["id"], **candidate["config"],
                            selection_step=best["step"], selection_validation_rmse=best["validation_rmse"],
                            n_selection_observations=best["n_selection_observations"],
                            source_completed_updates=record["completed_updates"], source_status=record["status"],
                            source_result_sha256=entry["result"]["sha256"],
                            source_git_blob_sha=entry["result"]["git_blob_sha"],
                            source_repository_path=entry["result"]["repository_path"], analysis_max_step=CAP,
                            new_development_updates=0)
            seed_rows.append(endpoint)
            candidate_rows.append(endpoint)
        scores.append({"candidate_id": candidate["id"], "config": candidate["config"], "n_seeds": 2,
                       "mean_min_validation_rmse": statistics.fmean(row["selection_validation_rmse"] for row in candidate_rows),
                       "mean_fixed20000_validation_rmse": statistics.fmean(row["validation_rmse"] for row in candidate_rows),
                       "n_valid_fixed20000_alpha": sum(row["spectral_fit_valid"] for row in candidate_rows),
                       "per_seed": [{key: row[key] for key in ("seed", "selection_step", "selection_validation_rmse",
                                                               "validation_rmse", "confidence_tail_alpha")} for row in candidate_rows]})
    winner = min(scores, key=lambda row: (row["mean_min_validation_rmse"], row["candidate_id"]))
    return seed_rows, scores, winner


def expected_frozen(plan, identity, seed_rows, scores, winner):
    return {"schema_version": 1, **identity, "experiment_id": plan["experiment_id"],
            "calibration_complete": True, "analysis_max_step": CAP, "n_development_records": 64,
            "n_candidates": 32, "complete_grid": True, "additional_development_updates": 0,
            "development_seeds": DEV_SEEDS, "confirmation_seeds": FRESH_SEEDS,
            "selection": plan["selection"], "selected_candidate_id": winner["candidate_id"],
            "selected": winner["config"], "selected_mean_min_validation_rmse": winner["mean_min_validation_rmse"],
            "candidate_scores": scores, "selection_evidence_sha256": canonical_sha256(seed_rows),
            "n_invalid_fixed20000_spectral_fits": sum(not row["spectral_fit_valid"] for row in seed_rows),
            "source_note": "Restricted observations from original longer trajectories; original status and completed updates remain unchanged."}


def validate_frozen_selection(root, plan, identity):
    seed_rows, scores, winner = compute_development(root, plan, identity)
    expected = expected_frozen(plan, identity, seed_rows, scores, winner)
    frozen = read_json(Path(root) / "frozen_selection.json")
    require(frozen == expected, "Frozen selection differs from the complete validation-only 20k recomputation")
    return frozen


def plot_development(output, rows, winner):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [row for row in rows if row["spectral_fit_valid"]]
    invalid = [row for row in rows if not row["spectral_fit_valid"]]
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    colors = {0.003: "#0072B2", 0.01: "#009E73", 0.03: "#E69F00", 0.1: "#CC79A7"}
    for lr, color in colors.items():
        group = [row for row in valid if row["lr"] == lr]
        ax.scatter([row["confidence_tail_alpha"] for row in group], [row["validation_rmse"] for row in group],
                   color=color, s=38, alpha=.85, label=f"LR {lr:g}", edgecolors="white", linewidths=.5)
    selected = [row for row in valid if row["candidate_id"] == winner["candidate_id"]]
    if selected:
        ax.scatter([row["confidence_tail_alpha"] for row in selected], [row["validation_rmse"] for row in selected],
                   s=120, facecolors="none", edgecolors="black", linewidths=1.3,
                   label=f"Selected: {winner['candidate_id']}")
    ax.axvline(2, color="#666666", linestyle="--", linewidth=1)
    ax.set(xlabel="Confidence/tail-weighted alpha at 20,000 updates",
           ylabel="Validation RMSE at 20,000 updates ($100,000 units)",
           title="California Housing: fixed-budget SGDM grid")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8, frameon=False)
    fig.text(.5, -.025, f"{len(valid)} valid-alpha points; {len(invalid)} failed fits retain their RMSE in the accompanying table.\n"
             "Each point is one candidate/seed and pairs metrics from identical weights. Lower RMSE is better.",
             ha="center", fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"grid_20000_validation_rmse_vs_alpha.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_development(root, plan, identity):
    root = Path(root)
    rows, scores, winner = compute_development(root, plan, identity)
    frozen = expected_frozen(plan, identity, rows, scores, winner)
    path = root / "frozen_selection.json"
    if path.exists():
        require(read_json(path) == frozen, "An existing frozen selection cannot change")
    else:
        write_json(path, frozen)
    output = root / "development_analysis"
    write_csv(output / "grid_20000_seed_values.csv", rows)
    flattened = [{"rank": rank, **{key: row[key] for key in ("candidate_id", "n_seeds", "mean_min_validation_rmse",
                  "mean_fixed20000_validation_rmse", "n_valid_fixed20000_alpha")}, **row["config"]}
                 for rank, row in enumerate(sorted(scores, key=lambda row: (row["mean_min_validation_rmse"], row["candidate_id"])), 1)]
    write_csv(output / "grid_candidate_scores.csv", flattened)
    summary = {"complete_grid": True, "n_development_records": 64, "n_candidates": 32,
               "analysis_max_step": CAP, "additional_development_updates": 0,
               "selected_candidate_id": winner["candidate_id"], "selected": winner["config"],
               "selected_mean_min_validation_rmse": winner["mean_min_validation_rmse"],
               "n_valid_fixed20000_alpha": sum(row["spectral_fit_valid"] for row in rows),
               "invalid_spectral_fits": [{key: row[key] for key in ("candidate_id", "seed", "validation_rmse", "fit_status", "spectral_diagnostic")}
                                         for row in rows if not row["spectral_fit_valid"]],
               "audit_errors": [], "source_provenance": plan["parent_source"]}
    write_json(output / "summary.json", summary)
    plot_development(output, rows, winner)
    lines = ["# SGDM grid restricted to 20,000 updates", "",
             "All 64 original development trajectories already reached 20,000 updates. This analysis reuses only their initialization, completed epochs through 20k, and fixed 20k endpoint. It performs zero additional development updates. The original longer histories remain unchanged.", "",
             f"Selected **{winner['candidate_id']}**, mean within-cap minimum validation RMSE **{winner['mean_min_validation_rmse']:.6f}** across seeds 3338 and 3339. No test metric or alpha entered selection.", "",
             f"Fixed-budget alpha is valid for {summary['n_valid_fixed20000_alpha']}/64 measurements. Failed fits remain undefined and retain their performance values in the grid.", "",
             "| Rank | Candidate | LR | Momentum | Weight decay | Dropout | Mean minimum validation RMSE ≤20k | Mean validation RMSE at20k |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['rank']} | {row['candidate_id']} | {row['lr']:g} | {row['momentum']:g} | {row['weight_decay']:g} | {row['dropout']:.4f} | {row['mean_min_validation_rmse']:.6f} | {row['mean_fixed20000_validation_rmse']:.6f} |" for row in flattened]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return frozen, summary


def audit_confirmation(root, plan, identity, frozen, seed):
    import torch
    from california_housing_benchmark import train as base
    from california_housing_sgdm_search.long_train import checkpoint_fingerprints
    from .run_one import validate_result
    directory = Path(root) / "runs" / f"seed_{seed}"
    record = read_json(directory / "result.json")
    validate_result(record, plan, identity, frozen["selected"], seed)
    frozen_hash = file_sha256(Path(root) / "frozen_selection.json")
    additions = {"plan_file_sha256": identity["plan_file_sha256"],
                 "source_inventory_sha256": identity["source_inventory_sha256"],
                 "orchestration_source_manifest_sha256": identity["source_manifest_sha256"],
                 "training_source_manifest_sha256": identity["training_source_manifest_sha256"],
                 "frozen_selection_sha256": frozen_hash, "github_run_id": identity["github_run_id"],
                 "launch_commit": identity["launch_commit"], "report_primary_estimand": "fixed_20000_updates"}
    for key, value in additions.items():
        require(record.get(key) == value, f"Confirmation metadata differs: {key}")
    execution, completion = (read_json(directory / name) for name in ("github_execution.json", "completion.json"))
    for item in (execution, completion):
        for key, value in {**identity, "seed": seed, "phase": "confirmation", "frozen_selection_sha256": frozen_hash}.items():
            require(item.get(key) == value, f"Confirmation execution/completion identity differs: {key}")
    require(completion.get("status") == "complete" and completion.get("completed_updates") == CAP,
            "Confirmation completion marker is not capped at 20000")
    require(completion.get("result_sha256") == file_sha256(directory / "result.json"), "Completed confirmation metrics changed")
    required = {"checkpoint_step_2000.pt", "checkpoint_step_20000.pt", "resume.pt", "best_checkpoint.pt", "final_weights.pt"}
    require(required.issubset(completion.get("retained_files", {})), "Completion omits required retained weights")
    states = {}
    for name in required:
        require(file_sha256(directory / name) == completion["retained_files"][name], f"Retained checkpoint hash differs: {name}")
        state = torch.load(directory / name, map_location="cpu", weights_only=False)
        for key, value in state["identity"].items():
            require(record.get(key) == value, f"Checkpoint identity differs: {name}/{key}")
        states[name] = state
    for step in (2000, CAP):
        state = states[f"checkpoint_step_{step}.pt"]
        point = next(row for row in record["history"] if row["step"] == step)
        require(state.get("checkpoint_kind") == "full_resume" and state.get("completed_updates") == step,
                "Full checkpoint has the wrong update budget")
        require(base.model_hash(state["model"]) == point["weights_sha256"], "Checkpoint and measured model differ")
    final = states["checkpoint_step_20000.pt"]
    require(checkpoint_fingerprints(states["resume.pt"]) == checkpoint_fingerprints(final), "Resume and final state differ")
    for name, expected_hash in (("best_checkpoint.pt", record["best_weights_sha256"]),
                                ("final_weights.pt", record["final_weights_sha256"])):
        require(base.model_hash(states[name]["model"]) == expected_hash, "Retained selected/final weights differ")
    minimum = minimum_validation_within_cap(record)
    selected = record["best_validation"]
    require(selected["step"] == minimum["step"] and math.isclose(selected["validation"]["rmse"], minimum["validation_rmse"], rel_tol=1e-10),
            "Selected confirmation checkpoint differs from declared within-cap validation minimum")
    trajectory = [checked_measurement(row, "sgdm_grid_selected", seed, test=True) for row in record["history"]]
    endpoint = next(row for row in trajectory if row["step"] == CAP)
    optional = checked_measurement(selected, "sgdm_grid_selected", seed, test=True)
    optional["endpoint"] = "best_validation_within_20000"
    return endpoint, trajectory, optional, record["execution"]


def audit_controls(root, plan, identity):
    inventory = read_json(Path(root) / "source_inventory.json")
    source = plan["controls_source"]
    entries = {(row["optimizer"], row["seed"]): row for row in inventory.get("controls", [])}
    expected = {(arm, seed) for arm in source["optimizers"] for seed in source["seeds"]}
    require(len(entries) == len(inventory.get("controls", [])) == 15 and set(entries) == expected,
            "All15 original optimizer control records are required")
    rows, runtimes = [], []
    for arm, seed in sorted(expected):
        path = f"inputs/controls/seed_{seed}/{arm}/result.json"
        record = verified_input(root, entries[arm, seed]["result"], path, source)
        require(record.get("optimizer") == arm and record.get("seed") == seed
                and record.get("data_manifest_sha256") == identity["data_manifest_sha256"], "Control identity/data differs")
        github = record.get("execution", {}).get("github", {})
        for key, value in (("GITHUB_RUN_ID", source["run_id"]), ("GITHUB_SHA", source["launch_commit"]),
                           ("GITHUB_REPOSITORY", source["repository"])):
            require(str(github.get(key)) == str(value), "Control runtime provenance differs")
        points = [row for row in record.get("history", []) if row.get("step") == CAP]
        require(len(points) == 1, "Control lacks a unique matching20k measurement")
        row = checked_measurement(points[0], "original_" + arm, seed, test=True, binding=False)
        row["source_result_sha256"] = entries[arm, seed]["result"]["sha256"]
        rows.append(row)
        runtimes.append({"cohort": "original_" + arm, "seed": seed, "execution": record["execution"]})
    return rows, runtimes


def aggregate_rows(rows):
    return {cohort: {metric: seed_statistics([row.get(metric) for row in rows if row["cohort"] == cohort], 5)
                    for metric in METRICS} for cohort in sorted({row["cohort"] for row in rows})}


def analyze(root, plan, identity):
    root = Path(root)
    frozen = validate_frozen_selection(root, plan, identity)
    errors, fresh, trajectories, selected, runtimes = [], [], [], [], []
    for seed in FRESH_SEEDS:
        try:
            endpoint, trajectory, optional, runtime = audit_confirmation(root, plan, identity, frozen, seed)
            fresh.append(endpoint); trajectories.extend(trajectory); selected.append(optional)
            runtimes.append({"cohort": "sgdm_grid_selected", "seed": seed, "execution": runtime})
        except Exception as error:
            errors.append(f"confirmation seed{seed}: {type(error).__name__}: {error}")
    controls = []
    try:
        controls, control_runtime = audit_controls(root, plan, identity)
        runtimes.extend(control_runtime)
    except Exception as error:
        errors.append(f"original controls: {type(error).__name__}: {error}")
    output = root / "analysis"
    all_rows = [*controls, *fresh]
    statistics_by_cohort = aggregate_rows(all_rows)
    effects = {}
    for control in sorted({row["cohort"] for row in controls}):
        effects[control] = {metric: welch_effect([row.get(metric) for row in fresh],
                                               [row.get(metric) for row in controls if row["cohort"] == control])
                            for metric in METRICS}
    summary = {"schema_version": 1, "complete_five_seeds": len(fresh) == 5 and not errors,
               "n_completed": len(fresh), "complete_grid": True, "n_development_records": 64,
               "horizon": CAP, "analysis_max_step": CAP, "additional_development_updates": 0,
               "confirmation_seeds": FRESH_SEEDS, "selected_candidate_id": frozen["selected_candidate_id"],
               "selected": frozen["selected"], "primary_estimand": "fixed_20000_updates",
               "statistics": statistics_by_cohort, "independent_seed_contrasts_new_minus_control": effects,
               "n_valid_confirmation_alpha": sum(row["spectral_fit_valid"] for row in fresh),
               "audit_errors": errors, "identity": identity,
               "interpretation": plan["comparison_policy"],
               "uncertainty": "Means weight seeds equally. Sample SD and two-sided95%Student-t CI describe seed uncertainty, not layer dispersion. Any incomplete alpha statistic explicitly reports its valid n.",
               "fit_failures": [{"cohort": row["cohort"], "seed": row["seed"], "diagnostic": row["spectral_diagnostic"]}
                                for row in all_rows if not row["spectral_fit_valid"]]}
    write_json(output / "summary.json", summary)
    write_json(output / "execution_cohort.json", {"identity": identity, "complete_five_seeds": summary["complete_five_seeds"],
                                                 "n_completed": len(fresh), "executions": runtimes})
    write_csv(output / "fixed_20000_seed_values.csv", all_rows)
    write_csv(output / "confirmation_trajectory_seed_values.csv", trajectories)
    write_csv(output / "optional_best_validation_within_20000.csv", selected)
    write_csv(output / "fixed_20000_statistics.csv", [{"cohort": cohort, "metric": metric, **value}
                                                     for cohort, stats in statistics_by_cohort.items() for metric, value in stats.items()])
    trajectory_stats = [{"step": step, "metric": metric, **seed_statistics([row.get(metric) for row in trajectories if row["step"] == step], 5)}
                        for step in HISTORY_STEPS for metric in ("test_rmse", "confidence_tail_alpha")]
    write_csv(output / "confirmation_trajectory_statistics.csv", trajectory_stats)
    lines = ["# California Housing: capped20k SGDM confirmation", "",
             f"Complete five-seed confirmation: **{summary['complete_five_seeds']}**; audited runs **{len(fresh)}/5**. All64 development prefixes were reused without further training.", "",
             f"Frozen winner: **{frozen['selected_candidate_id']}**, `{json.dumps(frozen['selected'], sort_keys=True)}`.", "",
             "All comparisons below use20,000 updates. New SGDM uses fresh seeds3340–3344; original controls use seeds2340–2344. Tuning and possibly dropout differ. These independent-seed comparisons do not isolate an optimizer-only causal effect.", "",
             "Lower RMSE is better. RMSE units are$100,000; it is not a percentage error. Alpha uses the ordinary fitted value with the prescribed confidence/tail weight. One eligible matrix means no layer-dispersion estimate.", "",
             "| Cohort | Test RMSE mean±95%CI | Test RMSE sampleSD | Alpha mean±95%CI | Alpha sampleSD |", "|---|---:|---:|---:|---:|"]
    def display(value, field):
        return "undefined" if value.get(field) is None else f"{value[field]:.6f}"
    for cohort, stats in statistics_by_cohort.items():
        rmse, alpha = stats["test_rmse"], stats["confidence_tail_alpha"]
        lines.append(f"| {cohort} | {display(rmse,'mean')}±{display(rmse,'ci95_half_width')} (n={rmse['n']}) | {display(rmse,'sample_sd')} | {display(alpha,'mean')}±{display(alpha,'ci95_half_width')} (n={alpha['n']}) | {display(alpha,'sample_sd')} |")
    if errors:
        lines += ["", "## Audit failures", "", *["- " + error for error in errors]]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if errors:
        raise ValueError("Final20k audit incomplete: " + "; ".join(errors))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("analyze-development", "analyze"), default="analyze")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    from .verify import load_context
    root, plan, identity = load_context(args.root)
    if args.development_only or args.command == "analyze-development":
        _, summary = analyze_development(root, plan, identity)
    else:
        summary = analyze(root, plan, identity)
    print(json.dumps({key: summary[key] for key in ("complete_grid", "n_development_records")}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Read saved short-GPT confirmations; never train, load weights, or refit spectra.

Kept in a subdirectory because the frozen runners hash their top-level *.py files.
Only SciPy (for Student-t quantiles) is needed beyond the Python standard library.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


OPTIMIZERS = ("AdamW", "SGDM")
CHECKPOINTS = ("endpoint", "validation_selected")
DEFAULT_SEEDS = (1340, 1341, 1342, 1343, 1344)
TOKEN = "transformer.wte.weight"
HEAD = "lm_head.weight"
MATRICES = (TOKEN, "transformer.wpe.weight") + tuple(
    f"transformer.h.{block}.{suffix}.weight"
    for block in range(4)
    for suffix in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
)
PERFORMANCE = tuple(
    f"{split}_{metric}"
    for split in ("train", "validation", "test")
    for metric in ("loss_nats_per_character", "perplexity", "next_character_accuracy")
)
MODEL_METRICS = (
    "mean_alpha", "mean_alpha_below_two", "any_layer_below_two",
    "valid_below_two_count", "valid_below_two_fraction", "valid_fit_count",
) + PERFORMANCE
UNCERTAINTY = (
    "95% paired Student-t interval across independent seed-pair differences; "
    "n is seeds, never matrices or checkpoints. Descriptive, especially at five "
    "pairs; binary/count intervals are approximate and unbounded. No multiple-"
    "comparison adjustment. Shared corpus, recipe search and prior test exposure "
    "are not represented by this interval."
)


def number(value):
    """Finite JSON numbers only; booleans are not spectral exponents."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(name):
    # This is the tied-weight contract of short_test/common.py, not a heuristic.
    return TOKEN if name == HEAD else name


def fit_value(row):
    fits = row.get("fits")
    if not isinstance(fits, list) or len(fits) != 1 or not isinstance(fits[0], dict):
        return None, "expected_exactly_one_fit"
    fit = fits[0]
    if fit.get("status") != "success":
        return None, "fit_not_successful"
    if not number(fit.get("alpha")):
        return None, "nonfinite_or_missing_alpha"
    return float(fit["alpha"]), "valid"


def normalize_spectrum(alpha):
    """Audit all 18 expected identities, deduplicating tied aliases once."""
    groups = defaultdict(list)
    issues = []
    alpha = alpha if isinstance(alpha, dict) else {}
    rows = alpha.get("matrices")
    if not isinstance(rows, list):
        rows = []
        issues.append("missing_matrices_list")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("matrix"), str):
            issues.append("unnamed_matrix_row")
            continue
        name = canonical(row["matrix"])
        if name not in MATRICES:
            issues.append(f"unexpected_matrix:{row['matrix']}")
            continue
        groups[name].append(row)

    matrices = []
    for name in MATRICES:
        candidates = groups[name]
        aliases = [TOKEN, HEAD] if name == TOKEN else [name]
        value, status = None, "missing_matrix"
        if candidates:
            value, status = fit_value(candidates[0])
            # A duplicate may be collapsed only when its saved fit and shape agree.
            signatures = {
                json.dumps([r.get("shape"), r.get("fits")], sort_keys=True)
                for r in candidates
            }
            if len(signatures) > 1:
                value, status = None, "conflicting_duplicate_fits"
            for row in candidates:
                declared = row.get("aliases", [row["matrix"]])
                if (not isinstance(declared, list) or
                        any(not isinstance(a, str) or canonical(a) != name for a in declared)):
                    value, status = None, "conflicting_alias_identity"
        matrices.append(dict(matrix=name, aliases=aliases, alpha=value,
                             below_two=value < 2 if value is not None else None,
                             status=status, source_rows=len(candidates)))

    values = [r["alpha"] for r in matrices if r["status"] == "valid"]
    valid = len(values)
    below = sum(a < 2 for a in values)
    complete = valid == len(MATRICES) and not issues
    mean = statistics.fmean(values) if complete else None
    # True is supported by one observed below-two fit; false requires full coverage.
    any_below = True if below else (False if complete else None)
    summary = dict(
        expected_fit_count=len(MATRICES), valid_fit_count=valid,
        missing_fit_count=len(MATRICES) - valid, all_fits_valid=complete,
        mean_alpha=mean, mean_alpha_below_two=mean < 2 if mean is not None else None,
        mean_valid_alpha=statistics.fmean(values) if values else None,
        any_layer_below_two=any_below,
        any_valid_layer_below_two=bool(below) if valid else None,
        valid_below_two_count=below,
        valid_below_two_fraction=below / valid if valid else None,
        missing_matrices=[r["matrix"] for r in matrices if r["status"] == "missing_matrix"],
        invalid_matrices=[r["matrix"] for r in matrices if r["status"] not in ("valid", "missing_matrix")],
        duplicate_rows_collapsed=sum(max(0, len(g) - 1) for g in groups.values()),
    )
    # Saved aggregate fields are audited, never used instead of the named fits.
    recomputed = {"unique_matrices": len(MATRICES), "successful_fits": valid,
                  "below_two": below, "mean_alpha": mean}
    for key, expected in recomputed.items():
        if key not in alpha:
            continue
        recorded = alpha[key]
        agrees = (recorded is None and expected is None) or (
            number(recorded) and number(expected) and
            math.isclose(recorded, expected, rel_tol=1e-10, abs_tol=1e-12)
        )
        if not agrees:
            issues.append(f"saved_{key}_disagrees_with_named_fits")
    return summary, matrices, issues


def normalize_checkpoint(result, checkpoint, path):
    payload = result[checkpoint]
    summary, matrices, issues = normalize_spectrum(payload.get("alpha"))
    row = dict(optimizer=result["job"]["recipe"]["optimizer"],
               seed=result["job"]["seed"], checkpoint=checkpoint,
               step=payload.get("step"), source=str(path), **summary)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    for split in ("train", "validation", "test"):
        split_metrics = metrics.get(split, {})
        if not isinstance(split_metrics, dict):
            split_metrics = {}
        for metric in ("loss_nats_per_character", "perplexity", "next_character_accuracy", "evaluated_targets"):
            key = f"{split}_{metric}"
            value = split_metrics.get(metric)
            row[key] = value if number(value) else None
            if row[key] is None:
                issues.append(f"missing_or_invalid_metric:{key}")
    row.update(matrices=matrices, issues=issues)
    return row


def seed_list(values):
    if (not isinstance(values, (list, tuple)) or not values or
            any(type(s) is not int for s in values) or len(set(values)) != len(values)):
        raise ValueError("Confirmation seeds must be a nonempty list of unique integers")
    return sorted(values)


def load_study(study, seeds=None):
    """Read one study; expose omissions and exclude ambiguous/mixed confirmations."""
    study = Path(study).resolve()
    if not study.is_dir():
        raise ValueError(f"Study directory does not exist: {study}")
    design_path, recipes_path = study / "design.json", study / "recipes.json"
    design = read_json(design_path) if design_path.exists() else {}
    recipes = read_json(recipes_path) if recipes_path.exists() else {}
    if not isinstance(design, dict) or not isinstance(recipes, dict):
        raise ValueError("design.json and recipes.json must contain objects")
    expected = seed_list(seeds if seeds is not None else design.get(
        "confirmation_seeds", recipes.get("confirmation_seeds", DEFAULT_SEEDS)))
    seed_source = ("command_line" if seeds is not None else "design.json" if "confirmation_seeds" in design
                   else "recipes.json" if "confirmation_seeds" in recipes else "frozen_five_seed_default")
    architecture = design.get("architecture", {})
    if not isinstance(architecture, dict) or architecture.get("n_layer", 4) != 4:
        raise ValueError("This command supports the frozen four-block, 18-matrix schema")
    issues = []
    if seed_source == "frozen_five_seed_default":
        issues.append("No seed manifest: expected seeds default to 1340-1344; use --seeds to override")
    design_hash = digest(design_path) if design_path.exists() else None
    selected = recipes.get("selected", {})
    if not isinstance(selected, dict):
        raise ValueError("recipes.json selected must be an optimizer-to-recipe object")
    if recipes.get("design_sha256") and design_hash and recipes["design_sha256"] != design_hash:
        raise ValueError("recipes.json does not belong to this design.json")

    entries, by_key = [], defaultdict(list)
    for path in sorted((study / "confirmation").glob("*/result.json")):
        entry = dict(path=str(path), sha256=digest(path), status="excluded", issues=[])
        entries.append(entry)
        try:
            result = read_json(path)
            job = result["job"]
            optimizer, seed = job["recipe"]["optimizer"], job["seed"]
            if optimizer not in OPTIMIZERS or type(seed) is not int:
                raise ValueError("invalid optimizer or seed")
            entry.update(optimizer=optimizer, seed=seed)
            by_key[(optimizer, seed)].append((result, entry))
            if job.get("phase") != "confirmation":
                entry["issues"].append("not_confirmation")
            if result.get("completed") is not True:
                entry["issues"].append("result_not_completed")
            if seed not in expected:
                entry["issues"].append("unexpected_seed")
            if not isinstance(result.get("design_sha256"), str) or not result["design_sha256"]:
                entry["issues"].append("missing_design_sha256")
            elif design_hash and result["design_sha256"] != design_hash:
                entry["issues"].append("design_sha256_mismatch")
            if selected and job["recipe"] != selected.get(optimizer):
                entry["issues"].append("recipe_differs_from_frozen_selection")
            if (path.parent / "numerical_failure.json").exists():
                entry["issues"].append("result_and_numerical_failure_both_present")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            entry["issues"].append(f"unreadable_result:{exc}")

    eligible = [(r, e) for group in by_key.values() for r, e in group if not e["issues"]]
    hashes = {r["design_sha256"] for r, _ in eligible}
    if len(hashes) > 1:
        for _, entry in eligible:
            entry["issues"].append("mixed_design_hashes")
    # Even without recipes.json, never silently pool different selected recipes.
    for optimizer in OPTIMIZERS:
        group = [(r, e) for r, e in eligible if e["optimizer"] == optimizer]
        signatures = {json.dumps(r["job"]["recipe"], sort_keys=True) for r, _ in group}
        if len(signatures) > 1:
            for _, entry in group:
                entry["issues"].append("multiple_recipes_for_optimizer")

    accepted = {}
    for key, group in by_key.items():
        if len(group) > 1:
            for _, entry in group:
                entry["issues"].append("duplicate_optimizer_seed")
        for result, entry in group:
            if not entry["issues"]:
                entry["status"] = "accepted"
                accepted[key] = (result, entry)

    # Jobs without result.json may be pending, interrupted, or numerically failed.
    pending = defaultdict(list)
    for path in sorted((study / "confirmation").glob("*/job.json")):
        try:
            job = read_json(path)
            key = (job["recipe"]["optimizer"], job["seed"])
            if key[0] not in OPTIMIZERS or type(key[1]) is not int:
                raise ValueError("invalid optimizer or seed")
            if not (path.parent / "result.json").exists():
                state = "numerical_failure" if (path.parent / "numerical_failure.json").exists() else "missing_result"
                pending[key].append(dict(path=str(path.parent), status=state))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"unreadable_job:{path}:{exc}")

    run_audit, models = [], []
    for seed in expected:
        for optimizer in OPTIMIZERS:
            key = (optimizer, seed)
            audit = dict(seed=seed, optimizer=optimizer, status="missing_result",
                         files=[e["path"] for _, e in by_key[key]], pending_jobs=pending[key])
            if key in accepted:
                result, entry = accepted[key]
                audit["status"] = "accepted"
                for checkpoint in CHECKPOINTS:
                    if isinstance(result.get(checkpoint), dict):
                        models.append(normalize_checkpoint(result, checkpoint, entry["path"]))
                    else:
                        entry["issues"].append(f"missing_checkpoint:{checkpoint}")
            elif by_key[key]:
                audit["status"] = "excluded_result"
            elif pending[key] and all(p["status"] == "numerical_failure" for p in pending[key]):
                audit["status"] = "numerical_failure"
            run_audit.append(audit)
    return models, dict(
        study=str(study), expected_seeds=expected, seed_source=seed_source,
        design_sha256=design_hash, result_design_sha256=sorted(hashes),
        recipes_sha256=digest(recipes_path) if recipes_path.exists() else None,
        recipes=selected, input_files=entries, runs=run_audit, issues=issues,
        test_disclosures=sorted({str(r.get("test_disclosure", "not recorded")) for r, _ in accepted.values()}),
    )


def paired_statistic(values, expected):
    """values is seed -> (AdamW, SGDM); one difference per independent pair."""
    seeds = sorted(values)
    differences = [float(values[s][0]) - float(values[s][1]) for s in seeds]
    n = len(seeds)
    mean = statistics.fmean(differences) if n else None
    sd = statistics.stdev(differences) if n > 1 else None
    half = None
    if n > 1:
        from scipy.stats import t
        half = float(t.ppf(.975, n - 1)) * sd / math.sqrt(n)
    return dict(
        n_pairs=n, paired_seeds=seeds, missing_pair_seeds=sorted(set(expected) - set(seeds)),
        planned_pairs=len(expected), all_planned_pairs=n == len(expected),
        adamw_mean=statistics.fmean(float(values[s][0]) for s in seeds) if n else None,
        sgdm_mean=statistics.fmean(float(values[s][1]) for s in seeds) if n else None,
        adamw_minus_sgdm_mean=mean, paired_sd=sd,
        ci95_low=mean - half if half is not None else None,
        ci95_high=mean + half if half is not None else None,
        differences=[dict(seed=s, adamw=values[s][0], sgdm=values[s][1],
                          adamw_minus_sgdm=d) for s, d in zip(seeds, differences)],
    )


def build_report(study, seeds=None):
    models, audit = load_study(study, seeds)
    expected = audit["expected_seeds"]
    indexed = {(r["checkpoint"], r["seed"], r["optimizer"]): r for r in models}
    paired_layers, statistics_rows, pairs = [], [], []
    for checkpoint in CHECKPOINTS:
        matched = {}
        for seed in expected:
            a, s = (indexed.get((checkpoint, seed, optimizer)) for optimizer in OPTIMIZERS)
            pairs.append(dict(checkpoint=checkpoint, seed=seed, complete=a is not None and s is not None,
                              missing_optimizers=[o for o, r in zip(OPTIMIZERS, (a, s)) if r is None]))
            if a is not None and s is not None:
                matched[seed] = (a, s)
                for am, sm in zip(a["matrices"], s["matrices"]):
                    row = dict(checkpoint=checkpoint, seed=seed, matrix=am["matrix"],
                               aliases=am["aliases"], adamw_alpha=am["alpha"], sgdm_alpha=sm["alpha"],
                               adamw_fit_status=am["status"], sgdm_fit_status=sm["status"],
                               adamw_below_two=am["below_two"], sgdm_below_two=sm["below_two"],
                               adamw_step=a["step"], sgdm_step=s["step"])
                    valid = am["alpha"] is not None and sm["alpha"] is not None
                    row["alpha_adamw_minus_sgdm"] = am["alpha"] - sm["alpha"] if valid else None
                    row["below_two_adamw_minus_sgdm"] = int(am["below_two"]) - int(sm["below_two"]) if valid else None
                    for metric in PERFORMANCE:
                        row[f"adamw_{metric}"], row[f"sgdm_{metric}"] = a[metric], s[metric]
                        row[f"{metric}_adamw_minus_sgdm"] = (
                            a[metric] - s[metric] if a[metric] is not None and s[metric] is not None else None)
                    paired_layers.append(row)

        for metric in MODEL_METRICS:
            values = {seed: (a[metric], s[metric]) for seed, (a, s) in matched.items()
                      if a[metric] is not None and s[metric] is not None
                      and (metric != "valid_below_two_count" or (a["valid_fit_count"] and s["valid_fit_count"]))}
            statistics_rows.append(dict(checkpoint=checkpoint, scope="model", matrix=None,
                                        metric=metric, **paired_statistic(values, expected)))
        for name in MATRICES:
            rows = [r for r in paired_layers if r["checkpoint"] == checkpoint and r["matrix"] == name]
            for metric in ("alpha", "below_two"):
                valid_rows = [r for r in rows if r[f"adamw_{metric}"] is not None and r[f"sgdm_{metric}"] is not None]
                values = {r["seed"]: (r[f"adamw_{metric}"], r[f"sgdm_{metric}"]) for r in valid_rows}
                stat = dict(checkpoint=checkpoint, scope="matrix", matrix=name, metric=metric,
                            **paired_statistic(values, expected))
                # Same-pair held-out context: never substitute a different seed subset.
                for performance in ("test_loss_nats_per_character", "test_next_character_accuracy"):
                    key = f"{performance}_adamw_minus_sgdm"
                    available = [r for r in valid_rows if r[key] is not None]
                    stat[f"{key}_mean"] = statistics.fmean(r[key] for r in available) if available else None
                    stat[f"{key}_seeds"] = [r["seed"] for r in available]
                statistics_rows.append(stat)
    audit["pairs"] = pairs
    complete = (all(p["complete"] for p in pairs) and
                all(r["all_fits_valid"] and not r["issues"] for r in models) and
                all(e["status"] == "accepted" and not e["issues"] for e in audit["input_files"]))
    return dict(schema_version=1, contrast="AdamW minus SGDM", alpha_threshold=2.0,
                expected_matrices=list(MATRICES), uncertainty=UNCERTAINTY,
                complete=complete, audit=audit, models=models,
                paired_layers=paired_layers, paired_statistics=statistics_rows)


def csv_value(value):
    return json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (list, dict)) else value


def write_csv(path, rows, minimum_fields):
    fields = list(dict.fromkeys(list(minimum_fields) + [key for row in rows for key in row]))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: csv_value(value) for key, value in row.items()} for row in rows)


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_report(report):
    audit = report["audit"]
    lines = ["# Paired layer alpha comparison", "",
             f"Status: **{'complete' if report['complete'] else 'incomplete / audit required'}**. "
             f"Expected seeds: {', '.join(map(str, audit['expected_seeds']))} ({audit['seed_source']}).",
             "", "All contrasts are **AdamW minus SGDM**. Thresholds are strictly alpha < 2.", "",
             "The full-model mean and its below-two flag require all 18 unique matrix fits. "
             "Any-layer is yes if a valid below-two fit exists, no only with complete coverage, "
             "and unknown otherwise. Count and fraction describe valid fits only; fractions "
             "use each model's valid-fit denominator. Missing fits are never counted as above two.", "",
             "No test-performance cutoff is imposed and neither alpha nor test metrics select "
             "checkpoints. Held-out metrics are shown alongside spectra for interpretation. "
             "These are separately calibrated recipes, not an isolated causal optimizer effect.", "",
             UNCERTAINTY, "", "## Input and pair audit", "",
             "| Seed | Optimizer | Result status |", "| --- | --- | --- |"]
    for run in audit["runs"]:
        lines.append(f"| {run['seed']} | {run['optimizer']} | {run['status']} |")
    issues = list(audit["issues"])
    for entry in audit["input_files"]:
        issues.extend(f"{entry['path']}: {issue}" for issue in entry["issues"])
    for model in report["models"]:
        label = f"{model['optimizer']} seed {model['seed']} {model['checkpoint']}"
        issues.extend(f"{label}: {issue}" for issue in model["issues"])
        issues.extend(f"{label}: {row['matrix']} — {row['status']}"
                      for row in model["matrices"] if row["status"] != "valid")
    if issues:
        lines += ["", "Audit details:", ""] + [f"- {fmt(issue)}" for issue in issues]
    for checkpoint in CHECKPOINTS:
        pairs = [p for p in audit["pairs"] if p["checkpoint"] == checkpoint]
        n = sum(p["complete"] for p in pairs)
        lines += ["", f"## {checkpoint}", "", f"Completed result pairs: {n}/{len(pairs)}.", ""]
        for pair in pairs:
            if not pair["complete"]:
                lines.append(f"- Seed {pair['seed']}: missing {', '.join(pair['missing_optimizers'])} checkpoint.")
        lines += ["", "### Models and held-out performance", "",
                  "| Seed | Optimizer | Step | Valid/18 | Mean alpha | Mean <2 | Any <2 | Count <2 | Fraction <2 | Validation loss | Test loss | Test perplexity | Test accuracy |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        keys = ("seed", "optimizer", "step", "valid_fit_count", "mean_alpha", "mean_alpha_below_two",
                "any_layer_below_two", "valid_below_two_count", "valid_below_two_fraction",
                "validation_loss_nats_per_character", "test_loss_nats_per_character", "test_perplexity",
                "test_next_character_accuracy")
        for row in report["models"]:
            if row["checkpoint"] == checkpoint:
                lines.append("| " + " | ".join(fmt(row[k]) for k in keys) + " |")
        lines += ["", "### Model paired differences", "",
                  "| Metric | Seed pairs | AdamW mean | SGDM mean | Difference | 95% interval |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in report["paired_statistics"]:
            if row["checkpoint"] == checkpoint and row["scope"] == "model":
                lines.append(f"| {row['metric']} | {row['n_pairs']} | {fmt(row['adamw_mean'])} | "
                             f"{fmt(row['sgdm_mean'])} | {fmt(row['adamw_minus_sgdm_mean'])} | "
                             f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] |")
        lines += ["", "### Named matrices", "",
                  "Below-two difference is a difference in seed-level proportions. Test-loss "
                  "difference uses the same valid layer pairs when performance is available; "
                  "its seed list is explicit in JSON/CSV. See paired_layers.csv for individual "
                  "alphas, fit failures, checkpoint steps and held-out metrics.", "",
                  "| Matrix | Seed pairs | AdamW alpha | SGDM alpha | Alpha difference (95% interval) | AdamW <2 fraction | SGDM <2 fraction | Below-two difference (95% interval) | Test-loss difference |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        lookup = {(r["matrix"], r["metric"]): r for r in report["paired_statistics"]
                  if r["checkpoint"] == checkpoint and r["scope"] == "matrix"}
        for name in MATRICES:
            row, below = lookup[(name, "alpha")], lookup[(name, "below_two")]
            lines.append(f"| {name} | {row['n_pairs']} | {fmt(row['adamw_mean'])} | "
                         f"{fmt(row['sgdm_mean'])} | {fmt(row['adamw_minus_sgdm_mean'])} "
                         f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | "
                         f"{fmt(below['adamw_mean'])} | {fmt(below['sgdm_mean'])} | "
                         f"{fmt(below['adamw_minus_sgdm_mean'])} "
                         f"[{fmt(below['ci95_low'])}, {fmt(below['ci95_high'])}] | "
                         f"{fmt(row['test_loss_nats_per_character_adamw_minus_sgdm_mean'])} |")
    lines += ["", "## Provenance", "",
              "Input paths and SHA256 hashes, excluded results, failed fits, missing pairs and "
              "per-metric seed lists are retained in layer_comparison.json. Tied token/output "
              "weights count once; packed QKV is one matrix. No checkpoint tensors are read.", ""]
    lines += [f"- Test disclosure: {fmt(value)}" for value in audit["test_disclosures"]]
    return "\n".join(lines) + "\n"


def write_report(report, output):
    output = Path(output).resolve()
    study = Path(report["audit"]["study"])
    # Keep every emitted artifact outside the inputs and frozen protocol records.
    if output == study or output == study / "confirmation" or study / "confirmation" in output.parents:
        raise ValueError("Use a separate report directory, outside confirmation/ and not the study root")
    output.mkdir(parents=True, exist_ok=True)
    (output / "layer_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(output / "models.csv", [{k: v for k, v in row.items() if k != "matrices"}
                                      for row in report["models"]], ("checkpoint", "seed", "optimizer"))
    write_csv(output / "paired_layers.csv", report["paired_layers"], ("checkpoint", "seed", "matrix"))
    write_csv(output / "paired_statistics.csv", report["paired_statistics"], ("checkpoint", "scope", "matrix", "metric"))
    (output / "report.md").write_text(markdown_report(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="Study root containing confirmation/*/result.json")
    parser.add_argument("--output", type=Path, required=True, help="Separate directory for JSON, CSV and Markdown reports")
    parser.add_argument("--seeds", nargs="+", type=int, help="Expected paired seeds; default: study manifest, else frozen five")
    parser.add_argument("--require-complete", action="store_true", help="Write the audit, then exit 1 if any results/fits/metrics are missing or invalid")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.study, args.seeds)
        write_report(report, args.output)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(output=str(args.output.resolve()), complete=report["complete"],
                          accepted_checkpoint_records=len(report["models"]),
                          completed_pairs={c: sum(p["complete"] for p in report["audit"]["pairs"] if p["checkpoint"] == c)
                                           for c in CHECKPOINTS})))
    return 1 if args.require_complete and not report["complete"] else 0


if __name__ == "__main__":
    sys.exit(main())

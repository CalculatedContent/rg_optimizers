"""Summarize the five paired TinyShakespeare seeds using only the standard library.

Losses are in nats per character. A seed's alpha requires one valid fit for each
of the 18 unique matrices. Each contributing seed has equal weight in arm means.
Paired effects are graft minus baseline; intervals are two-sided 95% Student t
intervals on seed differences. Fewer than five pairs are marked as partial.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
from tempfile import NamedTemporaryFile

ARMS = ("adamw", "sgdm", "graft")
SEEDS = tuple(range(1340, 1345))
CHECKPOINTS = ("endpoint", "validation_selected")
SPLITS = ("train", "validation", "test")
ALPHA = "confidence_tail_weighted_alpha"
LOSS = "loss_nats_per_character"
METRICS = tuple(f"{s}_{m}" for s in SPLITS for m in (LOSS, "next_character_accuracy")) + (ALPHA,)
T95 = {1: 12.70620474, 2: 4.30265273, 3: 3.18244631, 4: 2.77644511}
MATRIX_COUNT = 18


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def confidence_weight(fit):
    a, d = fit.get("alpha"), fit.get("D")
    tail, size = fit.get("num_pl_spikes"), fit.get("num_evals")
    if (fit.get("status") == "success" and finite(a) and a > 0 and finite(d) and 0 <= d <= 1
            and type(tail) is int and type(size) is int and 0 < tail <= size):
        return math.sqrt(tail) / (d + 0.02)
    return None


def extract_spectrum(spectrum):
    matrices = (spectrum or {}).get("matrices", [])
    rows, weights, terms, names = [], [], [], set()
    for matrix in matrices:
        name = matrix["matrix"]
        fits = matrix.get("fits") or []
        for index, fit in enumerate(fits or [{}], 1):
            weight = confidence_weight(fit)
            row = {"matrix": name, "aliases": matrix.get("aliases"), "shape": matrix.get("shape"),
                   "fit_index": index if fits else None, "confidence_weight": weight}
            row.update({f"fit_{key}": value for key, value in fit.items()})
            rows.append(row)
            if weight is not None and len(fits) == 1 and name not in names:
                weights.append(weight)
                terms.append(fit["alpha"] * weight)
        names.add(name)
    complete = len(matrices) == len(names) == len(weights) == MATRIX_COUNT
    summary = {"unique_matrices": len(names), "valid_matrix_fits": len(weights),
               ALPHA: math.fsum(terms) / math.fsum(weights) if complete else None}
    return summary, rows


def harvest(result):
    # A selected or final state can also appear in history. Share rows only
    # when both the optimizer step and raw matrix records agree.
    spectra, summaries = {}, {}
    for role, states in (("history", result.get("history", [])),
                         *((role, [result[role]]) for role in CHECKPOINTS)):
        for state in states:
            spectrum = state.get("alpha")
            if not spectrum:
                continue
            step = state["step"]
            key = (step, json.dumps(spectrum.get("matrices", []), sort_keys=True))
            if key not in spectra:
                summary, rows = extract_spectrum(spectrum)
                spectra[key] = ([], summary, rows)
            roles, summary, _ = spectra[key]
            if role not in roles:
                roles.append(role)
            if role in CHECKPOINTS:
                summaries[role] = summary
    matrix_rows = [dict(step=step, checkpoints="|".join(roles), **row)
                   for (step, _), (roles, _, rows) in spectra.items() for row in rows]
    seed_rows = []
    for role in CHECKPOINTS:
        state = result[role]
        values = {}
        for split in SPLITS:
            for key in (LOSS, "next_character_accuracy"):
                value = state["metrics"][split][key]
                if not finite(value):
                    raise ValueError(f"{role} {split} has nonfinite {key}")
                values[f"{split}_{key}"] = value
        seed_rows.append(dict(checkpoint=role, step=state["step"], **values,
                              **summaries.get(role, {"unique_matrices": 0, "valid_matrix_fits": 0, ALPHA: None})))
    return seed_rows, matrix_rows


def read_run(directory, arm, seed):
    result_file = directory / "result.json"
    failure_file = directory / "numerical_failure.json"
    progress_file = directory / "progress.json"
    status, detail = "missing", ""
    try:
        if result_file.exists():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if result["arm"] != arm or result["seed"] != seed or result.get("completed") is False:
                raise ValueError("result identity or completion flag disagrees with expected run")
            seeds, matrices = harvest(result)
            return "complete", "", seeds, matrices
        if failure_file.exists():
            failure = json.loads(failure_file.read_text(encoding="utf-8"))
            status = "failed"
            detail = f"step {failure.get('step', '?')}: {failure.get('error', 'numerical failure')}"
        elif progress_file.exists():
            progress = json.loads(progress_file.read_text(encoding="utf-8"))
            status, detail = "in_progress", f"last saved step {progress['step']}"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        status, detail = "invalid", str(error)
    return status, detail, [{"checkpoint": role} for role in CHECKPOINTS], []


def summarize(seed_rows):
    lookup = {(r["arm"], r["seed"], r["checkpoint"]): r for r in seed_rows}
    means, pairs = [], []
    for role in CHECKPOINTS:
        for metric in METRICS:
            for arm in ARMS:
                values = [lookup[arm, s, role].get(metric) for s in SEEDS]
                observed = [v for v in values if finite(v)]
                means.append(dict(arm=arm, checkpoint=role, metric=metric, n=len(observed),
                                  expected_n=len(SEEDS), mean=fmean(observed) if observed else None))
            for baseline in ("adamw", "sgdm"):
                base = {"checkpoint": role, "baseline": baseline, "metric": metric}
                differences = []
                for seed in SEEDS:
                    a, b = lookup["graft", seed, role].get(metric), lookup[baseline, seed, role].get(metric)
                    valid = finite(a) and finite(b)
                    delta = a - b if valid else None
                    if valid:
                        differences.append(delta)
                    pairs.append(dict(**base, level="seed", seed=seed, status="paired" if valid else "unavailable",
                                      n=int(valid), expected_n=1, graft=a, baseline_value=b, graft_minus_baseline=delta))
                n = len(differences)
                mean = fmean(differences) if n else None
                radius = T95[n - 1] * stdev(differences) / math.sqrt(n) if n >= 2 else None
                coverage = "complete" if n == len(SEEDS) else "partial" if n else "unavailable"
                pairs.append(dict(**base, level="mean", status=coverage, n=n, expected_n=len(SEEDS),
                                  graft_minus_baseline=mean, ci95_low=mean - radius if radius is not None else None,
                                  ci95_high=mean + radius if radius is not None else None))
    return means, pairs


def write_csv(path, columns, rows):
    # Include raw fields even when they occur in just one WeightWatcher fit.
    extra = sorted({k for row in rows for k in row}.difference(columns))
    with NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=[*columns, *extra])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                                 for k, v in row.items()})
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def build_report(output):
    output = Path(output)
    if not output.is_dir():
        raise ValueError(f"Output directory does not exist: {output}")
    seed_rows, matrix_rows = [], []
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = output / arm / f"seed_{seed:04d}"
            status, detail, states, matrices = read_run(run_dir, arm, seed)
            seed_rows.extend(dict(arm=arm, seed=seed, status=status, detail=detail, **r) for r in states)
            matrix_rows.extend(dict(arm=arm, seed=seed, **r) for r in matrices)
    means, pairs = summarize(seed_rows)
    write_csv(output / "per_seed.csv", ["arm", "seed", "status", "detail", "checkpoint", "step",
                                        *METRICS, "unique_matrices", "valid_matrix_fits"], seed_rows)
    write_csv(output / "per_matrix.csv", ["arm", "seed", "step", "checkpoints", "matrix", "aliases",
                                          "shape", "fit_index", "fit_status", "fit_alpha", "fit_D",
                                          "fit_num_pl_spikes", "fit_num_evals", "confidence_weight"], matrix_rows)
    write_csv(output / "arm_means.csv", ["arm", "checkpoint", "metric", "n", "expected_n", "mean"], means)
    write_csv(output / "paired_effects.csv", ["checkpoint", "baseline", "metric", "level", "seed", "status", "n",
                                             "expected_n", "graft", "baseline_value", "graft_minus_baseline",
                                             "ci95_low", "ci95_high"], pairs)
    return seed_rows, means, pairs


def display(seed_rows, means):
    lookup = {(r["arm"], r["checkpoint"], r["metric"]): r for r in means}
    for arm in ARMS:
        states = [r for r in seed_rows if r["arm"] == arm and r["checkpoint"] == "endpoint"]
        complete = sum(r["status"] == "complete" for r in states)
        print(f"{arm}: {complete}/{len(SEEDS)} completed")
        for r in states:
            if r["status"] != "complete":
                print(f"  seed {r['seed']} {r['status']} {r['detail']}".rstrip())
        for role in CHECKPOINTS:
            shown = []
            for name, metric in (("val loss", "validation_" + LOSS), ("test loss", "test_" + LOSS), ("alpha", ALPHA)):
                row = lookup[arm, role, metric]
                value = f"{row['mean']:.6f}" if row["n"] else "unavailable"
                shown.append(f"{name} {value} (n={row['n']})")
            print(f"  {role}: " + ", ".join(shown))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="directory containing the three arm directories")
    args = parser.parse_args()
    if not args.output.is_dir():
        parser.error(f"Output directory does not exist: {args.output}")
    rows, arm_means, _ = build_report(args.output)
    display(rows, arm_means)

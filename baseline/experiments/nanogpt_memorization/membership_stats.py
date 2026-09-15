#!/usr/bin/env python3
"""Quantify example-specific likelihood memorization from a checkpoint audit.

This is post-hoc, read-only statistics over teacher_scores_*.jsonl written by
``audit_checkpoints.py``. It does not load model weights or perform training.
The primary test compares a fixed cohort of early training examples with fixed
fresh examples and corrects each example for its own step-0 NLL.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

STEP_RE = re.compile(r"teacher_scores_(\d{8})\.jsonl$")


def auc(labels, scores):
    """ROC AUC with average ranks for ties; positive label is 1."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels/scores must be equal-length vectors")
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite score")
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def percentile_ci(values, level=0.95):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("invalid bootstrap values")
    alpha = (1.0 - level) / 2.0
    return tuple(float(x) for x in np.quantile(values, [alpha, 1.0 - alpha]))


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if len(pvalues) == 0:
        return np.array([], dtype=float)
    if np.any((pvalues < 0) | (pvalues > 1) | ~np.isfinite(pvalues)):
        raise ValueError("invalid p-value")
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def read_jsonl(path):
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL {path}:{i}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Invalid row {path}:{i}")
        rows.append(row)
    return rows


def load_audit(audit_dir):
    audit_dir = Path(audit_dir).expanduser().resolve()
    complete = audit_dir / "complete.json"
    protocol = audit_dir / "protocol.json"
    if not complete.is_file() or not protocol.is_file():
        raise ValueError("audit must contain complete.json and protocol.json")
    done = json.loads(complete.read_text())
    if done.get("status") != "complete" or done.get("new_training_updates") != 0:
        raise ValueError("only a completed read-only checkpoint audit is accepted")
    files = []
    for path in audit_dir.glob("teacher_scores_*.jsonl"):
        match = STEP_RE.fullmatch(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort()
    if len(files) < 2 or files[0][0] != 0:
        raise ValueError("step-0 and at least one later teacher-score file are required")
    return audit_dir, json.loads(protocol.read_text()), files


def background_rows(path):
    rows = [r for r in read_jsonl(path) if r.get("category") == "background"]
    for r in rows:
        if "eid" not in r or "variant" not in r or "nll" not in r:
            raise ValueError(f"missing required background fields in {path}")
        value = float(r["nll"])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite NLL in {path}")
    return rows


def by_variant(rows, variant):
    selected = [r for r in rows if r["variant"] == variant]
    result = {r["eid"]: r for r in selected}
    if len(result) != len(selected):
        raise ValueError(f"duplicate eid in {variant}")
    return result


def bootstrap_stats(seen_imp, fresh_imp, seen_raw, fresh_raw, reps, rng):
    effects, auc_corr, auc_raw = [], [], []
    ns, nf = len(seen_imp), len(fresh_imp)
    for _ in range(reps):
        si = rng.integers(0, ns, ns)
        fi = rng.integers(0, nf, nf)
        s_imp, f_imp = seen_imp[si], fresh_imp[fi]
        s_raw, f_raw = seen_raw[si], fresh_raw[fi]
        effects.append(float(s_imp.mean() - f_imp.mean()))
        labels = np.r_[np.ones(ns, dtype=np.int8), np.zeros(nf, dtype=np.int8)]
        auc_corr.append(auc(labels, np.r_[s_imp, f_imp]))
        auc_raw.append(auc(labels, np.r_[s_raw, f_raw]))
    return np.asarray(effects), np.asarray(auc_corr), np.asarray(auc_raw)


def permutation_stats(seen_imp, fresh_imp, seen_raw, fresh_raw, reps, rng):
    n1, n0 = len(seen_imp), len(fresh_imp)
    labels = np.r_[np.ones(n1, dtype=np.int8), np.zeros(n0, dtype=np.int8)]
    imp = np.r_[seen_imp, fresh_imp]
    raw = np.r_[seen_raw, fresh_raw]
    obs_effect = float(seen_imp.mean() - fresh_imp.mean())
    obs_auc_corr = auc(labels, imp)
    obs_auc_raw = auc(labels, raw)
    null_effect = np.empty(reps)
    null_auc_corr = np.empty(reps)
    null_auc_raw = np.empty(reps)
    for i in range(reps):
        perm = rng.permutation(labels)
        s = perm == 1
        null_effect[i] = imp[s].mean() - imp[~s].mean()
        null_auc_corr[i] = auc(perm, imp)
        null_auc_raw[i] = auc(perm, raw)
    p_effect = (1 + np.count_nonzero(null_effect >= obs_effect)) / (reps + 1)
    p_auc_corr = (1 + np.count_nonzero(null_auc_corr >= obs_auc_corr)) / (reps + 1)
    p_auc_raw = (1 + np.count_nonzero(null_auc_raw >= obs_auc_raw)) / (reps + 1)
    return float(p_effect), float(p_auc_corr), float(p_auc_raw)


def analyze(audit_dir, *, bootstrap=2000, permutations=10000, seed=20260915, output=None):
    if bootstrap < 100 or permutations < 100:
        raise ValueError("use at least 100 bootstrap/permutation replicates")
    audit_dir, protocol, files = load_audit(audit_dir)
    out = Path(output).expanduser().resolve() if output else audit_dir / "membership"
    out.mkdir(parents=True, exist_ok=False)

    step0 = background_rows(files[0][1])
    seen0 = by_variant(step0, "early_not_yet_seen")
    fresh0 = by_variant(step0, "fresh")
    if len(seen0) < 2 or len(fresh0) < 2:
        raise ValueError("step-0 fixed cohorts are missing")

    rng = np.random.default_rng(seed)
    rows = []
    trajectories_seen = []
    trajectories_fresh = []
    for step, path in files[1:]:
        current = background_rows(path)
        seen = by_variant(current, "seen_early")
        fresh = by_variant(current, "fresh")
        if set(seen) != set(seen0) or set(fresh) != set(fresh0):
            raise ValueError(f"fixed cohort identity changed at step {step}")
        seen_ids = sorted(seen)
        fresh_ids = sorted(fresh)
        seen_base = np.asarray([float(seen0[e]["nll"]) for e in seen_ids])
        fresh_base = np.asarray([float(fresh0[e]["nll"]) for e in fresh_ids])
        seen_now = np.asarray([float(seen[e]["nll"]) for e in seen_ids])
        fresh_now = np.asarray([float(fresh[e]["nll"]) for e in fresh_ids])
        seen_imp = seen_base - seen_now
        fresh_imp = fresh_base - fresh_now
        seen_raw_score = -seen_now
        fresh_raw_score = -fresh_now
        labels = np.r_[np.ones(len(seen_imp), dtype=np.int8), np.zeros(len(fresh_imp), dtype=np.int8)]
        corrected_score = np.r_[seen_imp, fresh_imp]
        raw_score = np.r_[seen_raw_score, fresh_raw_score]
        effect = float(seen_imp.mean() - fresh_imp.mean())
        corrected_auc = auc(labels, corrected_score)
        raw_auc = auc(labels, raw_score)

        boot_effect, boot_auc_corr, boot_auc_raw = bootstrap_stats(
            seen_imp, fresh_imp, seen_raw_score, fresh_raw_score, bootstrap, rng)
        p_effect, p_auc_corr, p_auc_raw = permutation_stats(
            seen_imp, fresh_imp, seen_raw_score, fresh_raw_score, permutations, rng)
        e_lo, e_hi = percentile_ci(boot_effect)
        a_lo, a_hi = percentile_ci(boot_auc_corr)
        r_lo, r_hi = percentile_ci(boot_auc_raw)
        rows.append({
            "step": step,
            "n_seen": len(seen_imp),
            "n_fresh": len(fresh_imp),
            "raw_fresh_minus_seen_nll": float(fresh_now.mean() - seen_now.mean()),
            "baseline_seen_minus_fresh_nll": float(seen_base.mean() - fresh_base.mean()),
            "baseline_corrected_membership_effect": effect,
            "effect_ci95_low": e_lo,
            "effect_ci95_high": e_hi,
            "auc_baseline_corrected": corrected_auc,
            "auc_corrected_ci95_low": a_lo,
            "auc_corrected_ci95_high": a_hi,
            "auc_raw_nll": raw_auc,
            "auc_raw_ci95_low": r_lo,
            "auc_raw_ci95_high": r_hi,
            "p_perm_effect_one_sided": p_effect,
            "p_perm_auc_corrected_one_sided": p_auc_corr,
            "p_perm_auc_raw_one_sided": p_auc_raw,
        })
        trajectories_seen.append(seen_imp)
        trajectories_fresh.append(fresh_imp)

    table = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    table["p_holm_effect"] = holm_adjust(table["p_perm_effect_one_sided"].to_numpy())
    table["p_holm_auc_corrected"] = holm_adjust(table["p_perm_auc_corrected_one_sided"].to_numpy())
    table["p_holm_auc_raw"] = holm_adjust(table["p_perm_auc_raw_one_sided"].to_numpy())

    # Global max-statistic correction for searching across saved checkpoints.
    seen_matrix = np.stack(trajectories_seen, axis=1)
    fresh_matrix = np.stack(trajectories_fresh, axis=1)
    all_matrix = np.vstack([seen_matrix, fresh_matrix])
    n1, n0 = len(seen_matrix), len(fresh_matrix)
    total = n1 + n0
    observed_effects = seen_matrix.mean(0) - fresh_matrix.mean(0)
    observed_aucs = table["auc_baseline_corrected"].to_numpy()
    ranks = np.empty_like(all_matrix, dtype=float)
    for col in range(all_matrix.shape[1]):
        values = all_matrix[:, col]
        order = np.argsort(values, kind="mergesort")
        i = 0
        while i < total:
            j = i + 1
            while j < total and values[order[j]] == values[order[i]]:
                j += 1
            ranks[order[i:j], col] = (i + 1 + j) / 2.0
            i = j
    max_effect_null = np.empty(permutations)
    max_auc_null = np.empty(permutations)
    total_sums = all_matrix.sum(0)
    for i in range(permutations):
        idx = rng.choice(total, n1, replace=False)
        member_sum = all_matrix[idx].sum(0)
        effect_null = member_sum / n1 - (total_sums - member_sum) / n0
        rank_sum = ranks[idx].sum(0)
        auc_null = (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        max_effect_null[i] = effect_null.max()
        max_auc_null[i] = auc_null.max()
    max_effect = float(observed_effects.max())
    max_auc = float(observed_aucs.max())
    p_global_effect = float((1 + np.count_nonzero(max_effect_null >= max_effect)) / (permutations + 1))
    p_global_auc = float((1 + np.count_nonzero(max_auc_null >= max_auc)) / (permutations + 1))

    table.to_csv(out / "membership_stats.csv", index=False)
    meta = {
        "source_audit": str(audit_dir),
        "source_run": protocol.get("source_run"),
        "audit_seed": protocol.get("audit_seed"),
        "background_examples": protocol.get("background_examples"),
        "bootstrap_replicates": bootstrap,
        "permutation_replicates": permutations,
        "statistics_seed": seed,
        "primary_score": "per-example NLL improvement from step 0",
        "primary_effect": "mean improvement(seen_early) - mean improvement(fresh)",
        "global_max_effect": max_effect,
        "global_max_effect_step": int(table.iloc[int(np.argmax(observed_effects))]["step"]),
        "global_max_effect_p": p_global_effect,
        "global_max_auc": max_auc,
        "global_max_auc_step": int(table.iloc[int(np.argmax(observed_aucs))]["step"]),
        "global_max_auc_p": p_global_auc,
    }
    (out / "membership_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    final = table.iloc[-1]
    best_effect = table.iloc[int(np.argmax(table["baseline_corrected_membership_effect"].to_numpy()))]
    best_auc = table.iloc[int(np.argmax(table["auc_baseline_corrected"].to_numpy()))]
    lines = [
        "# Baseline-corrected membership-likelihood analysis",
        "",
        "This is a post-hoc, single-model analysis of a completed read-only checkpoint audit. It performs no training and does not treat checkpoints or examples as independent model replications.",
        "",
        "## Primary design",
        "",
        "The same fixed early-training examples and the same fixed fresh examples are scored at initialization and at every saved checkpoint. Each example is converted to an NLL improvement from its own step-0 baseline. Positive membership effect means seen examples improved more than fresh examples after baseline difficulty is removed.",
        "",
        f"Fixed cohort sizes: {len(seen0)} seen, {len(fresh0)} fresh. Bootstrap replicates: {bootstrap}. Permutations: {permutations}.",
        "",
        "## Final checkpoint",
        "",
        f"- step: {int(final.step)}",
        f"- baseline-corrected membership effect: {final.baseline_corrected_membership_effect:.6f} nat/token (95% bootstrap CI {final.effect_ci95_low:.6f}, {final.effect_ci95_high:.6f})",
        f"- baseline-corrected membership ROC-AUC: {final.auc_baseline_corrected:.4f} (95% bootstrap CI {final.auc_corrected_ci95_low:.4f}, {final.auc_corrected_ci95_high:.4f})",
        f"- one-sided permutation p(effect): {final.p_perm_effect_one_sided:.6g}; Holm-adjusted across checkpoints: {final.p_holm_effect:.6g}",
        f"- one-sided permutation p(AUC): {final.p_perm_auc_corrected_one_sided:.6g}; Holm-adjusted across checkpoints: {final.p_holm_auc_corrected:.6g}",
        "",
        "## Strongest observed checkpoint (post-hoc)",
        "",
        f"- largest corrected effect: step {int(best_effect.step)}, {best_effect.baseline_corrected_membership_effect:.6f} nat/token",
        f"- largest corrected AUC: step {int(best_auc.step)}, AUC {best_auc.auc_baseline_corrected:.4f}",
        f"- global max-statistic permutation p(effect across all checkpoints): {p_global_effect:.6g}",
        f"- global max-statistic permutation p(AUC across all checkpoints): {p_global_auc:.6g}",
        "",
        "## Interpretation boundary",
        "",
        "A positive corrected effect or AUC above 0.5 supports example-specific likelihood memory in this trained model because it compares the same examples to their own initialization scores. It is not extractive recall and does not by itself establish the result across training seeds or architectures. Bootstrap intervals quantify cohort sampling variability within this model; they are not training-run confidence intervals. Independent training seeds are still required for a general claim.",
        "",
        "See `membership_stats.csv` for every saved checkpoint.",
    ]
    (out / "membership_report.md").write_text("\n".join(lines) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, help="Completed audit_checkpoints.py output directory")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", help="New output directory; default is <audit>/membership")
    args = parser.parse_args()
    try:
        out = analyze(args.audit, bootstrap=args.bootstrap, permutations=args.permutations,
                      seed=args.seed, output=args.output)
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Finished. Read: {out / 'membership_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

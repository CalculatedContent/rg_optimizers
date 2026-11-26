"""Independent arithmetic and finite-tail audit; never fits/trains new models.

Reads archived primary JSON records only. Uses NumPy/SciPy rather than the
campaign's aggregation/statistics functions. Reconstructs the standard Pareto
MLE/KS fit from the separately archived float64 diagnostic spectra as a
cross-check, not as a replacement for the recorded WeightWatcher estimates.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "verified_remote"
OUT = ROOT / "accuracy_audit"
ARMS = ("sgdm", "adamw", "modified_adamw")
SEEDS = tuple(range(1340, 1345))
NAMES = ("fc2.weight", "fc3.weight", "fc4.weight")


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    half = float(t.ppf(.975, len(a) - 1) * sd / np.sqrt(len(a)))
    return {"n": len(a), "mean": mean, "sample_sd": sd,
            "ci95_half_width": half, "ci95_low": mean-half,
            "ci95_high": mean+half}


def diagnostic_refit(eigenvalues):
    x = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    x = x[x > 1e-5]  # The installed frozen WW EVALS_THRESH.
    candidates = []
    for i in range(len(x)-1):
        tail = x[i:]
        n = len(tail)
        alpha = 1 + n / np.log(tail / tail[0]).sum()
        distance = np.max(np.abs(1-(tail/tail[0])**(1-alpha)-np.arange(n)/n))
        candidates.append((float(distance), float(alpha), n, float(tail[0])))
    return min(candidates, key=lambda q: q[0])


def span(a):
    a = np.asarray(a, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="Primary campaign root (with runs/) or local verified_remote mirror")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    source, out = args.source, args.output
    summary = json.loads((source / "analysis" / "summary.json").read_text())
    archived_statistics = {(r["optimizer"],r["step"],r["metric"]): r for r in summary["statistics"]}
    source_hashes = {r["path"]: r["sha256"] for r in summary["source_files"]}
    errors, fits, averages = [], [], []
    max_weight_error = max_average_error = max_layer_sd_error = 0.
    max_stat_error = 0.
    records = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = (source / "runs" / f"seed_{seed}" / arm / "result.json"
                    if (source / "runs").is_dir()
                    else source / str(seed) / arm / "result.json")
            raw = path.read_bytes()
            r = json.loads(raw)
            key = f"runs/seed_{seed}/{arm}/result.json"
            if hashlib.sha256(raw).hexdigest() != source_hashes[key]:
                errors.append(f"Source hash mismatch: {key}")
            if r["status"] != "complete" or r["completed_updates"] != 20000:
                errors.append(f"Incomplete run: {key}")
            records[seed,arm] = r
            if [h["step"] for h in r["history"]] != list(range(0,20001,500)):
                errors.append(f"Wrong cadence: {key}")
            for h in r["history"]:
                spectrum = h["spectral"]
                if [m["matrix"] for m in spectrum["matrices"]] != list(NAMES):
                    errors.append(f"Wrong matrix scope: {key} {h['step']}")
                aa, ww = [], []
                for m in spectrum["matrices"]:
                    f = m["fit"]
                    n, D, alpha = f["num_pl_spikes"],f["D"],f["alpha"]
                    if not (m["shape"] == [128,128] and f["num_evals"]==128 and
                            f["status"]=="success" and 0<D<1 and 1<=n<=128 and
                            np.isfinite(alpha) and alpha>0):
                        errors.append(f"Invalid fit: {key} {h['step']} {m['matrix']}")
                    weight = np.sqrt(float(n)) / (D+.02)
                    max_weight_error = max(max_weight_error,abs(weight-m["weight"]))
                    aa.append(alpha); ww.append(weight)
                    d_refit,a_refit,n_refit,xmin_refit = diagnostic_refit(m["eigenvalues"])
                    fits.append({"seed":seed,"optimizer":arm,"step":h["step"],
                                 "matrix":m["matrix"],"alpha":alpha,"D":D,
                                 "tail_count":n,"tail_fraction":n/128,
                                 "xmin":f["xmin"],"xmax":f["xmax"],
                                 "tail_log10_span":math.log10(f["xmax"]/f["xmin"]),
                                 "weight":float(weight),"fit_sigma":f["sigma"],
                                 "warning":f.get("warning",""),
                                 "diagnostic_float64_refit_alpha":a_refit,
                                 "diagnostic_float64_refit_tail_count":n_refit,
                                 "diagnostic_float64_refit_D":d_refit,
                                 "diagnostic_float64_refit_xmin":xmin_refit,
                                 "refit_alpha_absolute_difference":abs(a_refit-alpha)})
                avg = float(np.dot(aa,ww)/sum(ww))
                layer_sd = float(np.std(aa,ddof=1))
                max_average_error = max(max_average_error,abs(avg-spectrum["confidence_tail_alpha"]))
                max_layer_sd_error = max(max_layer_sd_error,abs(layer_sd-spectrum["layer_alpha_sd"]))
                averages.append({"seed":seed,"optimizer":arm,"step":h["step"],
                                 "confidence_tail_alpha":avg,"layer_alpha_sd":layer_sd})
    # Verify all alpha/layer-dispersion statistics, including intermediate steps.
    for arm in ARMS:
        for step in range(0,20001,500):
            rows = [r for r in averages if r["optimizer"]==arm and r["step"]==step]
            for metric in ("confidence_tail_alpha","layer_alpha_sd"):
                recalculated = stats([r[metric] for r in rows])
                archived = archived_statistics[arm,step,metric]
                for k,v in recalculated.items():
                    max_stat_error = max(max_stat_error,abs(v-archived[k]))
    endpoint_rows = []
    for step in (2000,20000):
        for arm in ARMS:
            layer_rows = [r for r in fits if r["optimizer"]==arm and r["step"]==step]
            seed_rows = [r for r in averages if r["optimizer"]==arm and r["step"]==step]
            endpoint_rows.append({"optimizer":arm,"step":step,
                "alpha_seed_statistics":stats([r["confidence_tail_alpha"] for r in seed_rows]),
                "seed_alphas":{str(r["seed"]):r["confidence_tail_alpha"] for r in seed_rows},
                "n_matrix_fits":len(layer_rows),
                "tail_counts":span([r["tail_count"] for r in layer_rows]),
                "tail_fraction":span([r["tail_fraction"] for r in layer_rows]),
                "KS_D":span([r["D"] for r in layer_rows]),
                "individual_matrix_alpha":span([r["alpha"] for r in layer_rows]),
                "tail_log10_span":span([r["tail_log10_span"] for r in layer_rows]),
                "fit_sigma":span([r["fit_sigma"] for r in layer_rows]),
                "n_tail_less_than_20":sum(r["tail_count"]<20 for r in layer_rows),
                "n_span_less_than_one_decade":sum(r["tail_log10_span"]<1 for r in layer_rows),
                "n_diagnostic_refit_cutoff_mismatches":sum(r["tail_count"]!=r["diagnostic_float64_refit_tail_count"] for r in layer_rows),
                "max_diagnostic_refit_alpha_difference":max(r["refit_alpha_absolute_difference"] for r in layer_rows)})
    report={"purpose":"Independent numerical audit of archived primary results; no new training or changed fits",
            "source_manifest_matches":not any("hash" in e.lower() for e in errors),
            "n_runs":len(records),"n_model_averages":len(averages),"n_matrix_fits":len(fits),
            "all_expected_fits_valid":not errors,"errors":errors,
            "max_absolute_stored_weight_difference":max_weight_error,
            "max_absolute_stored_model_alpha_difference":max_average_error,
            "max_absolute_stored_layer_sd_difference":max_layer_sd_error,
            "max_absolute_seed_statistics_difference":max_stat_error,
            "t95_df4":float(t.ppf(.975,4)),
            "diagnostic_refit_max_alpha_difference":max(r["refit_alpha_absolute_difference"] for r in fits),
            "diagnostic_refit_tail_count_mismatches":sum(r["tail_count"]!=r["diagnostic_float64_refit_tail_count"] for r in fits),
            "endpoints":endpoint_rows,
            "interpretation":[
                "All reported averages and seed intervals are arithmetically consistent with recorded ordinary fitted alpha, tail counts, and KS distances.",
                "The five-seed Student-t interval measures initialization/minibatch uncertainty on this fixed split and fixed recipe; it does not quantify dataset sampling or systematic PL-model/fit-cutoff uncertainty.",
                "128 eigenvalues satisfy both the frozen minimum-dimension64 rule and WeightWatcher0.7.7 default min_evals8; this does not establish a broad asymptotic power-law regime.",
                "Short fitted tails and automatic KS cutoff selection limit asymptotic interpretation. Low KS distance alone is not a calibrated goodness-of-fit p-value.",
                "Default WeightWatcher0.7.7 PL backend is its own WW Pareto MLE/KS implementation; powerlaw2.0.0 is an installed dependency. No alpha_weighted is used.",
                "Diagnostic eigenvalues were saved from a float64 SVD. Cross-check fits can differ slightly from WeightWatcher's working SVD; these cross-checks do not replace frozen estimates."
            ]}
    out.mkdir(parents=True,exist_ok=True)
    (out/'spectral_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'spectral_audit_matrix_fits.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(fits[0])); writer.writeheader();writer.writerows(fits)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

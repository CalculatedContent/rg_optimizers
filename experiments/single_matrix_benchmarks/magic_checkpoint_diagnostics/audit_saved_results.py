"""Audit saved MAGIC diagnostic records. No neural framework or inference is used.

Default inputs are the sibling immutable MAGIC archive and this folder's results.
The executed diagnose.py verifies checkpoint bytes; this script independently
checks all saved hash associations, metrics, tables, and the original failure.
"""
import argparse
import csv
import json
import math
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
SOURCE_ARCHIVE = "09dbaa4e095ff21fc8b1e130798d247b75c15913"
RESULTS_COMMIT = "a7262f96f1a64ffffcc45b921884bc4284fd9b7f"
STEPS = [0, 2000, 5000, 10000, 15000, 20000]
CASES = [f"{prefix}{i}" for prefix in "sa" for i in range(6)]
SEEDS = [3470, 3471]
# First failing checkpoint in the permanent partial summary at db31d29016bd.
ORIGINAL_FAILURES = {
    "s1-3471": 20000, "s2-3470": 15000, "s2-3471": 10000,
    "s3-3470": 5000, "s3-3471": 5000, "s4-3470": 15000,
    "s4-3471": 15000, "s5-3470": 10000, "s5-3471": 10000,
    "a1-3470": 10000, "a1-3471": 10000, "a2-3470": 5000,
    "a2-3471": 5000, "a3-3471": 20000, "a4-3470": 10000,
    "a4-3471": 10000, "a5-3470": 5000, "a5-3471": 5000,
}

def read_json(path):
    return json.loads(path.read_text())

def flatten(run, h):
    out = dict(case=run["case"], seed=run["seed"], track=run["config"]["track"],
               lr=run["config"]["lr"], momentum=run["config"].get("momentum"),
               step=h["step"], weights_sha256=h["weights_sha256"])
    for name in ("spectral", "initialization", "train", "validation"):
        for k, v in h[name].items():
            if v is None or isinstance(v, (str, int, float)):
                out[f"{name}_{k}"] = v
    if "delta_ww_fit" in h["initialization"]:
        for k in ("alpha", "D", "tail", "status"):
            out[f"delta_ww_{k}"] = h["initialization"]["delta_ww_fit"].get(k)
    for name in ("gradient_geometry", "hypothetical_next_update"):
        for layer, values in h.get(name, {}).get("layers", {}).items():
            for k, v in values.items():
                out[f"{name}_{layer}_{k}"] = v
    for name in ("truncation", "scale_interventions"):
        for sub, values in h.get(name, {}).items():
            for k, v in values.items():
                if v is None or isinstance(v, (int, float, str)):
                    out[f"{name}_{sub}_{k}"] = v
    return out

def key(row):
    return row["case"], int(row["seed"]), int(row["step"])

def check_csv(path, expected):
    actual = list(csv.DictReader(path.open()))
    lookup = {key(r): r for r in actual}
    assert len(lookup) == len(actual) == len(expected)
    for row in expected:
        got = lookup[key(row)]
        for column, raw in got.items():
            value = row.get(column)
            if value is None:
                assert raw == "", (key(row), column, raw)
            elif isinstance(value, (int, float)):
                assert math.isclose(float(raw), value, rel_tol=1e-12, abs_tol=1e-14), (key(row), column)
            else:
                assert raw == value, (key(row), column)

def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)

def mean_row(rows, identity):
    out = dict(identity)
    for column in rows[0]:
        values = [r.get(column) for r in rows]
        if column not in ("seed", "step") and all(isinstance(v, (int, float)) for v in values):
            out[column] = statistics.mean(values)
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=pathlib.Path, default=HERE / "results")
    parser.add_argument("--archive-grid", type=pathlib.Path, default=HERE.parent / "magic/results/grid")
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "results")
    args = parser.parse_args()
    summary = read_json(args.results / "summary.json")
    assert summary["complete"] and summary["n_runs"] == 24 and summary["n_checkpoints"] == 144
    assert summary["source_archive"] == SOURCE_ARCHIVE and not summary["errors"]
    assert summary["training_updates"] == summary["test_evaluations"] == 0
    rows, endpoints, first_failures, failures = [], [], {}, []
    replay_errors = {f"{part}_{m}": 0.0 for part in ("train", "validation") for m in ("ce", "accuracy", "auc")}
    gauge_max, gap_max, min_shape_share, logit_auc_max = 0.0, 0.0, 1.0, 0.0
    hashes, checkpoint_hashes = set(), set()
    for case in CASES:
        for seed in SEEDS:
            name = f"{case}-{seed}"
            run = read_json(args.results / (name + ".json"))
            source_path = args.archive_grid / name / "result.json"
            if not source_path.exists():
                source_path = args.archive_grid / (name + ".json")
            source = read_json(source_path)
            assert run["case"] == case and run["seed"] == seed and run["config"] == source["config"]
            assert source["status"] == "complete" and source["start_step"] == 0 and source["completed_updates"] == 20000
            assert [h["step"] for h in run["history"]] == [h["step"] for h in source["history"]] == STEPS
            for h, old in zip(run["history"], source["history"]):
                assert h["weights_sha256"] == old["weights_sha256"] == old["spectral_weights_sha256"]
                hashes.add(h["weights_sha256"])
                checkpoint_hashes.add(old["checkpoint_sha256"])
                assert set(old["metrics"]) == {"train", "validation"}
                for part in ("validation", "train"):
                    for metric, tol in (("ce", 3e-6), ("accuracy", 1 / old["metrics"][part]["n"] + 1e-9), ("auc", 1e-5)):
                        error = abs(h[part]["archive_replay_" + metric] - old["metrics"][part][metric])
                        assert error <= tol
                        assert math.isclose(error, h[part]["archive_replay_" + metric + "_abs_error"], abs_tol=1e-14)
                        replay_errors[f"{part}_{metric}"] = max(replay_errors[f"{part}_{metric}"], error)
                    assert abs(h[part]["ce"] - old["metrics"][part]["ce"]) < 3e-6
                    assert abs(h[part]["accuracy"] - old["metrics"][part]["accuracy"]) <= 1 / old["metrics"][part]["n"] + 1e-9
                    auc_gap = abs(h[part]["auc"] - old["metrics"][part]["auc"])
                    logit_auc_max = max(logit_auc_max, auc_gap)
                    if auc_gap >= 1e-5:
                        failures.append(dict(run=name, step=h["step"], partition=part, difference=auc_gap))
                        first_failures.setdefault(name, h["step"])
                for g in h["gradient_geometry"]["layers"].values():
                    gauge_max = max(gauge_max, abs(g.get("exact_relu_gauge_residual", 0.0)))
                row = flatten(run, h)
                rows.append(row)
                if h["step"] == 20000:
                    endpoints.append(row)
                    for kind, values in h["scale_interventions"].items():
                        if kind.startswith("compensated"):
                            gap_max = max(gap_max, values["max_abs_logit_difference"])
                    g = h["hypothetical_next_update"]["layers"]["fc2_weight"]
                    min_shape_share = min(min_shape_share, (g["relative_shape_update"] / g["relative_update_norm"]) ** 2)
    assert first_failures == ORIGINAL_FAILURES and all(x["partition"] == "validation" for x in failures)
    check_csv(args.results / "trajectory.csv", rows)
    check_csv(args.results / "endpoints.csv", endpoints)
    for case_summary in summary["cases"]:
        selected = [r for r in endpoints if r["case"] == case_summary["case"]]
        assert case_summary["n"] == len(selected) == 2
        for column, saved in case_summary["means"].items():
            expected = statistics.mean(r[column] for r in selected)
            assert math.isclose(saved, expected, rel_tol=1e-12, abs_tol=1e-14), (case_summary["case"], column)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = dict(source_archive=SOURCE_ARCHIVE, diagnostic_results_commit=RESULTS_COMMIT,
                 diagnostic_workflow=45846514114, completed_runs=24, checkpoint_records=144,
                 unique_weight_states=len(hashes), unique_expected_checkpoint_hashes=len(checkpoint_hashes),
                 neural_training_updates=0, test_evaluations=0, analysis_errors=[],
                 csv_and_json_agree=True, summary_means_agree=True,
                 original_failed_runs_explained=len(first_failures), original_failure_steps_match=True,
                 old_auc_gate_failing_validation_records=len(failures), max_logit_vs_probability_auc_difference=logit_auc_max,
                 maximum_archive_replay_errors=replay_errors,
                 maximum_compensated_binary_logit_gap_change=gap_max,
                 maximum_relu_gauge_gradient_residual=gauge_max,
                 minimum_fc2_hypothetical_shape_squared_share=min_shape_share,
                 failed_delta_fits=sum(r["delta_ww_status"] != "success" for r in endpoints),
                 checkpoint_binary_verification="Passed in the executed diagnose.py workflow; this saved-record audit does not reload checkpoint binaries.")
    (args.output / "audit_verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_csv(args.output / "case_means.csv", [mean_row([r for r in endpoints if r["case"] == c], dict(case=c, n=2)) for c in CASES])
    write_csv(args.output / "trajectory_means.csv", [mean_row([r for r in rows if r["case"] == c and r["step"] == s], dict(case=c, step=s, n=2)) for c in CASES for s in STEPS])
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()

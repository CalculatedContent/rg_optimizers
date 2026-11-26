"""Post-hoc units and baseline audit; never trains or changes the neural cohort."""
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
parser.add_argument("--source", type=Path, default=ROOT / "verified_remote")
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
data = np.load(args.data_dir / "prepared.npz", allow_pickle=False)
y_train = data["y_train_original"].astype(np.float64)
y_test = data["y_test_original"].astype(np.float64)
variance = float(np.var(y_test, ddof=0))


def measures(prediction):
    residual = np.asarray(prediction, dtype=np.float64) - y_test
    mse = float(np.mean(residual**2))
    return {
        "rmse_100000_usd": float(np.sqrt(mse)),
        "mae_100000_usd": float(np.mean(np.abs(residual))),
        "r2": 1 - mse / variance,
        "mape_percent": float(np.mean(np.abs(residual) / y_test) * 100),
    }


def seed_stats(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "sample_sd": sd,
            "ci95_halfwidth": half, "ci95": [mean - half, mean + half],
            "seed_values": values.tolist()}


train_design = np.column_stack((np.ones(len(y_train)), data["x_train"].astype(np.float64)))
test_design = np.column_stack((np.ones(len(y_test)), data["x_test"].astype(np.float64)))
coefficient, _, _, _ = np.linalg.lstsq(train_design, y_train, rcond=None)
output = {
    "scope": "Post-hoc descriptive audit of the already completed frozen cohort; no recipe selection or neural training.",
    "target": "Median house value by census block group, 1990 Census dataset, in units of 100000 USD.",
    "test_count": len(y_test),
    "test_target_mean_100000_usd": float(y_test.mean()),
    "test_target_median_100000_usd": float(np.median(y_test)),
    "test_target_population_sd_100000_usd": float(np.std(y_test)),
    "baselines": {
        "training_mean": measures(np.full_like(y_test, y_train.mean())),
        "ordinary_least_squares": measures(test_design @ coefficient),
    },
    "baseline_protocol": "Both baselines fit the original training partition only. OLS uses all eight standardized input features plus an intercept, no hyperparameters or test selection. Reported after the frozen experiment for context only.",
    "endpoints": [],
    "uncertainty": "Equal seed weights, sample SD (ddof=1), two-sided 95% Student-t intervals with df=4. A fixed train/validation/test partition is shared, so these quantify training-seed variation, not test-sample or split uncertainty.",
}
for step in (2000, 20000):
    for arm in ("sgdm", "adamw", "modified_adamw"):
        records = []
        for seed in range(1340, 1345):
            record_path = args.source / "runs" / f"seed_{seed}" / arm / "result.json"
            if not record_path.exists():
                record_path = args.source / str(seed) / arm / "result.json"
            r = json.loads(record_path.read_text())
            records.append(next(h["test"] for h in r["history"] if h["step"] == step))
        output["endpoints"].append({
            "step": step, "optimizer": arm,
            "rmse_usd": seed_stats([r["rmse"] * 100000 for r in records]),
            "mae_usd": seed_stats([r["mae"] * 100000 for r in records]),
            "r2": seed_stats([1 - r["mse"] / variance for r in records]),
            "rmse_as_percent_of_test_mean": seed_stats([r["rmse"] / y_test.mean() * 100 for r in records]),
            "normalization_note": "RMSE divided by the test mean is not MAPE; MAPE requires each individual prediction and target.",
        })
destination = args.output / "baselines_and_units.json"
destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
print(json.dumps({"baselines": output["baselines"], "test_mean_usd": y_test.mean() * 100000,
                  "endpoints": [{"step": row["step"], "optimizer": row["optimizer"],
                                 "rmse_usd": row["rmse_usd"]["mean"],
                                 "mae_usd": row["mae_usd"]["mean"],
                                 "r2": row["r2"]["mean"], "r2_ci_half": row["r2"]["ci95_halfwidth"]}
                                for row in output["endpoints"]]}, indent=2))

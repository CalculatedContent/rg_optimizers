"""Independent evaluation of all retained GitHub checkpoint weights; no training.

This audit does not import the training model or metric routine. It rebuilds
the forward pass from named state-dict tensors, reverses the archived target
scaler, and evaluates every retained endpoint against the fixed split.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import subprocess
import numpy as np
import torch
from scipy.stats import t

BASE = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=BASE / "repo")
parser.add_argument("--data-dir", type=Path, default=BASE / "data")
parser.add_argument("--source", type=Path, default=BASE / "verified_remote")
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--commit", default="5b11a1a4ee37b87113d35ef7b08973b5ecf4e288")
args = parser.parse_args()
OUT, REPO, COMMIT = args.output, args.repo, args.commit
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "checkpoints").mkdir(exist_ok=True)
PREFIX = "california_housing/results/2026-11-26_three_optimizers/runs"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
data = np.load(args.data_dir / "prepared.npz", allow_pickle=False)
manifest_hash = hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest()
rows = []
checks = []

def get_checkpoint(seed, arm, step):
    repo_path = f"{PREFIX}/seed_{seed}/{arm}/checkpoint_step_{step}.pt"
    local = OUT / "checkpoints" / f"seed_{seed}_{arm}_{step}.pt"
    if not local.exists():
        blob = subprocess.run(["git", "show", f"{COMMIT}:{repo_path}"], cwd=REPO,
                              capture_output=True, check=True).stdout
        local.write_bytes(blob)
    return seed, arm, step, local, repo_path

tasks = [(seed, arm, step) for seed in range(1340, 1345)
         for arm in ("sgdm", "adamw", "modified_adamw") for step in (2000, 20000)]
# Resolve immutable Git objects concurrently; no checkout or repository writes.
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    retained = list(pool.map(lambda args: get_checkpoint(*args), tasks))

def model_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

@torch.no_grad()
def predict(state, x, dtype=torch.float32):
    x = torch.from_numpy(x).to(dtype=dtype)
    outputs = []
    for chunk in x.split(1024):
        z = chunk
        for index in range(1, 6):
            z = torch.nn.functional.linear(z, state[f"fc{index}.weight"].to(dtype=dtype),
                                            state[f"fc{index}.bias"].to(dtype=dtype))
            if index < 5:
                z = torch.clamp_min(z, 0)
        outputs.append(z.squeeze(1).double().numpy())
    return np.concatenate(outputs)

for seed, arm, step, local, repo_path in retained:
    # This is the experiment's own immutable, trusted checkpoint, including RNG.
    checkpoint = torch.load(local, map_location="cpu", weights_only=False)
    record_path = args.source / "runs" / f"seed_{seed}" / arm / "result.json"
    if not record_path.exists():
        record_path = args.source / str(seed) / arm / "result.json"
    record = json.loads(record_path.read_text())
    saved = next(v for v in record["history"] if v["step"] == step)
    assert checkpoint["completed_updates"] == step
    assert checkpoint["identity"]["seed"] == seed
    assert checkpoint["identity"]["optimizer"] == arm
    assert checkpoint["identity"]["data_manifest_sha256"] == manifest_hash
    assert all(checkpoint["identity"][key] == record[key] for key in checkpoint["identity"])
    assert model_hash(checkpoint["model"]) == checkpoint["result"]["final_weights_sha256"]
    if step == 20000:
        assert model_hash(checkpoint["model"]) == record["final_weights_sha256"]
    checked = {"seed": seed, "optimizer": arm, "step": step, "repo_path": repo_path,
               "checkpoint_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
               "identity_matches": True, "weight_hash_matches": True}
    checks.append(checked)
    for split in ("train", "validation", "test"):
        pred_std = predict(checkpoint["model"], data[f"x_{split}"])
        predictions = pred_std * float(data["y_scale"]) + float(data["y_mean"])
        target = data[f"y_{split}_original"]
        residual = predictions - target
        squared_error = np.square(residual)
        total_squared_error = float(np.sum(squared_error))
        rmse = math.sqrt(total_squared_error / target.size)
        mae = float(np.sum(np.abs(residual)) / target.size)
        mse = total_squared_error / target.size
        r2 = 1 - total_squared_error / float(np.sum((target - np.mean(target)) ** 2))
        row = {"seed": seed, "optimizer": arm, "step": step, "split": split,
               "n": int(target.size), "rmse": rmse, "mae": mae, "mse": mse,
               "r2": r2, "rmse_dollars": rmse * 100000, "mae_dollars": mae * 100000,
               "reported_rmse": saved[split]["rmse"], "reported_mae": saved[split]["mae"],
               "reported_mse": saved[split]["mse"],
               "rmse_difference": rmse - saved[split]["rmse"],
               "mae_difference": mae - saved[split]["mae"],
               "mse_difference": mse - saved[split]["mse"],
               "prediction_min": float(predictions.min()), "prediction_max": float(predictions.max()),
               "median_absolute_error": float(np.median(np.abs(residual))),
               "mape_percent": float(100 * np.mean(np.abs(residual) / np.abs(target))),
               "median_absolute_percentage_error_percent": float(100 * np.median(np.abs(residual) / np.abs(target))),
               "mae_over_target_mean_percent": float(100 * mae / np.mean(target)),
               "max_absolute_error": float(np.max(np.abs(residual)))}
        if split == "test":
            alternate = predict(checkpoint["model"], data["x_test"], torch.float64)
            alternate = alternate * float(data["y_scale"]) + float(data["y_mean"])
            row["float64_rmse"] = float(np.sqrt(np.mean(np.square(alternate - target))))
            row["float64_vs_float32_rmse_difference"] = row["float64_rmse"] - rmse
            row["float64_max_prediction_difference"] = float(np.max(np.abs(alternate - predictions)))
        rows.append(row)
    print(f"checked {seed} {arm} {step}", flush=True)

summary = []
for arm in ("sgdm", "adamw", "modified_adamw"):
    for step in (2000, 20000):
        for split in ("train", "validation", "test"):
            group = [r for r in rows if (r["optimizer"], r["step"], r["split"]) == (arm, step, split)]
            entry = {"optimizer": arm, "step": step, "split": split, "n_seeds": len(group)}
            for metric in ("rmse", "mae", "r2", "mape_percent", "median_absolute_error", "median_absolute_percentage_error_percent", "mae_over_target_mean_percent"):
                v = np.array([r[metric] for r in group])
                mean = float(np.mean(v))
                sd = float(np.std(v, ddof=1))
                halfwidth = float(t.ppf(.975, 4) * sd / math.sqrt(5))
                entry[metric] = {"mean": mean, "sample_sd": sd, "ci95_halfwidth": halfwidth,
                                 "ci95_low": mean-halfwidth, "ci95_high": mean+halfwidth}
            summary.append(entry)

audit = {"source_commit": COMMIT, "checkpoint_count": len(checks),
         "split_evaluations": len(rows), "data_manifest_sha256": manifest_hash,
         "units": "target and errors are in 100000 USD; R2 is dimensionless",
         "posthoc_descriptive": ["r2", "mape_percent", "median_absolute_error", "median_absolute_percentage_error_percent", "mae_over_target_mean_percent"],
         "max_abs_rmse_difference": max(abs(r["rmse_difference"]) for r in rows),
         "max_abs_mae_difference": max(abs(r["mae_difference"]) for r in rows),
         "max_abs_mse_difference": max(abs(r["mse_difference"]) for r in rows),
         "max_abs_test_float64_vs_float32_rmse_difference": max(abs(r["float64_vs_float32_rmse_difference"]) for r in rows if r["split"] == "test"),
         "checks": checks, "summary": summary, "rows": rows}
(OUT / "checkpoint_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
fieldnames = sorted(set().union(*(row.keys() for row in rows)))
with (OUT / "checkpoint_audit.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({k: v for k, v in audit.items() if k not in ("checks", "summary", "rows")}, indent=2))
print(json.dumps([s for s in summary if s["split"] == "test"], indent=2))

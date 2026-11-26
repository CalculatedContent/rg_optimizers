"""Descriptive audit of the large early modified-AdamW residual; no exclusions."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
BASE = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=BASE / "data")
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
data = np.load(args.data_dir / "prepared.npz", allow_pickle=False)
raw = fetch_california_housing(data_home=str(args.data_dir / "sklearn_cache"), download_if_missing=False)

@torch.no_grad()
def predict(state, x):
    z = torch.from_numpy(x)
    for i in range(1, 6):
        z = torch.nn.functional.linear(z, state[f"fc{i}.weight"], state[f"fc{i}.bias"])
        if i < 5:
            z = torch.clamp_min(z, 0)
    return z.squeeze(1).double().numpy() * float(data["y_scale"]) + float(data["y_mean"])

checkpoint = torch.load(OUT / "checkpoints/seed_1341_modified_adamw_2000.pt", map_location="cpu", weights_only=False)
pred = predict(checkpoint["model"], data["x_test"])
y = data["y_test_original"]
residual = pred - y
squared = residual**2
order = np.argsort(squared)[::-1]
top = []
for index in order[:5]:
    idx = int(data["indices_test"][index])
    top.append({"test_position": int(index), "original_dataset_row_index": idx,
                "target": float(y[index]), "prediction": float(pred[index]),
                "error_dollars": float(residual[index] * 100000),
                "fraction_test_sse": float(squared[index] / np.sum(squared)),
                "features": dict(zip(raw.feature_names, map(float, raw.data[idx]))),
                "standardized_features": dict(zip(raw.feature_names, map(float, data["x_test"][index])))})
result = {"posthoc_descriptive_only": True, "no_observations_excluded": True,
          "seed": 1341, "optimizer": "modified_adamw", "step": 2000,
          "n_test": len(y), "rmse": float(np.sqrt(np.mean(squared))),
          "top_1_fraction_test_sse": float(squared[order[:1]].sum() / squared.sum()),
          "top_5_fraction_test_sse": float(squared[order[:5]].sum() / squared.sum()),
          "top_5_residuals": top}
result["ave_occupancy_maximum_by_split"] = {
    split: {"raw_maximum": float(raw.data[data[f"indices_{split}"], 5].max()),
            "standardized_maximum": float(data[f"x_{split}"][:, 5].max())}
    for split in ("train", "validation", "test")}
test_position = int(order[0])
comparators = []
for seed in range(1340, 1345):
    for arm in ("sgdm", "adamw", "modified_adamw"):
        for step in (2000, 20000):
            check = torch.load(OUT / f"checkpoints/seed_{seed}_{arm}_{step}.pt", map_location="cpu", weights_only=False)
            p = predict(check["model"], data["x_test"][test_position:test_position+1])
            comparators.append({"seed": seed, "optimizer": arm, "step": step,
                                "prediction_on_largest_residual_row": float(p[0])})
result["same_row_comparators"] = comparators
(OUT / "outlier_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

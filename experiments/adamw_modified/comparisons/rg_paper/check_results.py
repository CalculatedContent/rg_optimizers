"""Check the winner rule, layer-count analysis, and confidence caveat from saved tables."""
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
layers = pd.read_csv(root / "optimizer_checkpoints.csv")
model_id = ["cohort", "arm", "seed", "batch", "rate", "epoch"]
assert (layers.epoch > 0).all()
assert not layers.duplicated(model_id + ["layer"]).any()
assert layers.groupby(model_id).size().eq(3).all()
assert layers.gap.eq(layers.k_erg - layers.k_pl).all()
np.testing.assert_allclose(layers.weight, np.sqrt(layers.k_pl) / (layers.D + 0.02))
grouped = layers.assign(weighted=layers.weight * layers.alpha_layer).groupby(model_id)
np.testing.assert_allclose(grouped.weighted.sum() / grouped.weight.sum(), grouped.alpha_model.first())
models = layers.query("layer=='fc1'").copy()
assert len(models) == 1170
models["correct"] = np.rint(100 * models.test_accuracy_pct).astype(int)
np.testing.assert_allclose(models.correct, 100 * models.test_accuracy_pct)
models["distance"] = (models.alpha_model - 2).abs()
models["dataset"] = models.cohort.map({"Digits sweep": "Digits", "Fashion original": "Fashion",
                                      "Fashion continuation": "Fashion"})
assert models.dataset.notna().all()
best = (models.sort_values(["correct", "distance", "epoch", "seed", "batch", "rate"],
                          ascending=[False, True, True, True, True, True])
        .groupby(["dataset", "arm"]).head(1).set_index(["dataset", "arm"]).sort_index())
expected_accuracy = {
    ("Digits", "adamw"): 98.58, ("Digits", "magnitude_graft"): 98.67, ("Digits", "sgdm"): 98.58,
    ("Fashion", "adamw"): 90.35, ("Fashion", "magnitude_graft"): 90.26, ("Fashion", "sgdm"): 90.20,
}
assert best.test_accuracy_pct.round(2).to_dict() == expected_accuracy
corrected = best.loc[("Fashion", "magnitude_graft")]
assert corrected.epoch == 20 and round(corrected.alpha_model, 4) == 2.0023
print(best[["cohort", "epoch", "test_accuracy_pct", "alpha_model"]].to_string())
hidden = layers.query("epoch==120 and layer!='fc3'")
above = hidden.query("alpha_layer>2")
below = hidden.query("alpha_layer<2")
assert (len(above), sum(above.gap>0), len(below), sum(below.gap<0)) == (254, 246, 282, 133)
assert len(above) + len(below) == len(hidden) == 536
near = hidden[(hidden.alpha_layer-2).abs() <= .01]
assert (len(near), sum(near.gap>0), sum(near.gap<0), near.gap.min(), near.gap.max()) == (30,20,10,-53,68)
print("Signed gap counts verified across 536 hidden layers.")
confidence = pd.read_csv(root / "confidence_run_pairs.csv")
assert len(confidence)==292
eligible = confidence.query("flat25").dropna(subset=["gap99_60", "gap99_120"])
wider = eligible.gap99_120 > eligible.gap99_60 + 1e-10
print(eligible.assign(wider=wider).groupby([eligible.cohort.str.startswith("Digits").map({True:"Digits",False:"Fashion"}), "optimizer"]).wider.agg(["sum","size"]))

# The comparison table uses only seeds whose three optimizers finished.
central = models.query("batch==256 and rate==1 and epoch==120")
shared = central.groupby(["cohort", "seed"]).filter(lambda trial: trial.arm.nunique() == 3)
comparison = shared.groupby(["cohort", "arm"]).agg(
    seeds=("seed", "nunique"), alpha=("alpha_model", "mean"), accuracy=("test_accuracy_pct", "mean"))
for cohort, seeds, exponents, accuracies in [
        ("Digits sweep", 5, [2.025, 1.855, 2.043], [98.446, 98.220, 98.344]),
        ("Fashion original", 3, [1.992, 2.188, 2.646], [89.630, 89.777, 89.583]),
        ("Fashion continuation", 2, [2.016, 1.867, 2.067], [89.730, 89.810, 89.750])]:
    arm_order = ["sgdm", "adamw", "magnitude_graft"]
    row = comparison.loc[cohort].reindex(arm_order)
    assert row.seeds.eq(seeds).all()
    np.testing.assert_allclose(row.alpha.round(3), exponents)
    np.testing.assert_allclose(row.accuracy.round(3), accuracies)
counts = models.groupby("cohort").size()
assert counts.to_dict() == {"Digits sweep":540, "Fashion original":341, "Fashion continuation":289}
print("Table 7 shared-seed comparisons and Appendix B checkpoint counts verified.")

muon = pd.read_csv(root / "Muon_500_Reported.csv").set_index("epoch")
alpha = muon[["alpha_fc1", "alpha_fc2", "alpha_fc3"]]
np.testing.assert_allclose(alpha.mean(axis=1), muon.alpha)
np.testing.assert_allclose(alpha.loc[[200, 500]].std(axis=1).round(3), [0.339, 0.170])
print("Muon equal-weight layer averages and across-layer sample SDs verified.")

validation = pd.read_csv(root / "validation_endpoints.csv")
endpoints = models.query("epoch == 120").merge(validation, on=model_id, validate="one_to_one")
assert len(endpoints) == len(validation) == 268
winners = endpoints[endpoints.validation_accuracy_pct.eq(
    endpoints.groupby("cohort").validation_accuracy_pct.transform("max"))].set_index("cohort")
for cohort, arm, accuracy, alpha in [
        ("Digits sweep", "magnitude_graft", 98.42, 1.986),
        ("Fashion original", "sgdm", 91.06, 2.009),
        ("Fashion continuation", "magnitude_graft", 90.88, 2.102)]:
    win = winners.loc[cohort]
    assert (win.arm, round(win.validation_accuracy_pct, 2), round(win.alpha_model, 3)) == (arm, accuracy, alpha)
print("Validation-selected endpoint winners verified for all three sweeps.")

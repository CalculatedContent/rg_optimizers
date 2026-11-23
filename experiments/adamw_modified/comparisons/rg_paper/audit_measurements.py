"""Recompute paper summaries from the bundled observations. No training or network."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEY = ["seed", "batch", "rate", "arm", "epoch"]
COHORTS = {"digits": "Digits sweep", "fashion_original": "Fashion original",
           "fashion_continuation": "Fashion continuation"}


def verify_spectra(layers):
    """Recount modes after normalizing each layer by its full-spectrum mean."""
    checked = 0
    for filename, cohort in COHORTS.items():
        with np.load(ROOT / "inputs" / f"spectra_{filename}.npz") as stored:
            keys = pd.MultiIndex.from_arrays([stored[name] for name in KEY], names=KEY)
            assert keys.is_unique
            start = 0
            for layer in ["fc1", "fc2", "fc3"]:
                rows = layers[(layers.cohort == cohort) & (layers.layer == layer)].set_index(KEY)
                assert keys.difference(rows.index).empty and rows.index.difference(keys).empty
                rows = rows.loc[keys]
                widths = rows.M.unique()
                assert len(widths) == 1
                end = start + int(widths[0])
                values = np.sort(stored["eigenvalues"][:, start:end].astype(float))[:, ::-1]
                assert np.isfinite(values).all() and (values >= 0).all()
                with np.errstate(divide="ignore"):
                    cumulative = np.log(values / values.mean(axis=1, keepdims=True)).cumsum(axis=1)
                crossing = cumulative < 0
                rank = np.where(crossing.any(axis=1), crossing.argmax(axis=1) + 1, values.shape[1])
                # Independently select the cutoff as well as fitting the slope.
                # The left empirical CDF matches the archived estimator.
                fits, distances, tail_counts = [], [], []
                for spectrum in values:
                    positive = spectrum[spectrum > 1e-12][::-1]
                    candidates = []
                    for k in range(len(positive), (8 if layer == "fc3" else 2) - 1, -1):
                        tail = positive[-k:]
                        logs = np.log(tail / tail[0])
                        if logs.sum() <= 1e-12:
                            continue
                        estimate = 1 + (k - int(layer == "fc3")) / logs.sum()
                        fitted_cdf = -np.expm1((1 - estimate) * logs)
                        distance = np.abs(fitted_cdf - np.arange(k)/k).max()
                        candidates.append((distance, k, estimate))
                    distance, k, estimate = min(candidates, key=lambda candidate: candidate[0])
                    fits.append(estimate)
                    distances.append(distance)
                    tail_counts.append(k)
                np.testing.assert_array_equal(tail_counts, rows.k_pl)
                np.testing.assert_allclose(fits, rows.alpha_layer, rtol=1e-10)
                np.testing.assert_allclose(distances, rows.D, rtol=1e-10)
                np.testing.assert_allclose(np.sqrt(tail_counts)/(np.asarray(distances)+0.02), rows.weight)
                np.testing.assert_array_equal(rank, rows.k_erg)
                np.testing.assert_array_equal(rank - rows.k_pl, rows.gap)
                checked += len(rows)
                start = end
            assert start == stored["eigenvalues"].shape[1]
    assert checked == len(layers) == 3510
    return checked


def verify_calibration():
    counts = pd.read_csv(ROOT / "inputs/calibration_counts.csv")
    pairs = pd.read_csv(ROOT / "confidence_run_pairs.csv")
    keys = ["cohort", "seed", "setting", "optimizer"]
    counts["accuracy_change_input"] = 100 * counts.correct / counts.n
    counts["gap"] = ((counts.confidence_sum99 - counts.correct99) /
                     counts.n99.replace(0, np.nan)).abs()
    assert len(counts) == 2 * len(pairs) == 584
    for epoch, which in [(60, "before"), (120, "after")]:
        c = counts[counts.epoch == epoch]
        paired = pairs.merge(c, on=keys, validate="one_to_one")
        assert len(paired) == len(pairs)
        np.testing.assert_array_equal(paired.n99, paired[f"n99_{epoch}"])
        np.testing.assert_allclose(paired.gap, paired[f"gap99_{epoch}"], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(paired.accuracy_change_input, paired[f"acc_{which}"], atol=1e-9)
    table = counts.pivot(index=keys, columns="epoch", values=["gap", "accuracy_change_input"])
    stable = (table.accuracy_change_input[120] - table.accuracy_change_input[60]).abs() <= .25 + 1e-9
    gap = table.gap[stable].dropna()
    eligible = (gap[120] > gap[60] + 1e-10).rename("widened").reset_index()
    eligible["dataset"] = np.where(eligible.cohort.str.startswith("Digits"), "Digits", "Fashion")
    report = eligible.groupby(["dataset", "optimizer"]).widened.agg(["sum", "size"])
    assert list(report.itertuples(index=False, name=None)) == [
        (45, 45), (21, 40), (15, 46), (32, 32), (23, 23), (20, 20)]
    losses = {}
    scale_factors, scale_increases, total_increases = [], [], []
    trace_shares = {"fc1": [], "fc2": []}
    for seed in (33001, 33002):
        with np.load(ROOT / "inputs" / f"fashion_scores_{seed}.npz") as saved:
            labels = saved["labels"]
            def center(logits):
                logits = logits.astype(float)
                return logits - logits.mean(axis=1, keepdims=True)
            def image_loss(logits):
                return np.logaddexp.reduce(logits, axis=1) - logits[np.arange(len(labels)), labels]
            v60, v120 = [center(saved[f"validation_{e}"]) for e in (60, 120)]
            scale = (v60 * v120).sum() / (v60 * v60).sum()
            scale_factors.append(scale)
            test60, test120 = [center(saved[f"test_{e}"]) for e in (60, 120)]
            before, after = image_loss(test60), image_loss(test120)
            losses[seed] = (before, after)
            scale_increases.append((image_loss(scale * test60) - before).mean())
            total_increases.append((after - before).mean())
            for layer in trace_shares:
                pair = []
                for epoch in (60, 120):
                    values = saved[f"{layer}_{epoch}"]
                    pair.append(100 * np.sort(values)[-8:].sum() / values.sum())
                trace_shares[layer].append(pair)
    hard, total = [], []
    for seed, other in [(33001, 33002), (33002, 33001)]:
        before, after = losses[seed]
        delta = after - before
        hardest = np.argsort(-losses[other][0], kind="stable")[:1000]
        hard.append(delta[hardest].sum())
        total.append(delta.sum())
    share = 100 * sum(hard) / sum(total)
    explained = 100 * sum(scale_increases) / sum(total_increases)
    np.testing.assert_array_equal(np.round(scale_factors, 2), [1.41, 1.42])
    assert round(share) == 91 and round(explained) == 58
    focused = {"hardest_tenth_pct": round(share, 2), "validation_logit_scales": np.round(scale_factors, 4).tolist(),
               "log_loss_rise_reproduced_pct": round(explained, 2),
               "top8_trace_shares_pct_at_60_120": {l: np.mean(v,axis=0).round(3).tolist() for l,v in trace_shares.items()}}
    return report.reset_index().rename(columns={"sum": "widened", "size": "eligible"}).to_dict("records"), focused


def build_summary():
    layers = pd.read_csv(ROOT / "optimizer_checkpoints.csv")
    checked = verify_spectra(layers)
    cal, focused = verify_calibration()
    models = layers.query("layer == 'fc1'").copy()
    models["dataset"] = np.where(models.cohort.str.startswith("Digits"), "Digits", "Fashion")
    models["distance"] = abs(models.alpha_model - 2)
    ranks = models.sort_values(["test_accuracy_pct", "distance", "epoch", "seed", "batch", "rate"],
                               ascending=[False, True, True, True, True, True])
    best = ranks.groupby(["dataset", "arm"]).head(1)
    ends = models.query("batch==256 and rate==1 and epoch==120")
    shared = ends.groupby(["cohort", "seed"]).filter(lambda v: v.arm.nunique() == 3)
    central = shared.groupby(["cohort", "arm"]).agg(seeds=("seed", "size"),
             alpha_w=("alpha_model", "mean"), test_accuracy_pct=("test_accuracy_pct", "mean"))
    validation = pd.read_csv(ROOT / "validation_endpoints.csv")
    final = models.query("epoch==120").merge(validation, on=["cohort", *KEY], validate="one_to_one")
    val_best = final[final.validation_accuracy_pct == final.groupby("cohort").validation_accuracy_pct.transform("max")]
    h = layers.query("epoch==120 and layer!='fc3'").copy()
    nearest = h.loc[(h.alpha_layer - 2).abs().idxmin()]
    digits = layers.query("cohort=='Digits sweep' and epoch==120 and batch==256 and rate==1")
    delta = digits.pivot(index=["layer","seed"], columns="arm", values="gap").abs()
    shift = (delta.magnitude_graft - delta.adamw).groupby(level="layer")
    ci = 2.7764451051977987 * shift.std() / np.sqrt(5)  # Student t, df=4
    gap_effects = pd.DataFrame({"change":shift.mean(), "low95":shift.mean()-ci, "high95":shift.mean()+ci})
    np.testing.assert_allclose(gap_effects.loc[["fc1","fc2"]].round(1), [[15.2,5.7,24.7],[-39.2,-50.7,-27.7]])
    muon = pd.read_csv(ROOT/"Muon_500_Reported.csv").set_index("epoch").loc[[200,500]]
    alpha = muon[["alpha_fc1","alpha_fc2","alpha_fc3"]]
    return {
        "model_checkpoints": len(models), "layer_spectra_recounted": checked,
        "checkpoints_by_schedule": models.cohort.value_counts().to_dict(),
        "central_epoch120": central.round(3).reset_index().to_dict("records"),
        "best_test_checkpoints": best[["dataset","arm","cohort","seed","batch","rate","epoch","test_accuracy_pct","alpha_model"]].round(6).sort_values(["dataset","arm"]).to_dict("records"),
        "validation_winners": val_best[["cohort","seed","batch","rate","arm","validation_accuracy_pct","test_accuracy_pct","alpha_model"]].round(6).to_dict("records"),
        "erg_gap": {"epoch120_hidden_layers": len(h),
                    "positive_above_2": int(((h.alpha_layer>2)&(h.gap>0)).sum()), "above_2": int((h.alpha_layer>2).sum()),
                    "negative_below_2": int(((h.alpha_layer<2)&(h.gap<0)).sum()), "below_2": int((h.alpha_layer<2).sum()),
                    "closest_layer_alpha": round(nearest.alpha_layer,6), "closest_gap": int(nearest.gap),
                    "paired_digit_change_in_absolute_gap": gap_effects.round(3).reset_index().to_dict("records")},
        "muon_unweighted": pd.DataFrame({"alpha":alpha.mean(axis=1),"sd_across_layers":alpha.std(axis=1),
                                          "test_accuracy_pct":100*muon.test_acc}).round(3).reset_index().to_dict("records"),
        "confidence_pairs": 292, "high_confidence": cal, "paired_fashion_adamw": focused,
    }


if __name__ == "__main__":
    print(json.dumps(build_summary(), ensure_ascii=False, indent=2, allow_nan=False))
